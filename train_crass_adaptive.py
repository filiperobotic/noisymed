"""
Training Script with CRASS Adaptive for Noisy Labels.

CRASS Adaptive extends CRASS by dynamically adjusting lambda based on the
noise rate.  The hypothesis is that a fixed high lambda (e.g. 20) is too
aggressive under heavy noise, retaining too many noisy positive-class
samples.  Lowering lambda proportionally to the noise rate yields a better
risk-accuracy trade-off.

    lambda_adaptive = lambda_max * (1 - noise_rate)

Theoretical grounding:
    - At noise_rate=0:   lambda = lambda_max  (full clinical protection)
    - At noise_rate=0.2: lambda = 0.8 * lambda_max
    - At noise_rate=0.4: lambda = 0.6 * lambda_max
    - At noise_rate=0.6: lambda = 0.4 * lambda_max

Three adaptive strategies are provided (selected via --adaptive_strategy):
    v1: Linear interpolation  lambda_max -> lambda_min over [0, 0.5]
    v2: Proportional decay    lambda_max * (1 - noise_rate)
    v3: Lookup table          hand-tuned per noise rate

Everything else is identical to train_crass.py (GMM filtering + CRASS
thresholds from Proposition 1).

Usage:
    python train_crass_adaptive.py --dataset dermamnist_bin --noise_rate 0.2
    python train_crass_adaptive.py --dataset dermamnist_bin --noise_rate 0.4 --adaptive_strategy v2
    python train_crass_adaptive.py --dataset dermamnist_bin --noise_rate 0.4 --lambda_max 20 --weighted_loss
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import json
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, average_precision_score

from dataloaders import get_filter_dataloaders, DATASET_CONFIG


# ═══════════════════════════════════════════════════════════════════════════
# Helpers shared with other training scripts
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_class_weights(labels, device):
    labels = torch.as_tensor(labels).long().view(-1)
    labels = labels[labels >= 0]
    if labels.numel() == 0:
        return None
    n0 = int((labels == 0).sum().item())
    n1 = int((labels == 1).sum().item())
    n = n0 + n1
    if n0 == 0 or n1 == 0:
        return None
    w0 = n / (2.0 * n0)
    w1 = n / (2.0 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


def get_model(num_classes=2, in_channels=1, pretrained=False):
    model = models.resnet18(pretrained=pretrained)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def compute_confusion_matrix(outputs, targets, positive_class=1):
    _, predicted = torch.max(outputs, 1)
    TP = ((predicted == positive_class) & (targets == positive_class)).sum().item()
    TN = ((predicted != positive_class) & (targets != positive_class)).sum().item()
    FP = ((predicted == positive_class) & (targets != positive_class)).sum().item()
    FN = ((predicted != positive_class) & (targets == positive_class)).sum().item()
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}


def compute_bac(cm):
    TP, TN, FP, FN = cm['TP'], cm['TN'], cm['FP'], cm['FN']
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    return (sensitivity + specificity) / 2.0


def compute_clinical_risk(cm, lambda_risk):
    """Risk = (lambda * FN + FP) / N"""
    N = cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']
    if N == 0:
        return 0.0
    return (lambda_risk * cm['FN'] + cm['FP']) / N


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive Lambda Strategies
# ═══════════════════════════════════════════════════════════════════════════

def compute_adaptive_lambda_v1(noise_rate, lambda_max=20, lambda_min=1):
    """Linear interpolation: lambda decreases linearly from lambda_max to
    lambda_min as noise_rate goes from 0.0 to 0.5.

    - noise_rate=0.0 -> lambda = lambda_max (20)
    - noise_rate=0.5 -> lambda = lambda_min (1)
    """
    lam = lambda_max - (lambda_max - lambda_min) * (noise_rate / 0.5)
    return max(lambda_min, min(lambda_max, lam))


def compute_adaptive_lambda_v2(noise_rate, lambda_max=20, lambda_min=1):
    """Proportional decay: lambda = lambda_max * (1 - noise_rate).

    - noise_rate=0.0 -> lambda = 20
    - noise_rate=0.2 -> lambda = 16
    - noise_rate=0.4 -> lambda = 12
    - noise_rate=0.6 -> lambda = 8
    """
    lam = lambda_max * (1.0 - noise_rate)
    return max(lambda_min, lam)


def compute_adaptive_lambda_v3(noise_rate, lambda_max=20, lambda_min=1):
    """Lookup table with hand-tuned values per noise level.

    Designed to be aggressive at low noise and conservative at high noise.
    """
    lambda_table = {
        0.0: 20,
        0.1: 18,
        0.2: 15,
        0.3: 12,
        0.4: 8,
        0.5: 5,
        0.6: 3,
    }
    closest = min(lambda_table.keys(), key=lambda x: abs(x - noise_rate))
    return max(lambda_min, min(lambda_max, lambda_table[closest]))


ADAPTIVE_STRATEGIES = {
    'v1': compute_adaptive_lambda_v1,
    'v2': compute_adaptive_lambda_v2,
    'v3': compute_adaptive_lambda_v3,
}


# ═══════════════════════════════════════════════════════════════════════════
# CRASS: Clinical Risk-Aware Sample Selection (with adaptive lambda)
# ═══════════════════════════════════════════════════════════════════════════

def compute_optimal_thresholds(lambda_risk, c_noise=1.0, c_fp=1.0):
    """Proposition 1: theta*(y) = C_noise / (C_noise + C_y)."""
    c_fn = lambda_risk * c_fp
    theta_neg = c_noise / (c_noise + c_fp)   # = 0.5
    theta_pos = c_noise / (c_noise + c_fn)   # = 1 / (1 + lambda)
    return theta_neg, theta_pos


def eval_train_losses(model, eval_loader, device, n_samples):
    """Compute per-sample cross-entropy loss (unreduced)."""
    model.eval()
    CE = nn.CrossEntropyLoss(reduction='none')
    losses = torch.zeros(n_samples)
    with torch.no_grad():
        for images, labels, indices in eval_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = CE(outputs, labels)
            for b in range(images.size(0)):
                losses[indices[b]] = loss[b].cpu()
    return losses


def crass_filter_per_class(losses, labels, all_loss_history, noise_rate,
                           lambda_risk, positive_class=1):
    """Fit 2-component GMM per class and apply CRASS thresholds.

    Identical to train_crass.py but receives lambda_risk which may have been
    adapted.
    """
    labels = np.array(labels).flatten()
    n_samples = len(labels)
    unique_classes = np.unique(labels[labels >= 0])

    all_loss_history.append(losses.clone())

    theta_neg, theta_pos = compute_optimal_thresholds(lambda_risk)

    class_thresholds = {}
    for cls in unique_classes:
        cls = int(cls)
        if cls == positive_class:
            class_thresholds[cls] = theta_pos
        else:
            class_thresholds[cls] = theta_neg

    clean_mask = np.zeros(n_samples, dtype=bool)
    filter_info = {
        'lambda_risk': lambda_risk,
        'theta_neg': theta_neg,
        'theta_pos': theta_pos,
        'positive_class': positive_class,
        'per_class': {},
    }

    for cls in unique_classes:
        cls = int(cls)
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) == 0:
            continue

        cls_losses = losses[cls_idx]

        # Min-max normalise within class
        l_min = cls_losses.min()
        l_max = cls_losses.max()
        if l_max - l_min > 1e-8:
            cls_losses_norm = (cls_losses - l_min) / (l_max - l_min)
        else:
            cls_losses_norm = torch.zeros_like(cls_losses)

        # For very high noise, average over last 5 epochs
        if noise_rate >= 0.9 and len(all_loss_history) >= 5:
            history_stack = torch.stack(
                [h[cls_idx] for h in all_loss_history[-5:]]
            )
            avg_losses = history_stack.mean(0)
            a_min = avg_losses.min()
            a_max = avg_losses.max()
            if a_max - a_min > 1e-8:
                input_loss = ((avg_losses - a_min) /
                              (a_max - a_min)).reshape(-1, 1).numpy()
            else:
                input_loss = np.zeros((len(cls_idx), 1))
        else:
            input_loss = cls_losses_norm.reshape(-1, 1).numpy()

        # Fit GMM
        gmm = GaussianMixture(
            n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4
        )
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        clean_component = gmm.means_.argmin()
        prob_clean = prob[:, clean_component]

        # CRASS threshold
        threshold = class_thresholds[cls]
        cls_clean_mask = prob_clean > threshold
        clean_mask[cls_idx[cls_clean_mask]] = True

        n_total_cls = len(cls_idx)
        n_clean_cls = int(cls_clean_mask.sum())
        is_positive = (cls == positive_class)
        filter_info['per_class'][cls] = {
            'total': n_total_cls,
            'selected': n_clean_cls,
            'filtered': n_total_cls - n_clean_cls,
            'keep_ratio': n_clean_cls / n_total_cls if n_total_cls > 0 else 0.0,
            'threshold': threshold,
            'is_positive_class': is_positive,
            'gmm_means': gmm.means_.flatten().tolist(),
        }

    clean_indices = np.where(clean_mask)[0]
    return clean_indices, all_loss_history, filter_info


# ═══════════════════════════════════════════════════════════════════════════
# Training loops (identical to train_crass.py)
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch_full(model, train_loader, optimizer, device,
                     class_weights=None, positive_class=1):
    """Train one epoch on ALL samples (warmup)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []

    for images, labels, _ in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if class_weights is not None:
            loss = F.cross_entropy(outputs, labels, weight=class_weights)
        else:
            loss = F.cross_entropy(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_outputs.append(outputs.detach())
        all_targets.append(labels.detach())

    epoch_loss = running_loss / total
    accuracy = correct / total
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    cm = compute_confusion_matrix(all_outputs, all_targets, positive_class)

    return epoch_loss, accuracy, cm


def train_epoch_filtered(model, train_dataset, clean_indices, batch_size,
                         optimizer, device, class_weights=None, seed=42,
                         positive_class=1):
    """Train one epoch on the filtered (clean) subset."""
    subset = Subset(train_dataset, clean_indices)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []

    for batch in loader:
        images, labels, _ = batch
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if class_weights is not None:
            loss = F.cross_entropy(outputs, labels, weight=class_weights)
        else:
            loss = F.cross_entropy(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_outputs.append(outputs.detach())
        all_targets.append(labels.detach())

    epoch_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    all_outputs = torch.cat(all_outputs) if all_outputs else torch.tensor([])
    all_targets = torch.cat(all_targets) if all_targets else torch.tensor([])
    cm = (compute_confusion_matrix(all_outputs, all_targets, positive_class)
          if total > 0 else {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0})

    return epoch_loss, accuracy, cm


def evaluate(model, data_loader, device, class_weights=None,
             positive_class=1):
    """Evaluate model on validation/test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            if class_weights is not None:
                loss = F.cross_entropy(outputs, labels, weight=class_weights)
            else:
                loss = F.cross_entropy(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_outputs.append(outputs)
            all_targets.append(labels)

    epoch_loss = running_loss / total
    accuracy = correct / total

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    cm = compute_confusion_matrix(all_outputs, all_targets, positive_class)
    bac = compute_bac(cm)

    probs = F.softmax(all_outputs, dim=1)[:, positive_class].cpu().numpy()
    targets_np = (all_targets == positive_class).long().cpu().numpy()
    auc_roc = roc_auc_score(targets_np, probs)
    auprc = average_precision_score(targets_np, probs)

    return epoch_loss, accuracy, cm, bac, auc_roc, auprc


# ═══════════════════════════════════════════════════════════════════════════
# Main training routine
# ═══════════════════════════════════════════════════════════════════════════

def train(args):
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = DATASET_CONFIG[args.dataset]
    positive_class = config.get('positive_class', 1)
    print(f"Positive class: {positive_class} "
          f"({config['class_names'].get(positive_class, '?')})")

    # ── Compute adaptive lambda ──────────────────────────────────────────
    strategy_fn = ADAPTIVE_STRATEGIES[args.adaptive_strategy]
    lambda_adaptive = strategy_fn(
        args.noise_rate, args.lambda_max, args.lambda_min
    )

    theta_neg, theta_pos = compute_optimal_thresholds(lambda_adaptive)

    print(f"\nCRASS Adaptive Configuration:")
    print(f"  Strategy:       {args.adaptive_strategy}")
    print(f"  noise_rate:     {args.noise_rate}")
    print(f"  lambda_max:     {args.lambda_max}")
    print(f"  lambda_min:     {args.lambda_min}")
    print(f"  lambda_adaptive:{lambda_adaptive:.2f}")
    print(f"  theta_neg*:     {theta_neg:.4f}")
    print(f"  theta_pos*:     {theta_pos:.4f}")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = (
        f"{args.dataset}_noise{args.noise_rate}"
        f"_crass_adaptive"
        f"_{args.adaptive_strategy}"
        f"_lmax{args.lambda_max}"
        f"_warmup{args.warmup_epochs}"
        f"_{'weighted' if args.weighted_loss else 'standard'}"
        f"_{timestamp}"
    )
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save full config including adaptive info
    config_dict = vars(args).copy()
    config_dict['lambda_adaptive'] = lambda_adaptive
    config_dict['theta_neg'] = theta_neg
    config_dict['theta_pos'] = theta_pos
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Data ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Loading {args.dataset} dataset...")
    train_loader, eval_loader, val_loader, test_loader = get_filter_dataloaders(
        dataset=args.dataset, noise_rate=args.noise_rate,
        batch_size=args.batch_size, seed=args.seed, data_dir=args.data_dir
    )

    train_dataset = train_loader.dataset
    n_train = len(train_dataset)
    train_labels = np.array(train_dataset.labels).flatten()

    print(f"Train samples: {n_train}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    # Class weights ────────────────────────────────────────────────────────
    class_weights = None
    if args.weighted_loss:
        class_weights = infer_class_weights(train_labels, device)
        if class_weights is not None:
            print(f"\nUsing balanced class weights: "
                  f"{class_weights.cpu().tolist()}")
    else:
        print("\nUsing standard Cross Entropy loss")

    # Model ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Creating ResNet18 model...")
    model = get_model(
        num_classes=config['num_classes'],
        in_channels=config['in_channels'],
        pretrained=args.pretrained
    )
    model = model.to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Training ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Starting CRASS Adaptive training")
    print(f"  Strategy:         {args.adaptive_strategy}")
    print(f"  lambda_adaptive:  {lambda_adaptive:.2f}  "
          f"(from lambda_max={args.lambda_max})")
    print(f"  theta_neg* = {theta_neg:.4f}  |  theta_pos* = {theta_pos:.4f}")
    print(f"  warmup = {args.warmup_epochs} epochs")
    print("=" * 60)

    all_loss_history = []
    best_val_bac = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    best_val_cm = None
    best_val_auc_roc = 0.0
    best_val_auprc = 0.0
    best_val_risk = float('inf')

    filter_log = []

    for epoch in range(1, args.epochs + 1):
        is_warmup = epoch <= args.warmup_epochs

        if is_warmup:
            train_loss, train_acc, train_cm = train_epoch_full(
                model, train_loader, optimizer, device, class_weights,
                positive_class=positive_class
            )
            n_used = n_train
            epoch_filter_info = None
        else:
            # 1) Compute per-sample losses
            losses = eval_train_losses(model, eval_loader, device, n_train)

            # 2) GMM + CRASS filtering with adaptive lambda
            clean_indices, all_loss_history, epoch_filter_info = \
                crass_filter_per_class(
                    losses, train_labels, all_loss_history,
                    noise_rate=args.noise_rate,
                    lambda_risk=lambda_adaptive,
                    positive_class=positive_class
                )

            n_used = len(clean_indices)

            # 3) Train on selected subset
            train_loss, train_acc, train_cm = train_epoch_filtered(
                model, train_dataset, clean_indices,
                args.batch_size, optimizer, device,
                class_weights=class_weights, seed=args.seed,
                positive_class=positive_class
            )

        scheduler.step()

        # Validate ─────────────────────────────────────────────────────────
        val_loss, val_acc, val_cm, val_bac, val_auc_roc, val_auprc = \
            evaluate(model, val_loader, device, class_weights,
                     positive_class=positive_class)
        # Risk computed with the ADAPTIVE lambda (not lambda_max)
        val_risk = compute_clinical_risk(val_cm, lambda_adaptive)
        # Also compute risk with lambda_max for comparison
        val_risk_max = compute_clinical_risk(val_cm, args.lambda_max)

        # Logging
        phase_tag = "WARMUP" if is_warmup else "CRASS-A"
        filter_log_entry = {
            'epoch': epoch,
            'phase': phase_tag,
            'samples_used': int(n_used),
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_acc': float(val_acc),
            'val_bac': float(val_bac),
            'val_risk_adaptive': float(val_risk),
            'val_risk_max': float(val_risk_max),
        }
        if epoch_filter_info is not None:
            filter_log_entry['filter_info'] = epoch_filter_info
        filter_log.append(filter_log_entry)

        if epoch % args.print_freq == 0 or epoch == 1:
            print(f"\nEpoch [{epoch}/{args.epochs}]  ({phase_tag})  "
                  f"samples={n_used}/{n_train}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                  f"BAC: {val_bac:.4f}")
            print(f"          AUC-ROC: {val_auc_roc:.4f}, "
                  f"AUPRC: {val_auprc:.4f}")
            print(f"          TP: {val_cm['TP']}, TN: {val_cm['TN']}, "
                  f"FP: {val_cm['FP']}, FN: {val_cm['FN']}")
            print(f"          Risk (lambda_a={lambda_adaptive:.1f}): "
                  f"{val_risk:.4f}  |  "
                  f"Risk (lambda_max={args.lambda_max}): "
                  f"{val_risk_max:.4f}")
            if epoch_filter_info is not None:
                for cls, info in sorted(
                        epoch_filter_info['per_class'].items()):
                    pos_tag = (" (positive)" if info['is_positive_class']
                               else " (negative)")
                    print(
                        f"          Class {cls}{pos_tag}: "
                        f"kept {info['selected']}/{info['total']} "
                        f"({info['keep_ratio']:.1%}), "
                        f"theta*={info['threshold']:.4f}, "
                        f"GMM means="
                        f"{[f'{m:.4f}' for m in info['gmm_means']]}"
                    )

        # Best model (by BAC)
        if val_bac > best_val_bac:
            best_val_bac = val_bac
            best_val_acc = val_acc
            best_epoch = epoch
            best_val_cm = val_cm
            best_val_auc_roc = val_auc_roc
            best_val_auprc = val_auprc
            best_val_risk = val_risk
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_bac': val_bac,
                'val_cm': val_cm,
                'val_auc_roc': val_auc_roc,
                'val_auprc': val_auprc,
                'val_risk': val_risk,
                'lambda_adaptive': lambda_adaptive,
                'lambda_max': args.lambda_max,
                'adaptive_strategy': args.adaptive_strategy,
                'theta_neg': theta_neg,
                'theta_pos': theta_pos,
            }, os.path.join(output_dir, 'best_model.pth'))

    # ══════════════════════════════════════════════════════════════════════
    # Final evaluation
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"Loading best model from epoch {best_epoch}...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_cm, test_bac, test_auc_roc, test_auprc = \
        evaluate(model, test_loader, device, class_weights,
                 positive_class=positive_class)
    test_risk = compute_clinical_risk(test_cm, lambda_adaptive)
    test_risk_max = compute_clinical_risk(test_cm, args.lambda_max)

    test_sensitivity = (test_cm['TP'] / (test_cm['TP'] + test_cm['FN'])
                        if (test_cm['TP'] + test_cm['FN']) > 0 else 0.0)
    test_specificity = (test_cm['TN'] / (test_cm['TN'] + test_cm['FP'])
                        if (test_cm['TN'] + test_cm['FP']) > 0 else 0.0)

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS (Best Epoch: {best_epoch}, selected by BAC)")
    print(f"Method: CRASS Adaptive ({args.adaptive_strategy})")
    print(f"  noise_rate={args.noise_rate}  |  "
          f"lambda_adaptive={lambda_adaptive:.2f}  |  "
          f"theta_pos*={theta_pos:.4f}")
    print("=" * 60)

    print(f"\nValidation:")
    print(f"  Accuracy: {best_val_acc:.4f}, BAC: {best_val_bac:.4f}")
    print(f"  AUC-ROC: {best_val_auc_roc:.4f}, AUPRC: {best_val_auprc:.4f}")
    print(f"  TP: {best_val_cm['TP']}, TN: {best_val_cm['TN']}, "
          f"FP: {best_val_cm['FP']}, FN: {best_val_cm['FN']}")
    print(f"  Risk (lambda_a={lambda_adaptive:.1f}): {best_val_risk:.4f}")

    print(f"\nTest:")
    print(f"  Accuracy:    {test_acc:.4f}")
    print(f"  BAC:         {test_bac:.4f}")
    print(f"  Sensitivity: {test_sensitivity:.4f}")
    print(f"  Specificity: {test_specificity:.4f}")
    print(f"  AUC-ROC:     {test_auc_roc:.4f}")
    print(f"  AUPRC:       {test_auprc:.4f}")
    print(f"  TP: {test_cm['TP']}, TN: {test_cm['TN']}, "
          f"FP: {test_cm['FP']}, FN: {test_cm['FN']}")
    print(f"  Risk (lambda_a={lambda_adaptive:.1f}): {test_risk:.4f}")
    print(f"  Risk (lambda_max={args.lambda_max}): {test_risk_max:.4f}")

    # Save results ─────────────────────────────────────────────────────────
    results = {
        'dataset': args.dataset,
        'method': 'crass_adaptive',
        'adaptive_strategy': args.adaptive_strategy,
        'best_epoch': best_epoch,
        'best_metric': 'BAC',
        'noise_rate': args.noise_rate,
        'lambda_max': args.lambda_max,
        'lambda_min': args.lambda_min,
        'lambda_adaptive': lambda_adaptive,
        'theta_neg': theta_neg,
        'theta_pos': theta_pos,
        'warmup_epochs': args.warmup_epochs,
        'weighted_loss': args.weighted_loss,
        'class_weights': (class_weights.cpu().tolist()
                          if class_weights is not None else None),
        'validation': {
            'accuracy': best_val_acc, 'bac': best_val_bac,
            'auc_roc': best_val_auc_roc, 'auprc': best_val_auprc,
            'clinical_risk': best_val_risk,
            **best_val_cm
        },
        'test': {
            'accuracy': test_acc, 'bac': test_bac,
            'sensitivity': test_sensitivity,
            'specificity': test_specificity,
            'auc_roc': test_auc_roc, 'auprc': test_auprc,
            'clinical_risk': test_risk,
            'clinical_risk_lambda_max': test_risk_max,
            **test_cm
        },
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(output_dir, 'filter_log.json'), 'w') as f:
        json.dump(filter_log, f, indent=2)

    # Save results to TXT
    txt_path = os.path.join(output_dir, 'results.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EXPERIMENT RESULTS  --  CRASS Adaptive\n")
        f.write("(Clinical Risk-Aware Sample Selection with Adaptive Lambda)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Noise Rate: {args.noise_rate}\n")
        f.write(f"Adaptive Strategy: {args.adaptive_strategy}\n")
        f.write(f"Lambda_max: {args.lambda_max}\n")
        f.write(f"Lambda_min: {args.lambda_min}\n")
        f.write(f"Lambda_adaptive: {lambda_adaptive:.2f}\n")
        f.write(f"Theta_neg* (optimal): {theta_neg:.4f}\n")
        f.write(f"Theta_pos* (optimal): {theta_pos:.4f}\n")
        f.write(f"Warmup Epochs: {args.warmup_epochs}\n")
        f.write(f"Weighted Loss: {args.weighted_loss}\n")
        if class_weights is not None:
            f.write(f"Class Weights: {class_weights.cpu().tolist()}\n")
        f.write(f"Best Epoch: {best_epoch} (selected by BAC)\n\n")

        f.write("-" * 60 + "\n")
        f.write("VALIDATION RESULTS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy: {best_val_acc:.4f}\n")
        f.write(f"BAC: {best_val_bac:.4f}\n")
        f.write(f"AUC-ROC: {best_val_auc_roc:.4f}\n")
        f.write(f"AUPRC: {best_val_auprc:.4f}\n")
        f.write(f"Clinical Risk (lambda_a={lambda_adaptive:.1f}): "
                f"{best_val_risk:.4f}\n")
        f.write(f"TP: {best_val_cm['TP']}, TN: {best_val_cm['TN']}, "
                f"FP: {best_val_cm['FP']}, FN: {best_val_cm['FN']}\n\n")

        f.write("-" * 60 + "\n")
        f.write("TEST RESULTS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy:    {test_acc:.4f}\n")
        f.write(f"BAC:         {test_bac:.4f}\n")
        f.write(f"Sensitivity: {test_sensitivity:.4f}\n")
        f.write(f"Specificity: {test_specificity:.4f}\n")
        f.write(f"AUC-ROC:     {test_auc_roc:.4f}\n")
        f.write(f"AUPRC:       {test_auprc:.4f}\n")
        f.write(f"Clinical Risk (lambda_a={lambda_adaptive:.1f}): "
                f"{test_risk:.4f}\n")
        f.write(f"Clinical Risk (lambda_max={args.lambda_max}): "
                f"{test_risk_max:.4f}\n")
        f.write(f"TP: {test_cm['TP']}, TN: {test_cm['TN']}, "
                f"FP: {test_cm['FP']}, FN: {test_cm['FN']}\n")

    print(f"\nResults saved to: {output_dir}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Train with CRASS Adaptive '
                    '(adaptive lambda based on noise rate)'
    )

    # Data
    parser.add_argument('--dataset', type=str, default='dermamnist_bin',
                        choices=list(DATASET_CONFIG.keys()))
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results')

    # Noise
    parser.add_argument('--noise_rate', type=float, default=0.0,
                        help='Symmetric noise rate (0.0 to 1.0)')

    # CRASS Adaptive
    parser.add_argument('--lambda_max', type=float, default=20.0,
                        help='Maximum lambda (used at noise_rate=0)')
    parser.add_argument('--lambda_min', type=float, default=1.0,
                        help='Minimum lambda (floor)')
    parser.add_argument('--adaptive_strategy', type=str, default='v2',
                        choices=['v1', 'v2', 'v3'],
                        help='Strategy for computing adaptive lambda: '
                             'v1=linear, v2=proportional, v3=lookup')
    parser.add_argument('--warmup_epochs', type=int, default=10)

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Loss
    parser.add_argument('--weighted_loss', action='store_true')

    # Model
    parser.add_argument('--pretrained', action='store_true')

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_freq', type=int, default=10)

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("EXPERIMENT CONFIGURATION  --  CRASS Adaptive")
    print("=" * 60)
    print(f"Dataset:          {args.dataset} (64x64)")
    print(f"Model:            ResNet18")
    print(f"Noise Rate:       {args.noise_rate}")
    print(f"Adaptive Strategy:{args.adaptive_strategy}")
    print(f"Lambda Max:       {args.lambda_max}")
    print(f"Lambda Min:       {args.lambda_min}")
    print(f"Warmup Epochs:    {args.warmup_epochs}")
    print(f"Weighted Loss:    {args.weighted_loss}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Learning Rate:    {args.lr}")

    train(args)


if __name__ == '__main__':
    main()
