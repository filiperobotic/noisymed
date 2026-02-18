"""
Training Script with CRASS + Cost-Sensitive Loss for Noisy Labels.

CRASS (Clinical Risk-Aware Sample Selection) + Cost-Sensitive Loss:
    Combines the CRASS filtering approach (GMM per class + Proposition 1
    thresholds) with a cost-sensitive cross-entropy loss.  Instead of using
    balanced class weights inferred from label frequencies, the loss weights
    are derived directly from the clinical risk parameter lambda:

        w[positive_class] = lambda_risk
        w[other_class]    = 1.0

    This ensures that the same cost ratio that governs sample selection
    (CRASS thresholds) also governs the training loss, creating a unified
    risk-aware training pipeline.

    **Proposition 1 (Optimal per-class threshold):**
        theta*(y) = C_noise / (C_noise + C_y)

    With the simplification C_noise = C_FP = 1 and lambda = C_FN / C_FP:
        - Negative class (normal):  theta_0* = 1 / (1 + 1)      = 0.5
        - Positive class (disease): theta_1* = 1 / (1 + lambda)

    The cost-sensitive weights are always applied (no flag to disable).
    The same lambda_risk parameter controls both the CRASS thresholds and
    the loss weights.

    NOTE: positive_class varies by dataset. For breastmnist, positive_class=0
    (malignant). For others, positive_class=1. The weight vector is constructed
    accordingly from DATASET_CONFIG.

Strategy:
    1. Warmup phase (default 10 epochs): train on ALL samples with CS loss.
    2. Filter phase: per-epoch GMM fitting per class to estimate P(clean),
       then apply CRASS risk-aware thresholds for sample selection + CS loss.

Usage:
    python train_crass_cs.py --dataset pneumoniamnist --noise_rate 0.2 --lambda_risk 10
    python train_crass_cs.py --dataset breastmnist --noise_rate 0.4 --lambda_risk 5
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


def get_model(num_classes=2, in_channels=1, pretrained=False):
    model = models.resnet18(pretrained=pretrained)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def compute_confusion_matrix(outputs, targets, positive_class=1):
    """Compute confusion matrix elements.

    Args:
        outputs: model logits (N, num_classes)
        targets: ground-truth labels (N,)
        positive_class: which class index is the "positive" (disease/malignant).
    """
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
    """Compute normalised clinical risk.

    Risk = (lambda * FN + FP) / N

    This weighs false negatives (missed disease) by lambda relative to
    false positives.
    """
    N = cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']
    if N == 0:
        return 0.0
    return (lambda_risk * cm['FN'] + cm['FP']) / N


# ═══════════════════════════════════════════════════════════════════════════
# Cost-Sensitive Weights
# ═══════════════════════════════════════════════════════════════════════════

def compute_cost_sensitive_weights(positive_class, lambda_risk, device):
    """Compute cost-sensitive class weights from the clinical risk parameter.

    The weight for the positive (disease) class is set to lambda_risk,
    while the weight for the other class is 1.0.  This ensures that the
    same cost ratio driving CRASS thresholds also drives the loss function.

    Args:
        positive_class: int – index of the positive (disease) class.
        lambda_risk: float – cost ratio C_FN / C_FP.
        device: torch device.

    Returns:
        weights: Tensor of shape (2,) with cost-sensitive weights.
    """
    weights = torch.ones(2, dtype=torch.float32, device=device)
    weights[positive_class] = lambda_risk
    return weights


# ═══════════════════════════════════════════════════════════════════════════
# CRASS: Clinical Risk-Aware Sample Selection
# ═══════════════════════════════════════════════════════════════════════════

def compute_optimal_thresholds(lambda_risk, c_noise=1.0, c_fp=1.0):
    """Compute optimal per-class thresholds (Proposition 1).

    Proposition 1:
        theta*(y) = C_noise / (C_noise + C_y)

    where C_y = C_FN if y is the positive (disease) class,
          C_y = C_FP if y is the negative (normal) class.

    Args:
        lambda_risk: cost ratio lambda = C_FN / C_FP.
        c_noise: cost of training on a noisy sample (normalised to 1).
        c_fp: base false-positive cost (normalised to 1).

    Returns:
        theta_neg: optimal threshold for the negative class.
        theta_pos: optimal threshold for the positive class.
    """
    c_fn = lambda_risk * c_fp
    theta_neg = c_noise / (c_noise + c_fp)   # = 0.5 when normalised
    theta_pos = c_noise / (c_noise + c_fn)   # = 1 / (1 + lambda)
    return theta_neg, theta_pos


def eval_train_losses(model, eval_loader, device, n_samples):
    """Compute per-sample cross-entropy loss (unreduced) for the training set.

    Args:
        model: trained model (set to eval mode).
        eval_loader: DataLoader yielding (image, label, index).
        device: torch device.
        n_samples: total number of training samples.

    Returns:
        losses: Tensor of shape (n_samples,) with per-sample CE losses.
    """
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
    """Fit a 2-component GMM **per class** and apply CRASS thresholds.

    For each class independently:
        1. Select loss values belonging to that class.
        2. Min-max normalise them.
        3. Fit a 2-component GMM.
        4. Compute P(clean) = probability of belonging to the low-loss
           component.
        5. Apply the CRASS risk-aware threshold (Proposition 1):
           - Positive class uses theta_pos = 1 / (1 + lambda)
           - Negative class uses theta_neg = 0.5

    The ``positive_class`` argument maps the dataset's positive class to
    the correct CRASS cost.  For example, breastmnist has positive_class=0
    (malignant is class 0), so class 0 uses theta_pos and class 1 uses
    theta_neg.

    Args:
        losses: Tensor (n_samples,) – raw per-sample losses.
        labels: numpy array (n_samples,) – (noisy) labels.
        all_loss_history: list of Tensors – loss history from previous epochs.
        noise_rate: float – injected noise rate.
        lambda_risk: float – cost ratio C_FN / C_FP.
        positive_class: int – which class is the positive (disease) class.

    Returns:
        clean_indices: numpy array of global indices considered clean.
        all_loss_history: updated history list.
        filter_info: dict with per-class statistics for logging.
    """
    labels = np.array(labels).flatten()
    n_samples = len(labels)
    unique_classes = np.unique(labels[labels >= 0])

    # Append current losses to history
    all_loss_history.append(losses.clone())

    # Compute optimal thresholds
    theta_neg, theta_pos = compute_optimal_thresholds(lambda_risk)

    # Map class index → CRASS threshold
    # positive_class uses theta_pos (more permissive)
    # other classes use theta_neg (standard 0.5)
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

        # For high noise rates, average over last 5 epochs for stability
        if noise_rate >= 0.9 and len(all_loss_history) >= 5:
            history_stack = torch.stack(
                [h[cls_idx] for h in all_loss_history[-5:]]
            )
            avg_losses = history_stack.mean(0)
            a_min = avg_losses.min()
            a_max = avg_losses.max()
            if a_max - a_min > 1e-8:
                input_loss = ((avg_losses - a_min) / (a_max - a_min)).reshape(-1, 1).numpy()
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
        # Clean component = the one with the smaller mean
        clean_component = gmm.means_.argmin()
        prob_clean = prob[:, clean_component]

        # ═══════════════════════════════════════════════════════════════
        # CRASS: Apply risk-aware threshold (Proposition 1)
        # ═══════════════════════════════════════════════════════════════
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
# Training loops
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch_full(model, train_loader, optimizer, device,
                     cs_weights, positive_class=1):
    """Train one epoch on ALL samples (warmup / no filtering).

    train_loader yields (image, label, index) – index is ignored here.
    Cost-sensitive weights are always applied.
    """
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

        loss = F.cross_entropy(outputs, labels, weight=cs_weights)

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
                         optimizer, device, cs_weights, seed=42,
                         positive_class=1):
    """Train one epoch on the filtered (clean) subset only.

    Creates a temporary Subset + DataLoader from the clean indices.
    Cost-sensitive weights are always applied.
    """
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
        # Subset of IndexedNoisyMedMNISTDataset returns (image, label, index)
        images, labels, _ = batch
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = F.cross_entropy(outputs, labels, weight=cs_weights)

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
    cm = compute_confusion_matrix(all_outputs, all_targets, positive_class) if total > 0 else {
        'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    return epoch_loss, accuracy, cm


def evaluate(model, data_loader, device, cs_weights=None,
             positive_class=1):
    """Evaluate model on validation/test set (standard, no index)."""
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

            if cs_weights is not None:
                loss = F.cross_entropy(outputs, labels, weight=cs_weights)
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

    # Use P(positive_class) as the score for ROC/PRC
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

    # Compute optimal thresholds for display
    theta_neg, theta_pos = compute_optimal_thresholds(args.lambda_risk)
    print(f"\nCRASS Optimal Thresholds (Proposition 1):")
    print(f"  theta*(negative class) = {theta_neg:.4f}")
    print(f"  theta*(positive class) = {theta_pos:.4f}  "
          f"[lambda={args.lambda_risk}]")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = (
        f"{args.dataset}_noise{args.noise_rate}"
        f"_crass_cs"
        f"_lambda{args.lambda_risk}"
        f"_warmup{args.warmup_epochs}"
        f"_{timestamp}"
    )
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Data ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Loading {args.dataset} dataset...")
    train_loader, eval_loader, val_loader, test_loader = get_filter_dataloaders(
        dataset=args.dataset, noise_rate=args.noise_rate,
        batch_size=args.batch_size, seed=args.seed, data_dir=args.data_dir
    )

    train_dataset = train_loader.dataset  # IndexedNoisyMedMNISTDataset
    n_train = len(train_dataset)
    train_labels = np.array(train_dataset.labels).flatten()

    print(f"Train samples: {n_train}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    # Cost-sensitive weights ────────────────────────────────────────────────
    cs_weights = compute_cost_sensitive_weights(
        positive_class, args.lambda_risk, device
    )
    print(f"\nCost-sensitive weights (lambda={args.lambda_risk}):")
    print(f"  w[class 0] = {cs_weights[0].item():.4f}"
          f"  {'(positive)' if positive_class == 0 else '(negative)'}")
    print(f"  w[class 1] = {cs_weights[1].item():.4f}"
          f"  {'(positive)' if positive_class == 1 else '(negative)'}")

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
    print(f"Starting CRASS + Cost-Sensitive training")
    print(f"  Theoretical basis:  theta*(y) = 1 / (1 + C_y)  [Proposition 1]")
    print(f"  lambda (C_FN/C_FP): {args.lambda_risk}")
    print(f"  theta_neg* = {theta_neg:.4f}  |  theta_pos* = {theta_pos:.4f}")
    print(f"  warmup = {args.warmup_epochs} epochs")
    print(f"  Cost-sensitive weights: {cs_weights.cpu().tolist()}")
    print("=" * 60)

    all_loss_history = []  # list[Tensor(n_train,)]
    best_val_bac = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    best_val_cm = None
    best_val_auc_roc = 0.0
    best_val_auprc = 0.0
    best_val_risk = float('inf')

    # Keep track of filtering stats per epoch
    filter_log = []

    for epoch in range(1, args.epochs + 1):
        is_warmup = epoch <= args.warmup_epochs

        if is_warmup:
            # ── Warmup: train on all samples ──────────────────────────
            train_loss, train_acc, train_cm = train_epoch_full(
                model, train_loader, optimizer, device, cs_weights,
                positive_class=positive_class
            )
            n_used = n_train
            epoch_filter_info = None
        else:
            # ── CRASS filter phase ────────────────────────────────────
            # 1) Compute per-sample losses
            losses = eval_train_losses(model, eval_loader, device, n_train)

            # 2) GMM + CRASS filtering per class
            clean_indices, all_loss_history, epoch_filter_info = \
                crass_filter_per_class(
                    losses, train_labels, all_loss_history,
                    noise_rate=args.noise_rate,
                    lambda_risk=args.lambda_risk,
                    positive_class=positive_class
                )

            n_used = len(clean_indices)

            # 3) Train only on selected subset
            train_loss, train_acc, train_cm = train_epoch_filtered(
                model, train_dataset, clean_indices,
                args.batch_size, optimizer, device,
                cs_weights=cs_weights, seed=args.seed,
                positive_class=positive_class
            )

        scheduler.step()

        # Validate ─────────────────────────────────────────────────────────
        val_loss, val_acc, val_cm, val_bac, val_auc_roc, val_auprc = \
            evaluate(model, val_loader, device, cs_weights,
                     positive_class=positive_class)
        val_risk = compute_clinical_risk(val_cm, args.lambda_risk)

        # Logging ──────────────────────────────────────────────────────
        phase_tag = "WARMUP" if is_warmup else "CRASS"
        filter_log_entry = {
            'epoch': epoch,
            'phase': phase_tag,
            'samples_used': int(n_used),
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_acc': float(val_acc),
            'val_bac': float(val_bac),
            'val_risk': float(val_risk),
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
            print(f"          Clinical Risk (lambda={args.lambda_risk}): "
                  f"{val_risk:.4f}")
            if epoch_filter_info is not None:
                for cls, info in sorted(epoch_filter_info['per_class'].items()):
                    pos_tag = " (positive)" if info['is_positive_class'] else " (negative)"
                    print(f"          Class {cls}{pos_tag}: "
                          f"kept {info['selected']}/{info['total']} "
                          f"({info['keep_ratio']:.1%}), "
                          f"theta*={info['threshold']:.4f}, "
                          f"GMM means={[f'{m:.4f}' for m in info['gmm_means']]}")

        # Best model (by BAC) ──────────────────────────────────────────
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
                'lambda_risk': args.lambda_risk,
                'theta_neg': theta_neg,
                'theta_pos': theta_pos,
            }, os.path.join(output_dir, 'best_model.pth'))

    # ══════════════════════════════════════════════════════════════════════
    # Final evaluation on test set
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"Loading best model from epoch {best_epoch}...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_cm, test_bac, test_auc_roc, test_auprc = \
        evaluate(model, test_loader, device, cs_weights,
                 positive_class=positive_class)
    test_risk = compute_clinical_risk(test_cm, args.lambda_risk)

    # Compute sensitivity / specificity for the test set
    test_sensitivity = (test_cm['TP'] / (test_cm['TP'] + test_cm['FN'])
                        if (test_cm['TP'] + test_cm['FN']) > 0 else 0.0)
    test_specificity = (test_cm['TN'] / (test_cm['TN'] + test_cm['FP'])
                        if (test_cm['TN'] + test_cm['FP']) > 0 else 0.0)

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS (Best Epoch: {best_epoch}, selected by BAC)")
    print(f"Method: CRASS + Cost-Sensitive  |  lambda={args.lambda_risk}  |  "
          f"theta_pos*={theta_pos:.4f}")
    print(f"Cost-sensitive weights: {cs_weights.cpu().tolist()}")
    print("=" * 60)
    print(f"\nValidation:")
    print(f"  Accuracy: {best_val_acc:.4f}, BAC: {best_val_bac:.4f}")
    print(f"  AUC-ROC: {best_val_auc_roc:.4f}, AUPRC: {best_val_auprc:.4f}")
    print(f"  TP: {best_val_cm['TP']}, TN: {best_val_cm['TN']}, "
          f"FP: {best_val_cm['FP']}, FN: {best_val_cm['FN']}")
    print(f"  Clinical Risk: {best_val_risk:.4f}")

    print(f"\nTest:")
    print(f"  Accuracy:    {test_acc:.4f}")
    print(f"  BAC:         {test_bac:.4f}")
    print(f"  Sensitivity: {test_sensitivity:.4f}")
    print(f"  Specificity: {test_specificity:.4f}")
    print(f"  AUC-ROC:     {test_auc_roc:.4f}")
    print(f"  AUPRC:       {test_auprc:.4f}")
    print(f"  TP: {test_cm['TP']}, TN: {test_cm['TN']}, "
          f"FP: {test_cm['FP']}, FN: {test_cm['FN']}")
    print(f"  Clinical Risk (lambda={args.lambda_risk}): {test_risk:.4f}")

    # Save results ─────────────────────────────────────────────────────────
    results = {
        'dataset': args.dataset,
        'method': 'crass_cs',
        'best_epoch': best_epoch,
        'best_metric': 'BAC',
        'noise_rate': args.noise_rate,
        'lambda_risk': args.lambda_risk,
        'theta_neg': theta_neg,
        'theta_pos': theta_pos,
        'warmup_epochs': args.warmup_epochs,
        'cost_sensitive_weights': cs_weights.cpu().tolist(),
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
        f.write("EXPERIMENT RESULTS  --  CRASS + Cost-Sensitive\n")
        f.write("(Clinical Risk-Aware Sample Selection + Cost-Sensitive Loss)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Noise Rate: {args.noise_rate}\n")
        f.write(f"Lambda (C_FN/C_FP): {args.lambda_risk}\n")
        f.write(f"Theta_neg* (optimal): {theta_neg:.4f}\n")
        f.write(f"Theta_pos* (optimal): {theta_pos:.4f}\n")
        f.write(f"Warmup Epochs: {args.warmup_epochs}\n")
        f.write(f"Cost-Sensitive Weights: {cs_weights.cpu().tolist()}\n")
        f.write(f"  w[positive_class={positive_class}] = {args.lambda_risk}\n")
        f.write(f"  w[other_class] = 1.0\n")
        f.write(f"Best Epoch: {best_epoch} (selected by BAC)\n\n")
        f.write(f"Theoretical basis: Proposition 1\n")
        f.write(f"  theta*(y) = C_noise / (C_noise + C_y)\n")
        f.write(f"  theta_neg = 1/(1+1) = 0.5\n")
        f.write(f"  theta_pos = 1/(1+lambda) = {theta_pos:.4f}\n\n")

        f.write("-" * 60 + "\n")
        f.write("VALIDATION RESULTS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy: {best_val_acc:.4f}\n")
        f.write(f"BAC: {best_val_bac:.4f}\n")
        f.write(f"AUC-ROC: {best_val_auc_roc:.4f}\n")
        f.write(f"AUPRC: {best_val_auprc:.4f}\n")
        f.write(f"Clinical Risk: {best_val_risk:.4f}\n")
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
        f.write(f"Clinical Risk (lambda={args.lambda_risk}): {test_risk:.4f}\n")
        f.write(f"TP: {test_cm['TP']}, TN: {test_cm['TN']}, "
                f"FP: {test_cm['FP']}, FN: {test_cm['FN']}\n")

    print(f"\nResults saved to: {output_dir}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Train with CRASS + Cost-Sensitive Loss '
                    '(Clinical Risk-Aware Sample Selection + Cost-Sensitive CE)'
    )

    # Data
    parser.add_argument('--dataset', type=str, default='pneumoniamnist',
                        choices=list(DATASET_CONFIG.keys()),
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results')

    # Noise
    parser.add_argument('--noise_rate', type=float, default=0.0,
                        help='Symmetric noise rate (0.0 to 1.0)')

    # CRASS
    parser.add_argument('--lambda_risk', type=float, default=20.0,
                        help='Cost ratio lambda = C_FN / C_FP. '
                             'Higher values make the positive-class threshold '
                             'more permissive (fewer positive samples discarded) '
                             'AND increase the cost-sensitive weight on the '
                             'positive class')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs before filtering starts')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Model
    parser.add_argument('--pretrained', action='store_true')

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_freq', type=int, default=10)

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("EXPERIMENT CONFIGURATION  --  CRASS + Cost-Sensitive")
    print("(Clinical Risk-Aware Sample Selection + Cost-Sensitive Loss)")
    print("=" * 60)
    print(f"Dataset:       {args.dataset} (64x64)")
    print(f"Model:         ResNet18")
    print(f"Noise Rate:    {args.noise_rate}")
    print(f"Lambda:        {args.lambda_risk}  (C_FN / C_FP)")
    print(f"Warmup Epochs: {args.warmup_epochs}")
    print(f"Loss:          Cost-Sensitive CE (w[positive]={args.lambda_risk}, w[other]=1.0)")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Learning Rate: {args.lr}")

    train(args)


if __name__ == '__main__':
    main()
