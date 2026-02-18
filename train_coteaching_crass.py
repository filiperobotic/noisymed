"""
Training Script with Co-teaching + CRASS for Noisy Labels.

Co-teaching + CRASS combines the dual-network framework of Co-teaching
(Han et al., NeurIPS 2018) with per-class clinical risk-aware sample
selection from CRASS.

Key insight:
    Standard Co-teaching uses the same keep_rate for all classes.
    Co-teaching+CRASS uses *asymmetric* per-class keep rates:
        - Negative class (normal):  keep_rate = base_keep_rate  (standard)
        - Positive class (disease): keep_rate is boosted to retain more
          positive samples, reducing false negatives (missed diagnoses).

    The boost is derived from CRASS Proposition 1:
        boost = lambda / (1 + lambda)
        keep_rate_pos = base_keep_rate + (1 - base_keep_rate) * boost

    For lambda=20 and forget_rate=0.4 (i.e. base_keep_rate=0.6):
        boost = 20/21 = 0.952
        keep_rate_pos = 0.6 + 0.4 * 0.952 = 0.981
    So ~98% of positive-class samples are retained vs 60% of negatives.

Usage:
    python train_coteaching_crass.py --dataset dermamnist_bin --noise_rate 0.2 --lambda_risk 20
    python train_coteaching_crass.py --dataset dermamnist_bin --noise_rate 0.4 --lambda_risk 10 --weighted_loss
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

from sklearn.metrics import roc_auc_score, average_precision_score
from dataloaders import get_dataloaders, DATASET_CONFIG


# ═══════════════════════════════════════════════════════════════════════════
# Helpers (shared with other training scripts)
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
# Co-teaching + CRASS core
# ═══════════════════════════════════════════════════════════════════════════

def rate_schedule(epoch, forget_rate, num_gradual, exponent=1):
    """Forget rate schedule (same as standard Co-teaching)."""
    if epoch < num_gradual:
        return forget_rate * min(
            (epoch / num_gradual) ** exponent, 1.0
        )
    return forget_rate


def coteaching_crass_select(loss, labels, forget_rate, lambda_risk,
                            positive_class=1):
    """Select samples using per-class CRASS-aware keep rates.

    For each class independently:
        1. Compute class-specific keep_rate.
        2. Select the samples with the smallest loss within that class.

    The positive class gets a *higher* keep_rate (more permissive),
    derived from CRASS Proposition 1:
        boost = lambda / (1 + lambda)
        keep_rate_pos = base + (1 - base) * boost

    Args:
        loss: Tensor (B,) per-sample cross-entropy loss.
        labels: Tensor (B,) noisy labels.
        forget_rate: base fraction of samples to discard.
        lambda_risk: CRASS cost ratio C_FN / C_FP.
        positive_class: which class index is the positive (disease) class.

    Returns:
        keep_indices: Tensor of global batch indices to keep.
        info: dict with per-class selection statistics.
    """
    device = loss.device
    base_keep_rate = 1.0 - forget_rate

    # CRASS-derived boost for positive class
    boost_factor = lambda_risk / (1.0 + lambda_risk)

    keep_rate_neg = base_keep_rate
    keep_rate_pos = base_keep_rate + (1.0 - base_keep_rate) * boost_factor
    keep_rate_pos = min(0.99, keep_rate_pos)

    # Separate by class
    selected_indices = []
    info = {
        'base_keep_rate': base_keep_rate,
        'keep_rate_neg': keep_rate_neg,
        'keep_rate_pos': keep_rate_pos,
        'boost_factor': boost_factor,
    }

    for cls in [0, 1]:
        if cls == positive_class:
            keep_rate = keep_rate_pos
        else:
            keep_rate = keep_rate_neg

        cls_mask = (labels == cls)
        cls_indices = cls_mask.nonzero(as_tuple=True)[0]
        n_cls = len(cls_indices)

        if n_cls == 0:
            continue

        n_keep = max(1, int(keep_rate * n_cls))
        n_keep = min(n_keep, n_cls)

        # Select smallest-loss samples within this class
        cls_losses = loss[cls_indices]
        _, topk_idx = torch.topk(cls_losses, n_keep, largest=False)
        selected_indices.append(cls_indices[topk_idx])

        is_pos = (cls == positive_class)
        info[f'class_{cls}'] = {
            'total': n_cls,
            'kept': n_keep,
            'keep_rate': keep_rate,
            'is_positive': is_pos,
        }

    if selected_indices:
        keep_indices = torch.cat(selected_indices)
    else:
        keep_indices = torch.arange(len(labels), device=device)

    return keep_indices, info


def coteaching_crass_loss(logits1, logits2, labels, forget_rate,
                          lambda_risk, positive_class=1,
                          class_weights=None, use_crass=True):
    """Co-teaching loss with optional CRASS per-class selection.

    During warmup (use_crass=False), falls back to standard Co-teaching.

    Args:
        logits1, logits2: (B, C) logits from both networks.
        labels: (B,) noisy labels.
        forget_rate: current forget rate from schedule.
        lambda_risk: CRASS lambda.
        positive_class: positive class index.
        class_weights: optional per-class CE weights.
        use_crass: if False, use standard symmetric selection.

    Returns:
        loss1, loss2: scalar losses for each network.
        info1, info2: dicts with selection statistics.
    """
    # Per-sample losses
    loss1_vec = F.cross_entropy(logits1, labels, weight=class_weights,
                                reduction='none')
    loss2_vec = F.cross_entropy(logits2, labels, weight=class_weights,
                                reduction='none')

    if not use_crass or forget_rate < 1e-6:
        # Standard Co-teaching: symmetric keep
        batch_size = len(labels)
        keep_num = max(1, int((1 - forget_rate) * batch_size))

        _, idx1 = torch.sort(loss1_vec)
        idx_keep1 = idx1[:keep_num]
        _, idx2 = torch.sort(loss2_vec)
        idx_keep2 = idx2[:keep_num]

        info1 = {'mode': 'standard', 'kept': keep_num, 'total': batch_size}
        info2 = {'mode': 'standard', 'kept': keep_num, 'total': batch_size}
    else:
        # CRASS: per-class asymmetric selection
        idx_keep1, info1 = coteaching_crass_select(
            loss1_vec, labels, forget_rate, lambda_risk, positive_class
        )
        idx_keep2, info2 = coteaching_crass_select(
            loss2_vec, labels, forget_rate, lambda_risk, positive_class
        )
        info1['mode'] = 'crass'
        info2['mode'] = 'crass'

    # Cross-feed: Net1 trains on samples selected by Net2, and vice versa
    loss1 = F.cross_entropy(logits1[idx_keep2], labels[idx_keep2],
                            weight=class_weights)
    loss2 = F.cross_entropy(logits2[idx_keep1], labels[idx_keep1],
                            weight=class_weights)

    return loss1, loss2, info1, info2


# ═══════════════════════════════════════════════════════════════════════════
# Training / evaluation loops
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch_coteaching_crass(model1, model2, train_loader, optimizer1,
                                 optimizer2, device, forget_rate,
                                 lambda_risk, positive_class=1,
                                 class_weights=None, use_crass=True):
    """Train both networks for one epoch with Co-teaching + CRASS."""
    model1.train()
    model2.train()

    running_loss1 = 0.0
    running_loss2 = 0.0
    correct1 = 0
    correct2 = 0
    total = 0
    all_outputs1 = []
    all_outputs2 = []
    all_targets = []

    # Aggregate selection stats
    total_kept_neg = 0
    total_kept_pos = 0
    total_neg = 0
    total_pos = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits1 = model1(images)
        logits2 = model2(images)

        loss1, loss2, info1, info2 = coteaching_crass_loss(
            logits1, logits2, labels, forget_rate,
            lambda_risk, positive_class, class_weights, use_crass
        )

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        batch_size = labels.size(0)
        running_loss1 += loss1.item() * batch_size
        running_loss2 += loss2.item() * batch_size

        _, pred1 = torch.max(logits1.detach(), 1)
        _, pred2 = torch.max(logits2.detach(), 1)
        correct1 += (pred1 == labels).sum().item()
        correct2 += (pred2 == labels).sum().item()
        total += batch_size

        all_outputs1.append(logits1.detach())
        all_outputs2.append(logits2.detach())
        all_targets.append(labels.detach())

        # Accumulate per-class stats from network 1's selection
        if use_crass and 'class_0' in info1:
            for cls_key in ['class_0', 'class_1']:
                if cls_key in info1:
                    cls_info = info1[cls_key]
                    if cls_info['is_positive']:
                        total_kept_pos += cls_info['kept']
                        total_pos += cls_info['total']
                    else:
                        total_kept_neg += cls_info['kept']
                        total_neg += cls_info['total']

    all_outputs1 = torch.cat(all_outputs1)
    all_outputs2 = torch.cat(all_outputs2)
    all_targets = torch.cat(all_targets)

    cm1 = compute_confusion_matrix(all_outputs1, all_targets, positive_class)
    cm2 = compute_confusion_matrix(all_outputs2, all_targets, positive_class)

    epoch_stats = {
        'total_kept_neg': total_kept_neg,
        'total_neg': total_neg,
        'total_kept_pos': total_kept_pos,
        'total_pos': total_pos,
        'keep_rate_neg': total_kept_neg / total_neg if total_neg > 0 else 0,
        'keep_rate_pos': total_kept_pos / total_pos if total_pos > 0 else 0,
    }

    return (
        (running_loss1 / total, correct1 / total, cm1),
        (running_loss2 / total, correct2 / total, cm2),
        epoch_stats,
    )


def evaluate(model, data_loader, device, class_weights=None,
             positive_class=1):
    """Evaluate a single model."""
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


def evaluate_ensemble(model1, model2, data_loader, device,
                      class_weights=None, positive_class=1):
    """Evaluate ensemble (average of softmax outputs)."""
    model1.eval()
    model2.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_avg_outputs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits1 = model1(images)
            logits2 = model2(images)

            probs1 = F.softmax(logits1, dim=1)
            probs2 = F.softmax(logits2, dim=1)
            avg_probs = (probs1 + probs2) / 2.0

            avg_logits = (logits1 + logits2) / 2.0
            if class_weights is not None:
                loss = F.cross_entropy(avg_logits, labels,
                                       weight=class_weights)
            else:
                loss = F.cross_entropy(avg_logits, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(avg_probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_avg_outputs.append(avg_probs)
            all_targets.append(labels)

    epoch_loss = running_loss / total
    accuracy = correct / total

    all_avg_outputs = torch.cat(all_avg_outputs)
    all_targets = torch.cat(all_targets)

    _, predicted = torch.max(all_avg_outputs, 1)
    TP = ((predicted == positive_class) & (all_targets == positive_class)).sum().item()
    TN = ((predicted != positive_class) & (all_targets != positive_class)).sum().item()
    FP = ((predicted == positive_class) & (all_targets != positive_class)).sum().item()
    FN = ((predicted != positive_class) & (all_targets == positive_class)).sum().item()
    cm = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}

    bac = compute_bac(cm)

    probs_pos = all_avg_outputs[:, positive_class].cpu().numpy()
    targets_np = (all_targets == positive_class).long().cpu().numpy()
    auc_roc = roc_auc_score(targets_np, probs_pos)
    auprc = average_precision_score(targets_np, probs_pos)

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

    # CRASS keep-rate info
    base_keep = 1.0 - args.forget_rate
    boost = args.lambda_risk / (1.0 + args.lambda_risk)
    keep_pos = min(0.99, base_keep + (1.0 - base_keep) * boost)
    print(f"\nCo-teaching + CRASS keep rates (at full forget_rate):")
    print(f"  base_keep_rate (neg): {base_keep:.3f}")
    print(f"  keep_rate_pos:        {keep_pos:.3f}  "
          f"(boost={boost:.3f}, lambda={args.lambda_risk})")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = (
        f"{args.dataset}_noise{args.noise_rate}"
        f"_coteaching_crass"
        f"_lambda{args.lambda_risk}"
        f"_fr{args.forget_rate}_grad{args.num_gradual}"
        f"_{'weighted' if args.weighted_loss else 'standard'}"
        f"_{timestamp}"
    )
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Data ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Loading {args.dataset} dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=args.dataset, noise_rate=args.noise_rate,
        batch_size=args.batch_size, seed=args.seed, data_dir=args.data_dir
    )

    n_train = len(train_loader.dataset)
    print(f"Train samples: {n_train}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    # Class weights ────────────────────────────────────────────────────────
    class_weights = None
    if args.weighted_loss:
        class_weights = infer_class_weights(
            train_loader.dataset.labels, device
        )
        if class_weights is not None:
            print(f"\nUsing balanced class weights: "
                  f"{class_weights.cpu().tolist()}")
    else:
        print("\nUsing standard Cross Entropy loss")

    # Models ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Creating two ResNet18 models for Co-teaching + CRASS...")
    model1 = get_model(
        num_classes=config['num_classes'],
        in_channels=config['in_channels'],
        pretrained=args.pretrained
    ).to(device)
    model2 = get_model(
        num_classes=config['num_classes'],
        in_channels=config['in_channels'],
        pretrained=args.pretrained
    ).to(device)

    # Different random init for model 2
    torch.manual_seed(args.seed + 1)
    for m in model2.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    set_seed(args.seed)

    optimizer1 = optim.SGD(
        model1.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    optimizer2 = optim.SGD(
        model2.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, T_max=args.epochs
    )
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=args.epochs
    )

    # Training ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Starting Co-teaching + CRASS training")
    print(f"  forget_rate={args.forget_rate}  |  "
          f"num_gradual={args.num_gradual}  |  "
          f"lambda={args.lambda_risk}")
    print(f"  warmup={args.warmup_epochs} epochs (standard Co-teaching)")
    print("=" * 60)

    best_val_bac = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    best_val_cm = None
    best_val_auc_roc = 0.0
    best_val_auprc = 0.0

    training_log = []

    for epoch in range(1, args.epochs + 1):
        current_forget_rate = rate_schedule(
            epoch - 1, args.forget_rate, args.num_gradual, args.exponent
        )

        # Use standard Co-teaching during warmup, CRASS after
        use_crass = (epoch > args.warmup_epochs)

        (loss1, acc1, cm1), (loss2, acc2, cm2), epoch_stats = \
            train_epoch_coteaching_crass(
                model1, model2, train_loader, optimizer1, optimizer2,
                device, current_forget_rate,
                args.lambda_risk, positive_class,
                class_weights, use_crass
            )

        scheduler1.step()
        scheduler2.step()

        # Validate
        val_loss, val_acc, val_cm, val_bac, val_auc_roc, val_auprc = \
            evaluate_ensemble(
                model1, model2, val_loader, device, class_weights,
                positive_class=positive_class
            )
        val_risk = compute_clinical_risk(val_cm, args.lambda_risk)

        val_loss1, val_acc1, _, val_bac1, _, _ = evaluate(
            model1, val_loader, device, class_weights, positive_class
        )
        val_loss2, val_acc2, _, val_bac2, _, _ = evaluate(
            model2, val_loader, device, class_weights, positive_class
        )

        # Logging
        keep_ratio = 1.0 - current_forget_rate
        phase_tag = "STD" if not use_crass else "CRASS"
        log_entry = {
            'epoch': epoch,
            'phase': phase_tag,
            'forget_rate': float(current_forget_rate),
            'keep_ratio': float(keep_ratio),
            'net1_train_loss': float(loss1),
            'net1_train_acc': float(acc1),
            'net1_val_bac': float(val_bac1),
            'net2_train_loss': float(loss2),
            'net2_train_acc': float(acc2),
            'net2_val_bac': float(val_bac2),
            'ensemble_val_bac': float(val_bac),
            'ensemble_val_risk': float(val_risk),
        }
        if use_crass:
            log_entry['crass_stats'] = epoch_stats
        training_log.append(log_entry)

        if epoch % args.print_freq == 0 or epoch == 1:
            print(f"\nEpoch [{epoch}/{args.epochs}]  ({phase_tag})  "
                  f"keep={keep_ratio:.2%}")
            print(f"  Net1 Train - Loss: {loss1:.4f}, Acc: {acc1:.4f}")
            print(f"  Net2 Train - Loss: {loss2:.4f}, Acc: {acc2:.4f}")
            print(f"  Ensemble   - BAC: {val_bac:.4f}, "
                  f"AUC-ROC: {val_auc_roc:.4f}")
            print(f"               TP: {val_cm['TP']}, TN: {val_cm['TN']}, "
                  f"FP: {val_cm['FP']}, FN: {val_cm['FN']}")
            print(f"               Risk (lambda={args.lambda_risk}): "
                  f"{val_risk:.4f}")
            if use_crass and epoch_stats['total_pos'] > 0:
                print(f"  CRASS select - "
                      f"neg: {epoch_stats['keep_rate_neg']:.1%}  |  "
                      f"pos: {epoch_stats['keep_rate_pos']:.1%}")

        # Save best (by ensemble BAC)
        if val_bac > best_val_bac:
            best_val_bac = val_bac
            best_val_acc = val_acc
            best_epoch = epoch
            best_val_cm = val_cm
            best_val_auc_roc = val_auc_roc
            best_val_auprc = val_auprc
            torch.save({
                'epoch': epoch,
                'model1_state_dict': model1.state_dict(),
                'model2_state_dict': model2.state_dict(),
                'val_acc': val_acc,
                'val_bac': val_bac,
                'val_cm': val_cm,
                'val_auc_roc': val_auc_roc,
                'val_auprc': val_auprc,
            }, os.path.join(output_dir, 'best_model.pth'))

    # ══════════════════════════════════════════════════════════════════════
    # Final evaluation
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"Loading best models from epoch {best_epoch}...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model1.load_state_dict(checkpoint['model1_state_dict'])
    model2.load_state_dict(checkpoint['model2_state_dict'])

    # Ensemble test
    test_loss, test_acc, test_cm, test_bac, test_auc_roc, test_auprc = \
        evaluate_ensemble(
            model1, model2, test_loader, device, class_weights,
            positive_class=positive_class
        )
    test_risk = compute_clinical_risk(test_cm, args.lambda_risk)

    # Individual models
    test_loss1, test_acc1, test_cm1, test_bac1, test_auc1, test_auprc1 = \
        evaluate(model1, test_loader, device, class_weights, positive_class)
    test_loss2, test_acc2, test_cm2, test_bac2, test_auc2, test_auprc2 = \
        evaluate(model2, test_loader, device, class_weights, positive_class)

    test_sensitivity = (test_cm['TP'] / (test_cm['TP'] + test_cm['FN'])
                        if (test_cm['TP'] + test_cm['FN']) > 0 else 0.0)
    test_specificity = (test_cm['TN'] / (test_cm['TN'] + test_cm['FP'])
                        if (test_cm['TN'] + test_cm['FP']) > 0 else 0.0)

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS (Best Epoch: {best_epoch}, selected by BAC)")
    print(f"Method: Co-teaching + CRASS  |  lambda={args.lambda_risk}")
    print("=" * 60)

    print(f"\nValidation (Ensemble):")
    print(f"  Accuracy: {best_val_acc:.4f}, BAC: {best_val_bac:.4f}")
    print(f"  AUC-ROC: {best_val_auc_roc:.4f}, AUPRC: {best_val_auprc:.4f}")
    print(f"  TP: {best_val_cm['TP']}, TN: {best_val_cm['TN']}, "
          f"FP: {best_val_cm['FP']}, FN: {best_val_cm['FN']}")

    print(f"\nTest (Ensemble):")
    print(f"  Accuracy:    {test_acc:.4f}")
    print(f"  BAC:         {test_bac:.4f}")
    print(f"  Sensitivity: {test_sensitivity:.4f}")
    print(f"  Specificity: {test_specificity:.4f}")
    print(f"  AUC-ROC:     {test_auc_roc:.4f}")
    print(f"  AUPRC:       {test_auprc:.4f}")
    print(f"  TP: {test_cm['TP']}, TN: {test_cm['TN']}, "
          f"FP: {test_cm['FP']}, FN: {test_cm['FN']}")
    print(f"  Clinical Risk (lambda={args.lambda_risk}): {test_risk:.4f}")

    print(f"\nTest (Net1): BAC={test_bac1:.4f}, AUC-ROC={test_auc1:.4f}")
    print(f"Test (Net2): BAC={test_bac2:.4f}, AUC-ROC={test_auc2:.4f}")

    # Save results ─────────────────────────────────────────────────────────
    results = {
        'dataset': args.dataset,
        'method': 'coteaching_crass',
        'best_epoch': best_epoch,
        'best_metric': 'BAC',
        'noise_rate': args.noise_rate,
        'forget_rate': args.forget_rate,
        'num_gradual': args.num_gradual,
        'exponent': args.exponent,
        'lambda_risk': args.lambda_risk,
        'warmup_epochs': args.warmup_epochs,
        'weighted_loss': args.weighted_loss,
        'class_weights': (class_weights.cpu().tolist()
                          if class_weights is not None else None),
        'validation': {
            'accuracy': best_val_acc, 'bac': best_val_bac,
            'auc_roc': best_val_auc_roc, 'auprc': best_val_auprc,
            **best_val_cm,
        },
        'test_ensemble': {
            'accuracy': test_acc, 'bac': test_bac,
            'sensitivity': test_sensitivity,
            'specificity': test_specificity,
            'auc_roc': test_auc_roc, 'auprc': test_auprc,
            'clinical_risk': test_risk,
            **test_cm,
        },
        'test_net1': {
            'accuracy': test_acc1, 'bac': test_bac1,
            'auc_roc': test_auc1, 'auprc': test_auprc1,
            **test_cm1,
        },
        'test_net2': {
            'accuracy': test_acc2, 'bac': test_bac2,
            'auc_roc': test_auc2, 'auprc': test_auprc2,
            **test_cm2,
        },
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(output_dir, 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=2)

    # TXT
    txt_path = os.path.join(output_dir, 'results.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EXPERIMENT RESULTS  --  Co-teaching + CRASS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Noise Rate: {args.noise_rate}\n")
        f.write(f"Forget Rate: {args.forget_rate}\n")
        f.write(f"Lambda Risk: {args.lambda_risk}\n")
        f.write(f"Warmup Epochs: {args.warmup_epochs}\n")
        f.write(f"Weighted Loss: {args.weighted_loss}\n")
        if class_weights is not None:
            f.write(f"Class Weights: {class_weights.cpu().tolist()}\n")
        f.write(f"Best Epoch: {best_epoch} (selected by BAC)\n\n")
        f.write(f"Keep rates (at full forget_rate):\n")
        f.write(f"  Negative class: {base_keep:.3f}\n")
        f.write(f"  Positive class: {keep_pos:.3f}  "
                f"(boost={boost:.3f})\n\n")

        f.write("-" * 60 + "\n")
        f.write("VALIDATION RESULTS (Ensemble)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy: {best_val_acc:.4f}\n")
        f.write(f"BAC: {best_val_bac:.4f}\n")
        f.write(f"AUC-ROC: {best_val_auc_roc:.4f}\n")
        f.write(f"AUPRC: {best_val_auprc:.4f}\n")
        f.write(f"TP: {best_val_cm['TP']}, TN: {best_val_cm['TN']}, "
                f"FP: {best_val_cm['FP']}, FN: {best_val_cm['FN']}\n\n")

        f.write("-" * 60 + "\n")
        f.write("TEST RESULTS (Ensemble)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy:    {test_acc:.4f}\n")
        f.write(f"BAC:         {test_bac:.4f}\n")
        f.write(f"Sensitivity: {test_sensitivity:.4f}\n")
        f.write(f"Specificity: {test_specificity:.4f}\n")
        f.write(f"AUC-ROC:     {test_auc_roc:.4f}\n")
        f.write(f"AUPRC:       {test_auprc:.4f}\n")
        f.write(f"Clinical Risk (lambda={args.lambda_risk}): "
                f"{test_risk:.4f}\n")
        f.write(f"TP: {test_cm['TP']}, TN: {test_cm['TN']}, "
                f"FP: {test_cm['FP']}, FN: {test_cm['FN']}\n\n")

        f.write("-" * 60 + "\n")
        f.write(f"TEST (Net1): BAC={test_bac1:.4f}, "
                f"AUC-ROC={test_auc1:.4f}\n")
        f.write(f"TEST (Net2): BAC={test_bac2:.4f}, "
                f"AUC-ROC={test_auc2:.4f}\n")

    print(f"\nResults saved to: {output_dir}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Train with Co-teaching + CRASS '
                    '(per-class risk-aware sample selection)'
    )

    # Data
    parser.add_argument('--dataset', type=str, default='dermamnist_bin',
                        choices=list(DATASET_CONFIG.keys()))
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results')

    # Noise
    parser.add_argument('--noise_rate', type=float, default=0.0,
                        help='Symmetric noise rate (0.0 to 1.0)')

    # Co-teaching
    parser.add_argument('--forget_rate', type=float, default=None,
                        help='Forget rate (defaults to noise_rate)')
    parser.add_argument('--num_gradual', type=int, default=10,
                        help='Epochs for gradual forget-rate ramp')
    parser.add_argument('--exponent', type=float, default=1.0,
                        help='Ramp schedule exponent (1 = linear)')

    # CRASS
    parser.add_argument('--lambda_risk', type=float, default=20.0,
                        help='Cost ratio C_FN / C_FP for CRASS boost')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Epochs of standard Co-teaching before '
                             'switching to CRASS selection')

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

    if args.forget_rate is None:
        args.forget_rate = args.noise_rate

    print("\n" + "=" * 60)
    print("EXPERIMENT CONFIGURATION  --  Co-teaching + CRASS")
    print("=" * 60)
    print(f"Dataset:       {args.dataset} (64x64)")
    print(f"Model:         2 x ResNet18")
    print(f"Noise Rate:    {args.noise_rate}")
    print(f"Forget Rate:   {args.forget_rate}")
    print(f"Num Gradual:   {args.num_gradual}")
    print(f"Lambda Risk:   {args.lambda_risk}")
    print(f"Warmup Epochs: {args.warmup_epochs}")
    print(f"Weighted Loss: {args.weighted_loss}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Learning Rate: {args.lr}")

    train(args)


if __name__ == '__main__':
    main()
