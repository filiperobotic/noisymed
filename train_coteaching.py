"""
Training Script with Co-teaching for Noisy Labels.

Co-teaching (Han et al., NeurIPS 2018):
    Two networks are trained simultaneously.  At each mini-batch, each
    network selects the samples with the *smallest loss* and feeds them
    to the *other* network for parameter update.  The selection ratio
    R(t) starts at 1 (use all samples) and gradually decreases to
    1 - forget_rate over ``num_gradual`` epochs.

Key hyper-parameters:
    - forget_rate:  fraction of samples to discard (set = noise_rate)
    - num_gradual:  number of epochs to linearly ramp down the keep ratio
    - exponent:     controls the shape of the ramp (1 = linear)

Usage:
    python train_coteaching.py --dataset pneumoniamnist --noise_rate 0.2
    python train_coteaching.py --dataset breastmnist --noise_rate 0.4 --weighted_loss
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


# ═══════════════════════════════════════════════════════════════════════════
# Co-teaching core
# ═══════════════════════════════════════════════════════════════════════════

def rate_schedule(epoch, forget_rate, num_gradual, exponent=1):
    """Compute the *forget rate* at the current epoch.

    During the first ``num_gradual`` epochs the forget rate ramps up from 0
    to ``forget_rate`` following a polynomial schedule.  After that it stays
    constant at ``forget_rate``.

    Returns:
        float – the fraction of samples to **discard** at this epoch.
    """
    if epoch < num_gradual:
        return forget_rate * min(
            (epoch / num_gradual) ** exponent, 1.0
        )
    return forget_rate


def coteaching_loss(logits1, logits2, labels, forget_rate, class_weights=None):
    """Compute Co-teaching loss for one mini-batch.

    Steps:
        1. Compute per-sample CE loss for both networks.
        2. Each network selects the ``keep_num`` samples with the smallest
           loss.
        3. Network 1 is trained on the samples selected by Network 2 and
           vice-versa.

    Args:
        logits1: Tensor (B, C) – output logits of network 1.
        logits2: Tensor (B, C) – output logits of network 2.
        labels:  Tensor (B,)   – (noisy) labels.
        forget_rate: float     – fraction of samples to discard.
        class_weights: optional Tensor (C,) – per-class CE weights.

    Returns:
        loss1, loss2: scalar losses for network 1 and network 2.
    """
    # Per-sample losses (unreduced)
    loss1_vec = F.cross_entropy(logits1, labels, weight=class_weights,
                                reduction='none')
    loss2_vec = F.cross_entropy(logits2, labels, weight=class_weights,
                                reduction='none')

    # Number of samples to keep
    batch_size = len(labels)
    keep_num = max(1, int((1 - forget_rate) * batch_size))

    # Network 1 selects small-loss samples → feed to Network 2
    _, idx1 = torch.sort(loss1_vec)
    idx_keep1 = idx1[:keep_num]

    # Network 2 selects small-loss samples → feed to Network 1
    _, idx2 = torch.sort(loss2_vec)
    idx_keep2 = idx2[:keep_num]

    # Network 1 trains on samples selected by Network 2
    loss1 = F.cross_entropy(logits1[idx_keep2], labels[idx_keep2],
                            weight=class_weights)
    # Network 2 trains on samples selected by Network 1
    loss2 = F.cross_entropy(logits2[idx_keep1], labels[idx_keep1],
                            weight=class_weights)

    return loss1, loss2


# ═══════════════════════════════════════════════════════════════════════════
# Training / evaluation loops
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch_coteaching(model1, model2, train_loader, optimizer1,
                           optimizer2, device, forget_rate,
                           class_weights=None, positive_class=1):
    """Train both Co-teaching networks for one epoch.

    Returns:
        (loss1, acc1, cm1), (loss2, acc2, cm2)
    """
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

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass through both networks
        logits1 = model1(images)
        logits2 = model2(images)

        # Co-teaching loss
        loss1, loss2 = coteaching_loss(
            logits1, logits2, labels, forget_rate, class_weights
        )

        # Update network 1
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        # Update network 2
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

    all_outputs1 = torch.cat(all_outputs1)
    all_outputs2 = torch.cat(all_outputs2)
    all_targets = torch.cat(all_targets)

    cm1 = compute_confusion_matrix(all_outputs1, all_targets, positive_class)
    cm2 = compute_confusion_matrix(all_outputs2, all_targets, positive_class)

    return (
        (running_loss1 / total, correct1 / total, cm1),
        (running_loss2 / total, correct2 / total, cm2),
    )


def evaluate(model, data_loader, device, class_weights=None,
             positive_class=1):
    """Evaluate a single model on validation/test set."""
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
    """Evaluate using the average of both networks' predictions.

    For methods with two networks (Co-teaching, DivideMix, UNICON) the
    final prediction is the average of the softmax outputs of both models.
    """
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

            # Average softmax probabilities
            probs1 = F.softmax(logits1, dim=1)
            probs2 = F.softmax(logits2, dim=1)
            avg_probs = (probs1 + probs2) / 2.0

            # Loss computed on the average logits (for logging)
            avg_logits = (logits1 + logits2) / 2.0
            if class_weights is not None:
                loss = F.cross_entropy(avg_logits, labels, weight=class_weights)
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

    all_avg_outputs = torch.cat(all_avg_outputs)  # (N, C) probabilities
    all_targets = torch.cat(all_targets)

    # Confusion matrix from averaged probabilities
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

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = (
        f"{args.dataset}_noise{args.noise_rate}"
        f"_coteaching"
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
    print("Creating two ResNet18 models for Co-teaching...")
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

    # Use different random initialization for model 2
    # (set_seed already initialized model1; re-seed for model2)
    torch.manual_seed(args.seed + 1)
    for m in model2.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    # Restore original seed for training
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
    print(f"Starting Co-teaching training  |  "
          f"forget_rate={args.forget_rate}  |  "
          f"num_gradual={args.num_gradual}")
    print("=" * 60)

    best_val_bac = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    best_val_cm = None
    best_val_auc_roc = 0.0
    best_val_auprc = 0.0

    training_log = []

    for epoch in range(1, args.epochs + 1):
        # Current forget rate from schedule
        current_forget_rate = rate_schedule(
            epoch - 1, args.forget_rate, args.num_gradual, args.exponent
        )

        # Train both models
        (loss1, acc1, cm1), (loss2, acc2, cm2) = train_epoch_coteaching(
            model1, model2, train_loader, optimizer1, optimizer2,
            device, current_forget_rate, class_weights,
            positive_class=positive_class
        )

        scheduler1.step()
        scheduler2.step()

        # Validate using ensemble (average of both networks)
        val_loss, val_acc, val_cm, val_bac, val_auc_roc, val_auprc = \
            evaluate_ensemble(
                model1, model2, val_loader, device, class_weights,
                positive_class=positive_class
            )

        # Also evaluate individual models for logging
        val_loss1, val_acc1, _, val_bac1, _, _ = evaluate(
            model1, val_loader, device, class_weights,
            positive_class=positive_class
        )
        val_loss2, val_acc2, _, val_bac2, _, _ = evaluate(
            model2, val_loader, device, class_weights,
            positive_class=positive_class
        )

        # Logging
        keep_ratio = 1.0 - current_forget_rate
        log_entry = {
            'epoch': epoch,
            'forget_rate': float(current_forget_rate),
            'keep_ratio': float(keep_ratio),
            'net1_train_loss': float(loss1),
            'net1_train_acc': float(acc1),
            'net1_val_acc': float(val_acc1),
            'net1_val_bac': float(val_bac1),
            'net2_train_loss': float(loss2),
            'net2_train_acc': float(acc2),
            'net2_val_acc': float(val_acc2),
            'net2_val_bac': float(val_bac2),
            'ensemble_val_acc': float(val_acc),
            'ensemble_val_bac': float(val_bac),
            'ensemble_val_auc_roc': float(val_auc_roc),
            'ensemble_val_auprc': float(val_auprc),
        }
        training_log.append(log_entry)

        if epoch % args.print_freq == 0 or epoch == 1:
            print(f"\nEpoch [{epoch}/{args.epochs}]  "
                  f"keep={keep_ratio:.2%}")
            print(f"  Net1 Train - Loss: {loss1:.4f}, Acc: {acc1:.4f}")
            print(f"  Net2 Train - Loss: {loss2:.4f}, Acc: {acc2:.4f}")
            print(f"  Net1 Val   - Acc: {val_acc1:.4f}, "
                  f"BAC: {val_bac1:.4f}")
            print(f"  Net2 Val   - Acc: {val_acc2:.4f}, "
                  f"BAC: {val_bac2:.4f}")
            print(f"  Ensemble   - Acc: {val_acc:.4f}, "
                  f"BAC: {val_bac:.4f}")
            print(f"               AUC-ROC: {val_auc_roc:.4f}, "
                  f"AUPRC: {val_auprc:.4f}")
            print(f"               TP: {val_cm['TP']}, TN: {val_cm['TN']}, "
                  f"FP: {val_cm['FP']}, FN: {val_cm['FN']}")

        # Save best model (by ensemble BAC)
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
    # Final evaluation on test set
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"Loading best models from epoch {best_epoch}...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model1.load_state_dict(checkpoint['model1_state_dict'])
    model2.load_state_dict(checkpoint['model2_state_dict'])

    # Ensemble test evaluation
    test_loss, test_acc, test_cm, test_bac, test_auc_roc, test_auprc = \
        evaluate_ensemble(
            model1, model2, test_loader, device, class_weights,
            positive_class=positive_class
        )

    # Individual model test evaluations
    test_loss1, test_acc1, test_cm1, test_bac1, test_auc1, test_auprc1 = \
        evaluate(model1, test_loader, device, class_weights,
                 positive_class=positive_class)
    test_loss2, test_acc2, test_cm2, test_bac2, test_auc2, test_auprc2 = \
        evaluate(model2, test_loader, device, class_weights,
                 positive_class=positive_class)

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS (Best Epoch: {best_epoch}, selected by BAC)")
    print("=" * 60)

    print(f"\nValidation (Ensemble):")
    print(f"  Accuracy: {best_val_acc:.4f}, BAC: {best_val_bac:.4f}")
    print(f"  AUC-ROC: {best_val_auc_roc:.4f}, AUPRC: {best_val_auprc:.4f}")
    print(f"  TP: {best_val_cm['TP']}, TN: {best_val_cm['TN']}, "
          f"FP: {best_val_cm['FP']}, FN: {best_val_cm['FN']}")

    print(f"\nTest (Ensemble):")
    print(f"  Accuracy: {test_acc:.4f}, BAC: {test_bac:.4f}")
    print(f"  AUC-ROC: {test_auc_roc:.4f}, AUPRC: {test_auprc:.4f}")
    print(f"  TP: {test_cm['TP']}, TN: {test_cm['TN']}, "
          f"FP: {test_cm['FP']}, FN: {test_cm['FN']}")

    print(f"\nTest (Net1 individual):")
    print(f"  Accuracy: {test_acc1:.4f}, BAC: {test_bac1:.4f}, "
          f"AUC-ROC: {test_auc1:.4f}")

    print(f"\nTest (Net2 individual):")
    print(f"  Accuracy: {test_acc2:.4f}, BAC: {test_bac2:.4f}, "
          f"AUC-ROC: {test_auc2:.4f}")

    # Save results ─────────────────────────────────────────────────────────
    results = {
        'dataset': args.dataset,
        'method': 'coteaching',
        'best_epoch': best_epoch,
        'best_metric': 'BAC',
        'noise_rate': args.noise_rate,
        'forget_rate': args.forget_rate,
        'num_gradual': args.num_gradual,
        'exponent': args.exponent,
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
            'auc_roc': test_auc_roc, 'auprc': test_auprc,
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

    # Save results to TXT
    txt_path = os.path.join(output_dir, 'results.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EXPERIMENT RESULTS  --  Co-teaching\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Noise Rate: {args.noise_rate}\n")
        f.write(f"Forget Rate: {args.forget_rate}\n")
        f.write(f"Num Gradual: {args.num_gradual}\n")
        f.write(f"Exponent: {args.exponent}\n")
        f.write(f"Weighted Loss: {args.weighted_loss}\n")
        if class_weights is not None:
            f.write(f"Class Weights: {class_weights.cpu().tolist()}\n")
        f.write(f"Best Epoch: {best_epoch} (selected by BAC)\n\n")

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
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"BAC: {test_bac:.4f}\n")
        f.write(f"AUC-ROC: {test_auc_roc:.4f}\n")
        f.write(f"AUPRC: {test_auprc:.4f}\n")
        f.write(f"TP: {test_cm['TP']}, TN: {test_cm['TN']}, "
                f"FP: {test_cm['FP']}, FN: {test_cm['FN']}\n\n")

        f.write("-" * 60 + "\n")
        f.write("TEST RESULTS (Net1 individual)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy: {test_acc1:.4f}, BAC: {test_bac1:.4f}, "
                f"AUC-ROC: {test_auc1:.4f}\n\n")

        f.write("-" * 60 + "\n")
        f.write("TEST RESULTS (Net2 individual)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy: {test_acc2:.4f}, BAC: {test_bac2:.4f}, "
                f"AUC-ROC: {test_auc2:.4f}\n")

    print(f"\nResults saved to: {output_dir}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Train with Co-teaching for noisy labels '
                    '(Han et al., NeurIPS 2018)'
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

    # Co-teaching
    parser.add_argument('--forget_rate', type=float, default=None,
                        help='Forget rate for Co-teaching. '
                             'Defaults to noise_rate if not set.')
    parser.add_argument('--num_gradual', type=int, default=10,
                        help='Number of epochs for gradual forget-rate ramp')
    parser.add_argument('--exponent', type=float, default=1.0,
                        help='Exponent for the ramp schedule (1 = linear)')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Loss
    parser.add_argument('--weighted_loss', action='store_true',
                        help='Use balanced weighted cross entropy loss')

    # Model
    parser.add_argument('--pretrained', action='store_true')

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_freq', type=int, default=10)

    args = parser.parse_args()

    # Default forget_rate = noise_rate
    if args.forget_rate is None:
        args.forget_rate = args.noise_rate

    print("\n" + "=" * 60)
    print("EXPERIMENT CONFIGURATION  --  Co-teaching")
    print("=" * 60)
    print(f"Dataset:      {args.dataset} (64x64)")
    print(f"Model:        2 x ResNet18")
    print(f"Noise Rate:   {args.noise_rate}")
    print(f"Forget Rate:  {args.forget_rate}")
    print(f"Num Gradual:  {args.num_gradual}")
    print(f"Exponent:     {args.exponent}")
    print(f"Weighted Loss:{args.weighted_loss}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch Size:   {args.batch_size}")
    print(f"Learning Rate:{args.lr}")

    train(args)


if __name__ == '__main__':
    main()
