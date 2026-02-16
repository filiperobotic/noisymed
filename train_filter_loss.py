"""
Training Script with GMM-based Loss Filtering for Noisy Labels.

Strategy:
    1. Warmup phase (default 10 epochs): train on ALL samples (no filtering).
    2. Filter phase (remaining epochs): at the start of each epoch, compute
       per-sample cross-entropy losses, fit a 2-component GMM **per class**
       to distinguish clean vs noisy samples, and train only on the samples
       whose probability of being clean exceeds a threshold (default 0.5).

The GMM fitting follows the DivideMix approach:
    - Losses are min-max normalized per class.
    - A history of losses is kept; for high noise rates (>= 0.9) the last 5
      epochs are averaged for stability.
    - The clean component is identified as the one with the smaller mean.

Usage:
    python train_filter_loss.py --dataset pneumoniamnist --noise_rate 0.2 --weighted_loss
    python train_filter_loss.py --dataset dermamnist_bin --noise_rate 0.4 --warmup_epochs 15
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
# Helpers shared with train_baseline.py
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


# ═══════════════════════════════════════════════════════════════════════════
# GMM-based loss evaluation
# ═══════════════════════════════════════════════════════════════════════════

def eval_train_losses(model, eval_loader, device, n_samples):
    """Compute per-sample cross-entropy loss (unreduced) for the training set.

    Args:
        model: trained model (will be set to eval mode).
        eval_loader: DataLoader that yields (image, label, index).
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


def gmm_filter_per_class(losses, labels, all_loss_history, noise_rate,
                          threshold=0.5):
    """Fit a 2-component GMM **per class** and return clean sample indices.

    For each class independently:
        1. Select loss values belonging to that class.
        2. Min-max normalize them.
        3. If noise_rate >= 0.9, average last 5 epochs of history for stability.
        4. Fit a 2-component GMM.
        5. Compute P(clean) = probability of belonging to the low-loss component.
        6. Keep samples with P(clean) > threshold.

    Args:
        losses: Tensor (n_samples,) – raw per-sample losses for current epoch.
        labels: numpy array (n_samples,) – (noisy) labels.
        all_loss_history: list of Tensors – loss history from previous epochs.
        noise_rate: float – injected noise rate (used for history averaging).
        threshold: float – probability threshold to keep a sample.

    Returns:
        clean_indices: numpy array of global indices considered clean.
        all_loss_history: updated history list (with current losses appended).
        filter_info: dict with per-class statistics for logging.
    """
    labels = np.array(labels).flatten()
    n_samples = len(labels)
    unique_classes = np.unique(labels[labels >= 0])  # ignore marked negatives

    # Append current losses to history
    all_loss_history.append(losses.clone())

    clean_mask = np.zeros(n_samples, dtype=bool)
    filter_info = {}

    for cls in unique_classes:
        cls = int(cls)
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) == 0:
            continue

        cls_losses = losses[cls_idx]

        # Min-max normalize within class
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

        # Select clean samples
        cls_clean_mask = prob_clean > threshold
        clean_mask[cls_idx[cls_clean_mask]] = True

        n_total_cls = len(cls_idx)
        n_clean_cls = int(cls_clean_mask.sum())
        filter_info[cls] = {
            'total': n_total_cls,
            'clean': n_clean_cls,
            'filtered': n_total_cls - n_clean_cls,
            'gmm_means': gmm.means_.flatten().tolist(),
        }

    clean_indices = np.where(clean_mask)[0]
    return clean_indices, all_loss_history, filter_info


# ═══════════════════════════════════════════════════════════════════════════
# Training loops
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch_full(model, train_loader, optimizer, device,
                     class_weights=None, positive_class=1):
    """Train one epoch on ALL samples (warmup / no filtering).

    train_loader yields (image, label, index) – index is ignored here.
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
    """Train one epoch on the filtered (clean) subset only.

    Creates a temporary Subset + DataLoader from the clean indices.
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
    cm = compute_confusion_matrix(all_outputs, all_targets, positive_class) if total > 0 else {
        'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    return epoch_loss, accuracy, cm


def evaluate(model, data_loader, device, class_weights=None,
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

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = (
        f"{args.dataset}_noise{args.noise_rate}"
        f"_filter_loss"
        f"_warmup{args.warmup_epochs}"
        f"_thr{args.filter_threshold}"
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

    # Class weights ────────────────────────────────────────────────────────
    class_weights = None
    if args.weighted_loss:
        class_weights = infer_class_weights(train_labels, device)
        if class_weights is not None:
            print(f"\nUsing balanced class weights: {class_weights.cpu().tolist()}")
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
    print(f"Starting training  |  warmup={args.warmup_epochs} epochs  |  "
          f"filter threshold={args.filter_threshold}")
    print("=" * 60)

    all_loss_history = []  # list[Tensor(n_train,)]
    best_val_bac = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    best_val_cm = None
    best_val_auc_roc = 0.0
    best_val_auprc = 0.0

    # Keep track of filtering stats per epoch
    filter_log = []

    for epoch in range(1, args.epochs + 1):
        is_warmup = epoch <= args.warmup_epochs

        if is_warmup:
            # ── Warmup: train on all samples ──────────────────────────────
            train_loss, train_acc, train_cm = train_epoch_full(
                model, train_loader, optimizer, device, class_weights,
                positive_class=positive_class
            )
            n_used = n_train
            epoch_filter_info = None
        else:
            # ── Filter phase ──────────────────────────────────────────────
            # 1) Compute per-sample losses
            losses = eval_train_losses(model, eval_loader, device, n_train)

            # 2) GMM filtering per class
            clean_indices, all_loss_history, epoch_filter_info = \
                gmm_filter_per_class(
                    losses, train_labels, all_loss_history,
                    noise_rate=args.noise_rate,
                    threshold=args.filter_threshold
                )

            n_used = len(clean_indices)

            # 3) Train only on clean subset
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

        # Logging ──────────────────────────────────────────────────────────
        phase_tag = "WARMUP" if is_warmup else "FILTER"
        filter_log_entry = {
            'epoch': epoch,
            'phase': phase_tag,
            'samples_used': int(n_used),
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_acc': float(val_acc),
            'val_bac': float(val_bac),
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
            if epoch_filter_info is not None:
                for cls, info in sorted(epoch_filter_info.items()):
                    print(f"          Class {cls}: kept {info['clean']}"
                          f"/{info['total']} "
                          f"(filtered {info['filtered']}), "
                          f"GMM means={[f'{m:.4f}' for m in info['gmm_means']]}")

        # Best model (by BAC) ──────────────────────────────────────────────
        if val_bac > best_val_bac:
            best_val_bac = val_bac
            best_val_acc = val_acc
            best_epoch = epoch
            best_val_cm = val_cm
            best_val_auc_roc = val_auc_roc
            best_val_auprc = val_auprc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
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
    print(f"Loading best model from epoch {best_epoch}...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_cm, test_bac, test_auc_roc, test_auprc = \
        evaluate(model, test_loader, device, class_weights,
                 positive_class=positive_class)

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS (Best Epoch: {best_epoch}, selected by BAC)")
    print("=" * 60)
    print(f"\nValidation:")
    print(f"  Accuracy: {best_val_acc:.4f}, BAC: {best_val_bac:.4f}")
    print(f"  AUC-ROC: {best_val_auc_roc:.4f}, AUPRC: {best_val_auprc:.4f}")
    print(f"  TP: {best_val_cm['TP']}, TN: {best_val_cm['TN']}, "
          f"FP: {best_val_cm['FP']}, FN: {best_val_cm['FN']}")
    print(f"\nTest:")
    print(f"  Accuracy: {test_acc:.4f}, BAC: {test_bac:.4f}")
    print(f"  AUC-ROC: {test_auc_roc:.4f}, AUPRC: {test_auprc:.4f}")
    print(f"  TP: {test_cm['TP']}, TN: {test_cm['TN']}, "
          f"FP: {test_cm['FP']}, FN: {test_cm['FN']}")

    # Save results ─────────────────────────────────────────────────────────
    results = {
        'dataset': args.dataset,
        'method': 'filter_loss_gmm',
        'best_epoch': best_epoch,
        'best_metric': 'BAC',
        'noise_rate': args.noise_rate,
        'warmup_epochs': args.warmup_epochs,
        'filter_threshold': args.filter_threshold,
        'weighted_loss': args.weighted_loss,
        'class_weights': (class_weights.cpu().tolist()
                          if class_weights is not None else None),
        'validation': {
            'accuracy': best_val_acc, 'bac': best_val_bac,
            'auc_roc': best_val_auc_roc, 'auprc': best_val_auprc,
            **best_val_cm
        },
        'test': {
            'accuracy': test_acc, 'bac': test_bac,
            'auc_roc': test_auc_roc, 'auprc': test_auprc,
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
        f.write("EXPERIMENT RESULTS  —  Loss Filter (GMM per class)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Noise Rate: {args.noise_rate}\n")
        f.write(f"Warmup Epochs: {args.warmup_epochs}\n")
        f.write(f"Filter Threshold: {args.filter_threshold}\n")
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
        f.write(f"TP: {best_val_cm['TP']}, TN: {best_val_cm['TN']}, "
                f"FP: {best_val_cm['FP']}, FN: {best_val_cm['FN']}\n\n")
        f.write("-" * 60 + "\n")
        f.write("TEST RESULTS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"BAC: {test_bac:.4f}\n")
        f.write(f"AUC-ROC: {test_auc_roc:.4f}\n")
        f.write(f"AUPRC: {test_auprc:.4f}\n")
        f.write(f"TP: {test_cm['TP']}, TN: {test_cm['TN']}, "
                f"FP: {test_cm['FP']}, FN: {test_cm['FN']}\n")

    print(f"\nResults saved to: {output_dir}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Train with GMM-based loss filtering for noisy labels'
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

    # Filter
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs before filtering starts')
    parser.add_argument('--filter_threshold', type=float, default=0.5,
                        help='P(clean) threshold for keeping a sample')

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

    print("\n" + "=" * 60)
    print("EXPERIMENT CONFIGURATION  —  Loss Filter (GMM)")
    print("=" * 60)
    print(f"Dataset:          {args.dataset} (64x64)")
    print(f"Model:            ResNet18")
    print(f"Noise Rate:       {args.noise_rate}")
    print(f"Warmup Epochs:    {args.warmup_epochs}")
    print(f"Filter Threshold: {args.filter_threshold}")
    print(f"Weighted Loss:    {args.weighted_loss}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Learning Rate:    {args.lr}")

    train(args)


if __name__ == '__main__':
    main()
