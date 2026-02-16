"""
Training Script with DISC for Noisy Labels.

DISC (Li et al., CVPR 2023):
    Dynamic Instance-specific Selection and Correction.
    Key ideas:
    1. **EMA teacher**: An exponential moving average copy of the student
       network provides stable predictions for confidence estimation.
    2. **Two-view confidence**: Each sample is augmented twice; the average
       softmax output of the EMA teacher over both views estimates label
       confidence.
    3. **Memorization strength**: Tracks how consistently the model has
       correctly predicted each sample over training history (exponentially
       smoothed accuracy).
    4. **Dynamic threshold**: A per-sample threshold that combines the
       global noise rate, memorization strength, and a learnable/schedule
       component to decide whether to keep, correct, or discard a sample.
    5. **Label correction**: Samples above the confidence threshold use a
       corrected (teacher's) label; samples below are either kept with
       the original label or discarded.

Key hyper-parameters:
    - warmup_epochs:   warmup epochs (standard CE)
    - ema_decay:       EMA decay rate for the teacher (default 0.999)
    - threshold_init:  initial confidence threshold (default 0.5)

Usage:
    python train_disc.py --dataset pneumoniamnist --noise_rate 0.2
    python train_disc.py --dataset breastmnist --noise_rate 0.4 --weighted_loss
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
import torchvision.transforms as transforms
import json
import copy
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score
from PIL import Image

from dataloaders import get_filter_dataloaders, DATASET_CONFIG


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
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
    return torch.tensor([n / (2.0 * n0), n / (2.0 * n1)],
                        dtype=torch.float32, device=device)


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
# EMA helper
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def update_ema(student, teacher, decay):
    """Update teacher parameters as exponential moving average of student."""
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(decay).add_(s_param.data, alpha=1 - decay)


# ═══════════════════════════════════════════════════════════════════════════
# Two-view dataset
# ═══════════════════════════════════════════════════════════════════════════

class TwoViewIndexedDataset(Dataset):
    """Returns two augmented views + label + index for DISC."""

    def __init__(self, base_dataset, noisy_labels=None,
                 aug_transform=None, normalize=None):
        self.images = base_dataset.imgs
        self.labels = (np.array(noisy_labels).flatten()
                       if noisy_labels is not None
                       else np.array(base_dataset.labels).flatten())
        self.aug_transform = aug_transform
        self.normalize = normalize

    def __len__(self):
        return len(self.images)

    def _to_pil(self, image):
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return Image.fromarray(image, mode='L')
            elif image.shape[2] == 1:
                return Image.fromarray(image.squeeze(2), mode='L')
            else:
                return Image.fromarray(image, mode='RGB')
        return image

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        pil_image = self._to_pil(image)

        if self.aug_transform is not None:
            view1 = self.aug_transform(pil_image)
            view2 = self.aug_transform(pil_image)
        else:
            view1 = transforms.ToTensor()(pil_image)
            view2 = transforms.ToTensor()(pil_image)

        if self.normalize is not None:
            view1 = self.normalize(view1)
            view2 = self.normalize(view2)

        return view1, view2, torch.tensor(label, dtype=torch.long), idx


# ═══════════════════════════════════════════════════════════════════════════
# DISC core: confidence estimation + selection + correction
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_teacher_confidence(teacher, two_view_loader, device, n_samples,
                               num_classes):
    """Compute per-sample confidence from the EMA teacher using two views.

    For each sample:
        confidence[i] = average softmax probability for the noisy label
                        across both augmented views.

    Returns:
        confidence: Tensor (n_samples,)
        teacher_preds: Tensor (n_samples, num_classes) — average softmax.
    """
    teacher.eval()
    confidence = torch.zeros(n_samples)
    teacher_preds = torch.zeros(n_samples, num_classes)

    for view1, view2, labels, indices in two_view_loader:
        view1 = view1.to(device)
        view2 = view2.to(device)
        labels = labels.to(device)

        probs1 = F.softmax(teacher(view1), dim=1)
        probs2 = F.softmax(teacher(view2), dim=1)
        avg_probs = (probs1 + probs2) / 2.0

        for b in range(view1.size(0)):
            idx = indices[b].item()
            teacher_preds[idx] = avg_probs[b].cpu()
            # Confidence for the noisy label
            confidence[idx] = avg_probs[b, labels[b]].cpu()

    return confidence, teacher_preds


def update_memorization(memorization, model, eval_loader, device,
                        n_samples, ema_mem=0.9):
    """Update per-sample memorization strength.

    memorization[i] = ema_mem * memorization[i] + (1 - ema_mem) * correct[i]

    where correct[i] = 1 if the model correctly predicts sample i's noisy
    label, else 0.
    """
    model.eval()
    correct = torch.zeros(n_samples)

    with torch.no_grad():
        for images, labels, indices in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for b in range(images.size(0)):
                idx = indices[b].item()
                correct[idx] = float(predicted[b] == labels[b])

    memorization = ema_mem * memorization + (1 - ema_mem) * correct
    return memorization


def disc_select_and_correct(confidence, memorization, teacher_preds,
                            noisy_labels, noise_rate, threshold_init,
                            epoch, total_epochs, num_classes):
    """DISC sample selection and label correction.

    Three categories:
        1. **Clean + Keep**: High confidence → use corrected (teacher) label.
        2. **Ambiguous**: Medium confidence → keep noisy label.
        3. **Noisy + Discard**: Low confidence → discard.

    The threshold adapts dynamically:
        threshold = threshold_init * (1 - memorization[i]) * schedule(epoch)

    Args:
        confidence:     Tensor (N,) — teacher confidence for noisy label.
        memorization:   Tensor (N,) — per-sample memorization strength.
        teacher_preds:  Tensor (N, C) — teacher's average softmax.
        noisy_labels:   numpy (N,) — noisy labels.
        noise_rate:     float.
        threshold_init: float.
        epoch:          current epoch (1-indexed).
        total_epochs:   total epochs.
        num_classes:    number of classes.

    Returns:
        selected_indices: numpy array of indices to train on.
        corrected_labels: Tensor (N, C) — soft labels for selected samples.
        disc_info:        dict for logging.
    """
    n_samples = len(noisy_labels)

    # Dynamic schedule: starts high (conservative), decreases over time
    # This means early epochs are more conservative (keep more), later
    # epochs are more aggressive (discard more noisy)
    schedule = max(0.5, 1.0 - (epoch / total_epochs))

    # Per-sample dynamic threshold
    # Low memorization → higher threshold (harder to pass → discard)
    # High memorization → lower threshold (easier to pass → keep)
    thresholds = threshold_init * (1.0 - memorization) * schedule

    # Categorize
    clean_mask = confidence > (1.0 - thresholds)     # High confidence
    discard_mask = confidence < thresholds             # Low confidence
    ambiguous_mask = ~clean_mask & ~discard_mask       # In between

    # Selected = clean + ambiguous (discard only the clearly noisy)
    selected_mask = clean_mask | ambiguous_mask
    selected_indices = np.where(selected_mask.numpy())[0]

    # Corrected labels for selected samples
    corrected_labels = torch.zeros(n_samples, num_classes)
    for i in range(n_samples):
        if clean_mask[i]:
            # Use teacher's prediction (soft label)
            corrected_labels[i] = teacher_preds[i]
        else:
            # Keep noisy label (hard)
            corrected_labels[i, int(noisy_labels[i])] = 1.0

    disc_info = {
        'clean': int(clean_mask.sum()),
        'ambiguous': int(ambiguous_mask.sum()),
        'discarded': int(discard_mask.sum()),
        'selected': len(selected_indices),
        'avg_confidence': float(confidence.mean()),
        'avg_memorization': float(memorization.mean()),
        'schedule': float(schedule),
    }

    return selected_indices, corrected_labels, disc_info


# ═══════════════════════════════════════════════════════════════════════════
# Datasets for DISC training
# ═══════════════════════════════════════════════════════════════════════════

class CorrectedLabelDataset(Dataset):
    """Dataset with corrected soft labels for selected samples."""

    def __init__(self, base_dataset, indices, corrected_labels,
                 aug_transform=None, normalize=None):
        """
        Args:
            base_dataset: has .imgs attribute.
            indices: selected sample indices.
            corrected_labels: Tensor (N_total, C) soft labels.
            aug_transform: augmentation.
            normalize: normalization.
        """
        self.images = base_dataset.imgs
        self.indices = np.array(indices)
        self.corrected_labels = corrected_labels
        self.aug_transform = aug_transform
        self.normalize = normalize

    def __len__(self):
        return len(self.indices)

    def _to_pil(self, image):
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return Image.fromarray(image, mode='L')
            elif image.shape[2] == 1:
                return Image.fromarray(image.squeeze(2), mode='L')
            else:
                return Image.fromarray(image, mode='RGB')
        return image

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image = self.images[real_idx]
        pil_image = self._to_pil(image)

        if self.aug_transform is not None:
            image = self.aug_transform(pil_image)
        else:
            image = transforms.ToTensor()(pil_image)
        if self.normalize is not None:
            image = self.normalize(image)

        soft_label = self.corrected_labels[real_idx]
        return image, soft_label


# ═══════════════════════════════════════════════════════════════════════════
# Training loops
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch_warmup(model, train_loader, optimizer, device,
                       class_weights=None, positive_class=1):
    """Warmup: standard CE on all samples."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []

    for batch in train_loader:
        if len(batch) == 3:
            images, labels, _ = batch
        elif len(batch) == 4:
            images, _, labels, _ = batch  # two-view: use first view
        else:
            images, labels = batch
        images, labels = images.to(device), labels.to(device)

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

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    cm = compute_confusion_matrix(all_outputs, all_targets, positive_class)

    return running_loss / total, correct / total, cm


def train_epoch_disc(model, corrected_loader, optimizer, device,
                     class_weights=None, num_classes=2, positive_class=1):
    """DISC training epoch on selected samples with corrected soft labels.

    Loss: soft cross-entropy (CE with soft targets).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, soft_labels in corrected_loader:
        images = images.to(device)
        soft_labels = soft_labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Soft CE loss
        log_probs = F.log_softmax(outputs, dim=1)
        loss = -torch.mean(torch.sum(soft_labels * log_probs, dim=1))

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        _, hard_labels = torch.max(soft_labels, 1)
        correct += (predicted == hard_labels).sum().item()
        total += images.size(0)

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)

    return avg_loss, accuracy


def evaluate(model, data_loader, device, class_weights=None,
             positive_class=1):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            images, labels = images.to(device), labels.to(device)

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

def train_main(args):
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = DATASET_CONFIG[args.dataset]
    positive_class = config.get('positive_class', 1)
    num_classes = config['num_classes']
    in_channels = config['in_channels']
    print(f"Positive class: {positive_class} "
          f"({config['class_names'].get(positive_class, '?')})")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = (
        f"{args.dataset}_noise{args.noise_rate}"
        f"_disc"
        f"_warmup{args.warmup_epochs}"
        f"_ema{args.ema_decay}_thr{args.threshold_init}"
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
    train_loader, eval_loader, val_loader, test_loader = \
        get_filter_dataloaders(
            dataset=args.dataset, noise_rate=args.noise_rate,
            batch_size=args.batch_size, seed=args.seed, data_dir=args.data_dir
        )

    train_dataset = train_loader.dataset  # IndexedNoisyMedMNISTDataset
    n_train = len(train_dataset)
    train_labels = np.array(train_dataset.labels).flatten()

    print(f"Train samples: {n_train}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    # Transforms
    im_size = 64
    if in_channels == 1:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    else:
        normalize = transforms.Normalize(
            mean=[0.5] * in_channels, std=[0.5] * in_channels
        )
    train_aug_transform = transforms.Compose([
        transforms.RandomCrop(im_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # Two-view dataset for confidence estimation
    two_view_ds = TwoViewIndexedDataset(
        train_dataset, noisy_labels=train_labels,
        aug_transform=train_aug_transform, normalize=normalize
    )
    two_view_loader = DataLoader(
        two_view_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Class weights
    class_weights = None
    if args.weighted_loss:
        class_weights = infer_class_weights(train_labels, device)
        if class_weights is not None:
            print(f"\nUsing balanced class weights: "
                  f"{class_weights.cpu().tolist()}")
    else:
        print("\nUsing standard Cross Entropy loss")

    # Model + EMA teacher
    print("\n" + "=" * 60)
    print("Creating ResNet18 (student) + EMA teacher for DISC...")
    student = get_model(num_classes, in_channels, args.pretrained).to(device)
    teacher = copy.deepcopy(student)
    # Teacher does not require gradients
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = optim.SGD(student.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Per-sample memorization strength
    memorization = torch.zeros(n_train)

    # Training
    print("\n" + "=" * 60)
    print(f"Starting DISC training  |  warmup={args.warmup_epochs}  |  "
          f"EMA decay={args.ema_decay}  |  threshold_init={args.threshold_init}")
    print("=" * 60)

    best_val_bac = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    best_val_cm = None
    best_val_auc_roc = 0.0
    best_val_auprc = 0.0
    training_log = []

    for epoch in range(1, args.epochs + 1):
        is_warmup = epoch <= args.warmup_epochs

        if is_warmup:
            train_loss, train_acc, train_cm = train_epoch_warmup(
                student, train_loader, optimizer, device, class_weights,
                positive_class
            )
            # Update EMA teacher
            update_ema(student, teacher, args.ema_decay)
            n_selected = n_train
            disc_info = None
        else:
            # ── DISC phase ────────────────────────────────────────────

            # 1) Update memorization
            memorization = update_memorization(
                memorization, student, eval_loader, device, n_train,
                ema_mem=0.9
            )

            # 2) Teacher confidence (two views)
            confidence, teacher_preds = compute_teacher_confidence(
                teacher, two_view_loader, device, n_train, num_classes
            )

            # 3) Dynamic selection + correction
            selected_indices, corrected_labels, disc_info = \
                disc_select_and_correct(
                    confidence, memorization, teacher_preds,
                    train_labels, args.noise_rate, args.threshold_init,
                    epoch, args.epochs, num_classes
                )

            n_selected = len(selected_indices)

            if n_selected > 0:
                # 4) Train on selected samples with corrected labels
                corrected_ds = CorrectedLabelDataset(
                    train_dataset, selected_indices, corrected_labels,
                    aug_transform=train_aug_transform, normalize=normalize
                )
                corrected_loader = DataLoader(
                    corrected_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=False
                )
                train_loss, train_acc = train_epoch_disc(
                    student, corrected_loader, optimizer, device,
                    class_weights, num_classes, positive_class
                )
            else:
                train_loss, train_acc = 0.0, 0.0

            # 5) Update EMA teacher
            update_ema(student, teacher, args.ema_decay)

        scheduler.step()

        # Validate
        val_loss, val_acc, val_cm, val_bac, val_auc_roc, val_auprc = \
            evaluate(student, val_loader, device, class_weights,
                     positive_class)

        phase_tag = "WARMUP" if is_warmup else "DISC"
        log_entry = {
            'epoch': epoch,
            'phase': phase_tag,
            'n_selected': int(n_selected),
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_acc': float(val_acc),
            'val_bac': float(val_bac),
            'val_auc_roc': float(val_auc_roc),
        }
        if disc_info is not None:
            log_entry['disc_info'] = disc_info
        training_log.append(log_entry)

        if epoch % args.print_freq == 0 or epoch == 1:
            print(f"\nEpoch [{epoch}/{args.epochs}]  ({phase_tag})  "
                  f"selected={n_selected}/{n_train}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Acc: {val_acc:.4f}, BAC: {val_bac:.4f}")
            print(f"          AUC-ROC: {val_auc_roc:.4f}, "
                  f"AUPRC: {val_auprc:.4f}")
            print(f"          TP: {val_cm['TP']}, TN: {val_cm['TN']}, "
                  f"FP: {val_cm['FP']}, FN: {val_cm['FN']}")
            if disc_info is not None:
                print(f"          DISC: clean={disc_info['clean']}, "
                      f"ambiguous={disc_info['ambiguous']}, "
                      f"discarded={disc_info['discarded']}")
                print(f"          avg_conf={disc_info['avg_confidence']:.4f}, "
                      f"avg_mem={disc_info['avg_memorization']:.4f}")

        # Best model
        if val_bac > best_val_bac:
            best_val_bac = val_bac
            best_val_acc = val_acc
            best_epoch = epoch
            best_val_cm = val_cm
            best_val_auc_roc = val_auc_roc
            best_val_auprc = val_auprc
            torch.save({
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'val_acc': val_acc,
                'val_bac': val_bac,
                'val_cm': val_cm,
                'val_auc_roc': val_auc_roc,
                'val_auprc': val_auprc,
            }, os.path.join(output_dir, 'best_model.pth'))

    # ── Final test ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Loading best model from epoch {best_epoch}...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    student.load_state_dict(checkpoint['student_state_dict'])
    teacher.load_state_dict(checkpoint['teacher_state_dict'])

    # Test student
    test_loss, test_acc, test_cm, test_bac, test_auc_roc, test_auprc = \
        evaluate(student, test_loader, device, class_weights, positive_class)

    # Test teacher
    test_loss_t, test_acc_t, test_cm_t, test_bac_t, test_auc_t, test_auprc_t = \
        evaluate(teacher, test_loader, device, class_weights, positive_class)

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS (Best Epoch: {best_epoch}, selected by BAC)")
    print("=" * 60)

    print(f"\nValidation (Student):")
    print(f"  Accuracy: {best_val_acc:.4f}, BAC: {best_val_bac:.4f}")
    print(f"  AUC-ROC: {best_val_auc_roc:.4f}, AUPRC: {best_val_auprc:.4f}")
    print(f"  TP: {best_val_cm['TP']}, TN: {best_val_cm['TN']}, "
          f"FP: {best_val_cm['FP']}, FN: {best_val_cm['FN']}")

    print(f"\nTest (Student):")
    print(f"  Accuracy: {test_acc:.4f}, BAC: {test_bac:.4f}")
    print(f"  AUC-ROC: {test_auc_roc:.4f}, AUPRC: {test_auprc:.4f}")
    print(f"  TP: {test_cm['TP']}, TN: {test_cm['TN']}, "
          f"FP: {test_cm['FP']}, FN: {test_cm['FN']}")

    print(f"\nTest (Teacher/EMA):")
    print(f"  Accuracy: {test_acc_t:.4f}, BAC: {test_bac_t:.4f}")
    print(f"  AUC-ROC: {test_auc_t:.4f}, AUPRC: {test_auprc_t:.4f}")
    print(f"  TP: {test_cm_t['TP']}, TN: {test_cm_t['TN']}, "
          f"FP: {test_cm_t['FP']}, FN: {test_cm_t['FN']}")

    # Save results
    results = {
        'dataset': args.dataset,
        'method': 'disc',
        'best_epoch': best_epoch,
        'best_metric': 'BAC',
        'noise_rate': args.noise_rate,
        'warmup_epochs': args.warmup_epochs,
        'ema_decay': args.ema_decay,
        'threshold_init': args.threshold_init,
        'weighted_loss': args.weighted_loss,
        'class_weights': (class_weights.cpu().tolist()
                          if class_weights is not None else None),
        'validation': {
            'accuracy': best_val_acc, 'bac': best_val_bac,
            'auc_roc': best_val_auc_roc, 'auprc': best_val_auprc,
            **best_val_cm,
        },
        'test_student': {
            'accuracy': test_acc, 'bac': test_bac,
            'auc_roc': test_auc_roc, 'auprc': test_auprc,
            **test_cm,
        },
        'test_teacher': {
            'accuracy': test_acc_t, 'bac': test_bac_t,
            'auc_roc': test_auc_t, 'auprc': test_auprc_t,
            **test_cm_t,
        },
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(output_dir, 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=2)

    txt_path = os.path.join(output_dir, 'results.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EXPERIMENT RESULTS  --  DISC\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Noise Rate: {args.noise_rate}\n")
        f.write(f"Warmup Epochs: {args.warmup_epochs}\n")
        f.write(f"EMA Decay: {args.ema_decay}\n")
        f.write(f"Threshold Init: {args.threshold_init}\n")
        f.write(f"Weighted Loss: {args.weighted_loss}\n")
        if class_weights is not None:
            f.write(f"Class Weights: {class_weights.cpu().tolist()}\n")
        f.write(f"Best Epoch: {best_epoch} (selected by BAC)\n\n")

        f.write("-" * 60 + "\n")
        f.write("VALIDATION (Student)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy: {best_val_acc:.4f}\n")
        f.write(f"BAC: {best_val_bac:.4f}\n")
        f.write(f"AUC-ROC: {best_val_auc_roc:.4f}\n")
        f.write(f"AUPRC: {best_val_auprc:.4f}\n")
        f.write(f"TP: {best_val_cm['TP']}, TN: {best_val_cm['TN']}, "
                f"FP: {best_val_cm['FP']}, FN: {best_val_cm['FN']}\n\n")

        f.write("-" * 60 + "\n")
        f.write("TEST (Student)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"BAC: {test_bac:.4f}\n")
        f.write(f"AUC-ROC: {test_auc_roc:.4f}\n")
        f.write(f"AUPRC: {test_auprc:.4f}\n")
        f.write(f"TP: {test_cm['TP']}, TN: {test_cm['TN']}, "
                f"FP: {test_cm['FP']}, FN: {test_cm['FN']}\n\n")

        f.write("-" * 60 + "\n")
        f.write("TEST (Teacher/EMA)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy: {test_acc_t:.4f}\n")
        f.write(f"BAC: {test_bac_t:.4f}\n")
        f.write(f"AUC-ROC: {test_auc_t:.4f}\n")
        f.write(f"AUPRC: {test_auprc_t:.4f}\n")
        f.write(f"TP: {test_cm_t['TP']}, TN: {test_cm_t['TN']}, "
                f"FP: {test_cm_t['FP']}, FN: {test_cm_t['FN']}\n")

    print(f"\nResults saved to: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train with DISC for noisy labels '
                    '(Li et al., CVPR 2023)'
    )

    parser.add_argument('--dataset', type=str, default='pneumoniamnist',
                        choices=list(DATASET_CONFIG.keys()))
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--noise_rate', type=float, default=0.0)

    # DISC
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay for teacher model')
    parser.add_argument('--threshold_init', type=float, default=0.5,
                        help='Initial confidence threshold')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--weighted_loss', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_freq', type=int, default=10)

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("EXPERIMENT CONFIGURATION  --  DISC")
    print("=" * 60)
    print(f"Dataset:        {args.dataset} (64x64)")
    print(f"Model:          ResNet18 (Student + EMA Teacher)")
    print(f"Noise Rate:     {args.noise_rate}")
    print(f"Warmup:         {args.warmup_epochs}")
    print(f"EMA Decay:      {args.ema_decay}")
    print(f"Threshold Init: {args.threshold_init}")
    print(f"Weighted Loss:  {args.weighted_loss}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch Size:     {args.batch_size}")
    print(f"Learning Rate:  {args.lr}")

    train_main(args)


if __name__ == '__main__':
    main()
