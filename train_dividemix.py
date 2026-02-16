"""
Training Script with DivideMix for Noisy Labels.

DivideMix (Li et al., ICLR 2020):
    Combines two main ideas:
    1. **Co-divide**: Two networks maintain independent GMM models to split
       data into a clean (labeled) set and a noisy (unlabeled) set.  Each
       network's GMM provides the division used by the *other* network.
    2. **MixMatch-style semi-supervised learning**: The clean set is used as
       labeled data and the noisy set as unlabeled data.  MixUp is applied
       to both, and a sharpened pseudo-label is created for the unlabeled
       set from the average prediction of both networks.

Key hyper-parameters:
    - warmup_epochs: number of warmup epochs (standard CE training)
    - alpha:         Beta distribution parameter for MixUp (default 4)
    - lambda_u:      weight for the unsupervised (unlabeled) loss
    - T:             sharpening temperature
    - p_threshold:   GMM probability threshold for clean/noisy split

Usage:
    python train_dividemix.py --dataset pneumoniamnist --noise_rate 0.2
    python train_dividemix.py --dataset breastmnist --noise_rate 0.4 --weighted_loss
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
from torch.utils.data import DataLoader, Dataset
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, average_precision_score

from dataloaders import (
    get_filter_dataloaders, DATASET_CONFIG,
    NoisyMedMNISTDataset, IndexedNoisyMedMNISTDataset
)
import torchvision.transforms as transforms


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
# GMM-based sample division
# ═══════════════════════════════════════════════════════════════════════════

def eval_train_losses(model, eval_loader, device, n_samples):
    """Per-sample cross-entropy loss (unreduced)."""
    model.eval()
    CE = nn.CrossEntropyLoss(reduction='none')
    losses = torch.zeros(n_samples)
    with torch.no_grad():
        for images, labels, indices in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = CE(outputs, labels)
            for b in range(images.size(0)):
                losses[indices[b]] = loss[b].cpu()
    return losses


def gmm_divide(losses, labels, all_loss_history, noise_rate, p_threshold=0.5):
    """Fit a 2-component GMM **per class** and return clean probabilities.

    Returns:
        prob_clean: numpy array (n_samples,) with P(clean) for each sample.
        clean_indices: indices where P(clean) > p_threshold (labeled set).
        noisy_indices: indices where P(clean) <= p_threshold (unlabeled set).
        all_loss_history: updated list.
        filter_info: dict for logging.
    """
    labels = np.array(labels).flatten()
    n_samples = len(labels)
    unique_classes = np.unique(labels[labels >= 0])

    all_loss_history.append(losses.clone())

    prob_clean = np.zeros(n_samples)
    filter_info = {}

    for cls in unique_classes:
        cls = int(cls)
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) == 0:
            continue

        cls_losses = losses[cls_idx]

        # Min-max normalize
        l_min, l_max = cls_losses.min(), cls_losses.max()
        if l_max - l_min > 1e-8:
            cls_losses_norm = (cls_losses - l_min) / (l_max - l_min)
        else:
            cls_losses_norm = torch.zeros_like(cls_losses)

        # High noise: average last 5 epochs
        if noise_rate >= 0.9 and len(all_loss_history) >= 5:
            history_stack = torch.stack(
                [h[cls_idx] for h in all_loss_history[-5:]]
            )
            avg = history_stack.mean(0)
            a_min, a_max = avg.min(), avg.max()
            if a_max - a_min > 1e-8:
                input_loss = ((avg - a_min) / (a_max - a_min)).reshape(-1, 1).numpy()
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
        prob_clean[cls_idx] = prob[:, clean_component]

        n_clean = int((prob_clean[cls_idx] > p_threshold).sum())
        filter_info[cls] = {
            'total': len(cls_idx),
            'clean': n_clean,
            'noisy': len(cls_idx) - n_clean,
            'gmm_means': gmm.means_.flatten().tolist(),
        }

    clean_indices = np.where(prob_clean > p_threshold)[0]
    noisy_indices = np.where(prob_clean <= p_threshold)[0]

    return prob_clean, clean_indices, noisy_indices, all_loss_history, filter_info


# ═══════════════════════════════════════════════════════════════════════════
# MixMatch helpers
# ═══════════════════════════════════════════════════════════════════════════

def sharpen(probs, T):
    """Sharpen a probability distribution by raising to 1/T and re-normalizing."""
    sharp = probs ** (1.0 / T)
    return sharp / sharp.sum(dim=1, keepdim=True)


def mixup_data(x, y, alpha=1.0):
    """MixUp: mix pairs of (x, y) with Beta(alpha, alpha) samples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)  # Ensure lam >= 0.5
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y


# ═══════════════════════════════════════════════════════════════════════════
# Dataset helpers for DivideMix
# ═══════════════════════════════════════════════════════════════════════════

class LabeledDataset(Dataset):
    """Subset of training data labeled as 'clean' by GMM, with soft labels."""

    def __init__(self, base_dataset, indices, soft_labels=None,
                 aug_transform=None, normalize=None):
        """
        Args:
            base_dataset: Dataset wrapper (NoisyMedMNISTDataset or similar)
                          with .images and .labels attributes.
            indices: clean sample indices.
            soft_labels: if provided, Tensor (len(indices), num_classes) of
                         soft targets. Otherwise use hard noisy labels.
            aug_transform: augmentation pipeline (includes ToTensor).
            normalize: normalization transform.
        """
        # Support both .images (NoisyMedMNISTDataset) and .imgs (raw MedMNIST)
        self.images = getattr(base_dataset, 'images', None)
        if self.images is None:
            self.images = base_dataset.imgs
        self.indices = np.array(indices)
        self.noisy_labels = np.array(base_dataset.labels).flatten()
        self.soft_labels = soft_labels
        self.aug_transform = aug_transform
        self.normalize = normalize

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image = self.images[real_idx]

        # Convert to PIL
        if isinstance(image, np.ndarray):
            from PIL import Image
            if image.ndim == 2:
                pil_image = Image.fromarray(image, mode='L')
            elif image.shape[2] == 1:
                pil_image = Image.fromarray(image.squeeze(2), mode='L')
            else:
                pil_image = Image.fromarray(image, mode='RGB')
        else:
            pil_image = image

        if self.aug_transform is not None:
            image = self.aug_transform(pil_image)
        else:
            image = transforms.ToTensor()(pil_image)

        if self.normalize is not None:
            image = self.normalize(image)

        if self.soft_labels is not None:
            label = self.soft_labels[idx]  # (num_classes,) tensor
        else:
            label = torch.tensor(self.noisy_labels[real_idx], dtype=torch.long)

        return image, label


class UnlabeledDataset(Dataset):
    """Subset of training data labeled as 'noisy' by GMM – treated as unlabeled."""

    def __init__(self, base_dataset, indices, aug_transform=None,
                 normalize=None):
        self.images = getattr(base_dataset, 'images', None)
        if self.images is None:
            self.images = base_dataset.imgs
        self.indices = np.array(indices)
        self.aug_transform = aug_transform
        self.normalize = normalize

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image = self.images[real_idx]

        from PIL import Image
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                pil_image = Image.fromarray(image, mode='L')
            elif image.shape[2] == 1:
                pil_image = Image.fromarray(image.squeeze(2), mode='L')
            else:
                pil_image = Image.fromarray(image, mode='RGB')
        else:
            pil_image = image

        if self.aug_transform is not None:
            image = self.aug_transform(pil_image)
        else:
            image = transforms.ToTensor()(pil_image)

        if self.normalize is not None:
            image = self.normalize(image)

        return image


# ═══════════════════════════════════════════════════════════════════════════
# Training loops
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch_warmup(model, train_loader, optimizer, device,
                       class_weights=None, positive_class=1):
    """Warmup epoch: standard CE on all samples."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []

    for batch in train_loader:
        # train_loader may return (img, label, idx) if IndexedNoisyMedMNISTDataset
        if len(batch) == 3:
            images, labels, _ = batch
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


def train_epoch_dividemix(model, labeled_loader, unlabeled_loader,
                          optimizer, device, alpha, lambda_u, T,
                          class_weights=None, num_classes=2,
                          positive_class=1, other_model=None):
    """DivideMix training epoch with MixMatch-style semi-supervised learning.

    For the unlabeled set, pseudo-labels are generated from the *other*
    network's predictions (co-refinement).

    Args:
        model:            The network being trained.
        labeled_loader:   DataLoader for clean (labeled) samples.
        unlabeled_loader: DataLoader for noisy (unlabeled) samples.
        optimizer:        Optimizer for ``model``.
        device:           torch device.
        alpha:            Beta distribution parameter for MixUp.
        lambda_u:         Weight for the unlabeled loss.
        T:                Sharpening temperature.
        class_weights:    Optional per-class CE weights.
        num_classes:      Number of classes.
        positive_class:   Positive class index.
        other_model:      The *other* network (for pseudo-label generation).
    """
    model.train()
    if other_model is not None:
        other_model.eval()

    running_loss_x = 0.0  # labeled
    running_loss_u = 0.0  # unlabeled
    correct = 0
    total_labeled = 0

    # Iterate over labeled set; cycle unlabeled if shorter
    unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None and len(unlabeled_loader) > 0 else None

    for batch_x in labeled_loader:
        images_x, targets_x = batch_x
        images_x = images_x.to(device)

        # targets_x can be hard (long) or soft (float tensor of shape (B, C))
        if targets_x.dim() == 1 or (targets_x.dim() == 2 and targets_x.shape[1] == 1):
            # Hard label → one-hot soft label
            hard = targets_x.long().view(-1).to(device)
            targets_x_soft = torch.zeros(hard.size(0), num_classes, device=device)
            targets_x_soft.scatter_(1, hard.unsqueeze(1), 1.0)
        else:
            targets_x_soft = targets_x.to(device)

        batch_size_x = images_x.size(0)

        # ── Unlabeled batch ────────────────────────────────────────────
        if unlabeled_iter is not None:
            try:
                images_u = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                images_u = next(unlabeled_iter)
            images_u = images_u.to(device)

            # Pseudo-labels from the OTHER model
            with torch.no_grad():
                if other_model is not None:
                    pseudo_logits = other_model(images_u)
                else:
                    pseudo_logits = model(images_u)
                pseudo_probs = F.softmax(pseudo_logits, dim=1)
                targets_u_soft = sharpen(pseudo_probs, T)

            batch_size_u = images_u.size(0)
        else:
            images_u = None
            targets_u_soft = None
            batch_size_u = 0

        # ── Concatenate + MixUp ────────────────────────────────────────
        if images_u is not None and batch_size_u > 0:
            all_inputs = torch.cat([images_x, images_u], dim=0)
            all_targets = torch.cat([targets_x_soft, targets_u_soft], dim=0)
        else:
            all_inputs = images_x
            all_targets = targets_x_soft

        mixed_input, mixed_target = mixup_data(all_inputs, all_targets, alpha)

        # Split back
        mixed_input_x = mixed_input[:batch_size_x]
        mixed_target_x = mixed_target[:batch_size_x]
        if batch_size_u > 0:
            mixed_input_u = mixed_input[batch_size_x:]
            mixed_target_u = mixed_target[batch_size_x:]

        # ── Forward and loss ───────────────────────────────────────────
        optimizer.zero_grad()

        logits_x = model(mixed_input_x)
        # Labeled loss: CE with soft targets
        log_probs_x = F.log_softmax(logits_x, dim=1)
        loss_x = -torch.mean(torch.sum(mixed_target_x * log_probs_x, dim=1))

        if batch_size_u > 0:
            logits_u = model(mixed_input_u)
            # Unlabeled loss: MSE between softmax and soft targets
            probs_u = F.softmax(logits_u, dim=1)
            loss_u = F.mse_loss(probs_u, mixed_target_u)
        else:
            loss_u = torch.tensor(0.0, device=device)

        loss = loss_x + lambda_u * loss_u
        loss.backward()
        optimizer.step()

        running_loss_x += loss_x.item() * batch_size_x
        running_loss_u += loss_u.item() * max(batch_size_u, 1)

        # Track accuracy on labeled part (for logging)
        _, pred = torch.max(logits_x, 1)
        _, target_hard = torch.max(mixed_target_x, 1)
        correct += (pred == target_hard).sum().item()
        total_labeled += batch_size_x

    avg_loss_x = running_loss_x / max(total_labeled, 1)
    avg_loss_u = running_loss_u / max(total_labeled, 1)
    accuracy = correct / max(total_labeled, 1)

    return avg_loss_x, avg_loss_u, accuracy


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


def evaluate_ensemble(model1, model2, data_loader, device,
                      class_weights=None, positive_class=1):
    """Evaluate ensemble (average of softmax outputs)."""
    model1.eval()
    model2.eval()
    correct = 0
    total = 0
    all_avg_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            images, labels = images.to(device), labels.to(device)

            probs1 = F.softmax(model1(images), dim=1)
            probs2 = F.softmax(model2(images), dim=1)
            avg_probs = (probs1 + probs2) / 2.0

            _, predicted = torch.max(avg_probs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_avg_probs.append(avg_probs)
            all_targets.append(labels)

    accuracy = correct / total
    all_avg_probs = torch.cat(all_avg_probs)
    all_targets = torch.cat(all_targets)

    _, predicted = torch.max(all_avg_probs, 1)
    TP = ((predicted == positive_class) & (all_targets == positive_class)).sum().item()
    TN = ((predicted != positive_class) & (all_targets != positive_class)).sum().item()
    FP = ((predicted == positive_class) & (all_targets != positive_class)).sum().item()
    FN = ((predicted != positive_class) & (all_targets == positive_class)).sum().item()
    cm = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
    bac = compute_bac(cm)

    probs_pos = all_avg_probs[:, positive_class].cpu().numpy()
    targets_np = (all_targets == positive_class).long().cpu().numpy()
    auc_roc = roc_auc_score(targets_np, probs_pos)
    auprc = average_precision_score(targets_np, probs_pos)

    # Compute loss on avg logits for logging
    avg_logits = torch.log(all_avg_probs + 1e-8)
    if class_weights is not None:
        loss = F.cross_entropy(avg_logits, all_targets, weight=class_weights.to(
            avg_logits.device)).item()
    else:
        loss = F.cross_entropy(avg_logits, all_targets).item()

    return loss, accuracy, cm, bac, auc_roc, auprc


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
        f"_dividemix"
        f"_warmup{args.warmup_epochs}"
        f"_alpha{args.alpha}_lu{args.lambda_u}"
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

    # The underlying MedMNIST base dataset (for creating sub-datasets)
    train_dataset = train_loader.dataset  # IndexedNoisyMedMNISTDataset
    base_train_dataset = train_dataset.base_dataset  # raw MedMNIST
    n_train = len(train_dataset)
    train_labels = np.array(train_dataset.labels).flatten()

    print(f"Train samples: {n_train}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    # Transforms (same as dataloaders.py uses)
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

    # Class weights
    class_weights = None
    if args.weighted_loss:
        class_weights = infer_class_weights(train_labels, device)
        if class_weights is not None:
            print(f"\nUsing balanced class weights: "
                  f"{class_weights.cpu().tolist()}")
    else:
        print("\nUsing standard Cross Entropy loss")

    # Models ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Creating two ResNet18 models for DivideMix...")
    model1 = get_model(num_classes, in_channels, args.pretrained).to(device)

    torch.manual_seed(args.seed + 1)
    model2 = get_model(num_classes, in_channels, args.pretrained).to(device)
    for m in model2.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    set_seed(args.seed)

    optimizer1 = optim.SGD(model1.parameters(), lr=args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay)
    optimizer2 = optim.SGD(model2.parameters(), lr=args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay)
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.epochs)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.epochs)

    # Training ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Starting DivideMix training  |  warmup={args.warmup_epochs}  |  "
          f"alpha={args.alpha}  |  lambda_u={args.lambda_u}  |  T={args.T}")
    print("=" * 60)

    all_loss_history1 = []
    all_loss_history2 = []
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
            # ── Warmup: standard CE on all samples ─────────────────────
            _, acc1, cm1 = train_epoch_warmup(
                model1, train_loader, optimizer1, device, class_weights,
                positive_class
            )
            _, acc2, cm2 = train_epoch_warmup(
                model2, train_loader, optimizer2, device, class_weights,
                positive_class
            )
            n_clean = n_train
            n_noisy = 0
            epoch_filter_info = None
        else:
            # ── DivideMix phase ────────────────────────────────────────
            # 1) Model 1 computes losses → GMM divides for Model 2
            losses1 = eval_train_losses(model1, eval_loader, device, n_train)
            prob_clean1, clean_idx1, noisy_idx1, all_loss_history1, fi1 = \
                gmm_divide(losses1, train_labels, all_loss_history1,
                           args.noise_rate, args.p_threshold)

            # 2) Model 2 computes losses → GMM divides for Model 1
            losses2 = eval_train_losses(model2, eval_loader, device, n_train)
            prob_clean2, clean_idx2, noisy_idx2, all_loss_history2, fi2 = \
                gmm_divide(losses2, train_labels, all_loss_history2,
                           args.noise_rate, args.p_threshold)

            # ── Train Model 1 using Model 2's division ─────────────────
            labeled_ds1 = LabeledDataset(
                train_dataset, clean_idx2,
                aug_transform=train_aug_transform, normalize=normalize
            )
            unlabeled_ds1 = UnlabeledDataset(
                train_dataset, noisy_idx2,
                aug_transform=train_aug_transform, normalize=normalize
            )

            labeled_loader1 = DataLoader(
                labeled_ds1, batch_size=args.batch_size, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True
            ) if len(labeled_ds1) > 0 else None
            unlabeled_loader1 = DataLoader(
                unlabeled_ds1, batch_size=args.batch_size, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True
            ) if len(unlabeled_ds1) > 0 else None

            if labeled_loader1 is not None and len(labeled_loader1) > 0:
                train_epoch_dividemix(
                    model1, labeled_loader1, unlabeled_loader1,
                    optimizer1, device, args.alpha, args.lambda_u, args.T,
                    class_weights, num_classes, positive_class,
                    other_model=model2
                )

            # ── Train Model 2 using Model 1's division ─────────────────
            labeled_ds2 = LabeledDataset(
                train_dataset, clean_idx1,
                aug_transform=train_aug_transform, normalize=normalize
            )
            unlabeled_ds2 = UnlabeledDataset(
                train_dataset, noisy_idx1,
                aug_transform=train_aug_transform, normalize=normalize
            )

            labeled_loader2 = DataLoader(
                labeled_ds2, batch_size=args.batch_size, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True
            ) if len(labeled_ds2) > 0 else None
            unlabeled_loader2 = DataLoader(
                unlabeled_ds2, batch_size=args.batch_size, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True
            ) if len(unlabeled_ds2) > 0 else None

            if labeled_loader2 is not None and len(labeled_loader2) > 0:
                train_epoch_dividemix(
                    model2, labeled_loader2, unlabeled_loader2,
                    optimizer2, device, args.alpha, args.lambda_u, args.T,
                    class_weights, num_classes, positive_class,
                    other_model=model1
                )

            n_clean = len(clean_idx1)  # for logging
            n_noisy = len(noisy_idx1)
            epoch_filter_info = {'net1_gmm': fi1, 'net2_gmm': fi2}

        scheduler1.step()
        scheduler2.step()

        # Validate (ensemble)
        val_loss, val_acc, val_cm, val_bac, val_auc_roc, val_auprc = \
            evaluate_ensemble(model1, model2, val_loader, device,
                              class_weights, positive_class)

        # Logging
        phase_tag = "WARMUP" if is_warmup else "DIVIDEMIX"
        log_entry = {
            'epoch': epoch,
            'phase': phase_tag,
            'n_clean': int(n_clean),
            'n_noisy': int(n_noisy),
            'ensemble_val_acc': float(val_acc),
            'ensemble_val_bac': float(val_bac),
            'ensemble_val_auc_roc': float(val_auc_roc),
        }
        if epoch_filter_info is not None:
            log_entry['filter_info'] = epoch_filter_info
        training_log.append(log_entry)

        if epoch % args.print_freq == 0 or epoch == 1:
            print(f"\nEpoch [{epoch}/{args.epochs}]  ({phase_tag})  "
                  f"clean={n_clean}/{n_train}  noisy={n_noisy}")
            print(f"  Ensemble Val - Acc: {val_acc:.4f}, "
                  f"BAC: {val_bac:.4f}")
            print(f"                 AUC-ROC: {val_auc_roc:.4f}, "
                  f"AUPRC: {val_auprc:.4f}")
            print(f"                 TP: {val_cm['TP']}, TN: {val_cm['TN']}, "
                  f"FP: {val_cm['FP']}, FN: {val_cm['FN']}")

        # Best model (by ensemble BAC)
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

    test_loss, test_acc, test_cm, test_bac, test_auc_roc, test_auprc = \
        evaluate_ensemble(model1, model2, test_loader, device,
                          class_weights, positive_class)

    # Individual test
    _, test_acc1, test_cm1, test_bac1, test_auc1, test_auprc1 = \
        evaluate(model1, test_loader, device, class_weights, positive_class)
    _, test_acc2, test_cm2, test_bac2, test_auc2, test_auprc2 = \
        evaluate(model2, test_loader, device, class_weights, positive_class)

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

    print(f"\nTest (Net1): Acc={test_acc1:.4f}, BAC={test_bac1:.4f}, "
          f"AUC={test_auc1:.4f}")
    print(f"Test (Net2): Acc={test_acc2:.4f}, BAC={test_bac2:.4f}, "
          f"AUC={test_auc2:.4f}")

    # Save results
    results = {
        'dataset': args.dataset,
        'method': 'dividemix',
        'best_epoch': best_epoch,
        'best_metric': 'BAC',
        'noise_rate': args.noise_rate,
        'warmup_epochs': args.warmup_epochs,
        'alpha': args.alpha,
        'lambda_u': args.lambda_u,
        'T': args.T,
        'p_threshold': args.p_threshold,
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

    txt_path = os.path.join(output_dir, 'results.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EXPERIMENT RESULTS  --  DivideMix\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Noise Rate: {args.noise_rate}\n")
        f.write(f"Warmup Epochs: {args.warmup_epochs}\n")
        f.write(f"Alpha (MixUp): {args.alpha}\n")
        f.write(f"Lambda_u: {args.lambda_u}\n")
        f.write(f"T (sharpen): {args.T}\n")
        f.write(f"P threshold: {args.p_threshold}\n")
        f.write(f"Weighted Loss: {args.weighted_loss}\n")
        if class_weights is not None:
            f.write(f"Class Weights: {class_weights.cpu().tolist()}\n")
        f.write(f"Best Epoch: {best_epoch} (selected by BAC)\n\n")

        f.write("-" * 60 + "\n")
        f.write("VALIDATION (Ensemble)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy: {best_val_acc:.4f}\n")
        f.write(f"BAC: {best_val_bac:.4f}\n")
        f.write(f"AUC-ROC: {best_val_auc_roc:.4f}\n")
        f.write(f"AUPRC: {best_val_auprc:.4f}\n")
        f.write(f"TP: {best_val_cm['TP']}, TN: {best_val_cm['TN']}, "
                f"FP: {best_val_cm['FP']}, FN: {best_val_cm['FN']}\n\n")

        f.write("-" * 60 + "\n")
        f.write("TEST (Ensemble)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"BAC: {test_bac:.4f}\n")
        f.write(f"AUC-ROC: {test_auc_roc:.4f}\n")
        f.write(f"AUPRC: {test_auprc:.4f}\n")
        f.write(f"TP: {test_cm['TP']}, TN: {test_cm['TN']}, "
                f"FP: {test_cm['FP']}, FN: {test_cm['FN']}\n")

    print(f"\nResults saved to: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train with DivideMix for noisy labels '
                    '(Li et al., ICLR 2020)'
    )

    parser.add_argument('--dataset', type=str, default='pneumoniamnist',
                        choices=list(DATASET_CONFIG.keys()))
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--noise_rate', type=float, default=0.0)

    # DivideMix
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=4.0,
                        help='Beta distribution parameter for MixUp')
    parser.add_argument('--lambda_u', type=float, default=25.0,
                        help='Weight for unsupervised loss')
    parser.add_argument('--T', type=float, default=0.5,
                        help='Sharpening temperature')
    parser.add_argument('--p_threshold', type=float, default=0.5,
                        help='GMM probability threshold for clean/noisy split')

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
    print("EXPERIMENT CONFIGURATION  --  DivideMix")
    print("=" * 60)
    print(f"Dataset:       {args.dataset} (64x64)")
    print(f"Model:         2 x ResNet18")
    print(f"Noise Rate:    {args.noise_rate}")
    print(f"Warmup:        {args.warmup_epochs}")
    print(f"Alpha (MixUp): {args.alpha}")
    print(f"Lambda_u:      {args.lambda_u}")
    print(f"T (sharpen):   {args.T}")
    print(f"P threshold:   {args.p_threshold}")
    print(f"Weighted Loss: {args.weighted_loss}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Learning Rate: {args.lr}")

    train_main(args)


if __name__ == '__main__':
    main()
