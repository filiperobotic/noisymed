"""
Dataloaders for MedMNIST datasets with noisy label support.

Supported datasets:
    - pneumoniamnist: Chest X-Ray (grayscale, 2 classes)
    - breastmnist: Breast Ultrasound (grayscale, 2 classes)
    - dermamnist_bin: DermaMNIST binarized (RGB, 2 classes) — Malignant vs Benign
    - pathmnist_bin: PathMNIST binarized (RGB, 2 classes) — Malignant vs Benign

Usage:
    from dataloaders import get_dataloaders, DATASET_CONFIG

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset='pneumoniamnist', noise_rate=0.2, batch_size=128
    )
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from medmnist import PneumoniaMNIST, BreastMNIST, DermaMNIST, PathMNIST


DATASET_CONFIG = {
    'pneumoniamnist': {
        'class': PneumoniaMNIST,
        'num_classes': 2,
        'in_channels': 1,
        'class_names': {0: 'Normal', 1: 'Pneumonia'},
    },
    'breastmnist': {
        'class': BreastMNIST,
        'num_classes': 2,
        'in_channels': 1,
        'class_names': {0: 'Malignant', 1: 'Normal/Benign'},
    },
    'dermamnist_bin': {
        'class': DermaMNIST,
        'num_classes': 2,
        'in_channels': 3,
        'class_names': {0: 'Benign', 1: 'Malignant'},
        'binarize': {
            'malignant_classes': [0, 1, 4],
            'benign_classes': [2, 3, 5, 6],
        },
    },
    'pathmnist_bin': {
        'class': PathMNIST,
        'num_classes': 2,
        'in_channels': 3,
        'class_names': {0: 'Benign', 1: 'Malignant'},
        'binarize': {
            'malignant_classes': [7, 8],
            'benign_classes': [0, 1, 2, 3, 4, 5, 6],
        },
    },
}


def inject_symmetric_noise(labels, noise_rate, num_classes=2, seed=42):
    """
    Inject symmetric noise into labels.

    Args:
        labels: Original labels (numpy array)
        noise_rate: Probability of flipping each label (0.0 to 1.0)
        num_classes: Number of classes
        seed: Random seed for reproducibility

    Returns:
        noisy_labels: Labels with noise injected
        noise_mask: Boolean array indicating which labels were flipped
    """
    np.random.seed(seed)
    labels = labels.flatten()
    noisy_labels = labels.copy()
    n_samples = len(labels)

    # Determine which samples to flip
    flip_indices = np.random.choice(
        n_samples,
        size=int(noise_rate * n_samples),
        replace=False
    )

    noise_mask = np.zeros(n_samples, dtype=bool)
    noise_mask[flip_indices] = True

    # Flip selected labels to a different class
    for idx in flip_indices:
        original_label = labels[idx]
        possible_labels = list(range(num_classes))
        possible_labels.remove(original_label)
        noisy_labels[idx] = np.random.choice(possible_labels)

    actual_noise_rate = noise_mask.sum() / n_samples
    print(f"Injected noise rate: {actual_noise_rate:.2%}")
    print(f"Number of flipped labels: {noise_mask.sum()} / {n_samples}")

    return noisy_labels, noise_mask


class NoisyMedMNISTDataset(Dataset):
    """Wrapper dataset that applies noise to MedMNIST labels"""

    def __init__(self, base_dataset, noisy_labels=None, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

        # Get original images and labels
        self.images = base_dataset.imgs
        self.original_labels = base_dataset.labels.flatten()

        if noisy_labels is not None:
            self.labels = noisy_labels
        else:
            self.labels = self.original_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float() / 255.0
            if len(image.shape) == 3:
                image = image.permute(2, 0, 1)
            else:
                image = image.unsqueeze(0)

        # Apply normalization
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_dataloaders(dataset='pneumoniamnist', noise_rate=0.0, batch_size=128,
                    seed=42, data_dir='./data'):
    """
    Create train, validation and test dataloaders for a MedMNIST dataset.

    Args:
        dataset: Dataset name (key in DATASET_CONFIG)
        noise_rate: Symmetric noise rate for training labels
        batch_size: Batch size for dataloaders
        seed: Random seed
        data_dir: Root directory for data download

    Returns:
        train_loader, val_loader, test_loader
    """
    if dataset not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Available: {list(DATASET_CONFIG.keys())}"
        )

    config = DATASET_CONFIG[dataset]
    DatasetClass = config['class']
    num_classes = config['num_classes']
    in_channels = config['in_channels']

    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Normalization per channel count
    if in_channels == 1:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    else:
        normalize = transforms.Normalize(
            mean=[0.5] * in_channels, std=[0.5] * in_channels
        )

    # Load dataset splits
    train_dataset = DatasetClass(
        split='train', transform=None, download=True, root=data_dir, size=64
    )
    val_dataset = DatasetClass(
        split='val', transform=None, download=True, root=data_dir, size=64
    )
    test_dataset = DatasetClass(
        split='test', transform=None, download=True, root=data_dir, size=64
    )

    # Binarize labels if needed (for multi-class datasets mapped to binary)
    binarize_cfg = config.get('binarize')
    if binarize_cfg is not None:
        malignant_classes = binarize_cfg['malignant_classes']
        for ds in [train_dataset, val_dataset, test_dataset]:
            orig = np.array(ds.labels).squeeze().astype(np.int32)
            ds.labels = np.isin(orig, malignant_classes).astype(np.int32)
        print(f"\nBinarized labels: malignant classes {malignant_classes}")
        for name_split, ds in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
            n_mal = int((ds.labels == 1).sum())
            n_ben = int((ds.labels == 0).sum())
            total = len(ds.labels)
            print(f"  {name_split}: Malignant={n_mal} ({100*n_mal/total:.1f}%) | "
                  f"Benign={n_ben} ({100*n_ben/total:.1f}%) | Total={total}")

    # Print class distribution
    original_train_labels = train_dataset.labels.flatten()
    print(f"\nClass distribution (after binarization if applicable):")
    for cls_id in range(num_classes):
        count = int((original_train_labels == cls_id).sum())
        name = config['class_names'].get(cls_id, f'Class {cls_id}')
        print(f"  Class {cls_id} ({name}): {count}")

    # Inject noise into training labels
    noisy_labels = None
    if noise_rate > 0:
        noisy_labels, noise_mask = inject_symmetric_noise(
            original_train_labels, noise_rate,
            num_classes=num_classes, seed=seed
        )

        # Save noise information
        os.makedirs('noise_files', exist_ok=True)
        noise_info = {
            'dataset': dataset,
            'noise_rate': noise_rate,
            'noise_indices': noise_mask.tolist(),
            'original_labels': original_train_labels.tolist(),
            'noisy_labels': noisy_labels.tolist()
        }
        noise_path = f'noise_files/{dataset}_noise_{noise_rate}.json'
        with open(noise_path, 'w') as f:
            json.dump(noise_info, f)

    # Create wrapped datasets
    train_data = NoisyMedMNISTDataset(train_dataset, noisy_labels, normalize)
    val_data = NoisyMedMNISTDataset(val_dataset, None, normalize)
    test_data = NoisyMedMNISTDataset(test_dataset, None, normalize)

    # Create dataloaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader
