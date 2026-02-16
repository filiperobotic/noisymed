"""
Baseline Training Script for Medical Image Classification with Noisy Labels
Dataset: PneumoniaMNIST (64x64)
Model: ResNet18
Supports: Symmetric noise injection and weighted loss (balanced by class proportion)

Usage:
    python train_baseline.py --noise_rate 0.2 --weighted_loss
    python train_baseline.py --noise_rate 0.4
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import medmnist
from medmnist import PneumoniaMNIST
import json
from datetime import datetime


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def infer_class_weights(labels, device):
    """
    Infer balanced CE weights from labels.
    Weights are computed as: w_i = n_total / (2 * n_i)
    This gives higher weight to minority classes.
    
    Args:
        labels: Array of labels
        device: torch device
    
    Returns:
        torch.Tensor with weights [w0, w1] or None
    """
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


def get_dataloaders(noise_rate=0.0, batch_size=128, seed=42, data_dir='./data'):
    """Create train, validation and test dataloaders for PneumoniaMNIST"""
    
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    
    # Load PneumoniaMNIST with size 64x64
    train_dataset = PneumoniaMNIST(
        split='train', transform=None, download=True, root=data_dir, size=64
    )
    val_dataset = PneumoniaMNIST(
        split='val', transform=None, download=True, root=data_dir, size=64
    )
    test_dataset = PneumoniaMNIST(
        split='test', transform=None, download=True, root=data_dir, size=64
    )
    
    # Get original labels for class distribution
    original_train_labels = train_dataset.labels.flatten()
    class_counts = {
        0: int((original_train_labels == 0).sum()),
        1: int((original_train_labels == 1).sum())
    }
    print(f"\nOriginal class distribution:")
    print(f"  Class 0 (Normal): {class_counts[0]}")
    print(f"  Class 1 (Pneumonia): {class_counts[1]}")
    
    # Inject noise into training labels
    noisy_labels = None
    if noise_rate > 0:
        noisy_labels, noise_mask = inject_symmetric_noise(
            original_train_labels, noise_rate, num_classes=2, seed=seed
        )
        
        # Save noise information
        os.makedirs('noise_files', exist_ok=True)
        noise_info = {
            'noise_rate': noise_rate,
            'noise_indices': noise_mask.tolist(),
            'original_labels': original_train_labels.tolist(),
            'noisy_labels': noisy_labels.tolist()
        }
        with open(f'noise_files/pneumonia_noise_{noise_rate}.json', 'w') as f:
            json.dump(noise_info, f)
    
    # Create wrapped datasets
    train_data = NoisyMedMNISTDataset(train_dataset, noisy_labels, normalize)
    val_data = NoisyMedMNISTDataset(val_dataset, None, normalize)
    test_data = NoisyMedMNISTDataset(test_dataset, None, normalize)
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader


def get_model(num_classes=2, pretrained=False):
    """Create ResNet18 model adapted for grayscale medical images"""
    model = models.resnet18(pretrained=pretrained)
    
    # Modify first conv layer for single channel input (grayscale)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def compute_confusion_matrix(outputs, targets):
    """
    Compute confusion matrix elements
    
    Returns:
        dict with TP, TN, FP, FN
    """
    _, predicted = torch.max(outputs, 1)
    
    # Class 0 = Normal (Negative), Class 1 = Pneumonia (Positive)
    TP = ((predicted == 1) & (targets == 1)).sum().item()
    TN = ((predicted == 0) & (targets == 0)).sum().item()
    FP = ((predicted == 1) & (targets == 0)).sum().item()
    FN = ((predicted == 0) & (targets == 1)).sum().item()
    
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}


def train_epoch(model, train_loader, optimizer, device, class_weights=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Use weighted or standard cross entropy
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
    
    # Compute confusion matrix
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    cm = compute_confusion_matrix(all_outputs, all_targets)
    
    return epoch_loss, accuracy, cm


def evaluate(model, data_loader, device, class_weights=None):
    """Evaluate model on validation/test set"""
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
    
    # Compute confusion matrix
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    cm = compute_confusion_matrix(all_outputs, all_targets)
    
    return epoch_loss, accuracy, cm


def train(args):
    """Main training function"""
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"pneumonia_noise{args.noise_rate}_{'weighted' if args.weighted_loss else 'standard'}_{timestamp}"
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Get dataloaders
    print("\n" + "="*50)
    print("Loading PneumoniaMNIST dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        noise_rate=args.noise_rate, batch_size=args.batch_size, 
        seed=args.seed, data_dir=args.data_dir
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Compute class weights if using weighted loss
    class_weights = None
    if args.weighted_loss:
        class_weights = infer_class_weights(train_loader.dataset.labels, device)
        if class_weights is not None:
            print(f"\nUsing balanced class weights: {class_weights.cpu().tolist()}")
    else:
        print("\nUsing standard Cross Entropy loss")
    
    # Create model
    print("\n" + "="*50)
    print("Creating ResNet18 model...")
    model = get_model(num_classes=2, pretrained=args.pretrained)
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    
    best_val_acc = 0.0
    best_epoch = 0
    best_val_cm = None
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, train_cm = train_epoch(
            model, train_loader, optimizer, device, class_weights
        )
        
        # Validate
        val_loss, val_acc, val_cm = evaluate(model, val_loader, device, class_weights)
        
        scheduler.step()
        
        # Log progress
        if epoch % args.print_freq == 0 or epoch == 1:
            print(f"\nEpoch [{epoch}/{args.epochs}]")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"          TP: {val_cm['TP']}, TN: {val_cm['TN']}, FP: {val_cm['FP']}, FN: {val_cm['FN']}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_val_cm = val_cm
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_cm': val_cm,
            }, os.path.join(output_dir, 'best_model.pth'))
    
    # Load best model and evaluate on test set
    print("\n" + "="*50)
    print(f"Loading best model from epoch {best_epoch}...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_cm = evaluate(model, test_loader, device, class_weights)
    
    # Print final results
    print("\n" + "="*50)
    print(f"FINAL RESULTS (Best Epoch: {best_epoch})")
    print("="*50)
    print(f"\nValidation:")
    print(f"  Accuracy: {best_val_acc:.4f}")
    print(f"  TP: {best_val_cm['TP']}, TN: {best_val_cm['TN']}, FP: {best_val_cm['FP']}, FN: {best_val_cm['FN']}")
    
    print(f"\nTest:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  TP: {test_cm['TP']}, TN: {test_cm['TN']}, FP: {test_cm['FP']}, FN: {test_cm['FN']}")
    
    # Save results
    results = {
        'best_epoch': best_epoch,
        'noise_rate': args.noise_rate,
        'weighted_loss': args.weighted_loss,
        'class_weights': class_weights.cpu().tolist() if class_weights is not None else None,
        'validation': {'accuracy': best_val_acc, **best_val_cm},
        'test': {'accuracy': test_acc, **test_cm}
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train baseline model on PneumoniaMNIST with noisy labels')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results')
    
    # Noise parameters
    parser.add_argument('--noise_rate', type=float, default=0.0,
                        help='Symmetric noise rate (0.0 to 1.0)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Loss parameters
    parser.add_argument('--weighted_loss', action='store_true',
                        help='Use balanced weighted cross entropy loss')
    
    # Model parameters
    parser.add_argument('--pretrained', action='store_true')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_freq', type=int, default=10)
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("EXPERIMENT CONFIGURATION")
    print("="*50)
    print(f"Dataset: PneumoniaMNIST (64x64)")
    print(f"Model: ResNet18")
    print(f"Noise Rate: {args.noise_rate}")
    print(f"Weighted Loss: {args.weighted_loss}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    
    train(args)


if __name__ == '__main__':
    main()
