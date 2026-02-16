"""
Baseline Training Script for Medical Image Classification with Noisy Labels
Datasets: PneumoniaMNIST, BreastMNIST (64x64)
Model: ResNet18
Supports: Symmetric noise injection and weighted loss (balanced by class proportion)

Usage:
    python train_baseline.py --dataset pneumoniamnist --noise_rate 0.2 --weighted_loss
    python train_baseline.py --dataset breastmnist --noise_rate 0.4
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


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def get_model(num_classes=2, in_channels=1, pretrained=False):
    """Create ResNet18 model adapted for medical images"""
    model = models.resnet18(pretrained=pretrained)

    # Modify first conv layer to match input channels
    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

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

    # Class 0 = Negative, Class 1 = Positive
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

    # Compute AUC-ROC and AUPRC
    probs = F.softmax(all_outputs, dim=1)[:, 1].cpu().numpy()
    targets_np = all_targets.cpu().numpy()
    auc_roc = roc_auc_score(targets_np, probs)
    auprc = average_precision_score(targets_np, probs)

    return epoch_loss, accuracy, cm, auc_roc, auprc


def train(args):
    """Main training function"""
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = DATASET_CONFIG[args.dataset]

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = (
        f"{args.dataset}_noise{args.noise_rate}"
        f"_{'weighted' if args.weighted_loss else 'standard'}_{timestamp}"
    )
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Get dataloaders
    print("\n" + "="*50)
    print(f"Loading {args.dataset} dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=args.dataset, noise_rate=args.noise_rate,
        batch_size=args.batch_size, seed=args.seed, data_dir=args.data_dir
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
    model = get_model(
        num_classes=config['num_classes'],
        in_channels=config['in_channels'],
        pretrained=args.pretrained
    )
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
    best_val_auc_roc = 0.0
    best_val_auprc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, train_cm = train_epoch(
            model, train_loader, optimizer, device, class_weights
        )

        # Validate
        val_loss, val_acc, val_cm, val_auc_roc, val_auprc = evaluate(model, val_loader, device, class_weights)

        scheduler.step()

        # Log progress
        if epoch % args.print_freq == 0 or epoch == 1:
            print(f"\nEpoch [{epoch}/{args.epochs}]")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"          AUC-ROC: {val_auc_roc:.4f}, AUPRC: {val_auprc:.4f}")
            print(f"          TP: {val_cm['TP']}, TN: {val_cm['TN']}, FP: {val_cm['FP']}, FN: {val_cm['FN']}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_val_cm = val_cm
            best_val_auc_roc = val_auc_roc
            best_val_auprc = val_auprc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_cm': val_cm,
                'val_auc_roc': val_auc_roc,
                'val_auprc': val_auprc,
            }, os.path.join(output_dir, 'best_model.pth'))

    # Load best model and evaluate on test set
    print("\n" + "="*50)
    print(f"Loading best model from epoch {best_epoch}...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_cm, test_auc_roc, test_auprc = evaluate(model, test_loader, device, class_weights)

    # Print final results
    print("\n" + "="*50)
    print(f"FINAL RESULTS (Best Epoch: {best_epoch})")
    print("="*50)
    print(f"\nValidation:")
    print(f"  Accuracy: {best_val_acc:.4f}")
    print(f"  AUC-ROC: {best_val_auc_roc:.4f}, AUPRC: {best_val_auprc:.4f}")
    print(f"  TP: {best_val_cm['TP']}, TN: {best_val_cm['TN']}, FP: {best_val_cm['FP']}, FN: {best_val_cm['FN']}")

    print(f"\nTest:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  AUC-ROC: {test_auc_roc:.4f}, AUPRC: {test_auprc:.4f}")
    print(f"  TP: {test_cm['TP']}, TN: {test_cm['TN']}, FP: {test_cm['FP']}, FN: {test_cm['FN']}")

    # Save results
    results = {
        'dataset': args.dataset,
        'best_epoch': best_epoch,
        'noise_rate': args.noise_rate,
        'weighted_loss': args.weighted_loss,
        'class_weights': class_weights.cpu().tolist() if class_weights is not None else None,
        'validation': {'accuracy': best_val_acc, 'auc_roc': best_val_auc_roc, 'auprc': best_val_auprc, **best_val_cm},
        'test': {'accuracy': test_acc, 'auc_roc': test_auc_roc, 'auprc': test_auprc, **test_cm}
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save results to TXT file
    txt_path = os.path.join(output_dir, 'results.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("EXPERIMENT RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Noise Rate: {args.noise_rate}\n")
        f.write(f"Weighted Loss: {args.weighted_loss}\n")
        if class_weights is not None:
            f.write(f"Class Weights: {class_weights.cpu().tolist()}\n")
        f.write(f"Best Epoch: {best_epoch}\n\n")
        f.write("-" * 50 + "\n")
        f.write("VALIDATION RESULTS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Accuracy: {best_val_acc:.4f}\n")
        f.write(f"AUC-ROC: {best_val_auc_roc:.4f}\n")
        f.write(f"AUPRC: {best_val_auprc:.4f}\n")
        f.write(f"TP: {best_val_cm['TP']}, TN: {best_val_cm['TN']}, FP: {best_val_cm['FP']}, FN: {best_val_cm['FN']}\n\n")
        f.write("-" * 50 + "\n")
        f.write("TEST RESULTS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"AUC-ROC: {test_auc_roc:.4f}\n")
        f.write(f"AUPRC: {test_auprc:.4f}\n")
        f.write(f"TP: {test_cm['TP']}, TN: {test_cm['TN']}, FP: {test_cm['FP']}, FN: {test_cm['FN']}\n")

    print(f"\nResults saved to: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train baseline model on MedMNIST with noisy labels'
    )

    # Data parameters
    parser.add_argument('--dataset', type=str, default='pneumoniamnist',
                        choices=list(DATASET_CONFIG.keys()),
                        help='Dataset to use')
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
    print(f"Dataset: {args.dataset} (64x64)")
    print(f"Model: ResNet18")
    print(f"Noise Rate: {args.noise_rate}")
    print(f"Weighted Loss: {args.weighted_loss}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")

    train(args)


if __name__ == '__main__':
    main()
