"""
Baseline + Cost-Sensitive Loss Training Script for Medical Image Classification
with Noisy Labels.

Datasets: PneumoniaMNIST, BreastMNIST (64x64)
Model: ResNet18

Instead of standard cross-entropy or frequency-balanced weights, this script uses
clinical cost-sensitive weights for the cross-entropy loss:
    w[positive_class] = lambda_risk   (penalise missing disease / FN)
    w[other_class]    = 1.0           (standard FP cost)

The positive_class varies by dataset (breastmnist: 0 = malignant, others: 1).

Two variants in one script, controlled by --adaptive_lambda:
    1. Fixed lambda (default):
       method = 'baseline_cs'
       class_weights use lambda_risk directly.
    2. Adaptive lambda (--adaptive_lambda flag):
       method = 'baseline_cs_adaptive'
       lambda_effective = lambda_risk * (1 - noise_rate)
       This reduces the cost penalty proportionally to the noise level,
       acknowledging that some "positive" labels are corrupted negatives.

Usage:
    # Fixed lambda
    python train_baseline_cs.py --dataset pneumoniamnist --noise_rate 0.2 --lambda_risk 20
    # Adaptive lambda
    python train_baseline_cs.py --dataset breastmnist --noise_rate 0.4 --lambda_risk 20 --adaptive_lambda
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


def compute_cost_sensitive_weights(positive_class, lambda_risk, device):
    """Compute cost-sensitive class weights.

    Instead of frequency-balanced weights, use clinical cost weights:
        w[positive_class] = lambda_risk  (penalize FN)
        w[other_class] = 1.0             (standard FP cost)

    Args:
        positive_class: which class is positive (disease). 0 or 1.
        lambda_risk: cost ratio C_FN / C_FP.
        device: torch device.

    Returns:
        Tensor [w0, w1] on device.
    """
    weights = torch.ones(2, dtype=torch.float32, device=device)
    weights[positive_class] = lambda_risk
    return weights


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


def compute_confusion_matrix(outputs, targets, positive_class=1):
    """
    Compute confusion matrix elements.

    Args:
        outputs: model logits (N, num_classes)
        targets: ground-truth labels (N,)
        positive_class: which class index is the "positive" (disease/malignant).
                        Varies per dataset (e.g. breastmnist: positive_class=0).

    Returns:
        dict with TP, TN, FP, FN
    """
    _, predicted = torch.max(outputs, 1)

    TP = ((predicted == positive_class) & (targets == positive_class)).sum().item()
    TN = ((predicted != positive_class) & (targets != positive_class)).sum().item()
    FP = ((predicted == positive_class) & (targets != positive_class)).sum().item()
    FN = ((predicted != positive_class) & (targets == positive_class)).sum().item()

    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}


def compute_bac(cm):
    """
    Compute Balanced Accuracy (BAC) from confusion matrix.

    BAC = (Sensitivity + Specificity) / 2
        Sensitivity (Recall) = TP / (TP + FN)
        Specificity           = TN / (TN + FP)

    Args:
        cm: dict with keys 'TP', 'TN', 'FP', 'FN'

    Returns:
        float: Balanced Accuracy
    """
    TP, TN, FP, FN = cm['TP'], cm['TN'], cm['FP'], cm['FN']
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    return (sensitivity + specificity) / 2.0


def compute_clinical_risk(cm, lambda_risk):
    """Compute normalised clinical risk: Risk = (lambda * FN + FP) / N."""
    N = cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']
    if N == 0:
        return 0.0
    return (lambda_risk * cm['FN'] + cm['FP']) / N


def train_epoch(model, train_loader, optimizer, device, class_weights=None,
                positive_class=1):
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

        # Use cost-sensitive weighted cross entropy
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
    cm = compute_confusion_matrix(all_outputs, all_targets, positive_class)

    return epoch_loss, accuracy, cm


def evaluate(model, data_loader, device, class_weights=None, positive_class=1):
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
    cm = compute_confusion_matrix(all_outputs, all_targets, positive_class)

    # Compute BAC
    bac = compute_bac(cm)

    # Compute AUC-ROC and AUPRC
    # Use P(positive_class) as the score for ROC/PRC
    probs = F.softmax(all_outputs, dim=1)[:, positive_class].cpu().numpy()
    targets_np = (all_targets == positive_class).long().cpu().numpy()
    auc_roc = roc_auc_score(targets_np, probs)
    auprc = average_precision_score(targets_np, probs)

    return epoch_loss, accuracy, cm, bac, auc_roc, auprc


def train(args):
    """Main training function"""
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = DATASET_CONFIG[args.dataset]
    positive_class = config.get('positive_class', 1)
    print(f"Positive class: {positive_class} "
          f"({config['class_names'].get(positive_class, '?')})")

    # Determine method and effective lambda
    if args.adaptive_lambda:
        method = 'baseline_cs_adaptive'
        lambda_effective = args.lambda_risk * (1.0 - args.noise_rate)
        print(f"\nAdaptive lambda: lambda_risk * (1 - noise_rate) = "
              f"{args.lambda_risk} * {1.0 - args.noise_rate:.2f} = {lambda_effective:.4f}")
    else:
        method = 'baseline_cs'
        lambda_effective = args.lambda_risk

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.adaptive_lambda:
        exp_name = (
            f"{args.dataset}_noise{args.noise_rate}"
            f"_baseline_cs_adaptive_lambda{args.lambda_risk}_{timestamp}"
        )
    else:
        exp_name = (
            f"{args.dataset}_noise{args.noise_rate}"
            f"_baseline_cs_lambda{args.lambda_risk}_{timestamp}"
        )
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_dict = vars(args).copy()
    config_dict['method'] = method
    config_dict['lambda_effective'] = lambda_effective
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

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

    # Compute cost-sensitive class weights
    class_weights = compute_cost_sensitive_weights(
        positive_class, lambda_effective, device
    )
    print(f"\nCost-sensitive class weights: {class_weights.cpu().tolist()}")
    print(f"  w[class {positive_class} (positive)] = {lambda_effective:.4f}")
    print(f"  w[class {1 - positive_class} (negative)] = 1.0")

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

    best_val_bac = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    best_val_cm = None
    best_val_auc_roc = 0.0
    best_val_auprc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, train_cm = train_epoch(
            model, train_loader, optimizer, device, class_weights,
            positive_class=positive_class
        )

        # Validate
        val_loss, val_acc, val_cm, val_bac, val_auc_roc, val_auprc = evaluate(
            model, val_loader, device, class_weights,
            positive_class=positive_class
        )

        scheduler.step()

        # Log progress
        if epoch % args.print_freq == 0 or epoch == 1:
            print(f"\nEpoch [{epoch}/{args.epochs}]")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, BAC: {val_bac:.4f}")
            print(f"          AUC-ROC: {val_auc_roc:.4f}, AUPRC: {val_auprc:.4f}")
            print(f"          TP: {val_cm['TP']}, TN: {val_cm['TN']}, FP: {val_cm['FP']}, FN: {val_cm['FN']}")

        # Save best model based on BAC (Balanced Accuracy)
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

    # Load best model and evaluate on test set
    print("\n" + "="*50)
    print(f"Loading best model from epoch {best_epoch}...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_cm, test_bac, test_auc_roc, test_auprc = evaluate(
        model, test_loader, device, class_weights,
        positive_class=positive_class
    )

    # Compute clinical risk on test set
    test_clinical_risk = compute_clinical_risk(test_cm, args.lambda_risk)
    val_clinical_risk = compute_clinical_risk(best_val_cm, args.lambda_risk)

    # Print final results
    print("\n" + "="*50)
    print(f"FINAL RESULTS (Best Epoch: {best_epoch}, selected by BAC)")
    print("="*50)
    print(f"\nValidation:")
    print(f"  Accuracy: {best_val_acc:.4f}, BAC: {best_val_bac:.4f}")
    print(f"  AUC-ROC: {best_val_auc_roc:.4f}, AUPRC: {best_val_auprc:.4f}")
    print(f"  Clinical Risk: {val_clinical_risk:.4f}")
    print(f"  TP: {best_val_cm['TP']}, TN: {best_val_cm['TN']}, FP: {best_val_cm['FP']}, FN: {best_val_cm['FN']}")

    print(f"\nTest:")
    print(f"  Accuracy: {test_acc:.4f}, BAC: {test_bac:.4f}")
    print(f"  AUC-ROC: {test_auc_roc:.4f}, AUPRC: {test_auprc:.4f}")
    print(f"  Clinical Risk: {test_clinical_risk:.4f}")
    print(f"  TP: {test_cm['TP']}, TN: {test_cm['TN']}, FP: {test_cm['FP']}, FN: {test_cm['FN']}")

    # Save results
    results = {
        'method': method,
        'dataset': args.dataset,
        'best_epoch': best_epoch,
        'best_metric': 'BAC',
        'noise_rate': args.noise_rate,
        'lambda_risk': args.lambda_risk,
        'adaptive_lambda': args.adaptive_lambda,
        'lambda_effective': lambda_effective,
        'class_weights': class_weights.cpu().tolist(),
        'validation': {
            'accuracy': best_val_acc, 'bac': best_val_bac,
            'auc_roc': best_val_auc_roc, 'auprc': best_val_auprc,
            'clinical_risk': val_clinical_risk,
            **best_val_cm
        },
        'test': {
            'accuracy': test_acc, 'bac': test_bac,
            'auc_roc': test_auc_roc, 'auprc': test_auprc,
            'clinical_risk': test_clinical_risk,
            **test_cm
        }
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save results to TXT file
    txt_path = os.path.join(output_dir, 'results.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("EXPERIMENT RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Method: {method}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Noise Rate: {args.noise_rate}\n")
        f.write(f"Lambda Risk: {args.lambda_risk}\n")
        f.write(f"Adaptive Lambda: {args.adaptive_lambda}\n")
        f.write(f"Lambda Effective: {lambda_effective:.4f}\n")
        f.write(f"Class Weights: {class_weights.cpu().tolist()}\n")
        f.write(f"Best Epoch: {best_epoch} (selected by BAC)\n\n")
        f.write("-" * 50 + "\n")
        f.write("VALIDATION RESULTS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Accuracy: {best_val_acc:.4f}\n")
        f.write(f"BAC: {best_val_bac:.4f}\n")
        f.write(f"AUC-ROC: {best_val_auc_roc:.4f}\n")
        f.write(f"AUPRC: {best_val_auprc:.4f}\n")
        f.write(f"Clinical Risk: {val_clinical_risk:.4f}\n")
        f.write(f"TP: {best_val_cm['TP']}, TN: {best_val_cm['TN']}, FP: {best_val_cm['FP']}, FN: {best_val_cm['FN']}\n\n")
        f.write("-" * 50 + "\n")
        f.write("TEST RESULTS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"BAC: {test_bac:.4f}\n")
        f.write(f"AUC-ROC: {test_auc_roc:.4f}\n")
        f.write(f"AUPRC: {test_auprc:.4f}\n")
        f.write(f"Clinical Risk: {test_clinical_risk:.4f}\n")
        f.write(f"TP: {test_cm['TP']}, TN: {test_cm['TN']}, FP: {test_cm['FP']}, FN: {test_cm['FN']}\n")

    print(f"\nResults saved to: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train baseline + cost-sensitive loss model on MedMNIST with noisy labels'
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

    # Cost-sensitive parameters
    parser.add_argument('--lambda_risk', type=float, default=20.0,
                        help='Cost ratio C_FN / C_FP (default: 20.0)')
    parser.add_argument('--adaptive_lambda', action='store_true',
                        help='Use adaptive lambda = lambda_risk * (1 - noise_rate)')

    # Model parameters
    parser.add_argument('--pretrained', action='store_true')

    # Other parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_freq', type=int, default=10)

    args = parser.parse_args()

    # Determine method for display
    method = 'baseline_cs_adaptive' if args.adaptive_lambda else 'baseline_cs'

    print("\n" + "="*50)
    print("EXPERIMENT CONFIGURATION")
    print("="*50)
    print(f"Method: {method}")
    print(f"Dataset: {args.dataset} (64x64)")
    print(f"Model: ResNet18")
    print(f"Noise Rate: {args.noise_rate}")
    print(f"Lambda Risk: {args.lambda_risk}")
    print(f"Adaptive Lambda: {args.adaptive_lambda}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")

    train(args)


if __name__ == '__main__':
    main()
