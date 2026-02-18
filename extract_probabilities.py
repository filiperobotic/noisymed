"""
Extract probabilities from all saved models.

Scans the results directory for best_model.pth files, loads each model,
runs inference on the test set, and saves probabilities + labels as .npz.

Output format per experiment:
    {output_dir}/{exp_dir_name}_probs.npz
    containing: probs (N,), labels (N,), positive_class (scalar)

Usage:
    python extract_probabilities.py --results_dir ./results --output_dir ./probabilities
    python extract_probabilities.py --results_dir ./results --output_dir ./probabilities --device cpu
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader

from dataloaders import get_dataloaders, get_filter_dataloaders, DATASET_CONFIG


# ── Model definitions ────────────────────────────────────────────────────


def get_model(num_classes=2, in_channels=1):
    """Standard ResNet18 (baseline, coteaching, gmm_filter, crass, dividemix)."""
    model = models.resnet18(pretrained=False)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ResNet18WithProjection(nn.Module):
    """ResNet18 + MLP projection head (used by UNICON)."""

    def __init__(self, num_classes=2, in_channels=1, proj_dim=128):
        super().__init__()
        backbone = models.resnet18(pretrained=False)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.fc = nn.Linear(feat_dim, num_classes)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x, return_features=False, return_projection=False):
        features = self.backbone(x)
        logits = self.fc(features)
        if return_projection:
            proj = self.projector(features)
            proj = F.normalize(proj, dim=1)
            return logits, proj
        if return_features:
            return logits, features
        return logits


# ── Checkpoint key mapping ───────────────────────────────────────────────

# Maps method keyword (inferred from directory name) to:
#   model_type: "single", "dual", "student_teacher"
#   state_key:  key(s) for model state_dict in the checkpoint
#   model_class: which model constructor to use

METHOD_INFO = {
    'baseline':          {'model_type': 'single',          'state_key': 'model_state_dict',   'model_class': 'resnet18'},
    'coteaching':        {'model_type': 'dual',            'state_key': ('model1_state_dict', 'model2_state_dict'), 'model_class': 'resnet18'},
    'filter_loss':       {'model_type': 'single',          'state_key': 'model_state_dict',   'model_class': 'resnet18'},
    'crass':             {'model_type': 'single',          'state_key': 'model_state_dict',   'model_class': 'resnet18'},
    'dividemix':         {'model_type': 'dual',            'state_key': ('model1_state_dict', 'model2_state_dict'), 'model_class': 'resnet18'},
    'unicon':            {'model_type': 'dual',            'state_key': ('model1_state_dict', 'model2_state_dict'), 'model_class': 'unicon'},
    'disc':              {'model_type': 'student_teacher',  'state_key': 'student_state_dict', 'model_class': 'resnet18'},
    'gmm_filter_crass':  {'model_type': 'single',          'state_key': 'model_state_dict',   'model_class': 'resnet18'},
    'dividemix_crass':   {'model_type': 'dual',            'state_key': ('model1_state_dict', 'model2_state_dict'), 'model_class': 'resnet18'},
    'disc_crass':        {'model_type': 'student_teacher',  'state_key': 'student_state_dict', 'model_class': 'resnet18'},
}


def infer_method(dir_name):
    """Infer method from experiment directory name.

    Directory names follow the pattern:
        {dataset}_noise{rate}_{method}_{params}_{timestamp}
    e.g. pneumoniamnist_noise0.2_coteaching_fr0.2_grad10_20240101_120000
    """
    dir_lower = dir_name.lower()

    # Order matters: check more specific patterns first
    ordered_keys = [
        'gmm_filter_crass', 'dividemix_crass', 'disc_crass',
        'filter_loss', 'coteaching', 'dividemix', 'unicon',
        'disc', 'crass', 'baseline',
    ]
    for key in ordered_keys:
        if key in dir_lower:
            return key

    # Fallback: check for 'weighted'/'standard' (baseline pattern)
    if 'weighted' in dir_lower or 'standard' in dir_lower:
        return 'baseline'

    return None


def infer_dataset(dir_name):
    """Infer dataset name from experiment directory name."""
    for ds in DATASET_CONFIG:
        if dir_name.startswith(ds):
            return ds
    return None


def infer_noise_rate(dir_name):
    """Extract noise rate from directory name like ..._noise0.2_..."""
    import re
    match = re.search(r'noise([\d.]+)', dir_name)
    if match:
        return float(match.group(1))
    return None


def build_model(method_key, num_classes, in_channels, device):
    """Instantiate the right model for a given method."""
    info = METHOD_INFO[method_key]
    if info['model_class'] == 'unicon':
        return ResNet18WithProjection(num_classes, in_channels).to(device)
    return get_model(num_classes, in_channels).to(device)


def load_and_extract(checkpoint_path, method_key, test_loader,
                     num_classes, in_channels, positive_class, device):
    """Load checkpoint, run inference, return (probs, labels).

    For dual-network methods: returns average of softmax outputs.
    For student-teacher methods: uses the student network.
    For single-network methods: uses the single model.

    Returns:
        probs: np.ndarray (N,)  - P(positive_class) for each test sample
        labels: np.ndarray (N,) - binary labels (1 = positive_class)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    info = METHOD_INFO[method_key]

    if info['model_type'] == 'dual':
        # Two-network methods: ensemble average
        key1, key2 = info['state_key']
        model1 = build_model(method_key, num_classes, in_channels, device)
        model2 = build_model(method_key, num_classes, in_channels, device)
        model1.load_state_dict(checkpoint[key1])
        model2.load_state_dict(checkpoint[key2])
        model1.eval()
        model2.eval()

        all_probs = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                out1 = model1(images)
                out2 = model2(images)
                avg_probs = (F.softmax(out1, dim=1) + F.softmax(out2, dim=1)) / 2
                all_probs.append(avg_probs[:, positive_class].cpu().numpy())
                all_labels.append(labels.numpy())

    elif info['model_type'] == 'student_teacher':
        # DISC: use student
        model = build_model(method_key, num_classes, in_channels, device)
        model.load_state_dict(checkpoint[info['state_key']])
        model.eval()

        all_probs = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs[:, positive_class].cpu().numpy())
                all_labels.append(labels.numpy())

    else:
        # Single-network methods
        model = build_model(method_key, num_classes, in_channels, device)
        model.load_state_dict(checkpoint[info['state_key']])
        model.eval()

        all_probs = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs[:, positive_class].cpu().numpy())
                all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    # Convert labels to binary: 1 = positive_class, 0 = otherwise
    labels_binary = (labels == positive_class).astype(np.int32)

    return probs, labels_binary


# ── Test loader cache ────────────────────────────────────────────────────

_test_loader_cache = {}


def get_test_loader(dataset_name, data_dir='./data', batch_size=128):
    """Get (and cache) the test loader for a dataset."""
    if dataset_name in _test_loader_cache:
        return _test_loader_cache[dataset_name]

    # Use get_dataloaders with noise_rate=0 just to get the test loader
    _, _, test_loader = get_dataloaders(
        dataset=dataset_name, noise_rate=0.0,
        batch_size=batch_size, seed=42, data_dir=data_dir
    )
    _test_loader_cache[dataset_name] = test_loader
    return test_loader


# ── Main ─────────────────────────────────────────────────────────────────


def process_all_experiments(results_dir, output_dir, data_dir, device):
    """Scan results_dir for experiments, extract probabilities, save .npz."""
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    exp_dirs = sorted(os.listdir(results_dir))
    total = len(exp_dirs)
    processed = 0
    skipped = 0

    for i, exp_dir in enumerate(exp_dirs, 1):
        exp_path = os.path.join(results_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue

        checkpoint_path = os.path.join(exp_path, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            print(f"  [{i}/{total}] SKIP (no checkpoint): {exp_dir}")
            skipped += 1
            continue

        # Infer experiment parameters
        method_key = infer_method(exp_dir)
        dataset_name = infer_dataset(exp_dir)
        noise_rate = infer_noise_rate(exp_dir)

        if method_key is None or dataset_name is None:
            print(f"  [{i}/{total}] SKIP (cannot infer method/dataset): {exp_dir}")
            skipped += 1
            continue

        if method_key not in METHOD_INFO:
            print(f"  [{i}/{total}] SKIP (unknown method '{method_key}'): {exp_dir}")
            skipped += 1
            continue

        config = DATASET_CONFIG[dataset_name]
        num_classes = config['num_classes']
        in_channels = config['in_channels']
        positive_class = config.get('positive_class', 1)

        print(f"  [{i}/{total}] Processing: {exp_dir}")
        print(f"           method={method_key}, dataset={dataset_name}, "
              f"noise={noise_rate}, positive_class={positive_class}")

        try:
            test_loader = get_test_loader(dataset_name, data_dir)

            probs, labels = load_and_extract(
                checkpoint_path, method_key, test_loader,
                num_classes, in_channels, positive_class, device
            )

            # Save
            output_file = os.path.join(output_dir, f"{exp_dir}_probs.npz")
            np.savez(
                output_file,
                probs=probs,
                labels=labels,
                positive_class=positive_class,
                method=method_key,
                dataset=dataset_name,
                noise_rate=noise_rate if noise_rate is not None else -1,
            )
            print(f"           Saved: {output_file}")
            print(f"           probs shape={probs.shape}, "
                  f"label dist: pos={labels.sum()}, neg={(1-labels).sum()}")
            processed += 1

        except Exception as e:
            print(f"           ERROR: {e}")
            skipped += 1

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract probabilities from all saved models'
    )
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='./probabilities',
                        help='Directory to save probability files')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing dataset files')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference (cuda or cpu)')

    args = parser.parse_args()

    device_str = args.device
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_str = 'cpu'
    device = torch.device(device_str)

    print(f"Device: {device}")
    print(f"Results dir: {args.results_dir}")
    print(f"Output dir: {args.output_dir}")
    print()

    process_all_experiments(args.results_dir, args.output_dir, args.data_dir, device)
