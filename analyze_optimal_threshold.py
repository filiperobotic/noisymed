"""
Analyze optimal decision threshold for minimizing clinical risk.

For each experiment's saved probabilities, this script:
    1. Evaluates metrics at the default threshold (0.5)
    2. Sweeps thresholds to find the one minimizing Risk II (lambda-weighted)
    3. Generates per-experiment risk-vs-threshold plots
    4. Produces CSV comparison tables (fixed vs optimal threshold)

Usage:
    python analyze_optimal_threshold.py --probs_dir ./probabilities --output_dir ./threshold_analysis
    python analyze_optimal_threshold.py --probs_dir ./probabilities --output_dir ./threshold_analysis --lambda_risk 20
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Metric computation ───────────────────────────────────────────────────


def calculate_metrics_at_threshold(probs, labels, threshold, lambda_risk=20):
    """Compute classification metrics at a given decision threshold.

    Args:
        probs:  (N,) array of P(positive_class)
        labels: (N,) binary array (1 = positive, 0 = negative)
        threshold: decision boundary
        lambda_risk: weight for FN in Risk II (C_FN / C_FP)

    Returns:
        dict with TP, TN, FP, FN, sensitivity, specificity, BAC,
        risk_I, risk_II
    """
    preds = (probs >= threshold).astype(int)

    TP = int(((preds == 1) & (labels == 1)).sum())
    TN = int(((preds == 0) & (labels == 0)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())
    FN = int(((preds == 0) & (labels == 1)).sum())

    N = TP + TN + FP + FN

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    bac = (sensitivity + specificity) / 2.0

    risk_I = (FN + FP) / N if N > 0 else 0.0
    risk_II = (lambda_risk * FN + FP) / N if N > 0 else 0.0

    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'BAC': bac,
        'risk_I': risk_I,
        'risk_II': risk_II,
    }


def find_optimal_threshold(probs, labels, lambda_risk=20, metric='risk_II'):
    """Sweep thresholds and find the one minimizing the chosen risk metric.

    Args:
        probs:  (N,) P(positive)
        labels: (N,) binary
        lambda_risk: FN weight
        metric: 'risk_I' or 'risk_II'

    Returns:
        best_threshold, best_metrics (dict), all_results (list of dicts)
    """
    thresholds = np.arange(0.01, 1.00, 0.01)
    all_results = []

    best_threshold = 0.5
    best_risk = float('inf')
    best_metrics = None

    for thr in thresholds:
        m = calculate_metrics_at_threshold(probs, labels, thr, lambda_risk)
        m['threshold'] = round(float(thr), 2)
        all_results.append(m)

        if m[metric] < best_risk:
            best_risk = m[metric]
            best_threshold = m['threshold']
            best_metrics = m

    return best_threshold, best_metrics, all_results


# ── Plotting ─────────────────────────────────────────────────────────────


def plot_risk_vs_threshold(all_results, title, output_path):
    """Plot Risk I, Risk II, and BAC against threshold."""
    thresholds = [r['threshold'] for r in all_results]
    risk_I = [r['risk_I'] for r in all_results]
    risk_II = [r['risk_II'] for r in all_results]
    bac = [r['BAC'] for r in all_results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Risk', color='tab:red', fontsize=12)
    ax1.plot(thresholds, risk_I, 'r--', alpha=0.7, label='Risk I ($\\lambda$=1)')
    ax1.plot(thresholds, risk_II, 'r-', linewidth=2, label='Risk II ($\\lambda$=20)')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.6,
                label='Default threshold (0.5)')

    ax2 = ax1.twinx()
    ax2.set_ylabel('BAC', color='tab:blue', fontsize=12)
    ax2.plot(thresholds, bac, 'b-', linewidth=2, label='BAC')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Mark optimal threshold
    min_idx = int(np.argmin(risk_II))
    opt_thr = thresholds[min_idx]
    ax1.axvline(x=opt_thr, color='green', linestyle='--', linewidth=1.5,
                label=f'Optimal $\\theta^*$ = {opt_thr:.2f}')
    ax1.plot(opt_thr, risk_II[min_idx], 'go', markersize=8, zorder=5)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper right', fontsize=10)

    plt.title(title, fontsize=13)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Filename parsing ─────────────────────────────────────────────────────

# Method display names (matching analyze_results.py conventions)
METHOD_DISPLAY = {
    'baseline': 'Baseline+WL',
    'coteaching': 'Co-teaching',
    'filter_loss': 'GMM Filter',
    'crass': 'CRASS',
    'dividemix': 'DivideMix',
    'unicon': 'UNICON',
    'disc': 'DISC',
    'gmm_filter_crass': 'GMM+CRASS',
    'dividemix_crass': 'DivideMix+CRASS',
    'disc_crass': 'DISC+CRASS',
    'crass_adaptive': 'CRASS Adaptive',
    'coteaching_crass': 'CT+CRASS',
    'baseline_cs': 'Baseline+CS',
    'baseline_cs_adaptive': 'Baseline+CS-A',
    'crass_cs': 'CRASS+CS',
    'coteaching_cs': 'CT+CS',
    'crass_adaptive_cs': 'CRASS-A+CS',
}


def parse_probs_filename(filename):
    """Extract method, dataset, noise_rate from a _probs.npz file.

    We load the metadata from the .npz itself rather than parsing the
    filename, since the npz stores method/dataset/noise_rate.
    """
    pass  # We'll read from the npz directly


# ── Main analysis ────────────────────────────────────────────────────────


def analyze_all_experiments(probs_dir, output_dir, lambda_risk=20):
    """Process all probability files and produce analysis outputs."""
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    results_fixed = []
    results_optimal = []

    files = sorted([f for f in os.listdir(probs_dir) if f.endswith('_probs.npz')])
    if not files:
        print(f"No probability files found in {probs_dir}")
        return None

    print(f"Found {len(files)} probability files\n")

    for filename in files:
        filepath = os.path.join(probs_dir, filename)
        data = np.load(filepath, allow_pickle=True)

        probs = data['probs']
        labels = data['labels']
        method = str(data['method'])
        dataset = str(data['dataset'])
        noise_rate = float(data['noise_rate'])

        method_display = METHOD_DISPLAY.get(method, method)
        exp_label = f"{method_display} | {dataset} | noise={noise_rate}"

        # Threshold fixo (0.5)
        metrics_fixed = calculate_metrics_at_threshold(
            probs, labels, 0.5, lambda_risk
        )
        metrics_fixed.update({
            'method': method,
            'method_display': method_display,
            'dataset': dataset,
            'noise_rate': noise_rate,
            'threshold': 0.5,
        })
        results_fixed.append(metrics_fixed)

        # Threshold otimo
        best_thr, best_metrics, all_results = find_optimal_threshold(
            probs, labels, lambda_risk, metric='risk_II'
        )
        best_metrics.update({
            'method': method,
            'method_display': method_display,
            'dataset': dataset,
            'noise_rate': noise_rate,
            'threshold': best_thr,
        })
        results_optimal.append(best_metrics)

        # Plot
        plot_name = filename.replace('_probs.npz', '_risk_curve.png')
        plot_risk_vs_threshold(
            all_results,
            f"{exp_label}",
            os.path.join(plots_dir, plot_name)
        )

        print(f"{exp_label}:")
        print(f"  Threshold 0.5:   BAC={metrics_fixed['BAC']:.3f}, "
              f"Risk II={metrics_fixed['risk_II']:.3f}, "
              f"Sens={metrics_fixed['sensitivity']:.3f}, "
              f"Spec={metrics_fixed['specificity']:.3f}")
        print(f"  Threshold opt:   theta*={best_thr:.2f}, "
              f"BAC={best_metrics['BAC']:.3f}, "
              f"Risk II={best_metrics['risk_II']:.3f}, "
              f"Sens={best_metrics['sensitivity']:.3f}, "
              f"Spec={best_metrics['specificity']:.3f}")

    # ── Save tables ──────────────────────────────────────────────────────

    df_fixed = pd.DataFrame(results_fixed)
    df_optimal = pd.DataFrame(results_optimal)

    df_fixed.to_csv(
        os.path.join(output_dir, 'results_threshold_fixed.csv'), index=False
    )
    df_optimal.to_csv(
        os.path.join(output_dir, 'results_threshold_optimal.csv'), index=False
    )

    # Comparison table
    comparison = []
    for fixed, optimal in zip(results_fixed, results_optimal):
        risk_fixed = fixed['risk_II']
        risk_opt = optimal['risk_II']
        reduction = ((risk_fixed - risk_opt) / risk_fixed * 100
                     if risk_fixed > 0 else 0.0)

        comparison.append({
            'method': fixed['method'],
            'method_display': fixed['method_display'],
            'dataset': fixed['dataset'],
            'noise_rate': fixed['noise_rate'],
            'threshold_fixed': 0.5,
            'threshold_optimal': optimal['threshold'],
            'BAC_fixed': fixed['BAC'],
            'BAC_optimal': optimal['BAC'],
            'sensitivity_fixed': fixed['sensitivity'],
            'sensitivity_optimal': optimal['sensitivity'],
            'specificity_fixed': fixed['specificity'],
            'specificity_optimal': optimal['specificity'],
            'risk_II_fixed': risk_fixed,
            'risk_II_optimal': risk_opt,
            'risk_reduction_pct': reduction,
        })

    df_comparison = pd.DataFrame(comparison)
    df_comparison.to_csv(
        os.path.join(output_dir, 'comparison_fixed_vs_optimal.csv'), index=False
    )

    # Console summary
    print("\n" + "=" * 80)
    print(f"SUMMARY: Fixed (0.5) vs Optimal Threshold  [lambda={lambda_risk}]")
    print("=" * 80)

    summary_cols = [
        'method_display', 'dataset', 'noise_rate',
        'threshold_optimal', 'BAC_fixed', 'BAC_optimal',
        'risk_II_fixed', 'risk_II_optimal', 'risk_reduction_pct',
    ]
    print(df_comparison[summary_cols].to_string(index=False, float_format='%.3f'))

    print(f"\nResults saved to: {output_dir}")
    print(f"Plots saved to: {plots_dir}")

    return df_comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze optimal threshold for clinical risk minimization'
    )
    parser.add_argument('--probs_dir', type=str, default='./probabilities',
                        help='Directory with probability .npz files')
    parser.add_argument('--output_dir', type=str, default='./threshold_analysis',
                        help='Directory to save analysis outputs')
    parser.add_argument('--lambda_risk', type=float, default=20,
                        help='Lambda for Risk II = (lambda*FN + FP) / N')

    args = parser.parse_args()

    analyze_all_experiments(args.probs_dir, args.output_dir, args.lambda_risk)
