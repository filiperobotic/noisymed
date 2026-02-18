"""
Analyze and aggregate results from all noisy label experiments.

Scans the results directory for results.json files, aggregates them,
and produces:
    1. Console summary tables (per dataset, per noise level).
    2. CSV files for easy import into spreadsheets.
    3. LaTeX tables for paper inclusion.

Usage:
    python analyze_results.py                       # Default results dir
    python analyze_results.py --results_dir ./results --output_dir ./analysis
"""

import os
import argparse
import json
import csv
from collections import defaultdict


# Method display names and order
METHOD_ORDER = [
    'baseline', 'coteaching', 'filter_loss_gmm', 'crass',
    'dividemix', 'unicon', 'disc',
    'gmm_filter_crass', 'dividemix_crass', 'disc_crass',
    'crass_adaptive', 'coteaching_crass',
    'baseline_cs', 'baseline_cs_adaptive',
    'crass_cs', 'coteaching_cs', 'crass_adaptive_cs',
]
METHOD_DISPLAY = {
    'baseline': 'Baseline+WL',
    'coteaching': 'Co-teaching',
    'filter_loss_gmm': 'GMM Filter',
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

# For experiments that don't set a method field, infer from dir name
DIR_TO_METHOD = {
    'baseline_cs_adaptive': 'baseline_cs_adaptive',
    'baseline_cs': 'baseline_cs',
    'baseline': 'baseline',
    'coteaching_crass': 'coteaching_crass',
    'coteaching_cs': 'coteaching_cs',
    'coteaching': 'coteaching',
    'filter_loss': 'filter_loss_gmm',
    'filter': 'filter_loss_gmm',
    'gmm_filter_crass': 'gmm_filter_crass',
    'crass_adaptive_cs': 'crass_adaptive_cs',
    'crass_adaptive': 'crass_adaptive',
    'crass_cs': 'crass_cs',
    'crass': 'crass',
    'dividemix_crass': 'dividemix_crass',
    'dividemix': 'dividemix',
    'unicon': 'unicon',
    'disc_crass': 'disc_crass',
    'disc': 'disc',
}

DATASETS_ORDER = ['pneumoniamnist', 'breastmnist', 'dermamnist_bin', 'pathmnist_bin']
NOISE_RATES = [0.0, 0.2, 0.4, 0.6]

# Metrics to report
METRICS = ['bac', 'accuracy', 'auc_roc', 'auprc', 'clinical_risk']
METRIC_DISPLAY = {
    'bac': 'BAC',
    'accuracy': 'Acc',
    'auc_roc': 'AUC-ROC',
    'auprc': 'AUPRC',
    'clinical_risk': 'Risk',
}

# Clinical metrics
CLINICAL_METRICS = ['TP', 'TN', 'FP', 'FN']


def find_results_files(results_dir):
    """Recursively find all results.json files."""
    found = []
    for root, dirs, files in os.walk(results_dir):
        if 'results.json' in files:
            found.append(os.path.join(root, 'results.json'))
    return sorted(found)


def infer_method(result, dir_name):
    """Infer method name from result dict or directory name."""
    # From result
    method = result.get('method', '')
    if method:
        return method

    # From weighted_loss presence (baseline)
    if 'weighted_loss' in result and 'warmup_epochs' not in result and \
       'forget_rate' not in result:
        return 'baseline'

    # From directory name
    dir_lower = dir_name.lower()
    for key, val in DIR_TO_METHOD.items():
        if key in dir_lower:
            return val

    return 'unknown'


def get_test_metrics(result):
    """Extract test metrics, handling different result structures.

    Methods with two networks store results as test_ensemble, test_student, etc.
    We prefer: test_ensemble > test_student > test.
    """
    for key in ['test_ensemble', 'test_student', 'test']:
        if key in result:
            return result[key]
    return None


def load_all_results(results_dir):
    """Load and organize all experiment results.

    Returns:
        dict: {dataset: {noise_rate: {method: test_metrics_dict}}}
    """
    files = find_results_files(results_dir)
    print(f"Found {len(files)} result files in {results_dir}")

    organized = defaultdict(lambda: defaultdict(dict))

    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                result = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: Could not read {filepath}: {e}")
            continue

        dataset = result.get('dataset', 'unknown')
        noise_rate = result.get('noise_rate', -1)
        dir_name = os.path.basename(os.path.dirname(filepath))
        method = infer_method(result, dir_name)

        test_metrics = get_test_metrics(result)
        if test_metrics is None:
            print(f"  Warning: No test metrics in {filepath}")
            continue

        # Add extra info
        test_metrics['best_epoch'] = result.get('best_epoch', '?')

        # If method already exists for this combination, keep the newer one
        # (based on file modification time)
        existing = organized[dataset][noise_rate].get(method)
        if existing is not None:
            existing_path = existing.get('_filepath', '')
            # Keep newer
            if os.path.getmtime(filepath) > os.path.getmtime(existing_path):
                test_metrics['_filepath'] = filepath
                organized[dataset][noise_rate][method] = test_metrics
        else:
            test_metrics['_filepath'] = filepath
            organized[dataset][noise_rate][method] = test_metrics

    return organized


def print_summary_tables(organized):
    """Print summary tables to console."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for dataset in DATASETS_ORDER:
        if dataset not in organized:
            continue

        print(f"\n{'─' * 80}")
        print(f"Dataset: {dataset}")
        print(f"{'─' * 80}")

        # Header
        methods_present = []
        for method in METHOD_ORDER:
            for nr in NOISE_RATES:
                if method in organized[dataset].get(nr, {}):
                    if method not in methods_present:
                        methods_present.append(method)

        if not methods_present:
            print("  No results found.")
            continue

        # Print table for each metric
        for metric in METRICS:
            metric_name = METRIC_DISPLAY.get(metric, metric)
            print(f"\n  {metric_name}:")

            # Header row
            header = f"  {'Noise':>8}"
            for method in methods_present:
                header += f"  {METHOD_DISPLAY.get(method, method):>14}"
            print(header)
            print(f"  {'─' * (8 + 16 * len(methods_present))}")

            for nr in NOISE_RATES:
                row = f"  {nr:>8.1%}"
                for method in methods_present:
                    test = organized[dataset].get(nr, {}).get(method)
                    if test is not None and metric in test:
                        val = test[metric]
                        row += f"  {val:>14.4f}"
                    else:
                        row += f"  {'—':>14}"
                print(row)

        # Clinical metrics (TP, TN, FP, FN) for each noise rate
        print(f"\n  Clinical metrics (TP/TN/FP/FN):")
        header = f"  {'Noise':>8}"
        for method in methods_present:
            header += f"  {METHOD_DISPLAY.get(method, method):>20}"
        print(header)
        print(f"  {'─' * (8 + 22 * len(methods_present))}")

        for nr in NOISE_RATES:
            row = f"  {nr:>8.1%}"
            for method in methods_present:
                test = organized[dataset].get(nr, {}).get(method)
                if test is not None:
                    tp = test.get('TP', '?')
                    tn = test.get('TN', '?')
                    fp = test.get('FP', '?')
                    fn = test.get('FN', '?')
                    row += f"  {tp}/{tn}/{fp}/{fn}".rjust(20)
                else:
                    row += f"  {'—':>20}"
            print(row)


def save_csv(organized, output_dir):
    """Save results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # One CSV per dataset
    for dataset in DATASETS_ORDER:
        if dataset not in organized:
            continue

        csv_path = os.path.join(output_dir, f'{dataset}_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = ['noise_rate', 'method']
            header.extend([METRIC_DISPLAY.get(m, m) for m in METRICS])
            header.extend(CLINICAL_METRICS)
            header.append('best_epoch')
            writer.writerow(header)

            for nr in NOISE_RATES:
                for method in METHOD_ORDER:
                    test = organized[dataset].get(nr, {}).get(method)
                    if test is None:
                        continue
                    row = [nr, METHOD_DISPLAY.get(method, method)]
                    for metric in METRICS:
                        row.append(f"{test.get(metric, 0):.4f}")
                    for cm in CLINICAL_METRICS:
                        row.append(test.get(cm, 0))
                    row.append(test.get('best_epoch', '?'))
                    writer.writerow(row)

        print(f"  CSV saved: {csv_path}")

    # Combined CSV
    combined_path = os.path.join(output_dir, 'all_results.csv')
    with open(combined_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['dataset', 'noise_rate', 'method']
        header.extend([METRIC_DISPLAY.get(m, m) for m in METRICS])
        header.extend(CLINICAL_METRICS)
        header.append('best_epoch')
        writer.writerow(header)

        for dataset in DATASETS_ORDER:
            if dataset not in organized:
                continue
            for nr in NOISE_RATES:
                for method in METHOD_ORDER:
                    test = organized[dataset].get(nr, {}).get(method)
                    if test is None:
                        continue
                    row = [dataset, nr, METHOD_DISPLAY.get(method, method)]
                    for metric in METRICS:
                        row.append(f"{test.get(metric, 0):.4f}")
                    for cm in CLINICAL_METRICS:
                        row.append(test.get(cm, 0))
                    row.append(test.get('best_epoch', '?'))
                    writer.writerow(row)

    print(f"  Combined CSV saved: {combined_path}")


def save_latex(organized, output_dir):
    """Generate LaTeX tables."""
    os.makedirs(output_dir, exist_ok=True)

    for dataset in DATASETS_ORDER:
        if dataset not in organized:
            continue

        methods_present = []
        for method in METHOD_ORDER:
            for nr in NOISE_RATES:
                if method in organized[dataset].get(nr, {}):
                    if method not in methods_present:
                        methods_present.append(method)

        if not methods_present:
            continue

        tex_path = os.path.join(output_dir, f'{dataset}_table.tex')
        with open(tex_path, 'w') as f:
            n_methods = len(methods_present)
            col_spec = 'l' + 'c' * n_methods

            f.write(f"% Table for {dataset}\n")
            f.write(f"\\begin{{table}}[ht]\n")
            f.write(f"\\centering\n")
            dataset_escaped = dataset.replace('_', '\\_')
            f.write(f"\\caption{{Results on {dataset_escaped}}}\n")
            f.write(f"\\label{{tab:{dataset}}}\n")

            for metric in ['bac', 'auc_roc']:
                metric_name = METRIC_DISPLAY.get(metric, metric)
                f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
                f.write(f"\\toprule\n")

                # Header
                header = f"Noise"
                for method in methods_present:
                    header += f" & {METHOD_DISPLAY.get(method, method)}"
                header += " \\\\\n"
                f.write(header)
                f.write(f"\\midrule\n")

                # Find best per row for bolding
                for nr in NOISE_RATES:
                    row_vals = {}
                    for method in methods_present:
                        test = organized[dataset].get(nr, {}).get(method)
                        if test and metric in test:
                            row_vals[method] = test[metric]

                    best_method = max(row_vals, key=row_vals.get) if row_vals else None

                    row = f"{nr:.0%}"
                    for method in methods_present:
                        val = row_vals.get(method)
                        if val is not None:
                            if method == best_method:
                                row += f" & \\textbf{{{val:.4f}}}"
                            else:
                                row += f" & {val:.4f}"
                        else:
                            row += " & --"
                    row += " \\\\\n"
                    f.write(row)

                f.write(f"\\bottomrule\n")
                f.write(f"\\end{{tabular}}\n")
                f.write(f"\\quad\n")

            f.write(f"\\end{{table}}\n\n")

        print(f"  LaTeX saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze noisy label experiment results'
    )
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='./analysis',
                        help='Directory to save analysis outputs')

    args = parser.parse_args()

    print(f"Scanning results in: {args.results_dir}")
    organized = load_all_results(args.results_dir)

    if not organized:
        print("No results found!")
        return

    # Console summary
    print_summary_tables(organized)

    # CSV
    print(f"\nSaving CSV files to {args.output_dir}...")
    save_csv(organized, args.output_dir)

    # LaTeX
    print(f"\nSaving LaTeX tables to {args.output_dir}...")
    save_latex(organized, args.output_dir)

    print(f"\nAnalysis complete! Output in: {args.output_dir}")


if __name__ == '__main__':
    main()
