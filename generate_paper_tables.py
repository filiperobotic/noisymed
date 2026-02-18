"""
Generate publication-ready tables from threshold analysis results.

Produces:
    1. Console tables (per dataset)
    2. LaTeX tables ready for paper inclusion
    3. Summary CSV files

Usage:
    python generate_paper_tables.py
    python generate_paper_tables.py --analysis_dir ./threshold_analysis --output_dir ./paper_tables
    python generate_paper_tables.py --datasets dermamnist_bin pathmnist_bin
"""

import os
import argparse
import pandas as pd
import numpy as np


# ── Method ordering and display ──────────────────────────────────────────

METHOD_ORDER = [
    'baseline', 'coteaching', 'filter_loss', 'crass',
    'dividemix', 'unicon', 'disc',
    'gmm_filter_crass', 'dividemix_crass', 'disc_crass',
    'crass_adaptive', 'coteaching_crass',
    'baseline_cs', 'baseline_cs_adaptive',
    'crass_cs', 'coteaching_cs', 'crass_adaptive_cs',
]

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

NOISE_ORDER = [0.0, 0.2, 0.4, 0.6]

DATASET_DISPLAY = {
    'pneumoniamnist': 'PneumoniaMNIST',
    'breastmnist': 'BreastMNIST',
    'dermamnist_bin': 'DermaMNIST (bin)',
    'pathmnist_bin': 'PathMNIST (bin)',
}


def sort_by_method(df):
    """Sort dataframe rows following METHOD_ORDER."""
    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    df = df.copy()
    df['_rank'] = df['method'].map(method_rank).fillna(999)
    df = df.sort_values(['noise_rate', '_rank']).drop(columns='_rank')
    return df


# ── LaTeX generation ─────────────────────────────────────────────────────


def generate_latex_threshold_fixed(df_dataset, dataset_name, lambda_risk):
    """Table: Results at fixed threshold (0.5)."""
    dataset_display = DATASET_DISPLAY.get(dataset_name, dataset_name)
    df = sort_by_method(df_dataset)

    methods = [m for m in METHOD_ORDER if m in df['method'].values]
    n_methods = len(methods)

    lines = []
    lines.append(f"% Table: Fixed threshold results for {dataset_display}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{Results at $\\theta=0.5$ on {dataset_display} "
                 f"($\\lambda={int(lambda_risk)}$)}}")
    lines.append(f"\\label{{tab:{dataset_name}_fixed}}")
    lines.append(r"\begin{tabular}{ll" + "r" * 4 + "}")
    lines.append(r"\toprule")
    lines.append(r"Method & Noise & BAC & Sens. & Spec. & Risk II \\")
    lines.append(r"\midrule")

    for noise in NOISE_ORDER:
        subset = df[df['noise_rate'] == noise]
        if subset.empty:
            continue

        # Find best Risk II for this noise level (lowest)
        best_risk = subset['risk_II_fixed'].min()

        for _, row in subset.iterrows():
            if row['method'] not in methods:
                continue
            m_disp = METHOD_DISPLAY.get(row['method'], row['method'])
            risk_val = row['risk_II_fixed']
            risk_str = f"{risk_val:.3f}"
            if risk_val == best_risk:
                risk_str = f"\\textbf{{{risk_val:.3f}}}"

            lines.append(
                f"{m_disp} & {noise:.0%} & "
                f"{row['BAC_fixed']:.3f} & "
                f"{row['sensitivity_fixed']:.3f} & "
                f"{row['specificity_fixed']:.3f} & "
                f"{risk_str} \\\\"
            )
        lines.append(r"\midrule")

    # Remove last midrule, replace with bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_latex_threshold_optimal(df_dataset, dataset_name, lambda_risk):
    """Table: Results at optimal threshold."""
    dataset_display = DATASET_DISPLAY.get(dataset_name, dataset_name)
    df = sort_by_method(df_dataset)

    methods = [m for m in METHOD_ORDER if m in df['method'].values]

    lines = []
    lines.append(f"% Table: Optimal threshold results for {dataset_display}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{Results at optimal $\\theta^*$ on {dataset_display} "
                 f"($\\lambda={int(lambda_risk)}$)}}")
    lines.append(f"\\label{{tab:{dataset_name}_optimal}}")
    lines.append(r"\begin{tabular}{ll" + "r" * 5 + "}")
    lines.append(r"\toprule")
    lines.append(r"Method & Noise & $\theta^*$ & BAC & Sens. & Spec. & Risk II \\")
    lines.append(r"\midrule")

    for noise in NOISE_ORDER:
        subset = df[df['noise_rate'] == noise]
        if subset.empty:
            continue

        best_risk = subset['risk_II_optimal'].min()

        for _, row in subset.iterrows():
            if row['method'] not in methods:
                continue
            m_disp = METHOD_DISPLAY.get(row['method'], row['method'])
            risk_val = row['risk_II_optimal']
            risk_str = f"{risk_val:.3f}"
            if risk_val == best_risk:
                risk_str = f"\\textbf{{{risk_val:.3f}}}"

            lines.append(
                f"{m_disp} & {noise:.0%} & "
                f"{row['threshold_optimal']:.2f} & "
                f"{row['BAC_optimal']:.3f} & "
                f"{row['sensitivity_optimal']:.3f} & "
                f"{row['specificity_optimal']:.3f} & "
                f"{risk_str} \\\\"
            )
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_latex_comparison(df_dataset, dataset_name, lambda_risk):
    """Table: Comparison of fixed vs optimal threshold (risk reduction)."""
    dataset_display = DATASET_DISPLAY.get(dataset_name, dataset_name)
    df = sort_by_method(df_dataset)

    methods = [m for m in METHOD_ORDER if m in df['method'].values]

    lines = []
    lines.append(f"% Table: Risk reduction comparison for {dataset_display}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{Risk reduction with optimal threshold on {dataset_display} "
                 f"($\\lambda={int(lambda_risk)}$)}}")
    lines.append(f"\\label{{tab:{dataset_name}_comparison}}")
    lines.append(r"\begin{tabular}{ll" + "r" * 4 + "}")
    lines.append(r"\toprule")
    lines.append(r"Method & Noise & Risk ($\theta$=0.5) & $\theta^*$ & "
                 r"Risk ($\theta^*$) & Reduction \\")
    lines.append(r"\midrule")

    for noise in NOISE_ORDER:
        subset = df[df['noise_rate'] == noise]
        if subset.empty:
            continue

        for _, row in subset.iterrows():
            if row['method'] not in methods:
                continue
            m_disp = METHOD_DISPLAY.get(row['method'], row['method'])
            reduction = row['risk_reduction_pct']
            red_str = f"{reduction:.1f}\\%"
            if reduction > 20:
                red_str = f"\\textbf{{{reduction:.1f}\\%}}"

            lines.append(
                f"{m_disp} & {noise:.0%} & "
                f"{row['risk_II_fixed']:.3f} & "
                f"{row['threshold_optimal']:.2f} & "
                f"{row['risk_II_optimal']:.3f} & "
                f"{red_str} \\\\"
            )
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Console tables ───────────────────────────────────────────────────────


def print_console_tables(df, dataset_name, lambda_risk):
    """Print formatted console tables for a single dataset."""
    dataset_display = DATASET_DISPLAY.get(dataset_name, dataset_name)
    df = sort_by_method(df)

    print(f"\n{'=' * 80}")
    print(f"  {dataset_display}  (lambda={int(lambda_risk)})")
    print(f"{'=' * 80}")

    # Table 1: Fixed threshold
    print(f"\n  TABLE 1: Results at threshold = 0.5")
    print(f"  {'-' * 70}")
    cols1 = ['method_display', 'noise_rate', 'BAC_fixed',
             'sensitivity_fixed', 'specificity_fixed', 'risk_II_fixed']
    headers1 = ['Method', 'Noise', 'BAC', 'Sens', 'Spec', 'Risk II']
    print_df = df[cols1].copy()
    print_df.columns = headers1
    print(print_df.to_string(index=False, float_format='%.3f'))

    # Table 2: Optimal threshold
    print(f"\n  TABLE 2: Results at optimal threshold")
    print(f"  {'-' * 70}")
    cols2 = ['method_display', 'noise_rate', 'threshold_optimal',
             'BAC_optimal', 'sensitivity_optimal', 'specificity_optimal',
             'risk_II_optimal']
    headers2 = ['Method', 'Noise', 'theta*', 'BAC', 'Sens', 'Spec', 'Risk II']
    print_df = df[cols2].copy()
    print_df.columns = headers2
    print(print_df.to_string(index=False, float_format='%.3f'))

    # Table 3: Comparison
    print(f"\n  TABLE 3: Risk reduction (fixed vs optimal)")
    print(f"  {'-' * 70}")
    cols3 = ['method_display', 'noise_rate', 'risk_II_fixed',
             'threshold_optimal', 'risk_II_optimal', 'risk_reduction_pct']
    headers3 = ['Method', 'Noise', 'Risk (0.5)', 'theta*',
                'Risk (opt)', 'Reduction (%)']
    print_df = df[cols3].copy()
    print_df.columns = headers3
    print(print_df.to_string(index=False, float_format='%.3f'))


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-ready tables from threshold analysis'
    )
    parser.add_argument('--analysis_dir', type=str,
                        default='./threshold_analysis',
                        help='Directory with threshold analysis CSVs')
    parser.add_argument('--output_dir', type=str, default='./paper_tables',
                        help='Directory to save LaTeX tables')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Filter to specific datasets (e.g. dermamnist_bin)')
    parser.add_argument('--lambda_risk', type=float, default=20,
                        help='Lambda used in the analysis')

    args = parser.parse_args()

    comparison_path = os.path.join(args.analysis_dir,
                                   'comparison_fixed_vs_optimal.csv')
    if not os.path.exists(comparison_path):
        print(f"File not found: {comparison_path}")
        print("Run analyze_optimal_threshold.py first.")
        return

    df = pd.read_csv(comparison_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # Filter datasets if specified
    all_datasets = sorted(df['dataset'].unique())
    if args.datasets:
        all_datasets = [d for d in all_datasets if d in args.datasets]

    if not all_datasets:
        print("No matching datasets found.")
        return

    print(f"Datasets: {all_datasets}")
    print(f"Lambda: {args.lambda_risk}")

    all_latex = []

    for dataset_name in all_datasets:
        df_ds = df[df['dataset'] == dataset_name].copy()
        if df_ds.empty:
            continue

        # Console output
        print_console_tables(df_ds, dataset_name, args.lambda_risk)

        # LaTeX tables
        latex_fixed = generate_latex_threshold_fixed(
            df_ds, dataset_name, args.lambda_risk
        )
        latex_optimal = generate_latex_threshold_optimal(
            df_ds, dataset_name, args.lambda_risk
        )
        latex_comparison = generate_latex_comparison(
            df_ds, dataset_name, args.lambda_risk
        )

        all_latex.append(latex_fixed)
        all_latex.append("")
        all_latex.append(latex_optimal)
        all_latex.append("")
        all_latex.append(latex_comparison)
        all_latex.append("")

        # Save per-dataset LaTeX
        tex_path = os.path.join(args.output_dir,
                                f'{dataset_name}_threshold_tables.tex')
        with open(tex_path, 'w') as f:
            f.write(latex_fixed + "\n\n")
            f.write(latex_optimal + "\n\n")
            f.write(latex_comparison + "\n")
        print(f"\n  LaTeX saved: {tex_path}")

    # Save combined LaTeX
    combined_path = os.path.join(args.output_dir, 'all_threshold_tables.tex')
    with open(combined_path, 'w') as f:
        f.write("\n\n".join(all_latex))
    print(f"\nCombined LaTeX saved: {combined_path}")


if __name__ == "__main__":
    main()
