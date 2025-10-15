#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def load_data(csv_path: str) -> pd.DataFrame:
    """Load results CSV and coerce types."""
    df = pd.read_csv(csv_path)
    # Ensure numeric for analysis
    df['symptom_difference'] = pd.to_numeric(df['symptom_difference'], errors='coerce')
    return df


def analyze_wilcoxon_by_symptom(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each unique symptom_name:
    - Collect non-zero symptom_difference values across the entire dataset
    - Run two-sided Wilcoxon signed-rank test against 0
    - Summarize p-value, median difference, and direction/magnitude
    """
    results = []
    symptoms = sorted(df['symptom_name'].dropna().unique())

    for symptom in symptoms:
        diffs = df.loc[df['symptom_name'] == symptom, 'symptom_difference'].dropna()
        non_zero = diffs[diffs != 0]
        n = int(non_zero.shape[0])

        record = {
            'symptom_name': symptom,
            'n_non_zero': n,
        }

        if n < 3:
            record.update({
                'statistic': None,
                'p_value': None,
                'median_difference': None,
                'mean_difference': None,
                'std_difference': None,
                'pos_count': int((non_zero > 0).sum()),
                'neg_count': int((non_zero < 0).sum()),
                'pos_fraction': None,
                'neg_fraction': None,
                'mean_abs_difference': None,
                'median_abs_difference': None,
                'significant': False,
                'trend': 'insufficient_data',
                'error': 'Insufficient non-zero differences (<3)'
            })
            results.append(record)
            continue

        # Wilcoxon signed-rank test (two-sided) on non-zero differences
        stat, p = wilcoxon(
            non_zero,
            alternative='two-sided',
            zero_method='wilcox',
            correction=False,
            mode='auto'
        )

        median = float(np.median(non_zero))
        mean = float(np.mean(non_zero))
        std = float(np.std(non_zero, ddof=1)) if n > 1 else 0.0
        pos_count = int((non_zero > 0).sum())
        neg_count = int((non_zero < 0).sum())
        pos_frac = pos_count / n
        neg_frac = neg_count / n
        mean_abs = float(np.mean(np.abs(non_zero)))
        median_abs = float(np.median(np.abs(non_zero)))

        # Simple trend heuristic combining median sign and majority direction
        if median > 0 and pos_frac >= 0.6:
            trend = 'increase'
        elif median < 0 and neg_frac >= 0.6:
            trend = 'decrease'
        else:
            trend = 'mixed'

        record.update({
            'statistic': float(stat),
            'p_value': float(p),
            'median_difference': median,
            'mean_difference': mean,
            'std_difference': std,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'pos_fraction': float(pos_frac),
            'neg_fraction': float(neg_frac),
            'mean_abs_difference': mean_abs,
            'median_abs_difference': median_abs,
            'significant': bool(p < 0.05),
            'trend': trend,
        })
        results.append(record)

    return pd.DataFrame(results)


def print_summary(summary_df: pd.DataFrame) -> None:
    print("=" * 80)
    print("Wilcoxon Signed-Rank Test by Fairness Symptom")
    print("=" * 80)
    total = int(summary_df.shape[0])
    analyzed = summary_df.dropna(subset=['p_value'])

    print(f"Symptoms analyzed: {total}")
    print(f"Symptoms with sufficient data: {int(analyzed.shape[0])}")

    if not analyzed.empty:
        print("\nTop symptoms by significance (lowest p-values):")
        top = analyzed.sort_values('p_value').head(15)
        for _, row in top.iterrows():
            print(
                f"- {row['symptom_name']}: p={row['p_value']:.4g}, "
                f"median={row['median_difference']:.6f}, "
                f"n={int(row['n_non_zero'])}, "
                f"trend={row['trend']}, "
                f"significant={'Yes' if row['significant'] else 'No'}"
            )

        inc = int((analyzed['trend'] == 'increase').sum())
        dec = int((analyzed['trend'] == 'decrease').sum())
        mix = int((analyzed['trend'] == 'mixed').sum())
        print("\nTrend counts:")
        print(f"- increase: {inc}")
        print(f"- decrease: {dec}")
        print(f"- mixed: {mix}")


def save_outputs(summary_df: pd.DataFrame, csv_path: str, output_dir: str | None = None) -> None:
    base = Path(csv_path)
    out_dir = Path(output_dir) if output_dir else base.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "symptom_wilcoxon_summary.csv"
    out_json = out_dir / "symptom_wilcoxon_summary.json"

    summary_df.to_csv(out_csv, index=False)
    with open(out_json, 'w') as f:
        json.dump(summary_df.to_dict(orient='records'), f, indent=2)

    print(f"\nSaved summary CSV: {out_csv}")
    print(f"Saved summary JSON: {out_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Wilcoxon Signed-Rank Test per fairness symptom (two-sided), "
            "using non-zero symptom_difference values."
        )
    )
    parser.add_argument(
        "--csv",
        default="/Users/alfonsocannavale/Documents/Progetti/drama/results/results.csv",
        help="Path to results.csv",
    )
    parser.add_argument(
        "--outdir",
        default="/Users/alfonsocannavale/Documents/Progetti/drama/results",
        help="Directory to save outputs (CSV/JSON)",
    )
    args = parser.parse_args()

    df = load_data(args.csv)
    summary_df = analyze_wilcoxon_by_symptom(df)
    print_summary(summary_df)
    save_outputs(summary_df, args.csv, args.outdir)


if __name__ == "__main__":
    main()