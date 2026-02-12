import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests
import glob

# Path Configuration - Robust management of relative paths
# This script is located in drama/src/statistical_analysis/
# The project's base path is two levels above.
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = BASE_DIR / "results"

ALPHA = 0.05

def aggregate_results():
    """Aggregates all CSV files starting with 'results_' in the results folder."""
    print(f"Aggregating results from {RESULTS_DIR}...")
    
    # Find all results_*.csv files
    csv_files = list(RESULTS_DIR.glob("results_*.csv"))
    
    if not csv_files:
        print("No 'results_*.csv' files found in results folder.")
        return None
    
    print(f"Found {len(csv_files)} files: {[f.name for f in csv_files]}")
    
    # Load and concatenate files
    combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    
    # Save the aggregated file
    output_path = RESULTS_DIR / 'results.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"Aggregated results saved to: {output_path}")
    
    return combined_df

def main():
    print("="*80)
    print("   FADE - NORMALIZED FRAGILITY ANALYSIS")
    print("="*80)

    # 1. Data Loading (with Dynamic Aggregation)
    df = aggregate_results()
    
    if df is None:
        # Attempt to load an existing results.csv if aggregation fails (no files found)
        results_csv = RESULTS_DIR / 'results.csv'
        if results_csv.exists():
            print(f"Loading existing {results_csv}...")
            df = pd.read_csv(results_csv)
        else:
            print("Error: No data found to analyze.")
            return

    # Cleaning: Remove cases with infinite percentages or NaN (division by zero)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['symptom_percent_change'])
    
    # 2. Composite Fragility Score (CFS) Calculation
    df['abs_pct_change'] = df['symptom_percent_change'].abs()

    # Group by Experiment (Dataset + Feature + Mutation)
    experiment_fragility = df.groupby(['dataset', 'column', 'mutation_type', 'column_type'])['abs_pct_change'].agg(
        CFS_Mean='mean',      # Composite Fragility Score (Mean)
        CFS_Median='median',  # More robust to single outliers
        Max_Symptom_Change='max'
    ).reset_index()

    # -------------------------------------------------------------------------
    # 3. Statistical Analysis: Identification of "Fairness-Fragile Features"
    # -------------------------------------------------------------------------
    fragile_results = []
    datasets = experiment_fragility['dataset'].unique()
    
    for d_name in datasets:
        d_data = experiment_fragility[experiment_fragility['dataset'] == d_name].copy()
        
        # Define as "Fragile" the top 20% of reactivity for that dataset
        threshold = d_data['CFS_Mean'].quantile(0.80)
        d_data['Is_Fragile'] = d_data['CFS_Mean'] > threshold
        
        for idx, row in d_data.iterrows():
            fragile_results.append({
                'Dataset': d_name,
                'Feature': row['column'],
                'Mutation': row['mutation_type'],
                'CFS_Score': row['CFS_Mean'],
                'Is_High_Risk': row['Is_Fragile']
            })

    final_df = pd.DataFrame(fragile_results)
    
    # -------------------------------------------------------------------------
    # 4. Output and Saving of RQ1-B Results
    # -------------------------------------------------------------------------
    top_fragile = final_df[final_df['Is_High_Risk']].sort_values(by='CFS_Score', ascending=False)
    
    print(f"\n>>> ANALYSIS COMPLETE")
    print(f">>> Identified {len(top_fragile)} High-Risk Fragile Mutation Events.")
    
    print("\n[RQ Answer Data] Top 10 Most Fairness-Fragile Features Identified by FADE:")
    print(top_fragile[['Dataset', 'Feature', 'Mutation', 'CFS_Score']].head(10).to_string(index=False))

    # Saving to RESULTS_DIR
    analysis_output = RESULTS_DIR / 'fade_analysis.csv'
    final_df.to_csv(analysis_output, index=False)
    print(f"\n>>> Saved '{analysis_output}'")

    # -------------------------------------------------------------------------
    # 5. Symptom Sensitivity Analysis (RQ1 Part A)
    # -------------------------------------------------------------------------
    print("\n[Analysis] Running Symptom Sensitivity Analysis (Wilcoxon + Effect Size)...")
    
    def calculate_rank_biserial(pre, post):
        diffs = np.array(post) - np.array(pre)
        diffs = diffs[diffs != 0]
        if len(diffs) == 0: return 0.0
        n = len(diffs)
        ranks = rankdata(np.abs(diffs))
        pos_sum = np.sum(ranks[diffs > 0])
        neg_sum = np.sum(ranks[diffs < 0])
        return (pos_sum - neg_sum) / (n * (n + 1) / 2)

    symptom_stats = []
    unique_symptoms = df['symptom_name'].unique()

    for sym in unique_symptoms:
        sub_df = df[df['symptom_name'] == sym].copy()
        pre = sub_df['pre_symptom_value'].values
        post = sub_df['post_symptom_value'].values
        
        if np.any(pre != post):
            w_stat, p_val = wilcoxon(pre, post)
            effect_size = calculate_rank_biserial(pre, post)
            diffs = post - pre
            pos_count = np.sum(diffs > 1e-9)
            neg_count = np.sum(diffs < -1e-9)
            
            if pos_count > neg_count * 1.5: trend = "Increase"
            elif neg_count > pos_count * 1.5: trend = "Decrease"
            else: trend = "Mixed"
        else:
            p_val, effect_size, trend = 1.0, 0.0, "Stable"

        symptom_stats.append({
            'Symptom': sym,
            'Wilcoxon_p': p_val,
            'Effect_Size_r': effect_size,
            'Trend': trend,
            'Non_Zero_Count': np.sum(pre != post)
        })

    sym_df = pd.DataFrame(symptom_stats)
    
    if not sym_df.empty:
        reject, p_adj, _, _ = multipletests(sym_df['Wilcoxon_p'], alpha=0.05, method='fdr_bh')
        sym_df['p_adj'] = p_adj
        sym_df['Significant'] = p_adj < 0.05
    
    sym_df = sym_df.sort_values(by='p_adj')

    # Saving to RESULTS_DIR
    symptom_output = RESULTS_DIR / 'symptom_sensitivity_results.csv'
    sym_df.to_csv(symptom_output, index=False)
    print(f">>> Saved '{symptom_output}'")
    
    print("\nTop 10 Symptom Sensitivity Results:")
    print(sym_df[['Symptom', 'p_adj', 'Effect_Size_r', 'Trend']].head(10).to_string())
    print("-" * 80)

if __name__ == "__main__":
    main()