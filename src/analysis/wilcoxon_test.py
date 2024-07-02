from scipy.stats import wilcoxon
import pandas as pd

# Load the dataset
file_path = "../../results/results_20240702_095048.csv"
df = pd.read_csv(file_path)

# List of symptoms
symptoms = ["APD", "Gini Index", "Shannon Entropy", "Simpson Diversity", "Imbalance Ratio", "Kurtosis", "Skewness", "Mutual Information", "Normalized Mutual Information", "Kendall Tau", "Correlation Ratio"]

# Perform Wilcoxon Signed-Rank Test for each symptom
results_wilcoxon = {}
for symptom in symptoms:
    pre_values = df[f'pre_{symptom}']
    post_values = df[f'post_{symptom}']
    
    # Remove pairs where differences are zero
    non_zero_diff = (pre_values != post_values)
    pre_values = pre_values[non_zero_diff]
    post_values = post_values[non_zero_diff]
    
    if len(pre_values) > 0:
        # Perform the Wilcoxon Signed-Rank Test
        stat, p_value = wilcoxon(pre_values, post_values)
        # Store the results
        results_wilcoxon[symptom] = {'stat': stat, 'p_value': p_value}
    else:
        results_wilcoxon[symptom] = {'stat': None, 'p_value': None}

# Display the results
for symptom, result in results_wilcoxon.items():
    print(f"Wilcoxon Signed-Rank Test for {symptom}: Statistics={result['stat']}, p-value={result['p_value']}")
    if result['p_value'] and result['p_value'] < 0.05:
        print(f"There is a significant difference for {symptom}")
    else:
        print(f"There is no significant difference for {symptom}")