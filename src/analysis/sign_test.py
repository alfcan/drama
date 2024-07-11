from scipy.stats import binomtest
import pandas as pd

# Load the dataset
file_path = "../../results/results.csv"
df = pd.read_csv(file_path)

# List of symptoms
symptoms = ["APD", "Gini Index", "Shannon Entropy", "Simpson Diversity", "Imbalance Ratio", "Kurtosis", "Skewness", "Mutual Information", "Normalized Mutual Information", "Kendall Tau", "Correlation Ratio"]

# Perform Sign Test for each symptom
results_sign = {}
for symptom in symptoms:
    pre_values = df[f'pre_{symptom}']
    post_values = df[f'post_{symptom}']
    
    # Calculate the number of positive, negative, and zero differences
    differences = post_values - pre_values
    num_positive = sum(differences > 0)
    num_negative = sum(differences < 0)
    
    if num_positive + num_negative > 0:
        # Perform the Sign Test
        stat = min(num_positive, num_negative)
        p_value = binomtest(stat, n=num_positive + num_negative, p=0.5, alternative='two-sided').pvalue
    else:
        stat = None
        p_value = None
    
    # Store the results
    results_sign[symptom] = {'num_positive': num_positive, 'num_negative': num_negative, 'stat': stat, 'p_value': p_value}

# Display the results
for symptom, result in results_sign.items():
    print(f"Sign Test for {symptom}: num_positive={result['num_positive']}, num_negative={result['num_negative']}, stat={result['stat']}, p-value={result['p_value']}")
    if result['p_value'] and result['p_value'] < 0.05:
        print(f"There is a significant difference for {symptom}")
    else:
        print(f"There is no significant difference for {symptom}")