import numpy as np
import pandas as pd
from data.loader import load_dataset
from analysis.symptom_calculator import SymptomCalculator
from mutation.mutation_operator import MutationOperator
from datetime import datetime

# Function to check if a string contains a single word
def is_single_word(text):
    return len(text.split()) == 1

def store_results(column, column_type, mutation_type, is_sensitive, pre_symptoms, post_symptoms):
    pre_symptoms_prefixed = {f'pre_{key}': value for key, value in pre_symptoms.items()}
    post_symptoms_prefixed = {f'post_{key}': value for key, value in post_symptoms.items()}

    results.append({
        'column': column,
        'column_type': column_type, # 'text' or 'numeric
        'mutation_type': mutation_type,
        'is_sensitive': is_sensitive,
        **pre_symptoms_prefixed,
        **post_symptoms_prefixed
    })

def export_results_to_csv():
    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'../results/results_{timestamp}.csv'
    df_results.to_csv(filename, index=False)
    print(f'Results exported successfully to {filename}')

if __name__ == '__main__':
    
    # Load the dataset
    DATA_PATH = '../data/'
    file_path = input("Enter the name of the CSV data set to be analyzed (placed in the data folder): ")
    try:
        df = load_dataset(DATA_PATH+file_path)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        exit(1)

    # Display the columns of the dataset
    print("Columns in the dataset:")
    for col in df.columns:
        print(col)
    # Ask the user to specify the sensitive attributes and the target attribute
    sensitive_attributes = input("Enter the names of the sensitive attributes separated by commas: ").replace(" ", "").split(',')
    target_attribute = input("Enter the name of the target attribute: ")

    # Check that the attributes exist in the dataset
    for attr in sensitive_attributes:
        if attr not in df.columns:
            raise ValueError(f"The sensitive attribute {attr} does not exist in the dataset.")
            exit(1)
    if target_attribute not in df.columns:
        raise ValueError(f"The target attribute {target_attribute} does not exist in the dataset.")
        exit(1)

    results = []

    # Calculate the symptoms for each sensitive attribute
    for sensitive_attribute in sensitive_attributes:
        print(f"\n\n\nCalculating symptoms for the sensitive attribute: {sensitive_attribute}")
        privileged_condition = input(f"Enter the condition for the privileged group for {sensitive_attribute} (e.g., '{sensitive_attribute} >= 30' or '{sensitive_attribute} == \"White\"' or '{sensitive_attribute} < 20 or {sensitive_attribute} > 40'): ")
        unprivileged_condition = input(f"Enter the condition for the unprivileged group for {sensitive_attribute} (e.g., '{sensitive_attribute} < 30' or '{sensitive_attribute} != \"White\"'): ")

        # Create the symptom calculator instance
        symptom_calculator = SymptomCalculator(df, sensitive_attribute, target_attribute)

        # Calculate symptoms
        symptoms = symptom_calculator.calculate_symptoms(privileged_condition, unprivileged_condition)
        for symptom, value in symptoms.items():
            print(f"{symptom}: {value}")

        print("\n\n\nSymptoms analysis:")
        if symptoms['APD'] and symptoms['APD'] > 0.1:
            print(f"APD is high: {symptoms['APD']}, indicating potential bias.")
        if symptoms['Gini Index'] < 0.5:
            print(f"Gini Index is low: {symptoms['Gini Index']}, indicating potential bias.")
        if symptoms['Shannon Entropy'] < 0.5:
            print(f"Shannon Entropy is low: {symptoms['Shannon Entropy']}, indicating potential bias.")
        if symptoms['Simpson Diversity'] < 0.5:
            print(f"Simpson Diversity is low: {symptoms['Simpson Diversity']}, indicating potential bias.")
        if symptoms['Imbalance Ratio'] > 1.5:
            print(f"Imbalance Ratio is high: {symptoms['Imbalance Ratio']}, indicating potential bias.")
        if abs(symptoms['Kurtosis']) > 3:
            print(f"Kurtosis is high: {symptoms['Kurtosis']}, indicating potential bias.")
        if abs(symptoms['Skewness']) > 1:
            print(f"Skewness is high: {symptoms['Skewness']}, indicating potential bias.")
        if symptoms['Mutual Information'] > 0.5:
            print(f"Mutual Information is high: {symptoms['Mutual Information']}, indicating potential bias.")
        if symptoms['Normalized Mutual Information'] > 0.5:
            print(f"Normalized Mutual Information is high: {symptoms['Normalized Mutual Information']}, indicating potential bias.")
        if abs(symptoms['Kendall Tau']) > 0.5:
            print(f"Kendall Tau is high: {symptoms['Kendall Tau']}, indicating potential bias.")
        if symptoms['Correlation Ratio'] > 0.5:
            print(f"Correlation Ratio is high: {symptoms['Correlation Ratio']}, indicating potential bias.")

        print("\n\n\nBias detection:")
        bias_detection = symptom_calculator.detect_bias_symptoms(privileged_condition, unprivileged_condition)
        for symptom, flag in bias_detection.items():
            if flag:
                print(f"{symptom} indicates potential bias.")
        
        mutation_operator = MutationOperator(df)

        for col in df.columns:
            if col != target_attribute:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Apply a numeric mutation operator randomly
                    operators = ['increment_decrement_feature', 'swap_values', 'scale_values', 'discrete_binning']
                else:
                    # Check if all rows in the column contain a single word
                    if all(is_single_word(text) for text in df[col]):
                        # Exclude augment_text if all rows have a single word
                        operators = ['replace_synonyms', 'add_noise', 'random_category_assignment', 'swap_values']
                    else:
                        # Include augment_text among the available text-based operators and exclude random_category_assignment
                        operators = ['augment_text', 'replace_synonyms', 'add_noise', 'swap_values']
            
                for operator in operators:
                    print(f"\n\n\nApplying {operator} to column {col}.")
                    # Call the mutation operator method dynamically
                    if operator == 'increment_decrement_feature':
                        df_new = mutation_operator.increment_decrement_feature(col, increment=np.random.choice([True, False]), amount=np.random.uniform(1, 10), percentage=30)
                    elif operator == 'swap_values':
                        df_new = mutation_operator.swap_values(col, num_swaps=int(df.shape[0] * 0.1))
                    elif operator == 'scale_values':
                        df_new = mutation_operator.scale_values(col, scale_factor=np.random.uniform(0.5, 1.5), percentage=30)
                    elif operator == 'discrete_binning':
                        bins = np.linspace(df[col].min(), df[col].max(), num=4) # Split the range into 4 bins
                        df_new = mutation_operator.discrete_binning(col, bins=bins)
                    elif operator == 'random_category_assignment':
                        df_new = mutation_operator.random_category_assignment(col, percentage=30)
                    elif operator == 'augment_text':
                        df_new = mutation_operator.augment_text(col, percentage=20)
                    elif operator == 'replace_synonyms':
                        df_new = mutation_operator.replace_synonyms(col, row_percentage=20, word_percentage=20)
                    elif operator == 'add_noise':
                        df_new = mutation_operator.add_noise(col, noise_chance=0.1)
                    print("Mutation applied successfully.")

                    # Recalculating symptoms after mutation
                    symptom_calculator = SymptomCalculator(df_new, sensitive_attribute, target_attribute)
                    new_symptoms = symptom_calculator.calculate_symptoms(privileged_condition, unprivileged_condition)

                    column_type = 'numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'text'
                    store_results(col, column_type, operator, col in sensitive_attributes, symptoms, new_symptoms)

                    # Compare pre and post mutation symptoms
                    print(f"\nComparing symptoms for {col}:")
                    for symptom, value in new_symptoms.items():
                        old_value = symptoms.get(symptom, None)
                        if old_value:
                            change = value - old_value
                            if change != 0:
                                print(f"{symptom} changed by {change}")
                            else:
                                print(f"{symptom} remained the same")
            
    export_results_to_csv()      