import numpy as np
import pandas as pd
from data.loader import load_dataset
from analysis.symptom_calculator import SymptomCalculator
from mutation.mutation_operator import MutationOperator
from datetime import datetime


def store_results(column, dataset, column_type, mutation_type, is_sensitive, pre_symptoms, post_symptoms):
    pre_symptoms_prefixed = {f'pre_{key}': value for key, value in pre_symptoms.items()}
    post_symptoms_prefixed = {f'post_{key}': value for key, value in post_symptoms.items()}

    results.append({
        'column': column,
        'dataset': dataset,
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
        privileged_condition = privileged_condition.replace("\n", " ").strip()
        unprivileged_condition = unprivileged_condition.replace("\n", " ").strip()

        # Create the symptom calculator instance
        symptom_calculator = SymptomCalculator(df, sensitive_attribute, target_attribute)

        # Calculate symptoms
        pre_symptoms = symptom_calculator.calculate_symptoms(privileged_condition, unprivileged_condition)
        for symptom, value in pre_symptoms.items():
            print(f"{symptom}: {value}")

        print("\n\n\nPRE - Bias detection:")
        pre_bias_detection = symptom_calculator.detect_bias_symptoms(pre_symptoms, privileged_condition, unprivileged_condition)
        for symptom, flag in pre_bias_detection.items():
            if flag:
                print(f"{symptom} indicates potential bias.")
        
        mutation_operator = MutationOperator(df)

        for col in df.columns:
            if col != target_attribute:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Apply a numeric mutation operator randomly
                    operators = ['increment_decrement_feature', 'swap_values', 'scale_values', 'discrete_binning']
                else:
                    operators = ['augment_text', 'replace_synonyms', 'add_noise', 'random_category_assignment', 'swap_values']
                
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
                        df_new = mutation_operator.random_category_assignment(col, percentage=20)
                    elif operator == 'augment_text':
                        df_new = mutation_operator.augment_text(col, percentage=10)
                    elif operator == 'replace_synonyms':
                        df_new = mutation_operator.replace_synonyms(col, row_percentage=20, word_percentage=20)
                    elif operator == 'add_noise':
                        df_new = mutation_operator.add_noise(col, noise_chance=0.1)
                    print("Mutation applied successfully.")

                    # Recalculating symptoms after mutation
                    symptom_calculator = SymptomCalculator(df_new, sensitive_attribute, target_attribute)
                    post_symptoms = symptom_calculator.calculate_symptoms(privileged_condition, unprivileged_condition)

                    column_type = 'numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'text'
                    store_results(col, file_path, column_type, operator, col in sensitive_attributes, pre_symptoms, post_symptoms)

                    # Compare pre and post mutation symptoms
                    print(f"\nComparing symptoms for {col}:")
                    for symptom, value in post_symptoms.items():
                        old_value = pre_symptoms.get(symptom, None)
                        if old_value:
                            change = value - old_value
                            if change != 0:
                                print(f"{symptom} changed by {change}")
                            else:
                                print(f"{symptom} remained the same")

                    print("\n\n\nPOST - Bias detection:")
                    post_bias_detection = symptom_calculator.detect_bias_symptoms(post_symptoms, privileged_condition, unprivileged_condition)
                    for symptom, flag in post_bias_detection.items():
                        if flag:
                            print(f"{symptom} indicates potential bias.")
                    
                    for key in pre_bias_detection.keys():
                        if pre_bias_detection[key] != post_bias_detection[key]:
                            if pre_bias_detection[key] == 1 and post_bias_detection[key] == 0:
                                print(f"The symptom {key} is no longer present with the application of the mutation.")
                            elif pre_bias_detection[key] == 0 and post_bias_detection[key] == 1:
                                print(f"The symptom {key} appeared with the application of the mutation.")

            
    export_results_to_csv()      