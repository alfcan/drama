import numpy as np
import pandas as pd
import os
from datetime import datetime

# --- Imports for framework components ---
from data.loader import load_dataset
from data.preprocessor import DataPreprocessor
from analysis.symptom_calculator import SymptomCalculator
from mutation.mutation_operator import MutationOperator
from utils.result_handlers import export_streamlined_results
from utils.file_utils import create_feature_directory_structure
from utils.analysis_helpers import get_applicable_operators


if __name__ == '__main__':
    
    # Initialize preprocessor with only required preprocessing steps
    preprocessor = DataPreprocessor()
    
    # --- 1. Load and Preprocess Dataset ---
    DATA_PATH = '../data/'
    file_path_input = input("Enter the name of the CSV data set to be analyzed (placed in the data folder): ")
    
    try:
        df_raw = load_dataset(DATA_PATH + file_path_input)
        print(f"Loaded dataset with {len(df_raw)} rows and {len(df_raw.columns)} columns")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading dataset: {e}")
        exit(1)
    
    # Apply only missing value removal (keep raw format for mutations and other processing)
    df_cleaned, preprocessing_info = preprocessor.clean_data_only(df_raw)
    print(f"After cleaning (missing values removed): {len(df_cleaned)} rows and {len(df_cleaned.columns)} columns")
    
    # Store the cleaned raw dataset for mutations
    df_raw_cleaned = df_cleaned.copy()

    # --- 2. User Input for Target Attribute and Positive Label ---
    print("\nColumns in the dataset:")
    for col in df_cleaned.columns:
        print(f"- {col}")
        
    target_attribute = input("\nEnter the name of the target attribute (outcome variable): ")

    if target_attribute not in df_cleaned.columns:
        raise ValueError(f"The target attribute '{target_attribute}' does not exist in the dataset.")

    try:
        positive_label = int(input(f"Enter the positive label value for '{target_attribute}' (e.g., 1 for binary classification): "))
    except ValueError:
        print("Invalid input for positive label. Using default: 1")
        positive_label = 1 # Default value

    # --- 3. Identify All Features for Analysis ---
    # All features except the target will be treated as potential sensitive attributes for symptom calculation.
    all_features_for_symptom_analysis = [col for col in df_raw_cleaned.columns if col != target_attribute]
    # All features except the target will also be considered for mutations.
    all_features_for_mutation = [col for col in df_raw_cleaned.columns if col != target_attribute]

    print("\n" + "="*80)
    print("FAIRNESS ROBUSTNESS ANALYSIS FRAMEWORK")
    print("Sequential Feature Analysis - All applicable operators applied to all features")
    print("="*80)
    
    results = [] # Stores all analysis results
    transformed_datasets = [] # Stores metadata and actual transformed datasets

    # Initialize mutation operator with the clean base dataset
    mutation_operator = MutationOperator(df_raw_cleaned)
    
    # --- 4. Calculate Baseline Symptoms for All Potential Sensitive Features ---
    baseline_symptoms_by_feature = {}
    
    # Apply temporary encoding for baseline symptom calculation
    df_encoded_for_baseline = preprocessor.apply_temporary_encoding(df_raw_cleaned)
    baseline_symptom_calculator = SymptomCalculator(df_encoded_for_baseline, target_attribute, positive_label, df_raw_cleaned)

    print(f"\nCalculating baseline symptoms for {len(all_features_for_symptom_analysis)} potential sensitive attributes...")
    for feature in all_features_for_symptom_analysis:
        print(f"  - Processing baseline for feature: '{feature}'")
        pre_symptoms = baseline_symptom_calculator.calculate_symptoms_for_feature(feature)
        baseline_symptoms_by_feature[feature] = pre_symptoms
    
    # --- 5. Sequential Processing: Mutate Each Feature and Analyze Impact ---
    print(f"\nInitiating mutation analysis on {len(all_features_for_mutation)} features...")
    print(f"Target attribute '{target_attribute}' is excluded from mutations.")

    for feature_to_mutate in all_features_for_mutation:
        print(f"\n{'='*60}")
        print(f"Processing feature for mutation: '{feature_to_mutate}'")
        print(f"{'='*60}")
        
        column_type = 'numeric' if pd.api.types.is_numeric_dtype(df_raw_cleaned[feature_to_mutate]) else 'categorical'
        
        # Get applicable operators for the feature currently being mutated
        operators = get_applicable_operators(feature_to_mutate, df_raw_cleaned)
        print(f"Applicable operators for '{feature_to_mutate}': {operators}")
        
        for operator in operators:
            print(f"\n  Applying {operator} to '{feature_to_mutate}'...")
            
            try:
                # Apply mutation based on operator type
                df_mutated = df_raw_cleaned.copy() # Start with a fresh copy of the clean data for each mutation
                if operator == 'increment_decrement_feature':
                    df_mutated = mutation_operator.increment_decrement_feature(feature_to_mutate, percentage=20)
                elif operator == 'scale_values':
                    df_mutated = mutation_operator.scale_values(feature_to_mutate, percentage=20)
                elif operator == 'category_flip':
                    df_mutated = mutation_operator.category_flip(feature_to_mutate)
                elif operator == 'replace_synonyms':
                    # Assuming replace_synonyms works on a feature directly or infers text columns
                    df_mutated = mutation_operator.replace_synonyms(feature_to_mutate, row_percentage=10, word_percentage=15)
                elif operator == 'add_noise':
                    df_mutated = mutation_operator.add_noise(feature_to_mutate, percentage=10)
                else:
                    print(f"    Skipping unknown operator: {operator}")
                    continue
                
                print(f"    Mutation '{operator}' applied successfully to '{feature_to_mutate}'.")
                
                # Store the transformed dataset metadata and data itself
                transformed_datasets.append({
                    'original_file': file_path_input.replace('.csv', ''),
                    'column': feature_to_mutate,
                    'mutation': operator,
                    'column_type': column_type,
                    'is_sensitive': False,
                    'data': df_mutated.copy()
                })
                
                # Calculate post-mutation symptoms for ALL potential sensitive features on the MUTATED dataset
                # Apply temporary encoding for symptom calculation on mutated data
                # --- 3. Apply Temporary Encoding and Calculate Post-Mutation Symptoms ---
                df_mutated_encoded = preprocessor.apply_temporary_encoding(df_mutated)
                mutated_symptom_calculator = SymptomCalculator(df_mutated_encoded, target_attribute, positive_label, df_mutated)

                print(f"\n  Analyzing impact on {len(all_features_for_symptom_analysis)} potential sensitive attributes:")
                for sensitive_feature_for_symptoms in all_features_for_symptom_analysis:
                    baseline_symptoms = baseline_symptoms_by_feature[sensitive_feature_for_symptoms]
                    
                    # Calculate post-mutation symptoms for this sensitive feature using the mutated data
                    post_symptoms = mutated_symptom_calculator.calculate_symptoms_for_feature(sensitive_feature_for_symptoms)

                    # Store results for this specific combination of (mutated feature, operator, sensitive feature analyzed)
                    result_entry = {
                        'mutation_source_feature': feature_to_mutate, # The feature that was mutated
                        'dataset': file_path_input,
                        'mutation_source_feature_type': column_type,
                        'mutation_type': operator,
                        'sensitive_attr_analyzed': sensitive_feature_for_symptoms, # The feature whose symptoms are being analyzed
                        'sensitive_attr_type': 'numeric' if pd.api.types.is_numeric_dtype(df_raw_cleaned[sensitive_feature_for_symptoms]) else 'categorical',
                        **{f'pre_{key}': value for key, value in baseline_symptoms.items()}, # Baseline symptoms
                        **{f'post_{key}': value for key, value in post_symptoms.items()} # Post-mutation symptoms
                    }
                    results.append(result_entry)
                    
                    # --- Report Changes ---
                    print(f"    Impact on sensitive feature '{sensitive_feature_for_symptoms}':")
                    for symptom_name, post_value in post_symptoms.items():
                        pre_value = baseline_symptoms.get(symptom_name, None) # Get baseline value, or None if missing
                        
                        # Only report if both values exist and change is significant
                        if pre_value is not None and post_value is not None:
                            change = post_value - pre_value
                            if abs(change) > 0.01:  # Threshold for reporting
                                print(f"      - {symptom_name}: {pre_value:.3f} -> {post_value:.3f} (Δ{change:+.3f})")
                                if abs(change) > 0.1: # Threshold for significant alert
                                    print(f"        ⚠️  SIGNIFICANT CHANGE detected!")
                        elif pre_value is None and post_value is not None:
                            # Symptom was not applicable at baseline, but is now (or vice versa)
                            print(f"      - {symptom_name}: (N/A) -> {post_value:.3f}")
                        elif pre_value is not None and post_value is None:
                            print(f"      - {symptom_name}: {pre_value:.3f} -> (N/A)")

            except Exception as e:
                print(f"    Error applying {operator} to '{feature_to_mutate}': {e}")
                continue
    
    # --- 6. Summary Report and Export Results ---
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*80)
    print(f"✓ Sequential Feature Analysis completed.")
    print(f"  - {len(all_features_for_mutation)} features were mutated.")
    print(f"  - Impact was analyzed on {len(all_features_for_symptom_analysis)} potential sensitive attributes for each mutation.")
    
    print(f"\nTotal mutations applied: {len(transformed_datasets)}")
    print(f"Total result entries stored: {len(results)}")

    if results:
        # Export using your streamlined format (function from utils.result_handlers)
        export_streamlined_results(results, file_path_input)
        
        # Export transformed datasets in a hierarchical structure
        if transformed_datasets:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_results_path = '../results'
            
            # Group datasets by the feature that was mutated
            feature_groups = {}
            for dataset_info in transformed_datasets:
                feature_name = dataset_info['column'] # This is the feature_to_mutate
                if feature_name not in feature_groups:
                    feature_groups[feature_name] = []
                feature_groups[feature_name].append(dataset_info)
            
            # Create hierarchical structure and export each dataset
            for feature_name, datasets_list in feature_groups.items():
                feature_dir = create_feature_directory_structure(base_results_path, feature_name, timestamp)
                
                for dataset_info in datasets_list:
                    filename = f"transformed_{dataset_info['original_file']}_{dataset_info['column']}_{dataset_info['mutation']}.csv"
                    filepath = os.path.join(feature_dir, filename)
                    dataset_info['data'].to_csv(filepath, index=False)
                    print(f"  Exported: {filepath}")
            
            print(f"\n✓ All {len(transformed_datasets)} transformed datasets exported in hierarchical structure.")
            print(f"✓ Created {len(feature_groups)} feature-specific directories.")

        print("\n✓ Analysis results exported to CSV.")
    else:
        print("\nNo results to export (analysis might have failed or found no applicable operations).")
    
    print("\n" + "="*80)
    print("FRAMEWORK EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Check the '../results/' directory for:")
    print(f"  - Analysis results CSV file(s).")
    print(f"  - Transformed datasets (in feature-specific subfolders).")
    print("\nThank you for using the Fairness Robustness Analysis Framework!")
    print("="*80)