import numpy as np
import pandas as pd
import os
from data.loader import load_dataset
from data.preprocessor import DataPreprocessor
from analysis.symptom_calculator import SymptomCalculator
from mutation.mutation_operator import MutationOperator
from datetime import datetime
from utils.result_handlers import export_streamlined_results
from utils.file_utils import create_feature_directory_structure
from utils.analysis_helpers import get_applicable_operators, get_user_defined_conditions


if __name__ == '__main__':
    
    # Initialize preprocessor with only required preprocessing steps
    preprocessor = DataPreprocessor()
    
    # Load and preprocess dataset
    DATA_PATH = '../data/'
    file_path = input("Enter the name of the CSV data set to be analyzed (placed in the data folder): ")
    
    try:
        df_raw = load_dataset(DATA_PATH+file_path)
        print(f"Loaded dataset with {len(df_raw)} rows and {len(df_raw.columns)} columns")
    except (FileNotFoundError, ValueError) as e:
        print(e)
        exit(1)
    
    # Apply only missing value removal (keep raw format for mutations)
    df_cleaned, preprocessing_info = preprocessor.clean_data_only(df_raw)
    print(f"After cleaning (missing values removed): {len(df_cleaned)} rows and {len(df_cleaned.columns)} columns")
    
    # Store the cleaned raw dataset for mutations
    df_raw_cleaned = df_cleaned.copy()

    # Display the columns of the cleaned raw dataset
    print("Columns in the dataset:")
    for col in df_cleaned.columns:
        print(col)
    # Ask the user to specify the sensitive attributes and the target attribute
    sensitive_attributes = input("Enter the names of the sensitive attributes separated by commas: ").replace(" ", "").split(',')
    target_attribute = input("Enter the name of the target attribute: ")

    # Check that the attributes exist in the dataset
    for attr in sensitive_attributes:
        if attr not in df_cleaned.columns:
            raise ValueError(f"The sensitive attribute {attr} does not exist in the dataset.")
    if target_attribute not in df_cleaned.columns:
        raise ValueError(f"The target attribute {target_attribute} does not exist in the dataset.")

    # Start streamlined analysis
    print("\n" + "="*80)
    print("FAIRNESS ROBUSTNESS ANALYSIS FRAMEWORK")
    print("Sequential Feature Analysis - All operators applied to all features")
    print("="*80)
    
    results = []
    transformed_datasets = []
    
    # Initialize mutation operator
    mutation_operator = MutationOperator(df_raw_cleaned)
    
    # Get all features to analyze (exclude target attribute)
    all_features = [col for col in df_raw_cleaned.columns if col != target_attribute]
    
    print(f"\nAnalyzing {len(all_features)} features with all applicable operators...")
    
    # Calculate baseline symptoms for each sensitive attribute
    baseline_symptoms = {}
    for sensitive_attribute in sensitive_attributes:
        print(f"\nCalculating baseline symptoms for sensitive attribute: {sensitive_attribute}")
        
        # Apply temporary encoding for symptom calculation
        df_encoded_for_symptoms = preprocessor.apply_temporary_encoding(df_raw_cleaned)
        symptom_calculator = SymptomCalculator(df_encoded_for_symptoms, sensitive_attribute, target_attribute)
        
        # Get user-defined privileged and unprivileged conditions
        privileged_condition, unprivileged_condition = get_user_defined_conditions(sensitive_attribute, df_encoded_for_symptoms)
        pre_symptoms = symptom_calculator.calculate_symptoms(privileged_condition, unprivileged_condition)
        
        baseline_symptoms[sensitive_attribute] = {
            'symptoms': pre_symptoms,
            'privileged_condition': privileged_condition,
            'unprivileged_condition': unprivileged_condition
        }
    
    # Sequential processing: iterate through each feature (excluding sensitive attributes)
    all_features = [col for col in df_raw_cleaned.columns if col != target_attribute and col not in sensitive_attributes]
    
    print(f"\nAnalyzing {len(all_features)} non-sensitive features with all applicable operators...")
    print(f"Excluded from mutations: {len(sensitive_attributes)} sensitive attribute(s) + 1 target attribute")
    print(f"Sensitive attributes excluded: {', '.join(sensitive_attributes)}")
    print(f"Target attribute excluded: {target_attribute}")

    for feature in all_features:
        print(f"\n{'='*60}")
        print(f"Processing feature: {feature}")
        print(f"{'='*60}")
        
        # Since we've already filtered out sensitive attributes, this feature is non-sensitive
        is_sensitive = False
        
        # Get applicable operators for this feature
        operators = get_applicable_operators(feature, df_raw_cleaned)
        print(f"Applicable operators: {operators}")
        
        # Apply each operator to this feature
        for operator in operators:
            print(f"\nApplying {operator} to {feature}...")
            
            try:
                # Apply mutation based on operator type
                if operator == 'increment_decrement_feature':
                    df_mutated = mutation_operator.increment_decrement_feature(feature, percentage=20)
                elif operator == 'scale_values':
                    df_mutated = mutation_operator.scale_values(feature, percentage=20)
                elif operator == 'category_flip':
                    df_mutated = mutation_operator.category_flip(feature)
                elif operator == 'replace_synonyms':
                    df_mutated = mutation_operator.replace_synonyms(feature, row_percentage=10, word_percentage=15)
                elif operator == 'add_noise':
                    df_mutated = mutation_operator.add_noise(feature, percentage=10)
                
                print(f"Mutation {operator} applied successfully to {feature}")
                
                # Apply temporary encoding once for this mutated dataset
                df_mutated_encoded = preprocessor.apply_temporary_encoding(df_mutated)
                
                # Store the unique transformed dataset (only once per feature-operator combination)
                column_type = 'numeric' if pd.api.types.is_numeric_dtype(df_raw_cleaned[feature]) else 'categorical'
                transformed_datasets.append({
                    'original_file': file_path.replace('.csv', ''),
                    'column': feature,
                    'mutation': operator,
                    'column_type': column_type,
                    'is_sensitive': is_sensitive,
                    'data': df_mutated.copy()
                })
                
                # Calculate post-mutation symptoms for ALL sensitive attributes
                for sensitive_attribute in sensitive_attributes:
                    baseline = baseline_symptoms[sensitive_attribute]
                    
                    # Create symptom calculator for this sensitive attribute
                    symptom_calculator = SymptomCalculator(df_mutated_encoded, sensitive_attribute, target_attribute)
                    
                    post_symptoms = symptom_calculator.calculate_symptoms(
                        baseline['privileged_condition'], 
                        baseline['unprivileged_condition']
                    )
                    
                    # Store results for this sensitive attribute
                    result_entry = {
                        'column': feature,
                        'dataset': file_path,
                        'column_type': column_type,
                        'mutation_type': operator,
                        'is_sensitive': is_sensitive,
                        'sensitive_attr_analyzed': sensitive_attribute,
                        **{f'pre_{key}': value for key, value in baseline['symptoms'].items()},
                        **{f'post_{key}': value for key, value in post_symptoms.items()}
                    }
                    results.append(result_entry)
                    
                    # Report changes
                    for symptom, value in post_symptoms.items():
                        old_value = baseline['symptoms'].get(symptom, 0)
                        change = value - old_value
                        if abs(change) > 0.01:  # Only report significant changes
                            print(f"  {symptom}: {old_value:.3f} → {value:.3f} (Δ{change:+.3f})")
                            if abs(change) > 0.1:
                                print(f"    ⚠️  SIGNIFICANT CHANGE detected!")

            except Exception as e:
                print(f"Error applying {operator} to {feature}: {e}")
                continue
    
    # Summary report
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*80)
    print(f"✓ Sequential Feature Analysis completed")
    print(f"  - Processed {len(all_features)} features")
    print(f"  - Applied all applicable operators to each feature")
    print(f"  - Analyzed impact on {len(sensitive_attributes)} sensitive attribute(s)")
    
    print(f"\nTotal mutations applied: {len(transformed_datasets)}")
    print(f"Results stored: {len(results)} entries")

            
    # Export results and transformed datasets
    if results:
        # Export using new streamlined format only
        export_streamlined_results(results, file_path)
        
        # Keep hierarchical transformed datasets export
        if transformed_datasets:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_results_path = '../results'
            
            # Group datasets by feature for hierarchical organization
            feature_groups = {}
            for dataset_info in transformed_datasets:
                feature = dataset_info['column']
                if feature not in feature_groups:
                    feature_groups[feature] = []
                feature_groups[feature].append(dataset_info)
            
            # Create hierarchical structure and export datasets
            for feature, datasets in feature_groups.items():
                # Create dedicated subfolder for this feature
                feature_dir = create_feature_directory_structure(base_results_path, feature, timestamp)
                
                # Export each transformed dataset for this feature
                for dataset_info in datasets:
                    # Create filename without sensitive_attr since each dataset is unique per feature-operator
                    filename = f"transformed_{dataset_info['original_file']}_{dataset_info['column']}_{dataset_info['mutation']}.csv"
                    filepath = os.path.join(feature_dir, filename)
                    dataset_info['data'].to_csv(filepath, index=False)
                    print(f"Exported: {filepath}")
            
            print(f"\n✓ All {len(transformed_datasets)} transformed datasets exported in hierarchical structure.")
            print(f"✓ Created {len(feature_groups)} feature-specific directories.")

        print("\n✓ Results exported to CSV")
    
    print("\n" + "="*80)
    print("FRAMEWORK EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Check the results/ directory for:")
    print(f"  - Analysis results CSV files")
    print(f"  - Transformed datasets ({len(transformed_datasets)} files)")
    print("\nThank you for using the Fairness Robustness Analysis Framework!")
    print("="*80)