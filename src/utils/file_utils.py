import os
import pandas as pd
from datetime import datetime


def create_feature_directory_structure(base_results_path, feature_name, timestamp):
    """Create dedicated subfolder for each feature."""
    feature_dir = os.path.join(base_results_path, f"feature_{feature_name}_{timestamp}")
    os.makedirs(feature_dir, exist_ok=True)
    return feature_dir


def export_transformed_datasets_hierarchical(transformed_datasets, comprehensive_results):
    """Export transformed datasets with hierarchical structure and comprehensive final dataset."""
    if not transformed_datasets:
        print("No transformed datasets to export.")
        return
        
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
    exported_files = []
    for feature, datasets in feature_groups.items():
        # Create dedicated subfolder for this feature
        feature_dir = create_feature_directory_structure(base_results_path, feature, timestamp)
        
        # Export each transformed dataset for this feature
        for dataset_info in datasets:
            sensitive_attr = dataset_info.get('sensitive_attr', 'none')
            filename = f"transformed_{dataset_info['original_file']}_{dataset_info['column']}_{dataset_info['mutation']}_{sensitive_attr}.csv"
            filepath = os.path.join(feature_dir, filename)
            dataset_info['data'].to_csv(filepath, index=False)
            exported_files.append(filepath)
            print(f"Exported: {filepath}")
    
    # Export comprehensive final dataset
    if comprehensive_results:
        comprehensive_df = pd.DataFrame(comprehensive_results)
        comprehensive_filename = f'../results/comprehensive_analysis_{timestamp}.csv'
        comprehensive_df.to_csv(comprehensive_filename, index=False)
        print(f"\n✓ Comprehensive final dataset exported: {comprehensive_filename}")
    
    print(f"\n✓ All {len(transformed_datasets)} transformed datasets exported in hierarchical structure.")
    print(f"✓ Created {len(feature_groups)} feature-specific directories.")
    return exported_files