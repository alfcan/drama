import pandas as pd
from datetime import datetime


def export_streamlined_results(results, file_path):
    """Export results using the new streamlined symptom-based CSV format."""
    if not results:
        print("No results to export.")
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Transform results to new streamlined format
    streamlined_results = []
    
    for result in results:
        # Extract pre and post symptoms
        pre_symptoms = {k.replace('pre_', ''): v for k, v in result.items() if k.startswith('pre_')}
        post_symptoms = {k.replace('post_', ''): v for k, v in result.items() if k.startswith('post_')}
        
        # Create one row per symptom
        for symptom_name in pre_symptoms.keys():
            if symptom_name in post_symptoms:
                pre_value = pre_symptoms[symptom_name]
                post_value = post_symptoms[symptom_name]
                
                # Skip if either value is None
                if pre_value is None or post_value is None:
                    continue
                
                streamlined_entry = {
                    'column': result['mutation_source_feature'],
                    'dataset': result['dataset'],
                    'column_type': result['mutation_source_feature_type'],
                    'mutation_type': result['mutation_type'],
                    'is_sensitive': False,  # This field doesn't exist in the result dict, setting to False
                    'sensitive_attr_analyzed': result['sensitive_attr_analyzed'],
                    'symptom_name': symptom_name,
                    'pre_symptom_value': pre_value,
                    'post_symptom_value': post_value,
                    'symptom_difference': post_value - pre_value,
                    'symptom_abs_difference': abs(post_value - pre_value),
                    'symptom_percent_change': ((post_value - pre_value) / pre_value * 100) if pre_value != 0 else 0
                }
                
                streamlined_results.append(streamlined_entry)
    
    # Export streamlined results
    if streamlined_results:
        streamlined_df = pd.DataFrame(streamlined_results)
        results_filename = f'../results/results_{file_path.replace(".csv", "")}_{timestamp}.csv'
        streamlined_df.to_csv(results_filename, index=False)
        print(f"\n✓ Streamlined results exported: {results_filename}")
        return results_filename
    
    return None


def generate_comprehensive_final_dataset(results, file_path):
    """Generate comprehensive final dataset with calculated symptom differences."""
    comprehensive_results = []
    
    for result in results:
        # Extract pre and post symptoms
        pre_symptoms = {k.replace('pre_', ''): v for k, v in result.items() if k.startswith('pre_')}
        post_symptoms = {k.replace('post_', ''): v for k, v in result.items() if k.startswith('post_')}
        
        # Calculate differences for each symptom
        symptom_differences = {}
        for symptom in pre_symptoms.keys():
            if symptom in post_symptoms:
                difference = post_symptoms[symptom] - pre_symptoms[symptom]
                symptom_differences[f'{symptom}_difference'] = difference
                symptom_differences[f'{symptom}_abs_difference'] = abs(difference)
                symptom_differences[f'{symptom}_percent_change'] = (
                    (difference / pre_symptoms[symptom] * 100) if pre_symptoms[symptom] != 0 else 0
                )
        
        # Create comprehensive entry
        comprehensive_entry = {
            'source_dataset': file_path,
            'mutated_feature': result['column'],
            'mutation_operator': result['mutation_type'],
            'feature_type': result['column_type'],
            'is_sensitive_feature': result['is_sensitive'],
            'sensitive_attr_analyzed': result['sensitive_attr_analyzed'],
            # Pre-mutation symptoms
            **{f'pre_{k}': v for k, v in pre_symptoms.items()},
            # Post-mutation symptoms  
            **{f'post_{k}': v for k, v in post_symptoms.items()},
            # Calculated differences
            **symptom_differences,
            # Summary metrics
            'max_absolute_change': max([abs(v) for k, v in symptom_differences.items() if k.endswith('_difference')], default=0),
            'significant_change_detected': any([abs(v) > 0.1 for k, v in symptom_differences.items() if k.endswith('_difference')])
        }
        
        comprehensive_results.append(comprehensive_entry)
    
    return comprehensive_results