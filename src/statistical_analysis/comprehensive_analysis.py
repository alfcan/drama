"""
Comprehensive Statistical Analysis for Mutation Testing

This module provides a complete statistical analysis framework for mutation testing results,
combining both Sign Test and Wilcoxon Signed-Rank Test to identify which mutation operators
produce statistically significant changes in fairness symptoms.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from mutation_sign_test import MutationSignTest
from mutation_wilcoxon_test import MutationWilcoxonTest


class ComprehensiveMutationAnalysis:
    """
    Comprehensive analysis combining multiple statistical tests for mutation results.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize with the path to the results CSV file.
        
        Args:
            csv_path: Path to the CSV file containing mutation results
        """
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.sign_test = MutationSignTest(csv_path)
        self.wilcoxon_test = MutationWilcoxonTest(csv_path)
        
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run both Sign Test and Wilcoxon Signed-Rank Test analyses.
        
        Returns:
            Dictionary containing results from both tests
        """
        print("Running comprehensive statistical analysis...")
        print("=" * 60)
        
        # Run Sign Test
        print("\n1. Running Sign Test Analysis...")
        sign_analysis_results = self.sign_test.perform_comprehensive_analysis()
        sign_results = self.sign_test._generate_summary(sign_analysis_results)
        
        # Run Wilcoxon Test
        print("\n2. Running Wilcoxon Signed-Rank Test Analysis...")
        wilcoxon_results = self.wilcoxon_test.generate_summary()
        
        # Generate dataset mapping for significant combinations
        dataset_mapping = self._generate_dataset_mapping(wilcoxon_results.get('significant_combinations', []))
        
        # Combine results
        combined_results = {
            'sign_test_results': sign_results,
            'wilcoxon_test_results': wilcoxon_results,
            'data_summary': self._generate_data_summary(),
            'consensus_analysis': self._analyze_consensus(sign_results, wilcoxon_results),
            'dataset_mapping': dataset_mapping
        }
        
        return combined_results
    
    def _generate_dataset_mapping(self, significant_combinations: List[str]) -> Dict[str, Any]:
        """
        Generate mapping of significant combinations to their source datasets using the dataset column.
        
        Args:
            significant_combinations: List of significant mutation-column combinations
            
        Returns:
            Dictionary containing dataset mapping information
        """
        mapping = {
            'combination_to_dataset': {},
            'dataset_to_combinations': {},
            'summary': {}
        }
        
        # Extract unique datasets from the data
        unique_datasets = self.data['dataset'].unique()
        
        # Initialize dataset groups
        for dataset in unique_datasets:
            dataset_name = dataset.replace('.csv', '')  # Remove .csv extension for cleaner names
            mapping['dataset_to_combinations'][dataset_name] = []
        
        # Get all unique mutation types from the data to properly parse combinations
        unique_mutations = self.data['mutation_type'].unique()
        
        # Map each significant combination to its dataset
        for combination in significant_combinations:
            # Parse combination format: mutation_column
            # Try to match against known mutation types to find the correct split point
            mutation_type = None
            column_name = None
            
            # Try each mutation type to see if the combination starts with it
            for mut_type in unique_mutations:
                if combination.startswith(mut_type + '_'):
                    mutation_type = mut_type
                    column_name = combination[len(mut_type) + 1:]  # +1 for the underscore
                    break
            
            if mutation_type and column_name:
                # Find the dataset for this mutation-column combination
                matching_rows = self.data[
                    (self.data['mutation_type'] == mutation_type) & 
                    (self.data['column'] == column_name)
                ]
                
                if len(matching_rows) > 0:
                    # Use the first dataset found
                    dataset = matching_rows['dataset'].iloc[0]
                    dataset_name = dataset.replace('.csv', '')
                    
                    # Add to mapping
                    mapping['combination_to_dataset'][combination] = dataset_name
                    mapping['dataset_to_combinations'][dataset_name].append(combination)
                else:
                    # Handle case where combination is not found in data
                    mapping['combination_to_dataset'][combination] = 'unknown'
                    if 'unknown' not in mapping['dataset_to_combinations']:
                        mapping['dataset_to_combinations']['unknown'] = []
                    mapping['dataset_to_combinations']['unknown'].append(combination)
            else:
                # Handle case where we can't parse the combination
                mapping['combination_to_dataset'][combination] = 'unknown'
                if 'unknown' not in mapping['dataset_to_combinations']:
                    mapping['dataset_to_combinations']['unknown'] = []
                mapping['dataset_to_combinations']['unknown'].append(combination)
        
        # Generate summary statistics
        dataset_counts = {dataset: len(combinations) 
                         for dataset, combinations in mapping['dataset_to_combinations'].items() 
                         if combinations}  # Only include datasets with combinations
        
        mapping['summary'] = {
            'total_significant_combinations': len(significant_combinations),
            'datasets_with_significant_combinations': len(dataset_counts),
            'combinations_per_dataset': dataset_counts,
            'most_affected_dataset': max(dataset_counts.items(), key=lambda x: x[1])[0] if dataset_counts else None,
            'least_affected_dataset': min(dataset_counts.items(), key=lambda x: x[1])[0] if dataset_counts else None
        }
        
        return mapping
    
    def _generate_data_summary(self) -> Dict[str, Any]:
        """Generate summary statistics about the dataset."""
        summary = {
            'total_rows': len(self.data),
            'unique_mutations': len(self.data['mutation_type'].unique()),
            'unique_columns': len(self.data['column'].unique()),
            'unique_symptoms': len(self.data['symptom_name'].unique()),
            'unique_datasets': len(self.data['dataset'].unique()),
            'datasets': sorted(self.data['dataset'].unique().tolist()),
            'mutation_types': sorted(self.data['mutation_type'].unique().tolist()),
            'sensitive_attributes': sorted(self.data['column'].unique().tolist()),
            'symptoms': sorted(self.data['symptom_name'].unique().tolist()),
            'non_zero_differences': len(self.data[self.data['symptom_difference'] != 0]),
            'zero_differences': len(self.data[self.data['symptom_difference'] == 0]),
            'positive_differences': len(self.data[self.data['symptom_difference'] > 0]),
            'negative_differences': len(self.data[self.data['symptom_difference'] < 0])
        }
        
        # Add descriptive statistics
        non_zero_data = self.data[self.data['symptom_difference'] != 0]['symptom_difference']
        if len(non_zero_data) > 0:
            summary['difference_stats'] = {
                'mean': float(non_zero_data.mean()),
                'median': float(non_zero_data.median()),
                'std': float(non_zero_data.std()),
                'min': float(non_zero_data.min()),
                'max': float(non_zero_data.max()),
                'q25': float(non_zero_data.quantile(0.25)),
                'q75': float(non_zero_data.quantile(0.75))
            }
        
        return summary
    
    def _analyze_consensus(self, sign_results: Dict, wilcoxon_results: Dict) -> Dict[str, Any]:
        """
        Analyze consensus between Sign Test and Wilcoxon Test results.
        
        Args:
            sign_results: Results from Sign Test (summary format)
            wilcoxon_results: Results from Wilcoxon Test
            
        Returns:
            Dictionary containing consensus analysis
        """
        consensus = {
            'mutations': {},
            'columns': {},
            'summary': {}
        }
        
        # Get significant mutations from both tests
        sign_sig_mutations = set(sign_results.get('mutations_with_effects', []))
        wilcoxon_sig_mutations = set(wilcoxon_results.get('significant_mutations', []))
        
        # Get significant columns from both tests  
        sign_sig_columns = set()
        wilcoxon_sig_columns = set(wilcoxon_results.get('significant_columns', []))
        
        # Analyze mutation-level consensus
        all_mutations = sign_sig_mutations.union(wilcoxon_sig_mutations)
        for mutation in all_mutations:
            sign_sig = mutation in sign_sig_mutations
            wilcoxon_sig = mutation in wilcoxon_sig_mutations
            
            consensus['mutations'][mutation] = {
                'sign_test_significant': sign_sig,
                'wilcoxon_test_significant': wilcoxon_sig,
                'both_significant': sign_sig and wilcoxon_sig,
                'either_significant': sign_sig or wilcoxon_sig,
                'consensus': 'agree' if sign_sig == wilcoxon_sig else 'disagree'
            }
        
        # Analyze column-level consensus
        all_columns = sign_sig_columns.union(wilcoxon_sig_columns)
        for column in all_columns:
            sign_sig = column in sign_sig_columns
            wilcoxon_sig = column in wilcoxon_sig_columns
            
            consensus['columns'][column] = {
                'sign_test_significant': sign_sig,
                'wilcoxon_test_significant': wilcoxon_sig,
                'both_significant': sign_sig and wilcoxon_sig,
                'either_significant': sign_sig or wilcoxon_sig,
                'consensus': 'agree' if sign_sig == wilcoxon_sig else 'disagree'
            }
        
        # Generate summary statistics
        mutation_agreements = sum(1 for m in consensus['mutations'].values() if m['consensus'] == 'agree')
        column_agreements = sum(1 for c in consensus['columns'].values() if c['consensus'] == 'agree')
        
        both_sig_mutations = [m for m, data in consensus['mutations'].items() if data['both_significant']]
        both_sig_columns = [c for c, data in consensus['columns'].items() if data['both_significant']]
        
        consensus['summary'] = {
            'mutation_agreement_rate': mutation_agreements / len(consensus['mutations']) if consensus['mutations'] else 0,
            'column_agreement_rate': column_agreements / len(consensus['columns']) if consensus['columns'] else 0,
            'mutations_significant_in_both': both_sig_mutations,
            'columns_significant_in_both': both_sig_columns,
            'total_mutations_analyzed': len(consensus['mutations']),
            'total_columns_analyzed': len(consensus['columns'])
        }
        
        return consensus
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive text report of the analysis results.
        
        Args:
            results: Combined results from both tests
            
        Returns:
            Formatted text report
        """
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE MUTATION ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Data Summary
        data_summary = results['data_summary']
        report.append(f"\nDATASET SUMMARY")
        report.append("-" * 40)
        report.append(f"File: {Path(self.csv_path).name}")
        report.append(f"Total observations: {data_summary['total_rows']:,}")
        report.append(f"Datasets analyzed: {data_summary['unique_datasets']}")
        report.append(f"Mutation types: {data_summary['unique_mutations']}")
        report.append(f"Sensitive attributes: {data_summary['unique_columns']}")
        report.append(f"Fairness symptoms: {data_summary['unique_symptoms']}")
        report.append(f"Non-zero differences: {data_summary['non_zero_differences']:,} ({data_summary['non_zero_differences']/data_summary['total_rows']*100:.1f}%)")
        
        if 'difference_stats' in data_summary:
            stats = data_summary['difference_stats']
            report.append(f"\nDifference Statistics (non-zero only):")
            report.append(f"  Mean: {stats['mean']:.6f}")
            report.append(f"  Median: {stats['median']:.6f}")
            report.append(f"  Std Dev: {stats['std']:.6f}")
            report.append(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        
        # Dataset Mapping Analysis
        if 'dataset_mapping' in results:
            dataset_mapping = results['dataset_mapping']
            report.append(f"\nDATASET MAPPING ANALYSIS")
            report.append("-" * 40)
            mapping_summary = dataset_mapping['summary']
            report.append(f"Total significant combinations: {mapping_summary['total_significant_combinations']}")
            report.append(f"Datasets with significant effects: {mapping_summary['datasets_with_significant_combinations']}")
            
            if mapping_summary['most_affected_dataset']:
                report.append(f"Most affected dataset: {mapping_summary['most_affected_dataset']} ({mapping_summary['combinations_per_dataset'][mapping_summary['most_affected_dataset']]} combinations)")
            
            report.append(f"\nSignificant combinations per dataset:")
            for dataset, count in sorted(mapping_summary['combinations_per_dataset'].items()):
                report.append(f"  {dataset}: {count} combinations")
                # Show the actual combinations for each dataset
                combinations = dataset_mapping['dataset_to_combinations'][dataset]
                for combo in combinations[:5]:  # Show first 5 combinations
                    report.append(f"    • {combo}")
                if len(combinations) > 5:
                    report.append(f"    ... and {len(combinations) - 5} more")
        
        # Consensus Analysis
        consensus = results['consensus_analysis']
        report.append(f"\nCONSENSUS ANALYSIS")
        report.append("-" * 40)
        report.append(f"Mutation agreement rate: {consensus['summary']['mutation_agreement_rate']:.1%}")
        report.append(f"Column agreement rate: {consensus['summary']['column_agreement_rate']:.1%}")
        
        # Significant findings
        both_sig_mutations = consensus['summary']['mutations_significant_in_both']
        both_sig_columns = consensus['summary']['columns_significant_in_both']
        
        report.append(f"\nMUTATIONS SIGNIFICANT IN BOTH TESTS ({len(both_sig_mutations)}):")
        if both_sig_mutations:
            for mutation in both_sig_mutations:
                report.append(f"  ✓ {mutation}")
        else:
            report.append("  None")
        
        report.append(f"\nSENSITIVE ATTRIBUTES SIGNIFICANT IN BOTH TESTS ({len(both_sig_columns)}):")
        if both_sig_columns:
            for column in both_sig_columns:
                report.append(f"  ✓ {column}")
        else:
            report.append("  None")
        
        # Detailed results by test
        report.append(f"\nSIGN TEST RESULTS")
        report.append("-" * 40)
        sign_results = results['sign_test_results']
        sign_sig_mutations = sign_results.get('significant_mutations', [])
        sign_sig_columns = sign_results.get('significant_columns', [])
        
        report.append(f"Significant mutations: {len(sign_sig_mutations)}")
        for mutation in sign_sig_mutations:
            report.append(f"  • {mutation}")
        
        report.append(f"Significant columns: {len(sign_sig_columns)}")
        for column in sign_sig_columns:
            report.append(f"  • {column}")
        
        report.append(f"\nWILCOXON TEST RESULTS")
        report.append("-" * 40)
        wilcoxon_results = results['wilcoxon_test_results']
        wilcoxon_sig_mutations = wilcoxon_results.get('significant_mutations', [])
        wilcoxon_sig_columns = wilcoxon_results.get('significant_columns', [])
        
        report.append(f"Significant mutations: {len(wilcoxon_sig_mutations)}")
        for mutation in wilcoxon_sig_mutations:
            report.append(f"  • {mutation}")
        
        report.append(f"Significant columns: {len(wilcoxon_sig_columns)}")
        for column in wilcoxon_sig_columns:
            report.append(f"  • {column}")
                    
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], output_dir: str = None):
        """
        Save analysis results to files.
        
        Args:
            results: Combined results from analysis
            output_dir: Directory to save results (defaults to statistical_analysis folder)
        """
        if output_dir is None:
            output_dir = Path(self.csv_path).parent.parent / "src" / "statistical_analysis"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        # Convert results for JSON serialization
        json_results = convert_for_json(results)
        
        # Save JSON results
        json_path = output_dir / "comprehensive_analysis_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save text report
        report = self.generate_report(results)
        report_path = output_dir / "analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Report: {report_path}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of the analysis results."""
        print(self.generate_report(results))


def main():
    """Main function to run the comprehensive analysis."""
    path = "../../results/"
    dataset_name = input("Enter the dataset name (with extension):")

    csv_path = path + dataset_name
    
    print("Starting Comprehensive Mutation Analysis...")
    print("This analysis combines Sign Test and Wilcoxon Signed-Rank Test")
    print("to identify mutation operators with significant fairness impact.\n")
    
    # Initialize analysis
    analysis = ComprehensiveMutationAnalysis(csv_path)
    
    # Run complete analysis
    results = analysis.run_complete_analysis()
    
    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    analysis.print_summary(results)
    
    # Save results
    analysis.save_results(results)
    
    print("\n" + "=" * 80)
    print("Analysis completed successfully!")
    print("Check the generated files for detailed results and recommendations.")
    print("=" * 80)


if __name__ == "__main__":
    main()