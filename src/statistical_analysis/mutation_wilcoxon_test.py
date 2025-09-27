"""
Wilcoxon Signed-Rank Test for Mutation Analysis

This module performs Wilcoxon signed-rank tests on mutation results to determine
if mutation operators produce statistically significant changes in fairness symptoms.
The test is adapted to work with the current CSV format where each row represents
a specific column/mutation combination.
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


class MutationWilcoxonTest:
    """
    Performs Wilcoxon signed-rank tests on mutation results to identify
    statistically significant changes in fairness symptoms.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize with the path to the results CSV file.
        
        Args:
            csv_path: Path to the CSV file containing mutation results
        """
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.results = {}
        
    def get_unique_symptoms(self) -> List[str]:
        """Get list of unique symptoms from the data."""
        return sorted(self.data['symptom_name'].unique())
    
    def get_unique_mutations(self) -> List[str]:
        """Get list of unique mutation types from the data."""
        return sorted(self.data['mutation_type'].unique())
    
    def get_unique_columns(self) -> List[str]:
        """Get list of unique columns from the data."""
        return sorted(self.data['column'].unique())
    
    def wilcoxon_test_by_mutation(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform Wilcoxon signed-rank test for each mutation type across all symptoms.
        
        Returns:
            Dictionary with mutation types as keys and test results as values
        """
        results = {}
        
        for mutation in self.get_unique_mutations():
            mutation_data = self.data[self.data['mutation_type'] == mutation]
            
            # Get all symptom differences for this mutation
            differences = mutation_data['symptom_difference'].dropna()
            
            # Remove zero differences (pairs with no change)
            non_zero_differences = differences[differences != 0]
            
            if len(non_zero_differences) < 3:  # Need at least 3 non-zero differences
                results[mutation] = {
                    'statistic': None,
                    'p_value': None,
                    'n_pairs': len(differences),
                    'n_non_zero_pairs': len(non_zero_differences),
                    'significant': False,
                    'error': 'Insufficient non-zero differences for test'
                }
                continue
            
            try:
                # Perform Wilcoxon signed-rank test
                statistic, p_value = wilcoxon(non_zero_differences, alternative='two-sided')
                
                results[mutation] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'n_pairs': len(differences),
                    'n_non_zero_pairs': len(non_zero_differences),
                    'significant': p_value < 0.05,
                    'mean_difference': float(non_zero_differences.mean()),
                    'median_difference': float(non_zero_differences.median()),
                    'std_difference': float(non_zero_differences.std())
                }
                
            except Exception as e:
                results[mutation] = {
                    'statistic': None,
                    'p_value': None,
                    'n_pairs': len(differences),
                    'n_non_zero_pairs': len(non_zero_differences),
                    'significant': False,
                    'error': str(e)
                }
        
        return results
    
    def wilcoxon_test_by_column(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform Wilcoxon signed-rank test for each sensitive attribute column.
        
        Returns:
            Dictionary with column names as keys and test results as values
        """
        results = {}
        
        for column in self.get_unique_columns():
            column_data = self.data[self.data['column'] == column]
            
            # Get all symptom differences for this column
            differences = column_data['symptom_difference'].dropna()
            
            # Remove zero differences
            non_zero_differences = differences[differences != 0]
            
            if len(non_zero_differences) < 3:
                results[column] = {
                    'statistic': None,
                    'p_value': None,
                    'n_pairs': len(differences),
                    'n_non_zero_pairs': len(non_zero_differences),
                    'significant': False,
                    'error': 'Insufficient non-zero differences for test'
                }
                continue
            
            try:
                statistic, p_value = wilcoxon(non_zero_differences, alternative='two-sided')
                
                results[column] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'n_pairs': len(differences),
                    'n_non_zero_pairs': len(non_zero_differences),
                    'significant': p_value < 0.05,
                    'mean_difference': float(non_zero_differences.mean()),
                    'median_difference': float(non_zero_differences.median()),
                    'std_difference': float(non_zero_differences.std())
                }
                
            except Exception as e:
                results[column] = {
                    'statistic': None,
                    'p_value': None,
                    'n_pairs': len(differences),
                    'n_non_zero_pairs': len(non_zero_differences),
                    'significant': False,
                    'error': str(e)
                }
        
        return results
    
    def wilcoxon_test_by_mutation_and_column(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Perform Wilcoxon signed-rank test for each mutation-column combination.
        
        Returns:
            Nested dictionary with mutation types and columns as keys
        """
        results = {}
        
        for mutation in self.get_unique_mutations():
            results[mutation] = {}
            
            for column in self.get_unique_columns():
                subset_data = self.data[
                    (self.data['mutation_type'] == mutation) & 
                    (self.data['column'] == column)
                ]
                
                differences = subset_data['symptom_difference'].dropna()
                non_zero_differences = differences[differences != 0]
                
                if len(non_zero_differences) < 3:
                    results[mutation][column] = {
                        'statistic': None,
                        'p_value': None,
                        'n_pairs': len(differences),
                        'n_non_zero_pairs': len(non_zero_differences),
                        'significant': False,
                        'error': 'Insufficient non-zero differences for test'
                    }
                    continue
                
                try:
                    statistic, p_value = wilcoxon(non_zero_differences, alternative='two-sided')
                    
                    results[mutation][column] = {
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'n_pairs': len(differences),
                        'n_non_zero_pairs': len(non_zero_differences),
                        'significant': p_value < 0.05,
                        'mean_difference': float(non_zero_differences.mean()),
                        'median_difference': float(non_zero_differences.median()),
                        'std_difference': float(non_zero_differences.std())
                    }
                    
                except Exception as e:
                    results[mutation][column] = {
                        'statistic': None,
                        'p_value': None,
                        'n_pairs': len(differences),
                        'n_non_zero_pairs': len(non_zero_differences),
                        'significant': False,
                        'error': str(e)
                    }
        
        return results
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of all Wilcoxon test results.
        
        Returns:
            Dictionary containing summary statistics and significant findings
        """
        mutation_results = self.wilcoxon_test_by_mutation()
        column_results = self.wilcoxon_test_by_column()
        combined_results = self.wilcoxon_test_by_mutation_and_column()
        
        # Count significant results
        significant_mutations = [m for m, r in mutation_results.items() if r.get('significant', False)]
        significant_columns = [c for c, r in column_results.items() if r.get('significant', False)]
        
        significant_combinations = []
        for mutation, columns in combined_results.items():
            for column, result in columns.items():
                if result.get('significant', False):
                    significant_combinations.append(f"{mutation}_{column}")
        
        summary = {
            'total_mutations_tested': len(mutation_results),
            'significant_mutations': significant_mutations,
            'total_columns_tested': len(column_results),
            'significant_columns': significant_columns,
            'total_combinations_tested': sum(len(cols) for cols in combined_results.values()),
            'significant_combinations': significant_combinations,
            'mutation_results': mutation_results,
            'column_results': column_results,
            'combined_results': combined_results
        }
        
        return summary
    
    def print_results(self):
        """Print formatted results to console."""
        summary = self.generate_summary()
        
        print("=" * 80)
        print("WILCOXON SIGNED-RANK TEST RESULTS FOR MUTATION ANALYSIS")
        print("=" * 80)
        
        print(f"\nDataset: {Path(self.csv_path).name}")
        print(f"Total data points: {len(self.data)}")
        
        print("\n" + "=" * 50)
        print("RESULTS BY MUTATION TYPE")
        print("=" * 50)
        
        for mutation, result in summary['mutation_results'].items():
            print(f"\nMutation: {mutation}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            else:
                print(f"  Statistic: {result.get('statistic', 'N/A')}")
                print(f"  P-value: {result.get('p_value', 'N/A')}")
                print(f"  Significant: {'Yes' if result.get('significant') else 'No'}")
                print(f"  Non-zero pairs: {result.get('n_non_zero_pairs', 0)}")
                if result.get('mean_difference') is not None:
                    print(f"  Mean difference: {result['mean_difference']:.6f}")
                    print(f"  Median difference: {result['median_difference']:.6f}")
        
        print("\n" + "=" * 50)
        print("RESULTS BY SENSITIVE ATTRIBUTE")
        print("=" * 50)
        
        for column, result in summary['column_results'].items():
            print(f"\nColumn: {column}")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            else:
                print(f"  Statistic: {result.get('statistic', 'N/A')}")
                print(f"  P-value: {result.get('p_value', 'N/A')}")
                print(f"  Significant: {'Yes' if result.get('significant') else 'No'}")
                print(f"  Non-zero pairs: {result.get('n_non_zero_pairs', 0)}")
                if result.get('mean_difference') is not None:
                    print(f"  Mean difference: {result['mean_difference']:.6f}")
                    print(f"  Median difference: {result['median_difference']:.6f}")
        
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Significant mutations: {len(summary['significant_mutations'])}")
        if summary['significant_mutations']:
            print(f"  - {', '.join(summary['significant_mutations'])}")
        
        print(f"Significant columns: {len(summary['significant_columns'])}")
        if summary['significant_columns']:
            print(f"  - {', '.join(summary['significant_columns'])}")
        
        print(f"Significant combinations: {len(summary['significant_combinations'])}")
        if summary['significant_combinations']:
            for combo in summary['significant_combinations'][:10]:  # Show first 10
                print(f"  - {combo}")
            if len(summary['significant_combinations']) > 10:
                print(f"  ... and {len(summary['significant_combinations']) - 10} more")


def main():
    """Main function to run the Wilcoxon analysis."""
    csv_path = "/Users/alfonsocannavale/Documents/Progetti/drama/results/results_adult_20250916_185125.csv"
    
    print("Starting Wilcoxon Signed-Rank Test Analysis...")
    
    # Initialize and run analysis
    wilcoxon_test = MutationWilcoxonTest(csv_path)
    
    # Print results
    wilcoxon_test.print_results()
    
    # Save results to JSON
    summary = wilcoxon_test.generate_summary()
    output_path = "/Users/alfonsocannavale/Documents/Progetti/drama/src/statistical_analysis/wilcoxon_results.json"
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()