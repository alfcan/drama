"""
Sign Test for Mutation Operator Analysis

This module performs Sign Tests to determine if mutation operators produce
statistically significant changes in fairness and bias symptoms.

The Sign Test is a non-parametric test that compares paired observations
by counting the number of positive and negative differences.
"""

from scipy.stats import binomtest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class MutationSignTest:
    """
    Performs Sign Tests on mutation operator results to identify significant changes.
    """
    
    def __init__(self, results_file: str):
        """
        Initialize the Sign Test analyzer.
        
        Args:
            results_file: Path to the CSV file containing mutation results
        """
        self.results_file = results_file
        self.df = pd.read_csv(results_file)
        
    def get_unique_symptoms(self) -> List[str]:
        """Get list of unique symptoms from the dataset."""
        return sorted(self.df['symptom_name'].unique())
    
    def get_unique_mutations(self) -> List[str]:
        """Get list of unique mutation types from the dataset."""
        return sorted(self.df['mutation_type'].unique())
    
    def get_unique_columns(self) -> List[str]:
        """Get list of unique columns from the dataset."""
        return sorted(self.df['column'].unique())
    
    def perform_sign_test_by_mutation(self, mutation_type: str, 
                                    sensitive_attr: Optional[str] = None) -> Dict:
        """
        Perform Sign Test for a specific mutation type across all symptoms.
        
        Args:
            mutation_type: Type of mutation to analyze
            sensitive_attr: Optional filter for sensitive attribute
            
        Returns:
            Dictionary containing test results for each symptom
        """
        # Filter data for the specific mutation type
        mask = self.df['mutation_type'] == mutation_type
        if sensitive_attr:
            mask &= self.df['sensitive_attr_analyzed'] == sensitive_attr
            
        mutation_data = self.df[mask]
        
        if mutation_data.empty:
            return {}
        
        results = {}
        symptoms = self.get_unique_symptoms()
        
        for symptom in symptoms:
            symptom_data = mutation_data[mutation_data['symptom_name'] == symptom]
            
            if symptom_data.empty:
                continue
                
            # Get pre and post values
            pre_values = symptom_data['pre_symptom_value'].values
            post_values = symptom_data['post_symptom_value'].values
            
            # Calculate differences
            differences = post_values - pre_values
            
            # Count positive, negative, and zero differences
            num_positive = np.sum(differences > 0)
            num_negative = np.sum(differences < 0)
            num_zero = np.sum(differences == 0)
            
            # Perform Sign Test if there are non-zero differences
            if num_positive + num_negative > 0:
                stat = min(num_positive, num_negative)
                p_value = binomtest(stat, n=num_positive + num_negative, 
                                  p=0.5, alternative='two-sided').pvalue
            else:
                stat = None
                p_value = None
            
            results[symptom] = {
                'num_positive': num_positive,
                'num_negative': num_negative,
                'num_zero': num_zero,
                'total_observations': len(differences),
                'stat': stat,
                'p_value': p_value,
                'significant': p_value < 0.05 if p_value is not None else False,
                'effect_direction': 'increase' if num_positive > num_negative else 'decrease' if num_negative > num_positive else 'no_change'
            }
        
        return results
    
    def perform_sign_test_by_column(self, column_name: str, 
                                  sensitive_attr: Optional[str] = None) -> Dict:
        """
        Perform Sign Test for a specific column across all mutation types and symptoms.
        
        Args:
            column_name: Name of the column to analyze
            sensitive_attr: Optional filter for sensitive attribute
            
        Returns:
            Dictionary containing test results organized by mutation type and symptom
        """
        # Filter data for the specific column
        mask = self.df['column'] == column_name
        if sensitive_attr:
            mask &= self.df['sensitive_attr_analyzed'] == sensitive_attr
            
        column_data = self.df[mask]
        
        if column_data.empty:
            return {}
        
        results = {}
        mutations = column_data['mutation_type'].unique()
        
        for mutation in mutations:
            mutation_data = column_data[column_data['mutation_type'] == mutation]
            results[mutation] = self.perform_sign_test_by_mutation(mutation, sensitive_attr)
        
        return results
    
    def perform_comprehensive_analysis(self) -> Dict:
        """
        Perform comprehensive Sign Test analysis across all mutations and symptoms.
        
        Returns:
            Dictionary containing complete analysis results
        """
        results = {
            'by_mutation': {},
            'by_column': {},
            'by_sensitive_attr': {},
            'summary': {}
        }
        
        # Analysis by mutation type
        mutations = self.get_unique_mutations()
        for mutation in mutations:
            results['by_mutation'][mutation] = self.perform_sign_test_by_mutation(mutation)
        
        # Analysis by column
        columns = self.get_unique_columns()
        for column in columns:
            results['by_column'][column] = self.perform_sign_test_by_column(column)
        
        # Analysis by sensitive attribute
        sensitive_attrs = self.df['sensitive_attr_analyzed'].unique()
        for attr in sensitive_attrs:
            results['by_sensitive_attr'][attr] = {}
            for mutation in mutations:
                results['by_sensitive_attr'][attr][mutation] = \
                    self.perform_sign_test_by_mutation(mutation, attr)
        
        # Generate summary statistics
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary statistics from the analysis results."""
        summary = {
            'total_tests': 0,
            'significant_tests': 0,
            'mutations_with_effects': set(),
            'symptoms_most_affected': {},
            'columns_most_affected': {}
        }
        
        # Count tests and significant results
        for mutation, mutation_results in results['by_mutation'].items():
            for symptom, test_result in mutation_results.items():
                summary['total_tests'] += 1
                if test_result.get('significant', False):
                    summary['significant_tests'] += 1
                    summary['mutations_with_effects'].add(mutation)
                    
                    # Track most affected symptoms
                    if symptom not in summary['symptoms_most_affected']:
                        summary['symptoms_most_affected'][symptom] = 0
                    summary['symptoms_most_affected'][symptom] += 1
        
        # Convert set to list for JSON serialization
        summary['mutations_with_effects'] = list(summary['mutations_with_effects'])
        
        # Calculate significance rate
        if summary['total_tests'] > 0:
            summary['significance_rate'] = summary['significant_tests'] / summary['total_tests']
        else:
            summary['significance_rate'] = 0.0
        
        return summary
    
    def print_results(self, results: Dict, mutation_type: str = None):
        """
        Print formatted results of the Sign Test analysis.
        
        Args:
            results: Results dictionary from analysis
            mutation_type: Optional specific mutation type to print
        """
        if mutation_type and mutation_type in results:
            self._print_mutation_results(mutation_type, results[mutation_type])
        else:
            # Print summary
            if 'summary' in results:
                summary = results['summary']
                print("=== SIGN TEST ANALYSIS SUMMARY ===")
                print(f"Total tests performed: {summary['total_tests']}")
                print(f"Significant results: {summary['significant_tests']}")
                print(f"Significance rate: {summary['significance_rate']:.2%}")
                print(f"Mutations with significant effects: {', '.join(summary['mutations_with_effects'])}")
                print()
            
            # Print results by mutation
            if 'by_mutation' in results:
                for mutation, mutation_results in results['by_mutation'].items():
                    self._print_mutation_results(mutation, mutation_results)
    
    def _print_mutation_results(self, mutation_type: str, results: Dict):
        """Print results for a specific mutation type."""
        print(f"=== SIGN TEST RESULTS FOR {mutation_type.upper()} ===")
        
        significant_symptoms = []
        for symptom, result in results.items():
            print(f"\nSymptom: {symptom}")
            print(f"  Positive changes: {result['num_positive']}")
            print(f"  Negative changes: {result['num_negative']}")
            print(f"  No changes: {result['num_zero']}")
            print(f"  Total observations: {result['total_observations']}")
            
            if result['p_value'] is not None:
                print(f"  Test statistic: {result['stat']}")
                print(f"  p-value: {result['p_value']:.6f}")
                print(f"  Significant: {'Yes' if result['significant'] else 'No'}")
                print(f"  Effect direction: {result['effect_direction']}")
                
                if result['significant']:
                    significant_symptoms.append(symptom)
            else:
                print("  No non-zero differences found - test not applicable")
        
        if significant_symptoms:
            print(f"\nSignificant effects found in: {', '.join(significant_symptoms)}")
        else:
            print(f"\nNo significant effects found for {mutation_type}")
        print("-" * 60)


def main():
    """Main function to run the Sign Test analysis."""
    import sys
    
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = "../../results/results_adult_20250916_185125.csv"
    
    try:
        analyzer = MutationSignTest(results_file)
        
        print("Starting comprehensive Sign Test analysis...")
        results = analyzer.perform_comprehensive_analysis()
        
        analyzer.print_results(results)
        
        # Save results to file
        import json
        output_file = results_file.replace('.csv', '_sign_test_results.json')
        with open(output_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(results, f, indent=2, default=convert_numpy)
        
        print(f"\nDetailed results saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Results file '{results_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()