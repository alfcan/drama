import numpy as np
import pandas as pd
import random
import nltk
from nltk.corpus import wordnet
import string
import logging
import torch

# Configure logging for better error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('wordnet', quiet=True)

class MutationOperator:
    
    def __init__(self, dataframe):
        """
        Initialize the MutationOperator.
        
        Args:
            dataframe (pd.DataFrame): The input dataframe to operate on
        """
        self.dataframe = dataframe

    def _show_comparison(self, original_df, modified_df, modified_indices, feature_name, operation_name):
        """
        Display a comparison of original vs modified values for the specified rows and feature.
        
        Parameters:
        - original_df (pd.DataFrame): The original dataframe
        - modified_df (pd.DataFrame): The modified dataframe
        - modified_indices (list): List of indices that were modified
        - feature_name (str): The name of the feature that was modified
        - operation_name (str): The name of the operation performed
        """
        print(f"\n{'='*60}")
        print(f"COMPARISON: {operation_name} on feature '{feature_name}'")
        print(f"{'='*60}")
        print(f"Modified {len(modified_indices)} rows out of {len(original_df)} total rows")
        print(f"{'='*60}")
        
        # Create comparison dataframe
        comparison_data = []
        for idx in sorted(modified_indices):
            original_value = original_df.loc[idx, feature_name]
            modified_value = modified_df.loc[idx, feature_name]
            
            comparison_data.append({
                'Row_Index': idx,
                'Original_Value': original_value,
                'Modified_Value': modified_value,
                'Change': f"{original_value} → {modified_value}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display the comparison
        print("\nPRE- AND POST-CHANGE COMPARISON:")
        print("-" * 60)
        for _, row in comparison_df.iterrows():
            print(f"Row {row['Row_Index']:4d}: {row['Change']}")
        
        print(f"\n{'='*60}\n")

    def increment_decrement_feature(self, feature_name, percentage=20):
        """
        Value Increment/Decrement (Numeric)
        Simulates small measurement errors or natural fluctuations in the data by randomly 
        increasing or decreasing numeric feature values.
        
        Configuration: Modify values by a random amount within ±5% of the feature's range, 
        applied to 20% of the instances to simulate minor data noise.

        Parameters:
        - feature_name (str): The name of the feature to be modified.
        - percentage (float): The percentage of rows to be modified (default: 20).

        Returns:
        - pd.DataFrame: A new DataFrame with the modified feature values.
        """
        # Create a copy of the DataFrame to avoid modifying the original data
        modified_df = self.dataframe.copy()
        
        # Check if the feature_name is valid
        if feature_name not in modified_df.columns:
            raise ValueError(f"The feature {feature_name} does not exist in the DataFrame.")
        
        # Check if the percentage is valid
        if percentage < 0 or percentage > 100:
            raise ValueError("Percentage must be between 0 and 100.")
        
        # Check if the feature is numerical
        if not np.issubdtype(modified_df[feature_name].dtype, np.number):
            raise ValueError(f"The feature {feature_name} is not a numerical type.")
        
        # Calculate the feature range for ±5% variation
        feature_min = modified_df[feature_name].min()
        feature_max = modified_df[feature_name].max()
        feature_range = feature_max - feature_min
        max_variation = 0.05 * feature_range  # ±5% of range
        
        # Calculate the number of rows to modify
        num_rows = len(modified_df)
        num_rows_to_modify = int(num_rows * (percentage / 100.0))
        
        # Randomly select the indices of the rows to modify
        rows_to_modify = np.random.choice(modified_df.index, num_rows_to_modify, replace=False)
        
        # Save the original dtype of the column
        original_dtype = modified_df[feature_name].dtype
        modified_df[feature_name] = modified_df[feature_name].astype(float)

        # Apply random increment/decrement within ±5% of feature range
        for idx in rows_to_modify:
            # Generate random variation within ±5% of feature range
            variation = np.random.uniform(-max_variation, max_variation)
            modified_df.loc[idx, feature_name] += variation

        # Convert the column back to its original dtype
        modified_df[feature_name] = modified_df[feature_name].astype(original_dtype)
        
        # Show comparison of changes
        self._show_comparison(self.dataframe, modified_df, rows_to_modify, feature_name, "Increment/Decrement Feature")
        
        return modified_df
    
    def scale_values(self, feature_name, percentage=20):
        """
        Value Scaling (Numeric)
        Simulates magnitude or unit changes by multiplying or dividing a numeric feature 
        by a small random factor.
        
        Configuration: Scale values by a random factor between 0.8 and 1.2, applied to 
        20% of the instances to assess sensitivity to magnitude shifts.

        Parameters:
        - feature_name (str): The name of the feature to be scaled.
        - percentage (float): The percentage of rows to be modified (default: 20).

        Returns:
        - pd.DataFrame: A new DataFrame with scaled values.
        """
        # Create a copy of the DataFrame to avoid modifying the original data
        modified_df = self.dataframe.copy()
        
        # Validate the feature_name and parameters
        if feature_name not in modified_df.columns:
            raise ValueError(f"The feature {feature_name} does not exist in the DataFrame.")
        if percentage < 0 or percentage > 100:
            raise ValueError("Percentage must be between 0 and 100.")
        
        # Check if the feature is numerical
        if not np.issubdtype(modified_df[feature_name].dtype, np.number):
            raise ValueError(f"The feature {feature_name} is not a numerical type.")
        
        # Calculate the number of rows to modify
        num_rows_to_modify = int(len(modified_df) * (percentage / 100.0))
        rows_to_modify = np.random.choice(modified_df.index, num_rows_to_modify, replace=False)

        # Apply scaling with random factor between 0.8 and 1.2 to selected rows only
        for idx in rows_to_modify:
            scale_factor = random.uniform(0.8, 1.2)
            original_value = modified_df.loc[idx, feature_name]
            scaled_value = original_value * scale_factor
            
            if pd.api.types.is_integer_dtype(modified_df[feature_name]):
                modified_df.loc[idx, feature_name] = int(round(scaled_value))
            else:
                modified_df.loc[idx, feature_name] = scaled_value
        
        # Show comparison of changes
        self._show_comparison(self.dataframe, modified_df, rows_to_modify, feature_name, "Scale Values")
        
        return modified_df
    
    def category_flip(self, feature_name, percentage=15):
        """
        Category Flip (Categorical/Textual)
        Simulates data entry or labeling errors by randomly reassigning a subset of instances 
        from one category to another among the existing categories of a feature.
        
        Configuration: For each category, reassign 15% of its instances to another existing 
        category to simulate classification errors.

        Parameters:
        - feature_name (str): The name of the categorical feature to be modified.

        Returns:
        - pd.DataFrame: A new DataFrame with flipped categories.
        """
        # Create a copy of the DataFrame to avoid modifying the original data
        modified_df = self.dataframe.copy()
        
        # Check if the feature_name is valid
        if feature_name not in modified_df.columns:
            raise ValueError(f"The feature {feature_name} does not exist in the DataFrame.")
        
        # Get unique categories
        unique_categories = modified_df[feature_name].unique()
        
        if len(unique_categories) < 2:
            raise ValueError(f"Feature {feature_name} must have at least 2 categories for flipping.")
        
        # Track all modified indices
        all_modified_indices = []
        
        # For each category, reassign 15% of its instances to another category
        for category in unique_categories:
            # Get indices of instances with this category
            category_indices = modified_df[modified_df[feature_name] == category].index.tolist()
            
            if len(category_indices) == 0:
                continue
                
            # Calculate percentage of instances for this category
            num_to_flip = max(1, int(len(category_indices) * (percentage / 100.0)))
            
            # Randomly select instances to flip
            indices_to_flip = random.sample(category_indices, min(num_to_flip, len(category_indices)))
            all_modified_indices.extend(indices_to_flip)
            
            # Get other categories to flip to
            other_categories = [cat for cat in unique_categories if cat != category]
            
            # Flip selected instances to random other categories
            for idx in indices_to_flip:
                new_category = random.choice(other_categories)
                modified_df.loc[idx, feature_name] = new_category
        
        # Show comparison of changes
        self._show_comparison(self.dataframe, modified_df, all_modified_indices, feature_name, "Category Flip")
        
        return modified_df
    
    def replace_synonyms(self, text_column, row_percentage=10, word_percentage=15):
        """
        Synonym Replacement (Textual)
        Simulates natural linguistic variability by replacing a configurable fraction of words 
        in textual features with their synonyms, using lexical resources such as WordNet.
        
        Configuration: In 10% of the instances, replace 15% of their words with synonyms 
        to simulate linguistic variability.

        Parameters:
        - text_column (str): The name of the column containing text data.
        - row_percentage (int): The percentage of rows to modify (default: 10).
        - word_percentage (int): The percentage of words to replace with synonyms within each selected row (default: 15).

        Returns:
        - pd.DataFrame: A new DataFrame with the text data modified.
        """
        # Create a copy of the DataFrame to avoid modifying the original data
        modified_df = self.dataframe.copy()

        # Verify that the text column exists
        if text_column not in modified_df.columns:
            raise ValueError(f"The column {text_column} does not exist in the DataFrame.")
        if row_percentage < 0 or row_percentage > 100:
            raise ValueError("Row percentage must be between 0 and 100.")
        if word_percentage < 0 or word_percentage > 100:
            raise ValueError("Word percentage must be between 0 and 100.")

        # Determine the number of rows to modify based on the given row percentage
        num_rows = len(modified_df)
        num_rows_to_modify = int(num_rows * (row_percentage / 100))

        # Sample the rows to modify
        rows_to_modify = modified_df.sample(n=num_rows_to_modify).index

        # Apply synonym replacement to the sampled rows
        for idx in rows_to_modify:
            if isinstance(modified_df.at[idx, text_column], str):  # Make sure the value is a string, in the case of NaN values
                modified_df.at[idx, text_column] = self._synonym_replacement(modified_df.at[idx, text_column], word_percentage)

        # Show comparison of changes
        self._show_comparison(self.dataframe, modified_df, rows_to_modify, text_column, "Replace Synonyms")

        return modified_df
    
    def add_noise(self, text_column, percentage=10):
        """
        Noise Injection (Textual)
        Simulates common typographical errors by injecting controlled noise (e.g., random 
        character swaps, insertions) into textual features.
        
        For categorical features, creates a single mutation per unique category and applies
        it consistently to maintain category structure while introducing controlled noise.
        
        Configuration: Inject character-level noise (e.g., swaps, insertions) into 10% of 
        the instances to simulate typographical mistakes.

        Parameters:
        - text_column (str): The name of the column containing text data.
        - percentage (int): The percentage of rows to modify (default: 10).

        Returns:
        - pd.DataFrame: A new DataFrame with the noisy text data.
        """
        # Create a copy of the DataFrame to avoid modifying the original data
        modified_df = self.dataframe.copy()

        # Verify that the text column exists
        if text_column not in modified_df.columns:
            raise ValueError(f"The column {text_column} does not exist in the DataFrame.")
        if percentage < 0 or percentage > 100:
            raise ValueError("Percentage must be between 0 and 100.")

        unique_categories = modified_df[text_column].unique()
        is_categorical = len(unique_categories) <= 45  # Threshold for categorical vs free text
        
        # Calculate number of rows to modify
        num_rows = len(modified_df)
        num_rows_to_modify = int(num_rows * (percentage / 100))
        
        # Sample the rows to modify
        rows_to_modify = modified_df.sample(n=num_rows_to_modify).index
        
        if is_categorical:
            category_mutation_map = {}
            for category in unique_categories:
                if isinstance(category, str):
                    mutated_value = self._introduce_noise(category)
                    category_mutation_map[category] = mutated_value
                else:
                    category_mutation_map[category] = category
                        
            for idx in rows_to_modify:
                original_value = modified_df.at[idx, text_column]
                if original_value in category_mutation_map:
                    modified_df.at[idx, text_column] = category_mutation_map[original_value]
        
        else:
            # Original behavior for free text features
            for idx in rows_to_modify:
                if isinstance(modified_df.at[idx, text_column], str):
                    modified_df.at[idx, text_column] = self._introduce_noise(modified_df.at[idx, text_column])

        # Show comparison of changes
        self._show_comparison(self.dataframe, modified_df, rows_to_modify, text_column, "Add Noise")

        return modified_df
    
    def _synonym_replacement(self, text, word_percentage):
        """
        Replace a percentage of words in the text with their synonyms using WordNet.
        
        Parameters:
        - text (str): The input text.
        - word_percentage (int): The percentage of words to replace.
        
        Returns:
        - str: The text with some words replaced by synonyms.
        """
        words = text.split()
        num_words_to_replace = max(1, int(len(words) * (word_percentage / 100)))
        
        # Randomly select words to replace
        words_to_replace_indices = random.sample(range(len(words)), min(num_words_to_replace, len(words)))
        
        for idx in words_to_replace_indices:
            word = words[idx]
            synonyms = []
            
            # Get synonyms from WordNet
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word and synonym not in synonyms:
                        synonyms.append(synonym)
            
            # Replace the word with a random synonym if available
            if synonyms:
                words[idx] = random.choice(synonyms)
        
        return ' '.join(words)

    def _introduce_noise(self, text):
        """
        Introduce character-level noise into the text by performing random character operations.
        
        Parameters:
        - text (str): The input text.
        
        Returns:
        - str: The text with character-level noise introduced.
        """
        if len(text) < 2:
            return text
            
        text_list = list(text)
        noise_operations = ['swap', 'insert', 'delete', 'substitute']
        
        # Apply 1-3 random noise operations
        num_operations = random.randint(1, min(3, len(text_list)))
        
        for _ in range(num_operations):
            operation = random.choice(noise_operations)
            
            if operation == 'swap' and len(text_list) > 1:
                # Swap two adjacent characters
                idx = random.randint(0, len(text_list) - 2)
                text_list[idx], text_list[idx + 1] = text_list[idx + 1], text_list[idx]
                
            elif operation == 'insert':
                # Insert a random character
                idx = random.randint(0, len(text_list))
                char = random.choice(string.ascii_lowercase + ' ')
                text_list.insert(idx, char)
                
            elif operation == 'delete' and len(text_list) > 1:
                # Delete a random character
                idx = random.randint(0, len(text_list) - 1)
                text_list.pop(idx)
                
            elif operation == 'substitute':
                # Substitute a character with a random one
                idx = random.randint(0, len(text_list) - 1)
                char = random.choice(string.ascii_lowercase + ' ')
                text_list[idx] = char
        
        return ''.join(text_list)