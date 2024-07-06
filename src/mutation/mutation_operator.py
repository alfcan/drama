import numpy as np
import pandas as pd
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import nltk
from nltk.corpus import wordnet
import string
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm


nltk.download('wordnet')

class MutationOperator:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        # Initialise the paraphrase model
        self.tokenizer = T5Tokenizer.from_pretrained("prithivida/parrot_paraphraser_on_T5")
        self.model = T5ForConditionalGeneration.from_pretrained("prithivida/parrot_paraphraser_on_T5")
        self.paraphraser = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)

    def increment_decrement_feature(self, feature_name, increment=True, amount=1, percentage=100):
        """
        Increment or decrement the value of a numerical feature by a fixed amount on a random percentage of rows.

        Parameters:
        - feature_name (str): The name of the feature to be modified.
        - increment (bool): True if incrementing the value, False if decrementing.
        - amount (float): The amount by which to increment or decrement the feature values.
        - percentage (float): The percentage of rows to be modified.

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
        
        # Calculate the number of rows to modify
        num_rows = len(modified_df)
        num_rows_to_modify = int(num_rows * (percentage / 100.0))
        
        # Randomly select the indices of the rows to modify
        rows_to_modify = np.random.choice(modified_df.index, num_rows_to_modify, replace=False)
        
        # Save the original dtype of the column
        original_dtype = modified_df[feature_name].dtype

        # Cast the column to a float before modifying it
        if not np.issubdtype(modified_df[feature_name].dtype, np.number):
            raise ValueError(f"The feature {feature_name} is not a numerical type.")

        modified_df[feature_name] = modified_df[feature_name].astype(float)

        # Apply the increment or decrement to the selected rows
        if increment:
            modified_df.loc[rows_to_modify, feature_name] += amount
        else:
            modified_df.loc[rows_to_modify, feature_name] -= amount

         # Convert the column back to its original dtype
        modified_df[feature_name] = modified_df[feature_name].astype(original_dtype)
        
        return modified_df
    
    def swap_values(self, feature_name, num_swaps):
        """
        Randomly swap values between pairs of instances for a specified feature.

        Parameters:
        - feature_name (str): The name of the feature for which values are to be swapped.
        - num_swaps (int): The number of swaps to perform.

        Returns:
        - pd.DataFrame: A new DataFrame with swapped values.
        """
        # Create a copy of the DataFrame to avoid modifying the original data
        modified_df = self.dataframe.copy()
        
        # Check if the feature_name is valid
        if feature_name not in modified_df.columns:
            raise ValueError(f"The feature {feature_name} does not exist in the DataFrame.")
        if num_swaps > len(self.dataframe):
            raise ValueError("Number of swaps cannot exceed the number of rows in the DataFrame.")
        
        for _ in range(num_swaps):
            # Randomly pick two indices to swap
            idx1, idx2 = random.sample(range(len(modified_df)), 2)
            # Swap the values
            modified_df.at[idx1, feature_name], modified_df.at[idx2, feature_name] = modified_df.at[idx2, feature_name], modified_df.at[idx1, feature_name]

        return modified_df

    def scale_values(self, feature_name, scale_factor=None, scale_range=None, percentage=100):
        """
        Multiply or divide the values of a numeric feature by a random or fixed scale factor.

        Parameters:
        - feature_name (str): The name of the feature to be scaled.
        - scale_factor (float, optional): A fixed factor by which to scale the feature values.
        - scale_range (tuple, optional): A range (min, max) from which to randomly pick a scale factor.
        - percentage (float): The percentage of rows to be modified.

        Returns:
        - pd.DataFrame: A new DataFrame with scaled values.
        """
        # Create a copy of the DataFrame to avoid modifying the original data
        modified_df = self.dataframe.copy()
        
        # Validate the feature_name and scale parameters
        if feature_name not in modified_df.columns:
            raise ValueError(f"The feature {feature_name} does not exist in the DataFrame.")
        if scale_factor is None and scale_range is None:
            raise ValueError("Either scale_factor or scale_range must be specified.")
        if percentage < 0 or percentage > 100:
            raise ValueError("Percentage must be between 0 and 100.")

        # Determine the scale factor if a range is provided
        if scale_range is not None:
            scale_factor = random.uniform(*scale_range)
        
         # Calculate the number of rows to modify
        num_rows_to_modify = int(len(modified_df) * (percentage / 100.0))
        rows_to_modify = np.random.choice(modified_df.index, num_rows_to_modify, replace=False)

        # Apply the scaling
        modified_df[feature_name] = modified_df[feature_name] * scale_factor
        
        return modified_df
    
    def discrete_binning(self, feature_name, bins):
        """
        Convert continuous values of a feature into discrete categories based on specified bins.

        Parameters:
        - feature_name (str): The name of the feature to be binned.
        - bins (list or array): The edges defining the bins for discretization (e.g. bins = [0, 3, 6, 9], to have three bins: [0, 3], [3, 6], [6, 9]).

        Returns:
        - pd.DataFrame: A new DataFrame with the binned feature.
        """
        # Create a copy of the DataFrame to avoid modifying the original data
        modified_df = self.dataframe.copy()
        
        # Validate the feature_name
        if feature_name not in modified_df.columns:
            raise ValueError(f"The feature {feature_name} does not exist in the DataFrame.")
        
        # Apply the discretization
        modified_df[feature_name] = pd.cut(modified_df[feature_name], bins, labels=False, include_lowest=True)
        
        return modified_df
    
    def augment_text(self, text_column, percentage=10, batch_size=10):
        """
        Paraphrase text data in the specified column to introduce variability.

        Parameters:
        - text_column (str): The name of the column containing text data to be paraphrased.
        - percentage (int): The percentage of text data to paraphrase.
        - batch_size (int): The number of texts to paraphrase in a batch.

        Returns:
        - pd.DataFrame: A new DataFrame with the paraphrased text data.
        """
        # Create a copy of the DataFrame to avoid modifying the original data
        modified_df = self.dataframe.copy()

        # Verify that the text column exists
        if text_column not in modified_df.columns:
            raise ValueError(f"The column {text_column} does not exist in the DataFrame.")
        if percentage < 0 or percentage > 100:
            raise ValueError("Percentage must be between 0 and 100.")

        # Determine the number of rows to paraphrase based on the given percentage
        num_rows = len(modified_df)
        num_to_replace = int(num_rows * (percentage / 100))

        # Sample the rows to paraphrase
        rows_to_paraphrase = modified_df.sample(n=num_to_replace).index

         # Apply paraphrase function to the sampled rows in batches
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = []
            for i in range(0, num_to_replace, batch_size):
                batch_indices = rows_to_paraphrase[i:i + batch_size]
                batch_texts = modified_df.loc[batch_indices, text_column].tolist()
                futures.append(executor.submit(self._paraphrase_texts_batch, batch_texts))
            
            for future in tqdm(futures, desc="Paraphrasing", unit="batch"):
                paraphrased_texts = future.result()
                batch_indices = rows_to_paraphrase[futures.index(future) * batch_size : (futures.index(future) + 1) * batch_size]
                for j, idx in enumerate(batch_indices):
                    modified_df.at[idx, text_column] = paraphrased_texts[j]  
         
        return modified_df
    
    def replace_synonyms(self, text_column, row_percentage=10, word_percentage=10):
        """
        Replace a percentage of words in the text data with their synonyms.

        Parameters:
        - text_column (str): The name of the column containing text data.
        - row_percentage (int): The percentage of rows to modify.
        - word_percentage (int): The percentage of words to replace with synonyms within each selected row.

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

        return modified_df
    
    def add_noise(self, text_column, noise_chance=0.1):
        """
        Add typographical errors or random punctuation to text data.

        Parameters:
        - text_column (str): The name of the column containing text data.
        - noise_chance (float): The probability of introducing noise to each word (default is 10%).

        Returns:
        - pd.DataFrame: A new DataFrame with the noisy text data.
        """
        # Create a copy of the DataFrame to avoid modifying the original data
        modified_df = self.dataframe.copy()

        # Verify that the text column exists
        if text_column not in modified_df.columns:
            raise ValueError(f"The column {text_column} does not exist in the DataFrame.")

        modified_df[text_column] = modified_df[text_column].apply(self._introduce_noise, noise_chance=noise_chance)
        return modified_df
    
    def random_category_assignment(self, category_column, percentage):
        """
        Randomly assign a new category generated by an LLM to a specified percentage of rows in a feature.

        Parameters:
        - category_column (str): The name of the categorical column to modify.
        - percentage (int): The percentage of rows to modify.

        Returns:
        - pd.DataFrame: A new DataFrame with the modified categories.
        """
        # Create a copy of the DataFrame to avoid modifying the original data
        modified_df = self.dataframe.copy()

        # Verify that the category column exists
        if category_column not in modified_df.columns:
            raise ValueError(f"The column {category_column} does not exist in the DataFrame.")
        if percentage < 0 or percentage > 100:
            raise ValueError("Percentage must be between 0 and 100.")
    
        # Generate a new category
        existing_categories = modified_df[category_column].unique().tolist()
        random_index = random.randint(0, len(existing_categories) - 1)
        new_category = 'new_' + existing_categories[random_index]

        # Select random rows to modify
        num_rows_to_modify = int(len(modified_df) * (percentage / 100))
        rows_to_modify = random.sample(list(modified_df.index), num_rows_to_modify)

        # Assign the new category to the selected rows
        modified_df.loc[rows_to_modify, category_column] = new_category

        return modified_df

    def _paraphrase_texts_batch(self, texts, min_length=10, max_length=50):
        paraphrased_texts = []
        for text in texts:
            paraphrased_texts.append(self._paraphrase_text(text, min_length, max_length))
        return paraphrased_texts

    def _paraphrase_text(self, text, min_length=10, max_length=50):
        """
        Generate a paraphrased version of the input text using a pre-trained model.

        Parameters:
        - text (str): The text to be paraphrased.
        - min_length (int): The minimum length of the generated paraphrase.
        - max_length (int): The maximum length of the generated paraphrase.

        Returns:
        - str: The paraphrased text.
        """
        # Ensure min_length is less than max_length
        if min_length >= max_length:
            raise ValueError("min_length must be less than max_length.")
        
        try:
            paraphrased_results = self.paraphraser(text, min_length=min_length, max_length=max_length, truncation=True)
            paraphrased_text = paraphrased_results[0]['generated_text']
        except Exception as e:
            print(f"Error paraphrasing text: {e}")
            paraphrased_text = text  # Return original text in case of error
        return paraphrased_text

    def _synonym_replacement(self, text, percentage):
        """
        Replace a percentage of words in the input text with their synonyms using WordNet.

        Parameters:
        - text (str): The text in which words will be replaced with synonyms.
        - percentage (int): The percentage of words to replace with synonyms.

        Returns:
        - str: The text with a percentage of words replaced by their synonyms.
        """

        # Split the text into words
        words = text.split()
        num_words = len(words)
        # Determine the number of words to replace based on the given percentage
        num_to_replace = max(1, int(num_words * (percentage / 100)))
        # Randomly select the indices of the words to be replaced
        indices = random.sample(range(num_words), k=num_to_replace)
        # Create a copy of the words to modify
        new_words = words[:]
        
        for i in indices:
            # Get the synsets (sets of synonyms) for the current word
            synsets = wordnet.synsets(words[i])    
            
            # If there are synsets available
            if synsets:
                # Extract the synonyms from the first synset, excluding the original word
                synonyms = [lemma.name() for lemma in synsets[0].lemmas() if lemma.name() != words[i]]
                
                # If there are synonyms available
                if synonyms:
                    # Replace the word with a randomly chosen synonym
                    new_words[i] = random.choice(synonyms).replace('_', ' ')
        
        # Join the modified words back into a single string and return it
        return ' '.join(new_words)

    def _introduce_noise(self, text, noise_chance):
        """
        Introduce random punctuation or spaces into the input text to add noise without changing the semantics.

        Parameters:
        - text (str): The text to introduce noise into.
        - noise_chance (float): The probability of introducing noise to each word.

        Returns:
        - str: The text with introduced noise.
        """
        words = text.split()
        noisy_words = []
        for word in words:
            if random.random() < noise_chance:
                # Add random punctuation or space
                if random.choice([True, False]):
                    # Add random punctuation at the end of the word
                    word += random.choice(string.punctuation)
                else:
                    # Add random space within the word (if it's longer than 1 character)
                    if len(word) > 1:
                        index = random.randint(1, len(word) - 1)
                        word = word[:index] + ' ' + word[index:]
            noisy_words.append(word)
        return ' '.join(noisy_words)