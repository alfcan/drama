import pandas as pd
import json
import os
from typing import Dict, Any

class DataPreprocessor:
    """
    A streamlined data preprocessing module for DRAMA framework.
    
    This preprocessor implements the DRAMA framework's strict preprocessing requirements:
    1. Remove specified ignored features (e.g. identifiers)
    2. Remove all missing values from datasets
    3. Apply one-hot encoding to categorical columns
    
    No additional preprocessing steps (outlier detection, scaling, normalization, etc.) 
    are performed, ensuring compliance with DRAMA's methodology.
    
    Attributes:
        feature_types: Dictionary mapping column names to their identified types
        preprocessing_info: Tracking information about preprocessing steps applied
        config: Configuration dictionary loaded from JSON
    """
    
    def __init__(self):
        """
        Initialize the preprocessor and load dataset configuration.
        """
        self.feature_types = {}
        self.preprocessing_info = {
            'features_ignored': 0,
            'missing_handled': 0,
            'categorical_encoded': 0
        }
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load feature configuration from JSON file."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, '..', 'config', 'dataset_config.json')
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load dataset config. Using defaults. Error: {e}")
            return {"default": {"ignore_features": []}}

    def identify_feature_types(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Automatically identify and categorize feature types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping feature names to their type information
        """
        feature_types = {}
        
        for col in df.columns:
            col_info = {
                'type': None,
                'sub_type': None,
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique(),
                'dtype': str(df[col].dtype)
            }
            
            # Identify numeric features
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info['type'] = 'numeric'
                if pd.api.types.is_integer_dtype(df[col]):
                    col_info['sub_type'] = 'integer'
                else:
                    col_info['sub_type'] = 'float'
                    
            # Identify categorical features
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                col_info['type'] = 'categorical'
                if df[col].nunique() == 2:
                    col_info['sub_type'] = 'binary'
                else:
                    col_info['sub_type'] = 'nominal'
            
            # Identify boolean features
            elif pd.api.types.is_bool_dtype(df[col]):
                col_info['type'] = 'boolean'
                col_info['sub_type'] = 'boolean'
                
            feature_types[col] = col_info
            
        self.feature_types = feature_types
        return feature_types
    
    def drop_ignored_features(self, df: pd.DataFrame, dataset_name: str = None) -> pd.DataFrame:
        """
        Drop features specified in the configuration for the given dataset.
        
        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset file (e.g., 'adult.csv')
            
        Returns:
            DataFrame with ignored features removed
        """
        if dataset_name is None:
            return df
            
        ignored_features = []
        if dataset_name in self.config:
            ignored_features = self.config[dataset_name].get("ignore_features", [])
        else:
            ignored_features = self.config.get("default", {}).get("ignore_features", [])
            
        # Only try to drop features that actually exist in the dataframe
        features_to_drop = [f for f in ignored_features if f in df.columns]
        
        if features_to_drop:
            print(f"Dropping ignored features for {dataset_name}: {features_to_drop}")
            df_cleaned = df.drop(columns=features_to_drop)
            self.preprocessing_info['features_ignored'] = len(features_to_drop)
            return df_cleaned
            
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove all rows with missing values.
        
        This method implements the first required preprocessing step in DRAMA:
        complete removal of any rows containing missing values. This ensures
        data integrity without introducing bias through imputation methods.
        
        Note: This method also treats '?' values as missing values, which is
        common in some datasets (e.g., adult.csv, bank.csv).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values removed
            
        Raises:
            ValueError: If all rows are removed due to missing values
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        if df.empty:
            raise ValueError("Input DataFrame is empty")
            
        df_cleaned = df.copy()
        
        # Replace '?' with NaN to treat as missing values
        df_cleaned = df_cleaned.replace('?', pd.NA)
        
        missing_before = df_cleaned.isnull().sum().sum()
        
        if missing_before == 0:
            return df_cleaned
            
        df_cleaned = df_cleaned.dropna()
        
        if len(df_cleaned) == 0:
            raise ValueError("All rows contained missing values. Dataset is empty after preprocessing.")
                
        missing_after = df_cleaned.isnull().sum().sum()
        self.preprocessing_info['missing_handled'] = missing_before - missing_after
        
        return df_cleaned
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot encoding to categorical columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with one-hot encoded categorical features
        """
        df_encoded = df.copy()
        
        # Find categorical columns
        categorical_cols = [col for col, info in self.feature_types.items() 
                          if info['type'] == 'categorical' or info['type'] == 'boolean']
        
        if not categorical_cols:
            return df_encoded
        
        # Apply one-hot encoding
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, prefix=categorical_cols)
        
        encoded_count = len(df_encoded.columns) - len(df.columns)
        self.preprocessing_info['categorical_encoded'] = encoded_count
                
        return df_encoded
    
    def fit(self, df: pd.DataFrame):
        """Analyze the dataset to identify feature types."""
        self.feature_types = self.identify_feature_types(df)
        return self
        
    def transform(self, df: pd.DataFrame, dataset_name: str = None):
        """
        Apply required preprocessing steps: feature exclusion, missing value removal, and one-hot encoding.
        """
        df_processed = df.copy()
        
        # Step 0: Drop ignored features
        df_processed = self.drop_ignored_features(df_processed, dataset_name)
        
        # Step 1: Remove all missing values
        df_processed = self.handle_missing_values(df_processed)
        
        # Step 2: Apply one-hot encoding to categorical columns
        df_processed = self.encode_categorical_features(df_processed)
        
        return df_processed, self.preprocessing_info
    
    def fit_transform(self, df: pd.DataFrame, dataset_name: str = None):
        """
        Fit and transform the dataset in one step.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        try:
            # Note: fit() identifies types on the raw df. 
            # If we drop columns in transform, we might have identified types for dropped columns.
            # It's better to drop first, then identify types.
            
            # Custom pipeline here to ensure correct order:
            df_processed = df.copy()
            
            # 1. Drop features
            df_processed = self.drop_ignored_features(df_processed, dataset_name)
            
            # 2. Identify types on the potentially smaller dataframe
            self.fit(df_processed)
            
            # 3. Handle missing
            df_processed = self.handle_missing_values(df_processed)
            
            # 4. Encode
            df_processed = self.encode_categorical_features(df_processed)
            
            return df_processed, self.preprocessing_info
            
        except Exception as e:
            raise
            
    def clean_data_only(self, df: pd.DataFrame, dataset_name: str = None):
        """
        Apply feature exclusion and missing value removal without one-hot encoding.
        
        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset file
            
        Returns:
            tuple: (cleaned_dataframe, preprocessing_info_dict)
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        # Reset preprocessing info for this operation
        self.preprocessing_info = {
            'features_ignored': 0,
            'missing_handled': 0,
            'categorical_encoded': 0
        }
        
        df_cleaned = df.copy()
        
        # 1. Drop features
        df_cleaned = self.drop_ignored_features(df_cleaned, dataset_name)
        
        # 2. Identify feature types on the remaining data
        self.identify_feature_types(df_cleaned)
        
        # 3. Unconditionally remove missing values
        df_cleaned = self.handle_missing_values(df_cleaned)
        
        return df_cleaned, self.preprocessing_info
    
    def apply_temporary_encoding(self, df: pd.DataFrame):
        """
        Apply one-hot encoding temporarily for symptom calculation.
        
        This method applies one-hot encoding to a cleaned dataset
        for the purpose of calculating bias symptoms, but doesn't
        modify the original preprocessing info.
        
        Args:
            df: Input DataFrame (should be already cleaned)
            
        Returns:
            pd.DataFrame: DataFrame with one-hot encoded categorical features
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        # Make sure feature types are identified
        if not self.feature_types:
            self.identify_feature_types(df)
            
        # Apply one-hot encoding without modifying preprocessing_info
        df_encoded = df.copy()
        
        # Find categorical columns
        categorical_cols = [col for col, info in self.feature_types.items() 
                          if info['type'] == 'categorical' or info['type'] == 'boolean']
        
        if categorical_cols:
            # Apply one-hot encoding
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, prefix=categorical_cols)
        
        return df_encoded

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Return a summary of preprocessing steps applied.
        
        Returns:
            Dictionary containing preprocessing statistics
        """
        return {
            'preprocessing_steps_applied': ['feature_exclusion', 'missing_value_removal', 'one_hot_encoding'],
            'features_ignored': self.preprocessing_info['features_ignored'],
            'missing_values_removed': self.preprocessing_info['missing_handled'],
            'categorical_features_encoded': self.preprocessing_info['categorical_encoded'],
            'feature_types_identified': len(self.feature_types),
        }