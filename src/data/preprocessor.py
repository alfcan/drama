import pandas as pd
from typing import Dict, Any

class DataPreprocessor:
    """
    A streamlined data preprocessing module for DRAMA framework.
    
    This preprocessor implements the DRAMA framework's strict preprocessing requirements:
    1. Remove all missing values from datasets
    2. Apply one-hot encoding to categorical columns
    
    No additional preprocessing steps (outlier detection, scaling, normalization, etc.) 
    are performed, ensuring compliance with DRAMA's methodology.
    
    Attributes:
        feature_types: Dictionary mapping column names to their identified types
        preprocessing_info: Tracking information about preprocessing steps applied
    """
    
    def __init__(self):
        """
        Initialize the preprocessor with only required preprocessing steps.
        
        The preprocessor is configured to perform only the two mandatory steps:
        missing value removal and one-hot encoding for categorical variables.
        """
        self.feature_types = {}
        self.preprocessing_info = {
            'missing_handled': 0,
            'categorical_encoded': 0
        }
        
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
        
    def transform(self, df: pd.DataFrame):
        """Apply only the required preprocessing steps: missing value removal and one-hot encoding."""
        df_processed = df.copy()
        
        # Step 1: Remove all missing values
        df_processed = self.handle_missing_values(df_processed)
        
        # Step 2: Apply one-hot encoding to categorical columns
        df_processed = self.encode_categorical_features(df_processed)
        
        return df_processed, self.preprocessing_info
    
    def fit_transform(self, df: pd.DataFrame):
        """
        Fit and transform the dataset in one step.
        
        This convenience method performs both the analysis and preprocessing
        steps required by the DRAMA framework. It identifies feature types,
        removes missing values, and applies one-hot encoding to categorical
        features.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            tuple: (processed_dataframe, preprocessing_info_dict)
            
        Raises:
            TypeError: If input is not a pandas DataFrame
            ValueError: If preprocessing results in empty DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        try:
            return self.fit(df).transform(df)
        except Exception as e:
            raise
            
    def clean_data_only(self, df: pd.DataFrame):
        """
        Apply only missing value removal without one-hot encoding.
        
        This method is used when we need to apply mutations on raw data
        but still need clean data without categorical encoding.
        
        Args:
            df: Input DataFrame
            
        Returns:
            tuple: (cleaned_dataframe, preprocessing_info_dict)
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        # Reset preprocessing info for this operation
        self.preprocessing_info = {
            'missing_handled': 0,
            'categorical_encoded': 0
        }
        
        # Identify feature types
        self.identify_feature_types(df)
        
        # Only remove missing values
        df_cleaned = self.handle_missing_values(df)
        
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
            'preprocessing_steps_applied': ['missing_value_removal', 'one_hot_encoding'],
            'missing_values_removed': self.preprocessing_info['missing_handled'],
            'categorical_features_encoded': self.preprocessing_info['categorical_encoded'],
            'feature_types_identified': len(self.feature_types),
        }