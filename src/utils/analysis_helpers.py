import pandas as pd


def get_applicable_operators(column, df):
    """Determine applicable mutation operators based on column data type."""
    if pd.api.types.is_numeric_dtype(df[column]):
        return ['increment_decrement_feature', 'scale_values']
    else:
        return ['category_flip', 'replace_synonyms', 'add_noise']


def get_user_defined_conditions(sensitive_attribute, df):
    """
    Get user-defined privileged and unprivileged conditions for a sensitive attribute.
    
    Args:
        sensitive_attribute: Name of the sensitive attribute
        df: DataFrame to analyze
        
    Returns:
        tuple: (privileged_condition, unprivileged_condition)
    """
    print(f"\n--- Defining groups for sensitive attribute: {sensitive_attribute} ---")
    
    # Show unique values for the sensitive attribute to help user
    if sensitive_attribute in df.columns:
        unique_values = df[sensitive_attribute].unique()
        print(f"Unique values in '{sensitive_attribute}': {list(unique_values)}")
    else:
        # Check for one-hot encoded columns
        encoded_cols = [col for col in df.columns if col.startswith(f"{sensitive_attribute}_")]
        if encoded_cols:
            print(f"One-hot encoded columns found for '{sensitive_attribute}': {encoded_cols}")
            print("For one-hot encoded attributes, use format like: `attribute_value` == 1")
        else:
            print(f"Warning: '{sensitive_attribute}' not found in dataset columns")
    
    print("\nPlease define the conditions using pandas query syntax.")
    print("Examples:")
    print("  - For categorical: `gender` == 'Male'")
    print("  - For numerical: `age` >= 30")
    print("  - For one-hot encoded: `race_White` == 1")
    
    privileged_condition = input(f"Enter the condition for the PRIVILEGED group: ").strip()
    unprivileged_condition = input(f"Enter the condition for the UNPRIVILEGED group: ").strip()
    
    # Validate conditions by testing them on the dataframe
    try:
        priv_count = len(df.query(privileged_condition))
        unpriv_count = len(df.query(unprivileged_condition))
        print(f"Privileged group size: {priv_count} samples")
        print(f"Unprivileged group size: {unpriv_count} samples")
        
        if priv_count == 0:
            print("Warning: Privileged condition returns 0 samples!")
        if unpriv_count == 0:
            print("Warning: Unprivileged condition returns 0 samples!")
            
    except Exception as e:
        print(f"Error validating conditions: {e}")
        print("Please check your syntax and try again.")
        return get_user_defined_conditions(sensitive_attribute, df)
    
    return privileged_condition, unprivileged_condition