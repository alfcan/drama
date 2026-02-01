import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, kendalltau
from sklearn.metrics import mutual_info_score

# --- Helper Functions for Required Metrics ---

def calculate_gini_inequality(x: pd.Series) -> float:
    """
    Calculates the Gini Coefficient of Inequality for the frequency distribution.
    0 = Perfectly balanced (all categories have equal frequency).
    1 = Maximum inequality (one category dominates).
    Formula: G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mean(x))
    """
    # Get counts of each category
    counts = x.value_counts().values
    if len(counts) == 0:
        return 0.0
    
    # Gini coefficient calculation on the counts
    # Sort counts
    counts = np.sort(counts)
    n = len(counts)
    
    # Mean count
    mean_count = np.mean(counts)
    if mean_count == 0:
        return 0.0
        
    # Gini calculation
    # Using the formula: G = (2 * sum(i * x_i) / (n * sum(x_i))) - (n + 1) / n
    # where i is 1-based index
    index = np.arange(1, n + 1)
    g = (2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n + 1) / n
    return g

def calculate_shannon_entropy(x: pd.Series) -> float:
    """
    Calculates Shannon Entropy.
    """
    counts = x.value_counts(normalize=True)
    if len(counts) <= 1:
        return 0.0
    entropy = -np.sum(counts * np.log(counts + 1e-12))
    return entropy

def calculate_simpson_index(x: pd.Series) -> float:
    """
    Calculates Simpson Diversity Index (lambda).
    Measure the probability that two randomly selected entities represent the same type.
    Formula: lambda = sum(p_i^2)
    Range: [1/S, 1] where S is number of species/categories.
    """
    counts = x.value_counts(normalize=True).values
    lambda_val = np.sum(counts ** 2)
    return lambda_val

def calculate_imbalance_ratio(x: pd.Series) -> float:
    """
    Calculates Imbalance Ratio (IR).
    IR = min(frequency_counts) / max(frequency_counts).
    Range: [0, 1], where 1 is perfectly balanced.
    """
    counts = x.value_counts().values
    if len(counts) == 0:
        return 0.0
    return counts.min() / counts.max()

# --- Main SymptomCalculator Class ---

class SymptomCalculator:
    """
    A class to calculate strict Bias Symptoms for the separate FADE framework module.
    """

    def __init__(self, df: pd.DataFrame, target_attribute: str, positive_label: int = 1):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if target_attribute not in df.columns:
            raise ValueError(f"Target attribute '{target_attribute}' not found in DataFrame columns")
        
        self.df = df.copy()
        self.target_attribute = target_attribute
        self.positive_label = positive_label

    def infer_group_conditions(self, feature_name: str) -> tuple[str, str]:
        """
        Infer unprivileged and privileged groups based on target label Y.
        """
        if feature_name not in self.df.columns:
            raise ValueError(f"Feature '{feature_name}' not found.")
            
        feature_series = self.df[feature_name]
        y_series = self.df[self.target_attribute]
        
        # Determine if binary/categorical or continuous
        # Heuristic: < 20 unique values -> Categorical/Binary
        is_numeric = pd.api.types.is_numeric_dtype(feature_series)
        unique_count = feature_series.nunique()
        
        # Logic 1: Numerical Features (Continuous)
        if is_numeric and unique_count > 20: 
            # Split at Median
            median_val = feature_series.median()
            
            # Groups: <= Median and > Median
            mask_low = feature_series <= median_val
            mask_high = feature_series > median_val
            
            if mask_low.sum() == 0 or mask_high.sum() == 0:
                 return None, None # Cannot split

            rate_low = y_series[mask_low].mean() if mask_low.sum() > 0 else 0
            rate_high = y_series[mask_high].mean() if mask_high.sum() > 0 else 0
            
            # Unprivileged = Lower Rate
            if rate_low < rate_high:
                return f"`{feature_name}` <= {median_val}", f"`{feature_name}` > {median_val}"
            else:
                return f"`{feature_name}` > {median_val}", f"`{feature_name}` <= {median_val}"
                
        # Logic 2: Categorical / Binary / Textual
        else:
            # Calculate positive rate for every category
            # Ensure we handle all valid categories present in data
            means = self.df.groupby(feature_name)[y_series.name].mean()
            
            if means.empty:
                return None, None
                
            # Lowest rate = Unprivileged
            unprivileged_cat = means.idxmin()
            # Highest rate = Privileged
            privileged_cat = means.idxmax()
            
            if unprivileged_cat == privileged_cat:
                return None, None # No disparity found or only 1 category
                
            # Quote strings if needed
            def format_val(val):
                return f"'{val}'" if isinstance(val, str) else val
                
            return f"`{feature_name}` == {format_val(unprivileged_cat)}", f"`{feature_name}` == {format_val(privileged_cat)}"

    def _get_groups(self, unpriv_query, priv_query):
        """Helper to get dataframes for groups."""
        try:
            unpriv_df = self.df.query(unpriv_query)
            priv_df = self.df.query(priv_query)
            return unpriv_df, priv_df
        except Exception:
            return pd.DataFrame(), pd.DataFrame()

    def calculate_symptoms_for_feature(self, feature_name: str) -> dict:
        """
        Calculate the 12 specific bias symptoms.
        """
        symptoms = {}
        feature_series = self.df[feature_name]
        target_series = self.df[self.target_attribute]
        
        # --- Category A: Distributional and Diversity Metrics ---
        
        symptoms['Gini Index'] = calculate_gini_inequality(feature_series)
        symptoms['Shannon Entropy'] = calculate_shannon_entropy(feature_series)
        symptoms['Simpson Diversity'] = calculate_simpson_index(feature_series)
        symptoms['Imbalance Ratio'] = calculate_imbalance_ratio(feature_series)
        
        is_numeric = pd.api.types.is_numeric_dtype(feature_series)
        
        if is_numeric:
            symptoms['Kurtosis'] = kurtosis(feature_series, fisher=False, bias=False)
            symptoms['Skewness'] = skew(feature_series, bias=False)
        else:
            symptoms['Kurtosis'] = None
            symptoms['Skewness'] = None
            
        # --- Category B: Feature-Target Relationship Metrics ---
        
        # Mutual Information
        # Note: mutual_info_score expects cluster labels. 
        # If feature is continuous, this might need binning or mutual_info_classif.
        # However, user said "Use sklearn...mutual_info_classif".
        # But mutual_info_classif requires 2D array X.
        # Let's use mutual_info_score for discrete/categorical, 
        # For continuous, we should use mutual_info_classif properly?
        # User prompt: "Use sklearn.feature_selection.mutual_info_classif"
        
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        
        # Reshape for sklearn
        X = feature_series.values.reshape(-1, 1)
        y = target_series.values
        
        # Handle nan
        if feature_series.isnull().any():
             # Fill w/ median or mode just for calculation if needed, or drop
             # For now, let's assume pre-cleaned or skip
             pass
             
        try:
            if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 20: 
                 # Regression target?
                 mi = mutual_info_regression(X, y, discrete_features=not is_numeric)
            else:
                 # Classification target
                 # discrete_features='auto' usually handles it, but let's be safe
                 # If feature is object/string, we need to encode it for sklearn MI? 
                 # Sklearn MI only accepts numerical X usually?
                 # Actually mutual_info_classif handles pandas? No, usually expects numeric input or encoded.
                 
                 # Quick Encoding for MI calculation if non-numeric
                 if not is_numeric:
                     X_encoded = feature_series.astype('category').cat.codes.values.reshape(-1, 1)
                     mi = mutual_info_classif(X_encoded, y, discrete_features=True)
                 else:
                     mi = mutual_info_classif(X, y)
                     
            symptoms['Mutual Information'] = mi[0]
        except Exception:
            symptoms['Mutual Information'] = 0.0

        if is_numeric:
            try:
                # Kendall's Tau
                kt = kendalltau(feature_series, target_series)
                symptoms['Kendall Tau'] = kt.correlation
            except:
                symptoms['Kendall Tau'] = None
        else:
             symptoms['Kendall Tau'] = None

        # --- Category C: Ground-Truth Fairness Metrics ---
        
        unpriv_query, priv_query = self.infer_group_conditions(feature_name)
        
        if unpriv_query and priv_query:
            unpriv_df, priv_df = self._get_groups(unpriv_query, priv_query)
            
            if len(unpriv_df) > 0:
                p_unpriv = unpriv_df[self.target_attribute].mean() # P(Y=1 | Unpriv)
            else:
                p_unpriv = 0.0
                
            if len(priv_df) > 0:
                p_priv = priv_df[self.target_attribute].mean() # P(Y=1 | Priv)
            else:
                p_priv = 0.0
                
            # 9. Data Statistical Parity (DSP)
            symptoms['Data Statistical Parity'] = abs(p_unpriv - p_priv)
            
            # 10. UPP
            symptoms['Unprivileged Positive Probability'] = p_unpriv
            
            # 11. PPP
            symptoms['Privileged Positive Probability'] = p_priv
            
            # 12. Group Unbalance
            rate_global = target_series.mean()
            if rate_global > 0:
                 symptoms['Unprivileged Group Unbalance'] = p_unpriv / rate_global
                 symptoms['Privileged Group Unbalance'] = p_priv / rate_global
            else:
                 symptoms['Unprivileged Group Unbalance'] = 0.0
                 symptoms['Privileged Group Unbalance'] = 0.0
                 
        else:
            # Set None if groups couldn't be inferred
            symptoms['Data Statistical Parity'] = None
            symptoms['Unprivileged Positive Probability'] = None
            symptoms['Privileged Positive Probability'] = None
            symptoms['Unprivileged Group Unbalance'] = None
            symptoms['Privileged Group Unbalance'] = None
            
        return symptoms