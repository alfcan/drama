import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, kendalltau
from sklearn.metrics import mutual_info_score

# --- Helper Functions for Diversity/Distribution Metrics ---
# These functions are designed to operate on a single pandas Series.

def gini_diversity_index(x: pd.Series) -> float:
    """
    Calculates the Gini diversity index (similar to 1 - sum(p_i^2)).
    A value of 0 indicates complete dominance by one category, while 1 indicates high diversity.
    Normalized for the number of categories.
    """
    counts = x.value_counts(normalize=True).values
    m = len(counts)
    if m <= 1:
        # If there's only one category, there's no diversity, so often 0 or 1 depending on normalization.
        # Here, 1.0 implies 'no imbalance' or 'perfectly balanced in a trivial sense'.
        return 1.0
    f2 = np.sum(counts ** 2)
    # The (m / (m - 1)) factor normalizes it, so it can reach 1.0 for maximum diversity.
    return (m / (m - 1)) * (1.0 - f2) if (m - 1) > 0 else 0.0

def shannon_entropy_index(x: pd.Series) -> float:
    """
    Calculates the normalized Shannon Entropy.
    Normalized to range [0, 1], where 0 means no diversity (one category dominates)
    and 1 means maximum diversity (all categories equally frequent).
    """
    counts = x.value_counts(normalize=True)
    if len(counts) <= 1:
        # If only one category, entropy is 0, but normalized to 1.0 for consistency with 'balanced'
        return 1.0
    
    # Calculate Shannon entropy (using natural logarithm)
    entropy = -np.sum(counts * np.log(counts + 1e-12)) # Add epsilon to avoid log(0)
    
    # Calculate maximum possible entropy for the given number of categories
    max_entropy = np.log(len(counts))
    
    # Normalize entropy
    return entropy / max_entropy if max_entropy > 0 else 0.0

def simpson_diversity_index(x: pd.Series) -> float:
    """
    Calculates the normalized Simpson Diversity Index (1/sum(p_i^2)).
    A value near 0 means low diversity, near 1 means high diversity.
    Normalized for the number of categories.
    """
    counts = x.value_counts(normalize=True).values
    m = len(counts)
    if m <= 1:
        # If only one category, no diversity, 1.0 for consistency.
        return 1.0
    f2 = np.sum(counts ** 2)
    if f2 <= 0: # Avoid division by zero if all counts are zero (unlikely with normalize=True)
        return 1.0
    
    # The (1.0 / (m - 1)) factor normalizes it.
    return (1.0 / (m - 1)) * ((1.0 / f2) - 1.0) if (m - 1) > 0 else 0.0

# --- Main SymptomCalculator Class ---

class SymptomCalculator:
    """
    A class to calculate various fairness and distribution symptoms for a given dataset,
    treating individual features as potential sensitive attributes.
    It focuses on dataset-level characteristics, not model predictions.
    """

    def __init__(self, df: pd.DataFrame, target_attribute: str, positive_label: int = 1, original_df: pd.DataFrame = None):
        """
        Initialize the SymptomCalculator with a DataFrame and target information.
        
        Args:
            df: Input DataFrame (should be already cleaned and encoded if needed)
            target_attribute: Name of the target column
            positive_label: Value representing the positive class in the target
            original_df: Optional original DataFrame before encoding (for feature analysis)
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if target_attribute not in df.columns:
            raise ValueError(f"Target attribute '{target_attribute}' not found in DataFrame columns")
        
        self.df = df.copy()
        self.target_attribute = target_attribute
        self.positive_label = positive_label
        
        # Store original column names for reference
        self.original_columns = list(df.columns)
        
        # Use provided original_df or fallback to the encoded df
        if original_df is not None:
            self.original_df = original_df.copy()
        else:
            self.original_df = df.copy()  # Keep original for reference
            
        # Store mapping of original categorical columns to their encoded versions
        self.categorical_mappings = {}

    def _get_original_feature_name(self, encoded_feature_name: str) -> str:
        """Get the original feature name from an encoded feature name."""
        for original_col, mapping in self.categorical_mappings.items():
            if encoded_feature_name in mapping['encoded_columns']:
                return original_col
        return encoded_feature_name  # If not encoded, return as is

    def infer_group_conditions(self, feature_name: str) -> tuple[str, str]:
        """
        Automatically infers unprivileged and privileged group conditions for a given feature.
        Works with both numerical and one-hot encoded categorical features.
        
        Args:
            feature_name: Name of the feature to analyze
            
        Returns:
            tuple: (unprivileged_condition, privileged_condition) as pandas query strings
        """
        if feature_name not in self.df.columns:
            raise ValueError(f"Feature '{feature_name}' not found in the dataset columns: {list(self.df.columns)}")
        
        feature_series = self.df[feature_name]
        
        # Check if this is a binary/boolean feature (likely one-hot encoded)
        unique_values = feature_series.unique()
        if len(unique_values) == 2 and set(unique_values).issubset({0, 1, True, False}):
            # Binary feature: privileged = 1/True, unprivileged = 0/False
            privileged_condition = f"`{feature_name}` == 1"
            unprivileged_condition = f"`{feature_name}` == 0"
            return unprivileged_condition, privileged_condition
        
        # Check if this is a categorical feature with few unique values
        elif len(unique_values) <= 10 and not pd.api.types.is_numeric_dtype(feature_series):
            # Categorical feature: find categories with lowest/highest positive outcome rates
            category_means = {}
            for category in unique_values:
                mask = feature_series == category
                if mask.sum() > 0:  # Ensure category has instances
                    category_means[category] = self.df[mask][self.target_attribute].mean()
            
            if not category_means:
                raise ValueError(f"No valid categories found for feature '{feature_name}'")
            
            # Find privileged (highest mean) and unprivileged (lowest mean) categories
            privileged_category = max(category_means.keys(), key=lambda k: category_means[k])
            unprivileged_category = min(category_means.keys(), key=lambda k: category_means[k])
            
            # Handle string categories that might need quotes
            if isinstance(privileged_category, str):
                privileged_condition = f"`{feature_name}` == '{privileged_category}'"
                unprivileged_condition = f"`{feature_name}` == '{unprivileged_category}'"
            else:
                privileged_condition = f"`{feature_name}` == {privileged_category}"
                unprivileged_condition = f"`{feature_name}` == {unprivileged_category}"
            
            return unprivileged_condition, privileged_condition
        
        # Numerical feature: split at median
        else:
            median_value = feature_series.median()
            
            # Calculate mean target value for each half
            below_median_mask = feature_series <= median_value
            above_median_mask = feature_series > median_value
            
            # Ensure both groups have instances
            if below_median_mask.sum() == 0 or above_median_mask.sum() == 0:
                raise ValueError(f"Cannot split feature '{feature_name}' at median - one group would be empty")
            
            below_median_mean = self.df[below_median_mask][self.target_attribute].mean()
            above_median_mean = self.df[above_median_mask][self.target_attribute].mean()
            
            # Assign privileged/unprivileged based on positive outcome rates
            if above_median_mean >= below_median_mean:
                privileged_condition = f"`{feature_name}` > {median_value}"
                unprivileged_condition = f"`{feature_name}` <= {median_value}"
            else:
                privileged_condition = f"`{feature_name}` <= {median_value}"
                unprivileged_condition = f"`{feature_name}` > {median_value}"
            
            return unprivileged_condition, privileged_condition

    def _get_groups_by_query(self, unprivileged_query: str, privileged_query: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Helper method to divide the DataFrame into unprivileged and privileged groups
        based on provided Pandas query strings.

        Parameters:
        - unprivileged_query (str): A Pandas query string defining the unprivileged group.
                                    E.g., "`age` < 30", "`gender` == 'Female'".
        - privileged_query (str): A Pandas query string defining the privileged group.

        Returns:
        - tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
            (unprivileged_group_df, unprivileged_group_positive_series, privileged_group_df, privileged_group_positive_series)
        """
        # Execute queries to get group DataFrames
        unpriv_group_df = self.df.query(unprivileged_query)
        priv_group_df = self.df.query(privileged_query)
        
        # Filter for positive outcomes within each group, returning a Series of positive labels.
        # This approach ensures consistency with len() for counting positive instances.
        unpriv_group_positive_series = unpriv_group_df[self.target_attribute][unpriv_group_df[self.target_attribute] == self.positive_label]
        priv_group_positive_series = priv_group_df[self.target_attribute][priv_group_df[self.target_attribute] == self.positive_label]
        
        return unpriv_group_df, unpriv_group_positive_series, priv_group_df, priv_group_positive_series

    def _compute_probs_by_query(self, unprivileged_query: str, privileged_query: str) -> tuple[float, float]:
        """
        Calculates the probability of a positive outcome for the unprivileged and privileged groups.

        Parameters:
        - unprivileged_query (str): Query string for the unprivileged group.
        - privileged_query (str): Query string for the privileged group.

        Returns:
        - tuple[float, float]: (unprivileged_group_probability, privileged_group_probability)
        """
        unpriv_group_df, unpriv_group_positive_series, priv_group_df, priv_group_positive_series = \
            self._get_groups_by_query(unprivileged_query, privileged_query)
        
        # Calculate probabilities, handling division by zero
        unpriv_prob = len(unpriv_group_positive_series) / len(unpriv_group_df) if len(unpriv_group_df) > 0 else 0.0
        priv_prob = len(priv_group_positive_series) / len(priv_group_df) if len(priv_group_df) > 0 else 0.0
        
        return unpriv_prob, priv_prob

    def infer_group_conditions(self, feature_name: str) -> tuple[str, str]:
        """
        Automatically infers unprivileged and privileged group conditions (as query strings)
        for a given feature.

        - For numerical features: splits data at the median. The half with the lower
          positive outcome rate is considered unprivileged.
        - For categorical features: identifies categories with the highest and lowest
          positive outcome rates, designating the lowest as unprivileged.

        Parameters:
        - feature_name (str): The name of the feature to be treated as a sensitive attribute.

        Returns:
        - tuple[str, str]: (unprivileged_query_string, privileged_query_string)

        Raises:
        - ValueError: If the feature is not found or if groups cannot be meaningfully inferred.
        """
        if feature_name not in self.df.columns:
            raise ValueError(f"Feature '{feature_name}' not found in the dataset.")
            
        series = self.df[feature_name]
        y_series = self.df[self.target_attribute]

        # Handle numerical attributes
        if pd.api.types.is_numeric_dtype(series):
            # If numerical and effectively binary, treat as categorical with 2 values
            if series.nunique() == 2:
                val1, val2 = series.unique()
                mean1 = y_series[series == val1].mean()
                mean2 = y_series[series == val2].mean()
                
                if mean1 < mean2:
                    return f"`{feature_name}` == {val1}", f"`{feature_name}` == {val2}"
                else:
                    return f"`{feature_name}` == {val2}", f"`{feature_name}` == {val1}"
            
            # For continuous/multi-valued numerical attributes, split by median
            median_value = series.median()
            
            # Handle potential empty groups if median split is extreme (e.g., all values same)
            # Ensure there's data in both halves before calculating means
            low_group_mask = series < median_value
            high_group_mask = series >= median_value

            mean_low_half = y_series[low_group_mask].mean() if low_group_mask.any() else -np.inf # Use -inf to ensure it's "lower" if other is finite
            mean_high_half = y_series[high_group_mask].mean() if high_group_mask.any() else -np.inf

            # Determine which half has a lower mean target attribute value (thus unprivileged)
            if mean_low_half < mean_high_half:
                unprivileged_query = f"`{feature_name}` < {median_value}"
                privileged_query = f"`{feature_name}` >= {median_value}"
            else:
                unprivileged_query = f"`{feature_name}` >= {median_value}"
                privileged_query = f"`{feature_name}` < {median_value}"
            
            # Basic validation: ensure both groups have members (after split)
            if not self.df.query(unprivileged_query).empty and not self.df.query(privileged_query).empty:
                 return unprivileged_query, privileged_query
            else:
                raise ValueError(f"Cannot form distinct groups for numerical feature '{feature_name}' using median split (one group is empty).")


        # Handle categorical attributes
        elif pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
            means = self.df.groupby(feature_name)[y_series.name].mean()
            if means.empty:
                raise ValueError(f"Cannot infer groups for categorical feature '{feature_name}' (no data in categories).")
            
            privileged_cat = means.idxmax()
            unprivileged_cat = means.idxmin()

            # Construct query strings, quoting string values for Pandas query syntax
            unpriv_query = f"`{feature_name}` == {repr(unprivileged_cat)}" if isinstance(unprivileged_cat, str) else f"`{feature_name}` == {unprivileged_cat}"
            priv_query = f"`{feature_name}` == {repr(privileged_cat)}" if isinstance(privileged_cat, str) else f"`{feature_name}` == {privileged_cat}"
            
            # Ensure unprivileged and privileged categories are actually different
            if unprivileged_cat == privileged_cat:
                raise ValueError(f"All categories for '{feature_name}' have the same target attribute mean, cannot distinguish privileged/unprivileged.")
            
            return unpriv_query, priv_query
        else:
            raise ValueError(f"Feature '{feature_name}' has an unsupported dtype for group inference: {series.dtype}.")

    # --- Fairness Metrics (depend on group queries) ---

    def calculate_statistical_parity(self, unprivileged_query: str, privileged_query: str) -> float:
        """
        Calculates Statistical Parity (SP).
        SP = P(Y=positive | Unprivileged Group) - P(Y=positive | Privileged Group).
        A value of 0 indicates perfect parity. Positive values indicate higher positive
        outcome rate for the unprivileged group, negative for the privileged group.
        Matches original snippet's non-absolute definition.
        """
        unpriv_prob, priv_prob = self._compute_probs_by_query(unprivileged_query, privileged_query)
        return unpriv_prob - priv_prob

    def calculate_disparate_impact(self, unprivileged_query: str, privileged_query: str) -> float:
        """
        Calculates Disparate Impact (DI).
        DI = P(Y=positive | Unprivileged Group) / P(Y=positive | Privileged Group).
        A value of 1 indicates perfect parity. Values significantly above/below 1 indicate bias.
        Matches the second definition from the original snippet (which overwrites the first).
        """
        unpriv_prob, priv_prob = self._compute_probs_by_query(unprivileged_query, privileged_query)
        return unpriv_prob / priv_prob if priv_prob != 0 else 0.0

    def calculate_absolute_probability_difference(self, unprivileged_query: str, privileged_query: str) -> float:
        """
        Calculates Absolute Probability Difference (APD).
        APD = |P(Y=positive | Unprivileged Group) - P(Y=positive | Privileged Group)|.
        This is the absolute value of Statistical Parity. Not in original snippets, but in your prior code.
        """
        unpriv_prob, priv_prob = self._compute_probs_by_query(unprivileged_query, privileged_query)
        return abs(unpriv_prob - priv_prob)

    def calculate_group_ratio(self, unprivileged_query: str, privileged_query: str) -> tuple[float, float]:
        """
        Calculates a specific group ratio (w_obs / w_exp) for unprivileged and privileged groups.
        This ratio compares observed positive outcomes to expected positive outcomes within each group.
        Matches the definition from the original snippet.

        Returns:
        - tuple[float, float]: (unprivileged_group_ratio, privileged_group_ratio)
        """
        unpriv_group_df, unpriv_group_positive_series, priv_group_df, priv_group_positive_series = \
            self._get_groups_by_query(unprivileged_query, privileged_query)
        
        n_total = len(self.df)
        total_positives = (self.df[self.target_attribute] == self.positive_label).sum()

        def _ratio_calc(group_df: pd.DataFrame, group_pos_series: pd.Series) -> float:
            """Internal helper for calculating single group ratio."""
            if len(group_df) == 0 or n_total == 0 or total_positives == 0:
                return 0.0
            
            w_exp = (len(group_df) / n_total) * (total_positives / n_total)
            w_obs = len(group_pos_series) / n_total
            
            return w_obs / w_exp if w_exp != 0 else 0.0

        unpriv_ratio = _ratio_calc(unpriv_group_df, unpriv_group_positive_series)
        priv_ratio = _ratio_calc(priv_group_df, priv_group_positive_series)
            
        return unpriv_ratio, priv_ratio

    # --- Comprehensive Symptom Calculation Method ---

    def calculate_symptoms_for_feature(self, feature_name: str) -> dict:
        """
        Calculate comprehensive symptoms for a given feature, including:
        - Distribution/diversity metrics of the feature itself.
        - Relationship metrics between the feature and the target attribute.
        - Fairness metrics, if group conditions can be inferred for the feature.

        Parameters:
        - feature_name (str): The name of the feature for which to calculate symptoms.

        Returns:
        - dict: A dictionary containing all calculated symptom values. Values might be None
                if a metric is not applicable (e.g., non-numeric for Kurtosis) or
                if group conditions cannot be inferred.
        """
        symptoms = {}
        
        # Check if this is a categorical feature that was one-hot encoded
        encoded_columns = [col for col in self.df.columns if col.startswith(f"{feature_name}_")]
        is_one_hot_encoded = len(encoded_columns) > 0
        
        if is_one_hot_encoded:
            # Use original data for distribution metrics
            if feature_name in self.original_df.columns:
                feature_series = self.original_df[feature_name]
            else:
                raise KeyError(f"Original feature '{feature_name}' not found in original DataFrame")
                
            # --- Part 1: Distribution/Diversity Metrics (using original data) ---
            symptoms['Gini Index'] = gini_diversity_index(feature_series)
            symptoms['Shannon Entropy'] = shannon_entropy_index(feature_series)
            symptoms['Simpson Diversity'] = simpson_diversity_index(feature_series)
            
            # Imbalance Ratio: min count / max count for categories
            if feature_series.nunique() > 1:
                counts = feature_series.value_counts()
                symptoms['Imbalance Ratio'] = counts.min() / counts.max()
            else:
                symptoms['Imbalance Ratio'] = 0.0
                
            # Kurtosis and Skewness (not applicable to categorical)
            symptoms['Kurtosis'] = None
            symptoms['Skewness'] = None
            
            # --- Part 2: Relationship Metrics (using original data) ---
            symptoms['Mutual Information'] = mutual_info_score(feature_series, self.original_df[self.target_attribute])
            
            # Normalized Mutual Information
            mi = symptoms['Mutual Information']
            px = feature_series.value_counts(normalize=True).values
            hx = -np.sum(px * np.log(px + 1e-12)) if len(px) > 1 else 0.0
            py = self.original_df[self.target_attribute].value_counts(normalize=True).values
            hy = -np.sum(py * np.log(py + 1e-12)) if len(py) > 1 else 0.0
            denom = hx + hy
            symptoms['Normalized Mutual Information'] = mi / denom if denom > 0 else 0.0
            
            # Kendall Tau (not applicable to categorical vs target)
            symptoms['Kendall Tau'] = None
            
            # Correlation Ratio (categorical vs target)
            if pd.api.types.is_numeric_dtype(self.original_df[self.target_attribute]):
                categories = feature_series.unique()
                mean_total = self.original_df[self.target_attribute].mean()
                numerator = 0
                denominator_var_within = 0
                
                for category in categories:
                    subset_mask = feature_series == category
                    subset = self.original_df[subset_mask]
                    if len(subset) > 0:
                        mean_subset = subset[self.target_attribute].mean()
                        numerator += len(subset) * (mean_subset - mean_total) ** 2
                        denominator_var_within += (subset[self.target_attribute] - mean_subset).pow(2).sum()

                total_sum_of_squares = numerator + denominator_var_within
                symptoms['Correlation Ratio'] = np.sqrt(numerator / total_sum_of_squares) if total_sum_of_squares > 0 else 0.0
            else:
                symptoms['Correlation Ratio'] = None
                
            # --- Part 3: Fairness Metrics (aggregate results from all one-hot columns) ---
            # Initialize aggregated fairness metrics
            fairness_metrics = {
                'APD': [],
                'Statistical Parity': [],
                'Disparate Impact': [],
                'Unprivileged Unbalance': [],
                'Privileged Unbalance': [],
                'Unprivileged Pos Prob': [],
                'Privileged Pos Prob': [],
                'Pos Probability Diff': []
            }
            
            # Calculate fairness metrics for each one-hot encoded column
            for encoded_col in encoded_columns:
                try:
                    # For binary one-hot columns, we can define groups as 0 vs 1
                    unprivileged_query = f"{encoded_col} == 0"
                    privileged_query = f"{encoded_col} == 1"
                    
                    # Calculate metrics for this encoded column
                    apd = self.calculate_absolute_probability_difference(unprivileged_query, privileged_query)
                    stat_parity = self.calculate_statistical_parity(unprivileged_query, privileged_query)
                    disp_impact = self.calculate_disparate_impact(unprivileged_query, privileged_query)
                    
                    unpriv_ratio, priv_ratio = self.calculate_group_ratio(unprivileged_query, privileged_query)
                    unpriv_prob, priv_prob = self._compute_probs_by_query(unprivileged_query, privileged_query)
                    
                    # Store non-None values
                    if apd is not None:
                        fairness_metrics['APD'].append(apd)
                    if stat_parity is not None:
                        fairness_metrics['Statistical Parity'].append(stat_parity)
                    if disp_impact is not None:
                        fairness_metrics['Disparate Impact'].append(disp_impact)
                    if unpriv_ratio is not None:
                        fairness_metrics['Unprivileged Unbalance'].append(unpriv_ratio)
                    if priv_ratio is not None:
                        fairness_metrics['Privileged Unbalance'].append(priv_ratio)
                    if unpriv_prob is not None:
                        fairness_metrics['Unprivileged Pos Prob'].append(unpriv_prob)
                    if priv_prob is not None:
                        fairness_metrics['Privileged Pos Prob'].append(priv_prob)
                    if unpriv_prob is not None and priv_prob is not None:
                        fairness_metrics['Pos Probability Diff'].append(unpriv_prob - priv_prob)
                        
                except Exception as e:
                    # Skip this encoded column if calculation fails
                    continue
            
            # Aggregate fairness metrics (using mean of non-None values)
            for metric_name, values in fairness_metrics.items():
                if values:
                    symptoms[metric_name] = np.mean(values)
                else:
                    symptoms[metric_name] = None
                    
        else:
            # Original logic for non-encoded features
            if feature_name in self.df.columns:
                feature_series = self.df[feature_name]
            else:
                raise KeyError(f"Feature '{feature_name}' not found in DataFrame columns. "
                             f"Available columns: {list(self.df.columns)}")

            # --- Part 1: Distribution/Diversity Metrics of the Feature Itself ---
            symptoms['Gini Index'] = gini_diversity_index(feature_series)
            symptoms['Shannon Entropy'] = shannon_entropy_index(feature_series)
            symptoms['Simpson Diversity'] = simpson_diversity_index(feature_series)
            
            # Imbalance Ratio: min count / max count for categories
            if feature_series.nunique() > 1:
                counts = feature_series.value_counts()
                symptoms['Imbalance Ratio'] = counts.min() / counts.max()
            else:
                symptoms['Imbalance Ratio'] = 0.0

            # Kurtosis and Skewness (applicable only to numerical features)
            if pd.api.types.is_numeric_dtype(feature_series):
                symptoms['Kurtosis'] = kurtosis(feature_series, fisher=False, bias=False)
                symptoms['Skewness'] = skew(feature_series, bias=False)
            else:
                symptoms['Kurtosis'] = None
                symptoms['Skewness'] = None

            # --- Part 2: Relationship Metrics between the Feature and the Target ---
            symptoms['Mutual Information'] = mutual_info_score(feature_series, self.df[self.target_attribute])
            
            # Normalized Mutual Information
            mi = symptoms['Mutual Information']
            px = feature_series.value_counts(normalize=True).values
            hx = -np.sum(px * np.log(px + 1e-12)) if len(px) > 1 else 0.0
            py = self.df[self.target_attribute].value_counts(normalize=True).values
            hy = -np.sum(py * np.log(py + 1e-12)) if len(py) > 1 else 0.0
            denom = hx + hy
            symptoms['Normalized Mutual Information'] = mi / denom if denom > 0 else 0.0

            # Kendall Tau (applicable if both are numerical or ordinal)
            if pd.api.types.is_numeric_dtype(feature_series) and pd.api.types.is_numeric_dtype(self.df[self.target_attribute]):
                symptoms['Kendall Tau'] = kendalltau(feature_series, self.df[self.target_attribute]).correlation
            else:
                symptoms['Kendall Tau'] = None
            
            # Correlation Ratio
            if (pd.api.types.is_categorical_dtype(feature_series) or pd.api.types.is_object_dtype(feature_series)) \
               and pd.api.types.is_numeric_dtype(self.df[self.target_attribute]):
                categories = feature_series.unique()
                mean_total = self.df[self.target_attribute].mean()
                numerator = 0
                denominator_var_within = 0
                
                for category in categories:
                    subset_mask = feature_series == category
                    subset = self.df[subset_mask]
                    if len(subset) > 0:
                        mean_subset = subset[self.target_attribute].mean()
                        numerator += len(subset) * (mean_subset - mean_total) ** 2
                        denominator_var_within += (subset[self.target_attribute] - mean_subset).pow(2).sum()

                total_sum_of_squares = numerator + denominator_var_within
                symptoms['Correlation Ratio'] = np.sqrt(numerator / total_sum_of_squares) if total_sum_of_squares > 0 else 0.0
            elif pd.api.types.is_numeric_dtype(feature_series) and pd.api.types.is_numeric_dtype(self.df[self.target_attribute]):
                symptoms['Correlation Ratio'] = feature_series.corr(self.df[self.target_attribute], method='pearson')
            else:
                symptoms['Correlation Ratio'] = None

            # --- Part 3: Fairness Metrics (require group definitions) ---
            unprivileged_query, privileged_query = None, None
            try:
                unprivileged_query, privileged_query = self.infer_group_conditions(feature_name)
            except ValueError as e:
                pass

            if unprivileged_query and privileged_query:
                symptoms['APD'] = self.calculate_absolute_probability_difference(unprivileged_query, privileged_query)
                symptoms['Statistical Parity'] = self.calculate_statistical_parity(unprivileged_query, privileged_query)
                symptoms['Disparate Impact'] = self.calculate_disparate_impact(unprivileged_query, privileged_query)
                
                unpriv_ratio, priv_ratio = self.calculate_group_ratio(unprivileged_query, privileged_query)
                symptoms['Unprivileged Unbalance'] = unpriv_ratio
                symptoms['Privileged Unbalance'] = priv_ratio

                unpriv_prob, priv_prob = self._compute_probs_by_query(unprivileged_query, privileged_query)
                symptoms['Unprivileged Pos Prob'] = unpriv_prob
                symptoms['Privileged Pos Prob'] = priv_prob
                symptoms['Pos Probability Diff'] = unpriv_prob - priv_prob
            else:
                symptoms['APD'] = None
                symptoms['Statistical Parity'] = None
                symptoms['Disparate Impact'] = None
                symptoms['Unprivileged Unbalance'] = None
                symptoms['Privileged Unbalance'] = None
                symptoms['Unprivileged Pos Prob'] = None
                symptoms['Privileged Pos Prob'] = None
                symptoms['Pos Probability Diff'] = None
                
        return symptoms