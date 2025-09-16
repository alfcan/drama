import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, kendalltau
from sklearn.metrics import mutual_info_score

# --- Helper functions ---

def gini_fun(x, w=None):
    """Weighted Gini coefficient"""
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / (cumxw[-1] * cumw[-1])
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

class SymptomCalculator:
    def __init__(self, df, sensitive_attribute, target_attribute):
        """
        Initialize the SymptomCalculator with dataset, sensitive attribute, and target attribute.

        Parameters:
        - df (pd.DataFrame): The dataset as a pandas DataFrame.
        - sensitive_attribute (str): The sensitive attribute column name.
        - target_attribute (str): The target attribute column name.
        """
        self.df = df.copy()  # Make a copy of the dataset
        self.sensitive_attribute = sensitive_attribute
        self.target_attribute = target_attribute
        self._is_one_hot_encoded = None
        self._encoded_columns = None
        self._check_encoding_status()
    
    def _check_encoding_status(self):
        """Check if the sensitive attribute is one-hot encoded and cache the result."""
        if self.sensitive_attribute in self.df.columns:
            self._is_one_hot_encoded = False
            self._encoded_columns = [self.sensitive_attribute]
        else:
            # Check for one-hot encoded columns
            encoded_cols = [col for col in self.df.columns if col.startswith(f"{self.sensitive_attribute}_")]
            if encoded_cols:
                self._is_one_hot_encoded = True
                self._encoded_columns = encoded_cols
            else:
                raise ValueError(f"Sensitive attribute '{self.sensitive_attribute}' not found in dataset columns")
    
    def _get_sensitive_attribute_series(self):
        """Get a series representing the sensitive attribute values, handling one-hot encoding."""
        if not self._is_one_hot_encoded:
            return self.df[self.sensitive_attribute]
        else:
            # For one-hot encoded data, reconstruct the original categorical series
            result_series = pd.Series(index=self.df.index, dtype='object')
            for col in self._encoded_columns:
                category = col.replace(f"{self.sensitive_attribute}_", "")
                mask = self.df[col] == 1
                result_series[mask] = category
            return result_series

    def calculate_apd(self, privileged_condition, unprivileged_condition):
        """
        Calculate the Absolute Probability Difference (APD).

        Parameters:
        - privileged: Value of the privileged category of the sensitive attribute.
        - unprivileged: Value of the unprivileged category of the sensitive attribute.

        Returns:
        - float: The calculated APD value.
        """
        privileged = self.df.query(privileged_condition)
        unprivileged = self.df.query(unprivileged_condition)
        p_privileged = privileged[self.target_attribute].mean()
        p_unprivileged = unprivileged[self.target_attribute].mean()
        return abs(p_privileged - p_unprivileged)

    def calculate_gini_index(self):
        """
        Calculate the (normalized) Gini Index:
        G = (m/(m-1)) * (1 - sum_i f_i^2), where f_i are class frequencies (sum to 1).
        """
        sensitive_series = self._get_sensitive_attribute_series()
        counts = sensitive_series.value_counts(normalize=True).values
        m = len(counts)
        if m <= 1:
            return 1.0
        f2 = np.sum(counts ** 2)
        return (m / (m - 1)) * (1.0 - f2)

    def calculate_shannon_entropy(self):
        """Calculate the normalized Shannon Entropy (between 0-1)."""
        sensitive_series = self._get_sensitive_attribute_series()
        counts = sensitive_series.value_counts(normalize=True)
        if len(counts) <= 1:
            return 1.0  # Perfect balance for single category
        
        # Shannon entropy normalized to [0,1]
        entropy = -np.sum(counts * np.log(counts))
        max_entropy = np.log(len(counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def calculate_simpson_diversity(self):
        """Calculate the normalized Simpson Diversity Index as:
        D = (1/(m-1)) * ((1 / sum_i f_i^2) - 1), where f_i are class frequencies (sum to 1).
        """
        sensitive_series = self._get_sensitive_attribute_series()
        counts = sensitive_series.value_counts(normalize=True).values
        m = len(counts)
        if m <= 1:
            return 1.0
        f2 = np.sum(counts ** 2)
        if f2 <= 0:
            return 1.0
        return (1.0 / (m - 1)) * ((1.0 / f2) - 1.0)

    def calculate_imbalance_ratio(self):
        """Imbalance ratio as min/max"""
        sensitive_series = self._get_sensitive_attribute_series()
        counts = sensitive_series.value_counts(normalize=True).values
        return np.min(counts) / np.max(counts) if len(counts) > 1 else 0

    def calculate_kurtosis(self):
        """
        Calculate the Kurtosis (classical, not excess).
        """
        return kurtosis(self.df[self.target_attribute], fisher=False, bias=False)

    def calculate_skewness(self):
        """
        Calculate the Skewness (with bias correction).
        """
        return skew(self.df[self.target_attribute], bias=False)

    def calculate_mutual_information(self):
        """
        Calculate the Mutual Information between sensitive attribute and target attribute.

        Returns:
        - float: The calculated Mutual Information.
        """
        sensitive_series = self._get_sensitive_attribute_series()
        return mutual_info_score(sensitive_series, self.df[self.target_attribute])

    def calculate_normalized_mutual_information(self):
        """
        Calculate the Normalized Mutual Information between sensitive attribute and target attribute.
        Uses natural logarithm consistently and non-normalized entropies:
        NMI = I(X;Y) / (H(X) + H(Y))
        """
        mi = self.calculate_mutual_information()  # sklearn uses natural log for MI
        # H(X)
        sensitive_series = self._get_sensitive_attribute_series()
        px = sensitive_series.value_counts(normalize=True).values
        hx = -np.sum(px * np.log(px + 1e-12))
        # H(Y)
        py = self.df[self.target_attribute].value_counts(normalize=True).values
        hy = -np.sum(py * np.log(py + 1e-12))
        denom = hx + hy
        return mi / denom if denom > 0 else 0.0

    def calculate_kendall_tau(self):
        """
        Calculate the Kendall's Tau correlation coefficient between sensitive attribute and target attribute.

        Returns:
        - float: The calculated Kendall's Tau.
        """
        sensitive_series = self._get_sensitive_attribute_series()
        return kendalltau(sensitive_series, self.df[self.target_attribute]).correlation

    def calculate_correlation_ratio(self):
        """
        Calculate the Correlation Ratio between the sensitive attribute and the target attribute.

        Returns:
        - float: The calculated Correlation Ratio.
        """
        sensitive_series = self._get_sensitive_attribute_series()
        categories = sensitive_series.unique()
        mean_total = self.df[self.target_attribute].mean()
        numerator = 0
        denominator = 0
        for category in categories:
            subset_mask = sensitive_series == category
            subset = self.df[subset_mask]
            mean_subset = subset[self.target_attribute].mean()
            numerator += len(subset) * (mean_subset - mean_total) ** 2
            denominator += len(subset) * (subset[self.target_attribute] - mean_subset).var()
        denominator += len(self.df) * (self.df[self.target_attribute] - mean_total).var()
        return np.sqrt(numerator / denominator)

    # ------------------- Additional Metrics (bias_symptoms parity) -------------------
    def _get_groups(self, privileged_condition, unprivileged_condition):
        unpriv_group = self.df.query(unprivileged_condition)
        priv_group = self.df.query(privileged_condition)
        unpriv_group_pos = unpriv_group[self.target_attribute] == 1
        priv_group_pos = priv_group[self.target_attribute] == 1
        return unpriv_group, unpriv_group_pos, priv_group, priv_group_pos

    def _compute_probs(self, privileged_condition, unprivileged_condition):
        unpriv_group, unpriv_group_pos, priv_group, priv_group_pos = self._get_groups(privileged_condition, unprivileged_condition)
        unpriv_prob = unpriv_group_pos.mean() if len(unpriv_group) else 0
        priv_prob = priv_group_pos.mean() if len(priv_group) else 0
        return unpriv_prob, priv_prob

    def statistical_parity(self, privileged_condition, unprivileged_condition):
        unpriv_prob, priv_prob = self._compute_probs(privileged_condition, unprivileged_condition)
        return abs(unpriv_prob - priv_prob)

    def disparate_impact(self, privileged_condition, unprivileged_condition):
        unpriv_prob, priv_prob = self._compute_probs(privileged_condition, unprivileged_condition)
        return unpriv_prob / priv_prob if priv_prob != 0 else 0

    def _group_ratio(self, privileged_condition, unprivileged_condition):
        unpriv_group, unpriv_group_pos, priv_group, priv_group_pos = self._get_groups(privileged_condition, unprivileged_condition)
        n = len(self.df)
        total_pos = (self.df[self.target_attribute] == 1).sum()

        def _ratio(group, group_pos):
            if len(group) == 0 or total_pos == 0:
                return 0
            w_exp = (len(group) / n) * (total_pos / n)
            w_obs = group_pos.sum() / n
            return w_obs / w_exp if w_exp != 0 else 0

        return _ratio(unpriv_group, unpriv_group_pos), _ratio(priv_group, priv_group_pos)

    def infer_privileged_unprivileged_conditions(self):
        """
        Automatically infer privileged and unprivileged group conditions for the
        sensitive attribute based on the mean value of the target attribute.

        Logic:
        1. If the sensitive attribute is numerical, split the data at the median.
           The side with the higher mean target value is considered privileged.
        2. If the sensitive attribute is categorical, the category with the
           highest mean target value is privileged, while the category with the
           lowest mean target value is unprivileged.
        3. Handles one-hot encoded columns by detecting columns that start with the sensitive attribute name.

        Returns
        -------
        tuple(str, str)
            (privileged_condition, unprivileged_condition)
        """
        col = self.sensitive_attribute
        y = self.target_attribute
        
        # Check if the original column exists (not one-hot encoded)
        if col in self.df.columns:
            series = self.df[col]
        else:
            # Handle one-hot encoded columns
            encoded_cols = [c for c in self.df.columns if c.startswith(f"{col}_")]
            if not encoded_cols:
                raise ValueError(f"Sensitive attribute '{col}' not found in dataset columns. Available columns: {list(self.df.columns)[:10]}...")
            
            # For one-hot encoded categorical data, find the category with highest mean target value
            category_means = {}
            for encoded_col in encoded_cols:
                # Extract category name from column name (remove prefix)
                category = encoded_col.replace(f"{col}_", "")
                # Calculate mean target value for this category
                mask = self.df[encoded_col] == 1
                if mask.sum() > 0:  # Ensure there are samples in this category
                    category_means[category] = self.df[mask][y].mean()
            
            if not category_means:
                raise ValueError(f"No valid categories found for one-hot encoded attribute '{col}'")
            
            # Find privileged (highest mean) and unprivileged (lowest mean) categories
            privileged_category = max(category_means.keys(), key=lambda k: category_means[k])
            unprivileged_category = min(category_means.keys(), key=lambda k: category_means[k])
            
            privileged_condition = f"`{col}_{privileged_category}` == 1"
            unprivileged_condition = f"`{col}_{unprivileged_category}` == 1"
            
            return privileged_condition, unprivileged_condition
        
        # Original logic for non-encoded columns
        series = self.df[col]

        # Numerical attribute: median split
        if pd.api.types.is_numeric_dtype(series):
            median_val = series.median()
            cond_high = f"`{col}` >= {median_val}"
            cond_low = f"`{col}` < {median_val}"
            mean_high = self.df.query(cond_high)[y].mean()
            mean_low = self.df.query(cond_low)[y].mean()
            if mean_high >= mean_low:
                return cond_high, cond_low
            else:
                return cond_low, cond_high

        # Categorical attribute
        means = self.df.groupby(col)[y].mean()
        privileged_cat = means.idxmax()
        unprivileged_cat = means.idxmin()
        priv_cond = f"`{col}` == {repr(privileged_cat)}"
        unpriv_cond = f"`{col}` == {repr(unprivileged_cat)}"
        return priv_cond, unpriv_cond

    def calculate_symptoms(self, privileged_condition=None, unprivileged_condition=None):
        """
        Calculate all symptoms and store results.

        Returns:
        - dict: Dictionary containing all calculated symptom values.
        """
        unpriv_prob = priv_prob = None
        unpriv_ratio = priv_ratio = None
        dsp = di = pos_prob_diff = None
        if privileged_condition and unprivileged_condition:
            unpriv_prob, priv_prob = self._compute_probs(privileged_condition, unprivileged_condition)
            unpriv_ratio, priv_ratio = self._group_ratio(privileged_condition, unprivileged_condition)
            dsp = self.statistical_parity(privileged_condition, unprivileged_condition)
            di = self.disparate_impact(privileged_condition, unprivileged_condition)
            pos_prob_diff = unpriv_prob - priv_prob if unpriv_prob is not None else None

        symptoms = {
            'APD': self.calculate_apd(privileged_condition, unprivileged_condition) if privileged_condition and unprivileged_condition else None,
            'Gini Index': self.calculate_gini_index(),
            'Shannon Entropy': self.calculate_shannon_entropy(),
            'Simpson Diversity': self.calculate_simpson_diversity(),
            'Imbalance Ratio': self.calculate_imbalance_ratio(),
            'Skewness': self.calculate_skewness(),
            'Mutual Information': self.calculate_mutual_information(),
            'Normalized Mutual Information': self.calculate_normalized_mutual_information(),
            'Kendall Tau': self.calculate_kendall_tau(),
            'Correlation Ratio': self.calculate_correlation_ratio(),
            'Unprivileged Pos Prob': unpriv_prob,
            'Privileged Pos Prob': priv_prob,
            'Unprivileged Unbalance': unpriv_ratio,
            'Privileged Unbalance': priv_ratio,
            'Data Statistical Parity': dsp,
            'Disparate Impact': di,
            'Pos Probability Diff': pos_prob_diff
        }
        return symptoms

    def detect_bias_symptoms(self, symptoms):
        """
        Calculate symptoms and return a dictionary indicating if the dataset is affected by bias.

        Returns:
        - dict: Dictionary with symptom names as keys and boolean values indicating if the dataset is affected by bias.
        """
        # Initialize flags for all symptoms
        dsp_flag = False
        gini_flag = False
        shannon_flag = False
        simpson_flag = False
        ir_flag = False
        skewness_flag = False
        mutual_info_flag = False
        kendall_tau_flag = False
        unpriv_unbalance_flag = False
        priv_unbalance_flag = False
        upp_flag = False
        
        # Data Statistical Parity (DSP) - 1 indicates complete bias, 0 indicates optimal fairness
        if 'Data Statistical Parity' in symptoms and symptoms['Data Statistical Parity'] is not None:
            dsp_flag = abs(symptoms['Data Statistical Parity']) >= 1.0
        
        # Gini Index - After 0-1 normalisation, 0 indicates completely unbalanced variable (bias), 1 indicates completely balanced
        if symptoms['Gini Index'] is not None:
            gini_flag = symptoms['Gini Index'] <= 0.0
        
        # Shannon Diversity - After 0-1 normalisation, 0 indicates completely unbalanced variable (bias), 1 indicates completely balanced
        if symptoms['Shannon Entropy'] is not None:
            shannon_flag = symptoms['Shannon Entropy'] <= 0.0
        
        # Simpson Diversity - After 0-1 normalisation, 0 indicates completely unbalanced variable (bias), 1 indicates completely balanced
        if symptoms['Simpson Diversity'] is not None:
            simpson_flag = symptoms['Simpson Diversity'] <= 0.0
        
        # Imbalance Ratio - After 0-1 normalisation, 1 indicates completely unbalanced variable (bias), 0 indicates completely balanced
        if symptoms['Imbalance Ratio'] is not None:
            ir_flag = symptoms['Imbalance Ratio'] >= 1.0
        
        # Skewness - Value other than 0 indicates imbalance, tends to 0 for balanced data
        if symptoms['Skewness'] is not None:
            skewness_flag = abs(symptoms['Skewness']) > 0.0
        
        # Mutual Information - Greater than 0 indicates dependence, 0 indicates complete independence
        if symptoms['Mutual Information'] is not None:
            mutual_info_flag = symptoms['Mutual Information'] > 0.0
        
        # Kendall's Tau - 0 implies no correlation, no specific bias identification found
        if symptoms['Kendall Tau'] is not None:
            kendall_tau_flag = abs(symptoms['Kendall Tau']) > 0.5
        
        # Unprivileged Group Unbalance - Greater than 1 indicates over-sampling, less than 1 indicates under-sampling, 1 indicates balance
        if symptoms['Unprivileged Unbalance'] is not None:
            unpriv_unbalance_flag = symptoms['Unprivileged Unbalance'] != 1.0
        
        # Privileged Group Unbalance - Greater than 1 indicates over-sampling, less than 1 indicates under-sampling, 1 indicates balance
        if symptoms['Privileged Unbalance'] is not None:
            priv_unbalance_flag = symptoms['Privileged Unbalance'] != 1.0

        # Unprivileged Positive Probability (UPP) - Second most important variable for predicting SP and AO
        if symptoms['Unprivileged Pos Prob'] is not None:
            upp_flag = symptoms['Unprivileged Pos Prob'] != 0.5

        # Create bias detection dictionary with all symptoms
        bias_detection = {
            'Data Statistical Parity': dsp_flag,
            'Gini Index': gini_flag,
            'Shannon Entropy': shannon_flag,
            'Simpson Diversity': simpson_flag,
            'Imbalance Ratio': ir_flag,
            'Skewness': skewness_flag,
            'Mutual Information': mutual_info_flag,
            'Kendall Tau': kendall_tau_flag,
            'Unprivileged Unbalance': unpriv_unbalance_flag,
            'Privileged Unbalance': priv_unbalance_flag,
            'Unprivileged Pos Prob': upp_flag
        }

        return bias_detection
