import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, kendalltau
from sklearn.metrics import mutual_info_score

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

    def calculate_apd(self, privileged, unprivileged):
        """
        Calculate the Absolute Probability Difference (APD).

        Parameters:
        - privileged: Value of the privileged category of the sensitive attribute.
        - unprivileged: Value of the unprivileged category of the sensitive attribute.

        Returns:
        - float: The calculated APD value.
        """
        privileged = self.df[self.df[self.sensitive_attribute].isin([privileged])]
        unprivileged = self.df[self.df[self.sensitive_attribute].isin([unprivileged])]
        p_privileged = privileged[self.target_attribute].mean()
        p_unprivileged = unprivileged[self.target_attribute].mean()
        return abs(p_privileged - p_unprivileged)

    def calculate_gini_index(self):
        """
        Calculate the Gini Index.

        Returns:
        - float: The calculated Gini Index.
        """
        p = self.df[self.sensitive_attribute].value_counts(normalize=True).values
        return 1 - np.sum(np.square(p))

    def calculate_shannon_entropy(self):
        """
        Calculate the Shannon Entropy.

        Returns:
        - float: The calculated Shannon Entropy.
        """
        p = self.df[self.sensitive_attribute].value_counts(normalize=True).values
        return -np.sum(p * np.log2(p))

    def calculate_simpson_diversity(self):
        """
        Calculate the Simpson Diversity Index.

        Returns:
        - float: The calculated Simpson Diversity Index.
        """
        p = self.df[self.sensitive_attribute].value_counts(normalize=True).values
        return 1 - np.sum(p**2)

    def calculate_imbalance_ratio(self):
        """
        Calculate the Imbalance Ratio (IR).

        Returns:
        - float: The calculated Imbalance Ratio.
        """
        p = self.df[self.sensitive_attribute].value_counts(normalize=True).values
        return np.max(p) / np.min(p)

    def calculate_kurtosis(self):
        """
        Calculate the Kurtosis.

        Returns:
        - float: The calculated Kurtosis.
        """
        return kurtosis(self.df[self.target_attribute])

    def calculate_skewness(self):
        """
        Calculate the Skewness.

        Returns:
        - float: The calculated Skewness.
        """
        return skew(self.df[self.target_attribute])

    def calculate_mutual_information(self):
        """
        Calculate the Mutual Information between sensitive attribute and target attribute.

        Returns:
        - float: The calculated Mutual Information.
        """
        return mutual_info_score(self.df[self.sensitive_attribute], self.df[self.target_attribute])

    def calculate_normalized_mutual_information(self):
        """
        Calculate the Normalized Mutual Information between sensitive attribute and target attribute.

        Returns:
        - float: The calculated Normalized Mutual Information.
        """
        mutual_info = self.calculate_mutual_information()
        h_x = self.calculate_shannon_entropy()
        h_y = -np.sum(self.df[self.target_attribute].value_counts(normalize=True).values * 
                      np.log2(self.df[self.target_attribute].value_counts(normalize=True).values))
        return mutual_info / (h_x + h_y)

    def calculate_kendall_tau(self):
        """
        Calculate the Kendall's Tau correlation coefficient between sensitive attribute and target attribute.

        Returns:
        - float: The calculated Kendall's Tau.
        """
        return kendalltau(self.df[self.sensitive_attribute], self.df[self.target_attribute]).correlation

    def calculate_correlation_ratio(self):
        """
        Calculate the Correlation Ratio between the sensitive attribute and the target attribute.

        Returns:
        - float: The calculated Correlation Ratio.
        """
        categories = self.df[self.sensitive_attribute].unique()
        mean_total = self.df[self.target_attribute].mean()
        numerator = 0
        denominator = 0
        for category in categories:
            subset = self.df[self.df[self.sensitive_attribute] == category]
            mean_subset = subset[self.target_attribute].mean()
            numerator += len(subset) * (mean_subset - mean_total) ** 2
            denominator += len(subset) * (subset[self.target_attribute] - mean_subset).var()
        denominator += len(self.df) * (self.df[self.target_attribute] - mean_total).var()
        return np.sqrt(numerator / denominator)

    def calculate_symptoms(self, privileged=None, unprivileged=None):
        """
        Calculate all symptoms and store results.

        Parameters:
        - privileged: Value of the privileged category of the sensitive attribute (required for APD calculation).
        - unprivileged: Value of the unprivileged category of the sensitive attribute (required for APD calculation).

        Returns:
        - dict: Dictionary containing all calculated symptom values.
        """
        symptoms = {
            'APD': self.calculate_apd(privileged, unprivileged) if privileged and unprivileged else None,
            'Gini Index': self.calculate_gini_index(),
            'Shannon Entropy': self.calculate_shannon_entropy(),
            'Simpson Diversity': self.calculate_simpson_diversity(),
            'Imbalance Ratio': self.calculate_imbalance_ratio(),
            'Kurtosis': self.calculate_kurtosis(),
            'Skewness': self.calculate_skewness(),
            'Mutual Information': self.calculate_mutual_information(),
            'Normalized Mutual Information': self.calculate_normalized_mutual_information(),
            'Kendall Tau': self.calculate_kendall_tau(),
            'Correlation Ratio': self.calculate_correlation_ratio()
        }
        return symptoms

    def detect_bias_symptoms(self, privileged=None, unprivileged=None):
        """
        Calculate symptoms and return a dictionary indicating if the dataset is affected by bias.

        Parameters:
        - privileged: Value of the privileged category of the sensitive attribute (required for APD calculation).
        - unprivileged: Value of the unprivileged category of the sensitive attribute (required for APD calculation).

        Returns:
        - dict: Dictionary with symptom names as keys and boolean values indicating if the dataset is affected by bias.
        """
        symptoms = self.calculate_symptoms(privileged, unprivileged)

        apd_flag = False
        gini_flag = False
        shannon_flag = False
        simpson_flag = False
        ir_flag = False
        kurtosis_flag = False
        skewness_flag = False
        mi_flag = False
        nmi_flag = False
        kt_flag = False
        cr_flag = False

        if symptoms['APD'] and symptoms['APD'] > 0.1:
            apd_flag = True
        if symptoms['Gini Index'] < 0.5:
            gini_flag = True
        if symptoms['Shannon Entropy'] < 0.5:
            shannon_flag = True
        if symptoms['Simpson Diversity'] < 0.5:
            simpson_flag = True
        if symptoms['Imbalance Ratio'] > 1.5:
            ir_flag = True
        if abs(symptoms['Kurtosis']) > 3:
            kurtosis_flag = True
        if abs(symptoms['Skewness']) > 1:
            skewness_flag = True
        if symptoms['Mutual Information'] > 0.5:
            mi_flag = True
        if symptoms['Normalized Mutual Information'] > 0.5:
            nmi_flag = True
        if abs(symptoms['Kendall Tau']) > 0.5:
            kt_flag = True
        if symptoms['Correlation Ratio'] > 0.5:
            cr_flag = True

        bias_detection = {
            'APD': apd_flag,
            'Gini Index': gini_flag,
            'Shannon Entropy': shannon_flag,
            'Simpson Diversity': simpson_flag,
            'Imbalance Ratio': ir_flag,
            'Kurtosis': kurtosis_flag,
            'Skewness': skewness_flag,
            'Mutual Information': mi_flag,
            'Normalized Mutual Information': nmi_flag,
            'Kendall Tau': kt_flag,
            'Correlation Ratio': cr_flag
        }

        return bias_detection
