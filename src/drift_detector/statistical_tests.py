"""
Statistical Tests for Data Drift Detection
Supports various statistical tests for different data types
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, mannwhitneyu
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class StatisticalTests:
    """Collection of statistical tests for drift detection"""
    
    @staticmethod
    def detect_feature_type(series: pd.Series) -> str:
        """
        Automatically detect feature type
        
        Args:
            series: Input pandas Series
            
        Returns:
            Feature type: 'numeric', 'categorical', or 'binary'
        """
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            # Check if binary (only 0/1 or True/False)
            unique_vals = series.dropna().unique()
            if len(unique_vals) <= 2:
                return 'binary'
            return 'numeric'
        else:
            return 'categorical'
    
    @staticmethod
    def kolmogorov_smirnov_test(train: pd.Series, test: pd.Series, 
                                alpha: float = 0.05) -> Dict:
        """
        Kolmogorov-Smirnov test for continuous distributions
        
        Args:
            train: Training data series
            test: Test data series
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        # Remove NaN values
        train_clean = train.dropna()
        test_clean = test.dropna()
        
        if len(train_clean) < 2 or len(test_clean) < 2:
            return {
                'statistic': np.nan,
                'pvalue': 1.0,
                'drift_detected': False,
                'test_name': 'Kolmogorov-Smirnov'
            }
        
        statistic, pvalue = ks_2samp(train_clean, test_clean)
        
        return {
            'statistic': statistic,
            'pvalue': pvalue,
            'drift_detected': pvalue < alpha,
            'test_name': 'Kolmogorov-Smirnov',
            'alpha': alpha
        }
    
    @staticmethod
    def mann_whitney_u_test(train: pd.Series, test: pd.Series,
                           alpha: float = 0.05) -> Dict:
        """
        Mann-Whitney U test (non-parametric test for two independent samples)
        
        Args:
            train: Training data series
            test: Test data series
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        train_clean = train.dropna()
        test_clean = test.dropna()
        
        if len(train_clean) < 2 or len(test_clean) < 2:
            return {
                'statistic': np.nan,
                'pvalue': 1.0,
                'drift_detected': False,
                'test_name': 'Mann-Whitney U'
            }
        
        try:
            statistic, pvalue = mannwhitneyu(train_clean, test_clean, 
                                           alternative='two-sided')
        except ValueError:
            return {
                'statistic': np.nan,
                'pvalue': 1.0,
                'drift_detected': False,
                'test_name': 'Mann-Whitney U'
            }
        
        return {
            'statistic': statistic,
            'pvalue': pvalue,
            'drift_detected': pvalue < alpha,
            'test_name': 'Mann-Whitney U',
            'alpha': alpha
        }
    
    @staticmethod
    def chi_square_test(train: pd.Series, test: pd.Series,
                       alpha: float = 0.05) -> Dict:
        """
        Chi-square test for categorical data
        
        Args:
            train: Training data series
            test: Test data series
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        train_clean = train.dropna()
        test_clean = test.dropna()
        
        # Get all unique categories
        all_categories = set(train_clean.unique()) | set(test_clean.unique())
        
        if len(all_categories) < 2:
            return {
                'statistic': np.nan,
                'pvalue': 1.0,
                'drift_detected': False,
                'test_name': 'Chi-Square'
            }
        
        # Create contingency table
        train_counts = train_clean.value_counts().reindex(all_categories, fill_value=0)
        test_counts = test_clean.value_counts().reindex(all_categories, fill_value=0)
        
        contingency = np.array([train_counts.values, test_counts.values])
        
        # Check if we have enough samples
        if contingency.sum() < 2 or (contingency < 5).sum() > len(all_categories) * 0.5:
            # Use Fisher's exact test approximation or return conservative result
            return {
                'statistic': np.nan,
                'pvalue': 1.0,
                'drift_detected': False,
                'test_name': 'Chi-Square (insufficient data)'
            }
        
        try:
            statistic, pvalue, dof, expected = chi2_contingency(contingency)
        except:
            return {
                'statistic': np.nan,
                'pvalue': 1.0,
                'drift_detected': False,
                'test_name': 'Chi-Square'
            }
        
        return {
            'statistic': statistic,
            'pvalue': pvalue,
            'drift_detected': pvalue < alpha,
            'test_name': 'Chi-Square',
            'alpha': alpha,
            'degrees_of_freedom': dof
        }
    
    @staticmethod
    def psi_test(train: pd.Series, test: pd.Series, 
                bins: int = 10, threshold: float = 0.2) -> Dict:
        """
        Population Stability Index (PSI) test
        
        Args:
            train: Training data series
            test: Test data series
            bins: Number of bins for PSI calculation
            threshold: PSI threshold for drift detection
            
        Returns:
            Dictionary with test results
        """
        train_clean = train.dropna()
        test_clean = test.dropna()
        
        if len(train_clean) < 2 or len(test_clean) < 2:
            return {
                'psi': np.nan,
                'drift_detected': False,
                'test_name': 'PSI'
            }
        
        # Determine if numeric or categorical
        feature_type = StatisticalTests.detect_feature_type(train_clean)
        
        if feature_type == 'numeric':
            # Create bins based on training data
            _, bin_edges = pd.cut(train_clean, bins=bins, retbins=True, 
                                 duplicates='drop')
            
            # Handle edge cases
            if len(bin_edges) < 2:
                return {
                    'psi': np.nan,
                    'drift_detected': False,
                    'test_name': 'PSI'
                }
            
            # Ensure bin_edges cover test data range
            min_val = min(train_clean.min(), test_clean.min())
            max_val = max(train_clean.max(), test_clean.max())
            bin_edges[0] = min_val - 0.001
            bin_edges[-1] = max_val + 0.001
            
            train_binned = pd.cut(train_clean, bins=bin_edges, include_lowest=True)
            test_binned = pd.cut(test_clean, bins=bin_edges, include_lowest=True)
        else:
            # For categorical, use categories directly
            train_binned = train_clean
            test_binned = test_clean
        
        # Calculate distributions
        train_dist = train_binned.value_counts(normalize=True)
        test_dist = test_binned.value_counts(normalize=True)
        
        # Align distributions
        all_categories = set(train_dist.index) | set(test_dist.index)
        train_dist = train_dist.reindex(all_categories, fill_value=1e-6)
        test_dist = test_dist.reindex(all_categories, fill_value=1e-6)
        
        # Calculate PSI
        psi = np.sum((test_dist - train_dist) * np.log(test_dist / train_dist))
        
        return {
            'psi': psi,
            'drift_detected': psi > threshold,
            'test_name': 'PSI',
            'threshold': threshold
        }
    
    @staticmethod
    def wasserstein_distance(train: pd.Series, test: pd.Series) -> Dict:
        """
        Wasserstein (Earth Mover's) Distance
        
        Args:
            train: Training data series
            test: Test data series
            
        Returns:
            Dictionary with distance metric
        """
        from scipy.stats import wasserstein_distance
        
        train_clean = train.dropna()
        test_clean = test.dropna()
        
        if len(train_clean) < 1 or len(test_clean) < 1:
            return {
                'distance': np.nan,
                'drift_detected': False,
                'test_name': 'Wasserstein Distance'
            }
        
        # For numeric features
        if StatisticalTests.detect_feature_type(train_clean) == 'numeric':
            distance = wasserstein_distance(train_clean, test_clean)
            # Normalize by range for interpretability
            data_range = max(train_clean.max(), test_clean.max()) - \
                        min(train_clean.min(), test_clean.min())
            normalized_distance = distance / (data_range + 1e-6)
            
            return {
                'distance': distance,
                'normalized_distance': normalized_distance,
                'drift_detected': normalized_distance > 0.1,  # Threshold
                'test_name': 'Wasserstein Distance'
            }
        else:
            # For categorical, use label encoding
            le = LabelEncoder()
            all_values = pd.concat([train_clean, test_clean])
            le.fit(all_values)
            train_encoded = le.transform(train_clean)
            test_encoded = le.transform(test_clean)
            
            distance = wasserstein_distance(train_encoded, test_encoded)
            
            return {
                'distance': distance,
                'normalized_distance': distance / (len(le.classes_) + 1e-6),
                'drift_detected': distance > 0.1,
                'test_name': 'Wasserstein Distance'
            }
    
    @staticmethod
    def get_appropriate_test(feature_type: str, test_type: str = 'auto') -> str:
        """
        Get appropriate statistical test based on feature type
        
        Args:
            feature_type: Type of feature ('numeric', 'categorical', 'binary')
            test_type: Preferred test type ('auto', 'parametric', 'non-parametric')
            
        Returns:
            Recommended test name
        """
        if feature_type == 'numeric':
            if test_type == 'parametric':
                return 'kolmogorov_smirnov'
            else:
                return 'mann_whitney_u'  # More robust
        elif feature_type == 'categorical':
            return 'chi_square'
        else:  # binary
            return 'chi_square'
