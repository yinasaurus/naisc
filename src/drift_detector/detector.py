"""
Main Drift Detector Class
Automatically detects and quantifies data drift
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .statistical_tests import StatisticalTests
import warnings
warnings.filterwarnings('ignore')


class DriftDetector:
    """
    Main class for detecting data drift between training and test datasets
    """
    
    def __init__(self, alpha: float = 0.05, psi_threshold: float = 0.2):
        """
        Initialize DriftDetector
        
        Args:
            alpha: Significance level for statistical tests
            psi_threshold: Threshold for PSI test
        """
        self.alpha = alpha
        self.psi_threshold = psi_threshold
        self.stats_tests = StatisticalTests()
        self.feature_types = {}
        self.drift_results = {}
        
    def detect_feature_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically detect feature types for all columns
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary mapping feature names to types
        """
        feature_types = {}
        for col in df.columns:
            feature_types[col] = self.stats_tests.detect_feature_type(df[col])
        return feature_types
    
    def detect_drift(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                    features: Optional[List[str]] = None,
                    use_multiple_tests: bool = True) -> Dict:
        """
        Detect drift for all features or specified features
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            features: List of features to analyze (None = all)
            use_multiple_tests: Whether to use multiple tests per feature
            
        Returns:
            Dictionary with drift detection results
        """
        # Validate inputs
        if features is None:
            features = list(set(train_df.columns) & set(test_df.columns))
        
        # Detect feature types
        self.feature_types = self.detect_feature_types(train_df[features])
        
        drift_results = {}
        summary = {
            'total_features': len(features),
            'features_with_drift': 0,
            'drift_percentage': 0.0,
            'feature_results': {}
        }
        
        for feature in features:
            if feature not in train_df.columns or feature not in test_df.columns:
                continue
                
            train_series = train_df[feature]
            test_series = test_df[feature]
            
            feature_type = self.feature_types[feature]
            
            # Run appropriate tests
            test_results = {}
            
            if use_multiple_tests:
                # Run multiple tests for comprehensive analysis
                if feature_type == 'numeric':
                    # KS test
                    test_results['ks_test'] = self.stats_tests.kolmogorov_smirnov_test(
                        train_series, test_series, self.alpha
                    )
                    # Mann-Whitney U test
                    test_results['mann_whitney'] = self.stats_tests.mann_whitney_u_test(
                        train_series, test_series, self.alpha
                    )
                    # PSI test
                    test_results['psi'] = self.stats_tests.psi_test(
                        train_series, test_series, threshold=self.psi_threshold
                    )
                    # Wasserstein distance
                    test_results['wasserstein'] = self.stats_tests.wasserstein_distance(
                        train_series, test_series
                    )
                else:  # categorical or binary
                    # Chi-square test
                    test_results['chi_square'] = self.stats_tests.chi_square_test(
                        train_series, test_series, self.alpha
                    )
                    # PSI test
                    test_results['psi'] = self.stats_tests.psi_test(
                        train_series, test_series, threshold=self.psi_threshold
                    )
            else:
                # Use single most appropriate test
                test_name = self.stats_tests.get_appropriate_test(feature_type)
                if test_name == 'kolmogorov_smirnov':
                    test_results[test_name] = self.stats_tests.kolmogorov_smirnov_test(
                        train_series, test_series, self.alpha
                    )
                elif test_name == 'mann_whitney_u':
                    test_results[test_name] = self.stats_tests.mann_whitney_u_test(
                        train_series, test_series, self.alpha
                    )
                elif test_name == 'chi_square':
                    test_results[test_name] = self.stats_tests.chi_square_test(
                        train_series, test_series, self.alpha
                    )
            
            # Determine overall drift status
            drift_detected = any(
                result.get('drift_detected', False) 
                for result in test_results.values()
            )
            
            # Calculate drift severity
            severity = self._calculate_drift_severity(test_results, feature_type)
            
            # Store results
            drift_results[feature] = {
                'feature_type': feature_type,
                'drift_detected': drift_detected,
                'severity': severity,
                'test_results': test_results,
                'train_stats': self._calculate_statistics(train_series, feature_type),
                'test_stats': self._calculate_statistics(test_series, feature_type)
            }
            
            if drift_detected:
                summary['features_with_drift'] += 1
        
        summary['drift_percentage'] = (
            summary['features_with_drift'] / summary['total_features'] * 100
            if summary['total_features'] > 0 else 0
        )
        summary['feature_results'] = drift_results
        
        self.drift_results = summary
        return summary
    
    def _calculate_drift_severity(self, test_results: Dict, 
                                  feature_type: str) -> str:
        """
        Calculate drift severity based on test results
        
        Args:
            test_results: Dictionary of test results
            feature_type: Type of feature
            
        Returns:
            Severity level: 'low', 'medium', or 'high'
        """
        severity_scores = []
        
        for test_name, result in test_results.items():
            if not result.get('drift_detected', False):
                continue
            
            if test_name == 'psi':
                psi_value = result.get('psi', 0)
                if psi_value > 0.5:
                    severity_scores.append(3)  # High
                elif psi_value > 0.25:
                    severity_scores.append(2)  # Medium
                else:
                    severity_scores.append(1)  # Low
            elif test_name in ['ks_test', 'mann_whitney', 'chi_square']:
                pvalue = result.get('pvalue', 1.0)
                if pvalue < 0.001:
                    severity_scores.append(3)  # High
                elif pvalue < 0.01:
                    severity_scores.append(2)  # Medium
                else:
                    severity_scores.append(1)  # Low
            elif test_name == 'wasserstein':
                norm_dist = result.get('normalized_distance', 0)
                if norm_dist > 0.3:
                    severity_scores.append(3)
                elif norm_dist > 0.15:
                    severity_scores.append(2)
                else:
                    severity_scores.append(1)
        
        if not severity_scores:
            return 'none'
        
        avg_severity = np.mean(severity_scores)
        if avg_severity >= 2.5:
            return 'high'
        elif avg_severity >= 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_statistics(self, series: pd.Series, 
                             feature_type: str) -> Dict:
        """
        Calculate descriptive statistics for a feature
        
        Args:
            series: Input series
            feature_type: Type of feature
            
        Returns:
            Dictionary with statistics
        """
        stats_dict = {
            'count': len(series),
            'missing_count': series.isna().sum(),
            'missing_percentage': series.isna().sum() / len(series) * 100
        }
        
        if feature_type == 'numeric':
            stats_dict.update({
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'q25': series.quantile(0.25),
                'q75': series.quantile(0.75)
            })
        else:  # categorical or binary
            value_counts = series.value_counts()
            stats_dict.update({
                'unique_count': series.nunique(),
                'mode': value_counts.index[0] if len(value_counts) > 0 else None,
                'mode_frequency': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'value_distribution': value_counts.to_dict()
            })
        
        return stats_dict
    
    def get_drift_summary(self) -> pd.DataFrame:
        """
        Get summary of drift detection results as DataFrame
        
        Returns:
            DataFrame with drift summary
        """
        if not self.drift_results:
            return pd.DataFrame()
        
        summary_data = []
        for feature, results in self.drift_results['feature_results'].items():
            summary_data.append({
                'feature': feature,
                'feature_type': results['feature_type'],
                'drift_detected': results['drift_detected'],
                'severity': results['severity']
            })
        
        return pd.DataFrame(summary_data)
    
    def get_features_with_drift(self, severity: Optional[str] = None) -> List[str]:
        """
        Get list of features with detected drift
        
        Args:
            severity: Filter by severity ('low', 'medium', 'high', None for all)
            
        Returns:
            List of feature names
        """
        if not self.drift_results:
            return []
        
        features = []
        for feature, results in self.drift_results['feature_results'].items():
            if results['drift_detected']:
                if severity is None or results['severity'] == severity:
                    features.append(feature)
        
        return features
