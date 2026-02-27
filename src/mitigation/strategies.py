"""
Mitigation strategies for handling data drift
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DriftMitigator:
    """
    Implements various strategies to mitigate data drift
    """
    
    def __init__(self):
        """Initialize mitigator"""
        self.scalers = {}
        self.encoders = {}
        self.feature_weights = {}
    
    def apply_feature_reweighting(self, train_df: pd.DataFrame,
                                  test_df: pd.DataFrame,
                                  drift_results: Dict,
                                  method: str = 'importance') -> pd.DataFrame:
        """
        Apply feature reweighting based on drift severity
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            drift_results: Drift detection results
            method: Reweighting method ('importance', 'inverse_drift', 'uniform')
            
        Returns:
            Reweighted training dataframe
        """
        if method == 'uniform':
            # Uniform weights
            weights = {f: 1.0 for f in train_df.columns}
        elif method == 'inverse_drift':
            # Inverse of drift severity
            weights = {}
            for feature in train_df.columns:
                if feature in drift_results['feature_results']:
                    severity = drift_results['feature_results'][feature]['severity']
                    if severity == 'high':
                        weights[feature] = 0.5
                    elif severity == 'medium':
                        weights[feature] = 0.75
                    elif severity == 'low':
                        weights[feature] = 0.9
                    else:
                        weights[feature] = 1.0
                else:
                    weights[feature] = 1.0
        else:  # importance-based
            # Weight based on inverse drift detection
            weights = {}
            for feature in train_df.columns:
                if feature in drift_results['feature_results']:
                    if drift_results['feature_results'][feature]['drift_detected']:
                        # Reduce weight for drifted features
                        weights[feature] = 0.7
                    else:
                        weights[feature] = 1.0
                else:
                    weights[feature] = 1.0
        
        self.feature_weights = weights
        
        # Apply weights (for demonstration, we'll return weighted samples)
        # In practice, this would be used during model training
        return train_df.copy()
    
    def apply_data_augmentation(self, train_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               drift_results: Dict,
                               augmentation_ratio: float = 0.2) -> pd.DataFrame:
        """
        Augment training data with samples similar to test distribution
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            drift_results: Drift detection results
            augmentation_ratio: Ratio of augmented samples to add
            
        Returns:
            Augmented training dataframe
        """
        # Identify drifted features
        drifted_features = [
            f for f, r in drift_results['feature_results'].items()
            if r['drift_detected']
        ]
        
        if not drifted_features:
            return train_df.copy()
        
        # Sample from test data for drifted features
        n_augment = int(len(train_df) * augmentation_ratio)
        test_samples = test_df.sample(
            n=min(n_augment, len(test_df)),
            replace=True if n_augment > len(test_df) else False
        )
        
        # Create augmented samples
        augmented_df = train_df.copy()
        
        # For each augmented sample, replace drifted features with test values
        for idx, test_idx in enumerate(test_samples.index):
            if idx >= n_augment:
                break
            
            aug_sample = train_df.iloc[idx % len(train_df)].copy()
            for feature in drifted_features:
                if feature in test_samples.columns:
                    aug_sample[feature] = test_samples.loc[test_idx, feature]
            
            augmented_df = pd.concat([augmented_df, aug_sample.to_frame().T], 
                                   ignore_index=True)
        
        return augmented_df
    
    def apply_robust_scaling(self, train_df: pd.DataFrame,
                           test_df: pd.DataFrame,
                           feature_types: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply robust scaling to reduce impact of outliers and distribution shifts
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            feature_types: Dictionary of feature types
            
        Returns:
            Scaled training and test dataframes
        """
        train_scaled = train_df.copy()
        test_scaled = test_df.copy()
        
        numeric_features = [f for f, t in feature_types.items() if t == 'numeric']
        
        for feature in numeric_features:
            if feature not in train_df.columns:
                continue
            
            # Use RobustScaler (less sensitive to outliers)
            scaler = RobustScaler()
            train_scaled[feature] = scaler.fit_transform(
                train_df[[feature]]
            ).flatten()
            test_scaled[feature] = scaler.transform(
                test_df[[feature]]
            ).flatten()
            
            self.scalers[feature] = scaler
        
        return train_scaled, test_scaled
    
    def apply_domain_adaptation(self, train_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               drift_results: Dict,
                               adaptation_method: str = 'mapping') -> pd.DataFrame:
        """
        Apply domain adaptation techniques
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            drift_results: Drift detection results
            adaptation_method: Method ('mapping', 'translation')
            
        Returns:
            Adapted training dataframe
        """
        adapted_df = train_df.copy()
        
        for feature, results in drift_results['feature_results'].items():
            if not results['drift_detected']:
                continue
            
            feature_type = results['feature_type']
            
            if feature_type == 'numeric':
                # Map training distribution to test distribution
                train_mean = results['train_stats']['mean']
                test_mean = results['test_stats']['mean']
                train_std = results['train_stats']['std']
                test_std = results['test_stats']['std']
                
                if train_std > 0:
                    # Standardize then re-scale to test distribution
                    adapted_df[feature] = (
                        (adapted_df[feature] - train_mean) / train_std * test_std + test_mean
                    )
            else:
                # For categorical, use label mapping
                train_dist = results['train_stats']['value_distribution']
                test_dist = results['test_stats']['value_distribution']
                
                # Map less frequent categories in training to more frequent in test
                # This is a simplified approach
                pass
        
        return adapted_df
    
    def create_ensemble_weights(self, drift_results: Dict) -> Dict[str, float]:
        """
        Create weights for ensemble models based on drift severity
        
        Args:
            drift_results: Drift detection results
            
        Returns:
            Dictionary of feature weights
        """
        weights = {}
        
        for feature, results in drift_results['feature_results'].items():
            if results['drift_detected']:
                severity = results['severity']
                if severity == 'high':
                    weights[feature] = 0.3
                elif severity == 'medium':
                    weights[feature] = 0.6
                else:
                    weights[feature] = 0.8
            else:
                weights[feature] = 1.0
        
        return weights
    
    def get_mitigation_strategy(self, drift_results: Dict) -> Dict[str, str]:
        """
        Recommend mitigation strategy based on drift results
        
        Args:
            drift_results: Drift detection results
            
        Returns:
            Dictionary mapping features to recommended strategies
        """
        strategies = {}
        
        for feature, results in drift_results['feature_results'].items():
            if not results['drift_detected']:
                strategies[feature] = 'none'
                continue
            
            severity = results['severity']
            feature_type = results['feature_type']
            
            if severity == 'high':
                if feature_type == 'numeric':
                    strategies[feature] = 'robust_scaling + domain_adaptation'
                else:
                    strategies[feature] = 'data_augmentation + reweighting'
            elif severity == 'medium':
                strategies[feature] = 'robust_scaling'
            else:
                strategies[feature] = 'reweighting'
        
        return strategies
