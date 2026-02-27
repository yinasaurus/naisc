"""
Adaptive training techniques for handling data drift
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


class AdaptiveTrainer:
    """
    Adaptive training techniques that account for data drift
    """
    
    def __init__(self, model_type: str = 'auto'):
        """
        Initialize adaptive trainer
        
        Args:
            model_type: Type of model ('auto', 'rf', 'xgb', 'lgb')
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = {}
        self.training_history = []
    
    def train_with_drift_awareness(self, X_train: pd.DataFrame,
                                   y_train: pd.Series,
                                   X_test: pd.DataFrame,
                                   y_test: Optional[pd.Series],
                                   drift_results: Dict,
                                   task_type: str = 'classification',
                                   use_weights: bool = True) -> Dict:
        """
        Train model with drift-aware techniques
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels (optional)
            drift_results: Drift detection results
            task_type: 'classification' or 'regression'
            use_weights: Whether to use feature weights
            
        Returns:
            Dictionary with training results and model
        """
        # Get feature weights based on drift
        if use_weights:
            feature_weights = self._calculate_feature_weights(drift_results, X_train.columns)
        else:
            feature_weights = {f: 1.0 for f in X_train.columns}
        
        # Select model
        if self.model_type == 'auto':
            model = self._select_best_model(X_train, y_train, task_type)
        else:
            model = self._create_model(self.model_type, task_type)
        
        # Train with sample weights if applicable
        sample_weights = self._calculate_sample_weights(
            X_train, drift_results, feature_weights
        )
        
        # Fit model with sample weights if supported
        try:
            if hasattr(model, 'fit'):
                fit_params = {}
                # Check if model supports sample_weight
                import inspect
                sig = inspect.signature(model.fit)
                if 'sample_weight' in sig.parameters:
                    fit_params['sample_weight'] = sample_weights
                model.fit(X_train, y_train, **fit_params)
        except Exception as e:
            # Fallback to fitting without sample weights
            model.fit(X_train, y_train)
        
        self.model = model
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        results = {
            'model': model,
            'feature_weights': feature_weights,
            'train_predictions': train_pred,
            'test_predictions': test_pred
        }
        
        if task_type == 'classification':
            results['train_accuracy'] = accuracy_score(y_train, train_pred)
            if y_test is not None:
                results['test_accuracy'] = accuracy_score(y_test, test_pred)
                results['classification_report'] = classification_report(
                    y_test, test_pred, output_dict=True
                )
        else:
            results['train_rmse'] = np.sqrt(mean_squared_error(y_train, train_pred))
            if y_test is not None:
                results['test_rmse'] = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                X_train.columns,
                model.feature_importances_
            ))
            results['feature_importance'] = self.feature_importance
        
        return results
    
    def train_ensemble_with_drift(self, X_train: pd.DataFrame,
                                  y_train: pd.Series,
                                  X_test: pd.DataFrame,
                                  y_test: Optional[pd.Series],
                                  drift_results: Dict,
                                  task_type: str = 'classification') -> Dict:
        """
        Train ensemble model with drift-aware weighting
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels (optional)
            drift_results: Drift detection results
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with ensemble results
        """
        models = []
        predictions = []
        weights = []
        
        # Train multiple models
        for model_name in ['rf', 'xgb', 'lgb']:
            model = self._create_model(model_name, task_type)
            
            feature_weights = self._calculate_feature_weights(
                drift_results, X_train.columns
            )
            sample_weights = self._calculate_sample_weights(
                X_train, drift_results, feature_weights
            )
            
            # Fit with sample weights if supported
            try:
                import inspect
                sig = inspect.signature(model.fit)
                if 'sample_weight' in sig.parameters:
                    model.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    model.fit(X_train, y_train)
            except:
                model.fit(X_train, y_train)
            models.append(model)
            
            pred = model.predict(X_test)
            predictions.append(pred)
            
            # Weight based on feature drift
            drift_score = self._calculate_drift_score(drift_results, X_train.columns)
            weights.append(1.0 / (1.0 + drift_score))
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Ensemble prediction
        if task_type == 'classification':
            # Voting
            predictions_array = np.array(predictions)
            ensemble_pred = []
            for i in range(len(X_test)):
                votes = predictions_array[:, i]
                ensemble_pred.append(np.bincount(votes.astype(int)).argmax())
            ensemble_pred = np.array(ensemble_pred)
        else:
            # Weighted average
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        results = {
            'models': models,
            'ensemble_predictions': ensemble_pred,
            'model_weights': weights.tolist(),
            'individual_predictions': predictions
        }
        
        if y_test is not None:
            if task_type == 'classification':
                results['ensemble_accuracy'] = accuracy_score(y_test, ensemble_pred)
            else:
                results['ensemble_rmse'] = np.sqrt(
                    mean_squared_error(y_test, ensemble_pred)
                )
        
        return results
    
    def _create_model(self, model_type: str, task_type: str) -> Any:
        """Create model instance"""
        if model_type == 'rf':
            if task_type == 'classification':
                return RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                return RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'xgb':
            if task_type == 'classification':
                return xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            else:
                return xgb.XGBRegressor(random_state=42)
        elif model_type == 'lgb':
            if task_type == 'classification':
                return lgb.LGBMClassifier(random_state=42, verbose=-1)
            else:
                return lgb.LGBMRegressor(random_state=42, verbose=-1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _select_best_model(self, X_train: pd.DataFrame,
                          y_train: pd.Series,
                          task_type: str) -> Any:
        """Select best model based on cross-validation"""
        models = {
            'rf': self._create_model('rf', task_type),
            'xgb': self._create_model('xgb', task_type),
            'lgb': self._create_model('lgb', task_type)
        }
        
        best_score = -np.inf
        best_model = None
        
        for name, model in models.items():
            try:
                scores = cross_val_score(
                    model, X_train, y_train, cv=3, scoring='accuracy' 
                    if task_type == 'classification' else 'neg_mean_squared_error'
                )
                score = scores.mean()
                
                if score > best_score:
                    best_score = score
                    best_model = model
            except:
                continue
        
        return best_model if best_model else models['rf']
    
    def _calculate_feature_weights(self, drift_results: Dict,
                                   features: List[str]) -> Dict[str, float]:
        """Calculate feature weights based on drift"""
        weights = {}
        
        for feature in features:
            if feature in drift_results['feature_results']:
                results = drift_results['feature_results'][feature]
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
            else:
                weights[feature] = 1.0
        
        return weights
    
    def _calculate_sample_weights(self, X_train: pd.DataFrame,
                                  drift_results: Dict,
                                  feature_weights: Dict[str, float]) -> np.ndarray:
        """Calculate sample weights based on feature drift"""
        # Simple approach: weight samples based on how similar they are to test distribution
        # This is a simplified version
        n_samples = len(X_train)
        weights = np.ones(n_samples)
        
        # Could be enhanced with more sophisticated weighting
        return weights
    
    def _calculate_drift_score(self, drift_results: Dict,
                               features: List[str]) -> float:
        """Calculate overall drift score"""
        total_drift = 0.0
        count = 0
        
        for feature in features:
            if feature in drift_results['feature_results']:
                results = drift_results['feature_results'][feature]
                if results['drift_detected']:
                    severity = results['severity']
                    if severity == 'high':
                        total_drift += 3.0
                    elif severity == 'medium':
                        total_drift += 2.0
                    else:
                        total_drift += 1.0
                    count += 1
        
        return total_drift / max(count, 1)
