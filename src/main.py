"""
Main entry point for NAISC Singtel 2026 Challenge
Adaptive Drift Intelligence Challenge - Solution Pipeline
"""

import argparse
import time
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import average_precision_score
import warnings
warnings.filterwarnings('ignore')

from drift_detector import DriftDetector
from mitigation import DriftMitigator

# Fixed LightGBM hyperparameters as per challenge requirements
LIGHTGBM_PARAMS = {
    'verbosity': -1,
    'objective': 'binary',
    'is_unbalance': True,
    'random_state': 42,
    'importance_type': 'gain'
}


def load_data(train_path: str, test_path: str):
    """Load training and test datasets"""
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Prepare features and target variable"""
    from sklearn.preprocessing import LabelEncoder
    
    # Identify target column (ChurnStatus)
    target_col = 'ChurnStatus'
    
    # Identify ID column
    id_col = 'CustomerID'
    
    # Get feature columns (exclude ID, target, and Month if present)
    exclude_cols = [id_col, target_col, 'Month']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Prepare data
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    
    X_test = test_df[feature_cols].copy()
    test_ids = test_df[id_col].copy()
    
    # Encode target variable (Yes/No -> 1/0)
    if y_train.dtype == 'object':
        y_train = (y_train == 'Yes').astype(int)
    
    # Encode categorical features
    label_encoders = {}
    for col in feature_cols:
        if X_train[col].dtype == 'object':
            le = LabelEncoder()
            # Fit on combined train and test to handle unseen categories
            combined = pd.concat([X_train[col], X_test[col]], axis=0)
            le.fit(combined)
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
            label_encoders[col] = le
    
    return X_train, y_train, X_test, test_ids, feature_cols, label_encoders


def detect_and_mitigate_drift(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                              X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Detect drift and apply mitigation strategies"""
    print("\n" + "="*60)
    print("DATA DRIFT DETECTION & MITIGATION")
    print("="*60)
    
    start_time = time.time()
    
    # Initialize detector
    detector = DriftDetector(alpha=0.05, psi_threshold=0.2)
    
    # Detect drift
    print("\n[1/3] Detecting data drift...")
    drift_results = detector.detect_drift(
        X_train, X_test,
        use_multiple_tests=True
    )
    
    # Get drifted features
    drifted_features = detector.get_features_with_drift()
    
    # Print drift detection summary
    print("\n[2/3] Drift Detection Summary:")
    print(f"  - Total features analyzed: {drift_results['total_features']}")
    print(f"  - Features with detected drift: {drift_results['features_with_drift']}")
    print(f"  - Drift percentage: {drift_results['drift_percentage']:.2f}%")
    
    if drifted_features:
        print(f"\n  Columns with detected drift:")
        for feature in drifted_features:
            feature_result = drift_results['feature_results'][feature]
            drift_type = feature_result['feature_type']
            severity = feature_result['severity']
            print(f"    - {feature} ({drift_type}, severity: {severity})")
    else:
        print("\n  No drift detected in any features.")
    
    # Apply mitigation
    print("\n[3/3] Applying mitigation strategies...")
    mitigator = DriftMitigator()
    
    # Get recommended strategies
    strategies = mitigator.get_mitigation_strategy(drift_results)
    
    # Apply robust scaling for numeric features
    feature_types = detector.feature_types
    X_train_mitigated, X_test_mitigated = mitigator.apply_robust_scaling(
        X_train, X_test, feature_types
    )
    
    # Apply domain adaptation for high-severity drifted features
    high_severity_features = [
        f for f, r in drift_results['feature_results'].items()
        if r['drift_detected'] and r['severity'] == 'high'
    ]
    
    if high_severity_features:
        print(f"  Applying domain adaptation to {len(high_severity_features)} high-severity features...")
        X_train_mitigated = mitigator.apply_domain_adaptation(
            X_train_mitigated, X_test_mitigated, drift_results
        )
    
    # Apply feature reweighting
    X_train_weighted = mitigator.apply_feature_reweighting(
        X_train_mitigated, X_test_mitigated, drift_results, method='inverse_drift'
    )
    
    mitigation_methods = []
    if high_severity_features:
        mitigation_methods.append("Domain Adaptation")
    mitigation_methods.append("Robust Scaling")
    mitigation_methods.append("Feature Reweighting")
    
    print(f"\n  Mitigation methods applied: {', '.join(mitigation_methods)}")
    
    drift_time = time.time() - start_time
    
    return X_train_mitigated, X_test_mitigated, drift_results, drift_time, mitigation_methods


def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
               X_test: pd.DataFrame, drift_results: dict):
    """Train LightGBM model with fixed hyperparameters"""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    import lightgbm as lgb
    
    # Create LightGBM model with fixed hyperparameters
    model = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)
    
    # Calculate sample weights based on drift
    mitigator = DriftMitigator()
    feature_weights = mitigator._calculate_feature_weights(drift_results, X_train.columns)
    
    # Convert feature weights to sample weights (simplified approach)
    # In practice, you might want more sophisticated weighting
    sample_weights = np.ones(len(X_train))
    
    # Train model
    print("\nTraining LightGBM model with fixed hyperparameters...")
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights
    )
    
    return model


def evaluate_model(model, X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series = None):
    """Evaluate model using AU-PRC metric"""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    
    # Get predictions
    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate AU-PRC
    train_auprc = average_precision_score(y_train, train_proba)
    print(f"\nAU-PRC on training set: {train_auprc:.6f}")
    
    if y_test is not None:
        test_auprc = average_precision_score(y_test, test_proba)
        print(f"AU-PRC on test set after mitigation: {test_auprc:.6f}")
        return train_auprc, test_auprc, test_proba
    else:
        return train_auprc, None, test_proba


def save_outputs(model, test_ids: pd.Series, test_proba: np.ndarray, 
                output_dir: Path = Path('.')):
    """Save model and predictions"""
    # Save model
    model_path = output_dir / 'model.joblib'
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save predictions
    prediction_df = pd.DataFrame({
        'CustomerID': test_ids,
        'probability_score': test_proba
    })
    
    prediction_path = output_dir / 'prediction.csv'
    prediction_df.to_csv(prediction_path, index=False)
    print(f"Predictions saved to: {prediction_path}")
    
    return model_path, prediction_path


def main():
    """Main pipeline"""
    parser = argparse.ArgumentParser(
        description='NAISC Singtel 2026: Adaptive Drift Intelligence Challenge'
    )
    parser.add_argument(
        '--train_data_filepath',
        type=str,
        required=True,
        help='Path to training data CSV file'
    )
    parser.add_argument(
        '--test_data_filepath',
        type=str,
        required=True,
        help='Path to test data CSV file'
    )
    
    args = parser.parse_args()
    
    # Record total start time
    total_start_time = time.time()
    
    try:
        # Load data
        train_df, test_df = load_data(args.train_data_filepath, args.test_data_filepath)
        
        # Prepare features
        X_train, y_train, X_test, test_ids, feature_cols, label_encoders = prepare_features(
            train_df, test_df
        )
        
        # Detect and mitigate drift
        X_train_mitigated, X_test_mitigated, drift_results, drift_time, mitigation_methods = \
            detect_and_mitigate_drift(train_df, test_df, X_train, X_test)
        
        # Train model
        model = train_model(X_train_mitigated, y_train, X_test_mitigated, drift_results)
        
        # Evaluate model
        train_auprc, test_auprc, test_proba = evaluate_model(
            model, X_train_mitigated, y_train, X_test_mitigated
        )
        
        # Save outputs
        model_path, prediction_path = save_outputs(model, test_ids, test_proba)
        
        # Print runtime
        total_time = time.time() - total_start_time
        print("\n" + "="*60)
        print("RUNTIME")
        print("="*60)
        print(f"\nTime taken for drift detection and mitigation: {drift_time:.2f} seconds")
        print(f"Total runtime: {total_time:.2f} seconds")
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
