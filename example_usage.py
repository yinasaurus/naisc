"""
Example usage of the Adaptive Drift Intelligence framework
"""

import pandas as pd
import numpy as np
from src.drift_detector import DriftDetector
from src.visualization import DriftVisualizer
from src.mitigation import DriftMitigator, AdaptiveTrainer

def generate_sample_data():
    """Generate sample training and test data with intentional drift"""
    np.random.seed(42)
    
    # Training data
    n_train = 1000
    train_data = {
        'feature_1': np.random.normal(0, 1, n_train),
        'feature_2': np.random.normal(5, 2, n_train),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_train, p=[0.5, 0.3, 0.2]),
        'feature_4': np.random.exponential(2, n_train),
        'target': np.random.choice([0, 1], n_train)
    }
    train_df = pd.DataFrame(train_data)
    
    # Test data with drift
    n_test = 500
    test_data = {
        'feature_1': np.random.normal(0.5, 1.2, n_test),  # Mean shift
        'feature_2': np.random.normal(6, 2.5, n_test),  # Distribution shift
        'feature_3': np.random.choice(['A', 'B', 'C'], n_test, p=[0.3, 0.4, 0.3]),  # Categorical drift
        'feature_4': np.random.exponential(3, n_test),  # Scale shift
        'target': np.random.choice([0, 1], n_test)
    }
    test_df = pd.DataFrame(test_data)
    
    return train_df, test_df

def main():
    """Main example workflow"""
    print("=" * 60)
    print("Adaptive Drift Intelligence - Example Usage")
    print("=" * 60)
    
    # 1. Generate or load data
    print("\n1. Loading data...")
    train_df, test_df = generate_sample_data()
    print(f"   Training data: {train_df.shape}")
    print(f"   Test data: {test_df.shape}")
    
    # 2. Initialize detector
    print("\n2. Initializing drift detector...")
    detector = DriftDetector(alpha=0.05, psi_threshold=0.2)
    
    # 3. Detect drift
    print("\n3. Detecting drift...")
    drift_results = detector.detect_drift(
        train_df.drop(columns=['target']),
        test_df.drop(columns=['target']),
        use_multiple_tests=True
    )
    
    # 4. Display results
    print("\n4. Drift Detection Results:")
    print(f"   Total features analyzed: {drift_results['total_features']}")
    print(f"   Features with drift: {drift_results['features_with_drift']}")
    print(f"   Drift percentage: {drift_results['drift_percentage']:.2f}%")
    
    # Get features with drift
    drifted_features = detector.get_features_with_drift()
    print(f"\n   Features with detected drift: {drifted_features}")
    
    # 5. Display detailed results for each feature
    print("\n5. Detailed Feature Analysis:")
    for feature, results in drift_results['feature_results'].items():
        print(f"\n   Feature: {feature}")
        print(f"   - Type: {results['feature_type']}")
        print(f"   - Drift detected: {results['drift_detected']}")
        print(f"   - Severity: {results['severity']}")
        
        # Show test results
        for test_name, test_result in results['test_results'].items():
            if 'pvalue' in test_result:
                print(f"   - {test_result['test_name']}: p-value = {test_result['pvalue']:.4f}")
            elif 'psi' in test_result:
                print(f"   - PSI: {test_result['psi']:.4f}")
    
    # 6. Apply mitigation
    print("\n6. Applying mitigation strategies...")
    mitigator = DriftMitigator()
    
    # Get recommended strategies
    strategies = mitigator.get_mitigation_strategy(drift_results)
    print("\n   Recommended mitigation strategies:")
    for feature, strategy in strategies.items():
        if strategy != 'none':
            print(f"   - {feature}: {strategy}")
    
    # Apply robust scaling
    feature_types = detector.feature_types
    train_scaled, test_scaled = mitigator.apply_robust_scaling(
        train_df.drop(columns=['target']),
        test_df.drop(columns=['target']),
        feature_types
    )
    print("\n   Applied robust scaling to numeric features")
    
    # 7. Adaptive training
    print("\n7. Training model with drift awareness...")
    trainer = AdaptiveTrainer(model_type='auto')
    
    X_train = train_scaled
    y_train = train_df['target']
    X_test = test_scaled
    y_test = test_df['target']
    
    training_results = trainer.train_with_drift_awareness(
        X_train, y_train, X_test, y_test,
        drift_results, task_type='classification', use_weights=True
    )
    
    print(f"\n   Training accuracy: {training_results['train_accuracy']:.4f}")
    print(f"   Test accuracy: {training_results['test_accuracy']:.4f}")
    
    # 8. Feature importance
    if 'feature_importance' in training_results:
        print("\n   Top 5 most important features:")
        importance = training_results['feature_importance']
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, imp in sorted_features[:5]:
            print(f"   - {feature}: {imp:.4f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
