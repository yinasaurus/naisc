"""
Streamlit Dashboard for Data Drift Detection and Mitigation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.drift_detector import DriftDetector
from src.visualization import DriftVisualizer
from src.mitigation import DriftMitigator, AdaptiveTrainer
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Adaptive Drift Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = DriftDetector()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = DriftVisualizer()
if 'mitigator' not in st.session_state:
    st.session_state.mitigator = DriftMitigator()
if 'trainer' not in st.session_state:
    st.session_state.trainer = AdaptiveTrainer()
if 'drift_results' not in st.session_state:
    st.session_state.drift_results = None
if 'train_df' not in st.session_state:
    st.session_state.train_df = None
if 'test_df' not in st.session_state:
    st.session_state.test_df = None

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">🛡️ Adaptive Drift Intelligence Challenge</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Guarding Model Integrity in a Shifting Data World")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📥 Data Upload", "🔍 Drift Detection", "📊 Visualizations", 
         "🔧 Mitigation", "🤖 Adaptive Training", "📈 Results"]
    )
    
    if page == "📥 Data Upload":
        data_upload_page()
    elif page == "🔍 Drift Detection":
        drift_detection_page()
    elif page == "📊 Visualizations":
        visualizations_page()
    elif page == "🔧 Mitigation":
        mitigation_page()
    elif page == "🤖 Adaptive Training":
        adaptive_training_page()
    elif page == "📈 Results":
        results_page()

def data_upload_page():
    """Data upload page"""
    st.header("📥 Data Upload")
    st.markdown("Upload your training and test datasets to begin drift detection.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Dataset")
        train_file = st.file_uploader(
            "Upload Training CSV",
            type=['csv'],
            key='train_upload'
        )
        
        if train_file is not None:
            try:
                train_df = pd.read_csv(train_file)
                st.session_state.train_df = train_df
                st.success(f"✅ Training data loaded: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
                
                with st.expander("View Training Data"):
                    st.dataframe(train_df.head(100))
                    st.write(f"**Shape:** {train_df.shape}")
                    st.write(f"**Columns:** {list(train_df.columns)}")
            except Exception as e:
                st.error(f"Error loading training data: {str(e)}")
    
    with col2:
        st.subheader("Test Dataset")
        test_file = st.file_uploader(
            "Upload Test CSV",
            type=['csv'],
            key='test_upload'
        )
        
        if test_file is not None:
            try:
                test_df = pd.read_csv(test_file)
                st.session_state.test_df = test_df
                st.success(f"✅ Test data loaded: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
                
                with st.expander("View Test Data"):
                    st.dataframe(test_df.head(100))
                    st.write(f"**Shape:** {test_df.shape}")
                    st.write(f"**Columns:** {list(test_df.columns)}")
            except Exception as e:
                st.error(f"Error loading test data: {str(e)}")
    
    # Generate sample data option
    st.markdown("---")
    st.subheader("Or Generate Sample Data")
    if st.button("Generate Sample Data with Drift"):
        train_df, test_df = generate_sample_data_with_drift()
        st.session_state.train_df = train_df
        st.session_state.test_df = test_df
        st.success("✅ Sample data generated!")
        st.rerun()

def drift_detection_page():
    """Drift detection page"""
    st.header("🔍 Drift Detection")
    
    if st.session_state.train_df is None or st.session_state.test_df is None:
        st.warning("⚠️ Please upload training and test datasets first.")
        return
    
    train_df = st.session_state.train_df
    test_df = st.session_state.test_df
    
    # Configuration
    st.subheader("Detection Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        alpha = st.slider("Significance Level (α)", 0.01, 0.1, 0.05, 0.01)
    with col2:
        psi_threshold = st.slider("PSI Threshold", 0.1, 0.5, 0.2, 0.05)
    with col3:
        use_multiple_tests = st.checkbox("Use Multiple Tests", value=True)
    
    # Feature selection
    common_features = list(set(train_df.columns) & set(test_df.columns))
    selected_features = st.multiselect(
        "Select Features to Analyze",
        common_features,
        default=common_features[:min(10, len(common_features))]
    )
    
    if st.button("🔍 Detect Drift", type="primary"):
        with st.spinner("Analyzing data drift..."):
            # Update detector parameters
            st.session_state.detector.alpha = alpha
            st.session_state.detector.psi_threshold = psi_threshold
            
            # Detect drift
            drift_results = st.session_state.detector.detect_drift(
                train_df,
                test_df,
                features=selected_features if selected_features else None,
                use_multiple_tests=use_multiple_tests
            )
            
            st.session_state.drift_results = drift_results
            
            st.success("✅ Drift detection completed!")
    
    # Display results
    if st.session_state.drift_results:
        results = st.session_state.drift_results
        
        st.markdown("---")
        st.subheader("📊 Detection Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Features", results['total_features'])
        with col2:
            st.metric("Features with Drift", results['features_with_drift'])
        with col3:
            st.metric("Drift Percentage", f"{results['drift_percentage']:.2f}%")
        with col4:
            no_drift = results['total_features'] - results['features_with_drift']
            st.metric("No Drift", no_drift)
        
        # Detailed results
        st.markdown("---")
        st.subheader("📋 Detailed Results")
        
        summary_df = st.session_state.detector.get_drift_summary()
        st.dataframe(summary_df, use_container_width=True)
        
        # Features with drift
        drifted_features = st.session_state.detector.get_features_with_drift()
        if drifted_features:
            st.markdown("**Features with Detected Drift:**")
            st.write(", ".join(drifted_features))

def visualizations_page():
    """Visualizations page"""
    st.header("📊 Data Drift Visualizations")
    
    if st.session_state.train_df is None or st.session_state.test_df is None:
        st.warning("⚠️ Please upload datasets first.")
        return
    
    if st.session_state.drift_results is None:
        st.warning("⚠️ Please run drift detection first.")
        return
    
    train_df = st.session_state.train_df
    test_df = st.session_state.test_df
    drift_results = st.session_state.drift_results
    
    # Summary visualization
    st.subheader("Overall Drift Summary")
    fig_summary = st.session_state.visualizer.plot_drift_summary(drift_results)
    st.plotly_chart(fig_summary, use_container_width=True)
    
    # Feature selection for detailed view
    st.markdown("---")
    st.subheader("Feature-Level Analysis")
    
    feature_list = list(drift_results['feature_results'].keys())
    selected_feature = st.selectbox("Select Feature", feature_list)
    
    if selected_feature:
        feature_results = drift_results['feature_results'][selected_feature]
        feature_type = feature_results['feature_type']
        
        # Distribution comparison
        st.markdown(f"### Distribution Comparison: {selected_feature}")
        fig_dist = st.session_state.visualizer.plot_distribution_comparison(
            train_df[selected_feature],
            test_df[selected_feature],
            selected_feature,
            feature_type
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Statistical test results
        st.markdown(f"### Statistical Test Results: {selected_feature}")
        fig_tests = st.session_state.visualizer.plot_statistical_test_results(
            selected_feature,
            feature_results['test_results']
        )
        st.plotly_chart(fig_tests, use_container_width=True)
        
        # Statistics comparison
        st.markdown(f"### Statistics Comparison: {selected_feature}")
        fig_stats = st.session_state.visualizer.plot_feature_statistics_comparison(
            feature_results['train_stats'],
            feature_results['test_stats'],
            selected_feature,
            feature_type
        )
        st.plotly_chart(fig_stats, use_container_width=True)
        
        # Display test results
        with st.expander("View Detailed Test Results"):
            for test_name, test_result in feature_results['test_results'].items():
                st.json(test_result)

def mitigation_page():
    """Mitigation strategies page"""
    st.header("🔧 Drift Mitigation Strategies")
    
    if st.session_state.train_df is None or st.session_state.test_df is None:
        st.warning("⚠️ Please upload datasets first.")
        return
    
    if st.session_state.drift_results is None:
        st.warning("⚠️ Please run drift detection first.")
        return
    
    train_df = st.session_state.train_df
    test_df = st.session_state.test_df
    drift_results = st.session_state.drift_results
    
    st.subheader("Mitigation Strategy Selection")
    
    mitigation_method = st.selectbox(
        "Select Mitigation Method",
        ["Feature Reweighting", "Data Augmentation", "Robust Scaling", 
         "Domain Adaptation", "Recommended Strategy"]
    )
    
    if st.button("Apply Mitigation", type="primary"):
        with st.spinner("Applying mitigation strategy..."):
            if mitigation_method == "Feature Reweighting":
                reweighting_method = st.selectbox(
                    "Reweighting Method",
                    ["importance", "inverse_drift", "uniform"]
                )
                mitigated_df = st.session_state.mitigator.apply_feature_reweighting(
                    train_df, test_df, drift_results, method=reweighting_method
                )
                st.success("✅ Feature reweighting applied!")
                
            elif mitigation_method == "Data Augmentation":
                augmentation_ratio = st.slider("Augmentation Ratio", 0.1, 0.5, 0.2, 0.05)
                mitigated_df = st.session_state.mitigator.apply_data_augmentation(
                    train_df, test_df, drift_results, augmentation_ratio
                )
                st.success(f"✅ Data augmentation applied! New size: {len(mitigated_df)}")
                
            elif mitigation_method == "Robust Scaling":
                feature_types = st.session_state.detector.feature_types
                train_scaled, test_scaled = st.session_state.mitigator.apply_robust_scaling(
                    train_df, test_df, feature_types
                )
                st.success("✅ Robust scaling applied!")
                st.session_state.train_df = train_scaled
                st.session_state.test_df = test_scaled
                
            elif mitigation_method == "Domain Adaptation":
                mitigated_df = st.session_state.mitigator.apply_domain_adaptation(
                    train_df, test_df, drift_results
                )
                st.success("✅ Domain adaptation applied!")
                
            elif mitigation_method == "Recommended Strategy":
                strategies = st.session_state.mitigator.get_mitigation_strategy(drift_results)
                st.subheader("Recommended Strategies by Feature")
                strategy_df = pd.DataFrame([
                    {'Feature': f, 'Strategy': s} 
                    for f, s in strategies.items()
                ])
                st.dataframe(strategy_df, use_container_width=True)
    
    # Display mitigation recommendations
    st.markdown("---")
    st.subheader("Mitigation Recommendations")
    strategies = st.session_state.mitigator.get_mitigation_strategy(drift_results)
    
    for feature, strategy in strategies.items():
        if strategy != 'none':
            with st.expander(f"Feature: {feature}"):
                st.write(f"**Recommended Strategy:** {strategy}")
                feature_results = drift_results['feature_results'][feature]
                st.write(f"**Severity:** {feature_results['severity']}")
                st.write(f"**Feature Type:** {feature_results['feature_type']}")

def adaptive_training_page():
    """Adaptive training page"""
    st.header("🤖 Adaptive Model Training")
    
    if st.session_state.train_df is None or st.session_state.test_df is None:
        st.warning("⚠️ Please upload datasets first.")
        return
    
    if st.session_state.drift_results is None:
        st.warning("⚠️ Please run drift detection first.")
        return
    
    train_df = st.session_state.train_df
    test_df = st.session_state.test_df
    drift_results = st.session_state.drift_results
    
    st.subheader("Training Configuration")
    
    # Identify target column (assume last column or 'target')
    target_col = st.selectbox(
        "Select Target Column",
        train_df.columns.tolist(),
        index=len(train_df.columns) - 1
    )
    
    task_type = st.radio("Task Type", ["classification", "regression"])
    
    model_type = st.selectbox(
        "Model Type",
        ["auto", "rf", "xgb", "lgb"]
    )
    
    use_weights = st.checkbox("Use Feature Weights", value=True)
    
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model with drift awareness..."):
            # Prepare data
            feature_cols = [c for c in train_df.columns if c != target_col]
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_test = test_df[feature_cols]
            y_test = test_df[target_col] if target_col in test_df.columns else None
            
            # Train
            st.session_state.trainer.model_type = model_type
            results = st.session_state.trainer.train_with_drift_awareness(
                X_train, y_train, X_test, y_test,
                drift_results, task_type, use_weights
            )
            
            st.session_state.training_results = results
            st.success("✅ Model training completed!")
    
    # Display results
    if 'training_results' in st.session_state:
        results = st.session_state.training_results
        
        st.markdown("---")
        st.subheader("Training Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if task_type == 'classification':
                st.metric("Train Accuracy", f"{results.get('train_accuracy', 0):.4f}")
                if 'test_accuracy' in results:
                    st.metric("Test Accuracy", f"{results['test_accuracy']:.4f}")
            else:
                st.metric("Train RMSE", f"{results.get('train_rmse', 0):.4f}")
                if 'test_rmse' in results:
                    st.metric("Test RMSE", f"{results['test_rmse']:.4f}")
        
        with col2:
            if 'feature_importance' in results:
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame([
                    {'Feature': f, 'Importance': imp}
                    for f, imp in results['feature_importance'].items()
                ]).sort_values('Importance', ascending=False)
                st.dataframe(importance_df.head(10), use_container_width=True)

def results_page():
    """Results and export page"""
    st.header("📈 Results & Export")
    
    if st.session_state.drift_results is None:
        st.warning("⚠️ No results available. Please run drift detection first.")
        return
    
    # Summary
    st.subheader("Detection Summary")
    results = st.session_state.drift_results
    
    summary_text = f"""
    **Total Features Analyzed:** {results['total_features']}
    
    **Features with Drift:** {results['features_with_drift']}
    
    **Drift Percentage:** {results['drift_percentage']:.2f}%
    
    **Features without Drift:** {results['total_features'] - results['features_with_drift']}
    """
    st.markdown(summary_text)
    
    # Export options
    st.markdown("---")
    st.subheader("Export Results")
    
    if st.button("Download Drift Report (CSV)"):
        summary_df = st.session_state.detector.get_drift_summary()
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="drift_report.csv",
            mime="text/csv"
        )

def generate_sample_data_with_drift():
    """Generate sample data with intentional drift"""
    np.random.seed(42)
    
    n_train = 1000
    n_test = 500
    
    # Create training data
    train_data = {
        'feature_1': np.random.normal(0, 1, n_train),
        'feature_2': np.random.normal(5, 2, n_train),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_train, p=[0.5, 0.3, 0.2]),
        'feature_4': np.random.exponential(2, n_train),
        'target': np.random.choice([0, 1], n_train)
    }
    train_df = pd.DataFrame(train_data)
    
    # Create test data with drift
    test_data = {
        'feature_1': np.random.normal(0.5, 1.2, n_test),  # Mean shift
        'feature_2': np.random.normal(6, 2.5, n_test),  # Distribution shift
        'feature_3': np.random.choice(['A', 'B', 'C'], n_test, p=[0.3, 0.4, 0.3]),  # Categorical drift
        'feature_4': np.random.exponential(3, n_test),  # Scale shift
        'target': np.random.choice([0, 1], n_test)
    }
    test_df = pd.DataFrame(test_data)
    
    return train_df, test_df

if __name__ == "__main__":
    main()
