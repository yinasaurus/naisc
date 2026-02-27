"""
Visualization components for data drift analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')


class DriftVisualizer:
    """Create visualizations for data drift analysis"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.color_palette = {
            'train': '#3498db',
            'test': '#e74c3c',
            'drift': '#f39c12',
            'no_drift': '#2ecc71'
        }
    
    def plot_distribution_comparison(self, train_series: pd.Series,
                                    test_series: pd.Series,
                                    feature_name: str,
                                    feature_type: str) -> go.Figure:
        """
        Create distribution comparison plot
        
        Args:
            train_series: Training data
            test_series: Test data
            feature_name: Name of the feature
            feature_type: Type of feature
            
        Returns:
            Plotly figure
        """
        if feature_type == 'numeric':
            return self._plot_numeric_distribution(
                train_series, test_series, feature_name
            )
        else:
            return self._plot_categorical_distribution(
                train_series, test_series, feature_name
            )
    
    def _plot_numeric_distribution(self, train_series: pd.Series,
                                   test_series: pd.Series,
                                   feature_name: str) -> go.Figure:
        """Plot numeric distribution comparison"""
        fig = go.Figure()
        
        # Histogram for training data
        fig.add_trace(go.Histogram(
            x=train_series.dropna(),
            name='Training Data',
            opacity=0.7,
            marker_color=self.color_palette['train'],
            nbinsx=30
        ))
        
        # Histogram for test data
        fig.add_trace(go.Histogram(
            x=test_series.dropna(),
            name='Test Data',
            opacity=0.7,
            marker_color=self.color_palette['test'],
            nbinsx=30
        ))
        
        fig.update_layout(
            title=f'Distribution Comparison: {feature_name}',
            xaxis_title=feature_name,
            yaxis_title='Frequency',
            barmode='overlay',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def _plot_categorical_distribution(self, train_series: pd.Series,
                                      test_series: pd.Series,
                                      feature_name: str) -> go.Figure:
        """Plot categorical distribution comparison"""
        train_counts = train_series.value_counts()
        test_counts = test_series.value_counts()
        
        # Get all categories
        all_categories = set(train_counts.index) | set(test_counts.index)
        
        train_normalized = train_counts.reindex(all_categories, fill_value=0) / len(train_series)
        test_normalized = test_counts.reindex(all_categories, fill_value=0) / len(test_series)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(all_categories),
            y=train_normalized.values,
            name='Training Data',
            marker_color=self.color_palette['train']
        ))
        
        fig.add_trace(go.Bar(
            x=list(all_categories),
            y=test_normalized.values,
            name='Test Data',
            marker_color=self.color_palette['test']
        ))
        
        fig.update_layout(
            title=f'Distribution Comparison: {feature_name}',
            xaxis_title=feature_name,
            yaxis_title='Proportion',
            barmode='group',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def plot_drift_summary(self, drift_results: Dict) -> go.Figure:
        """
        Create summary visualization of drift detection results
        
        Args:
            drift_results: Results from DriftDetector
            
        Returns:
            Plotly figure
        """
        summary_df = pd.DataFrame({
            'Feature': [],
            'Drift Status': [],
            'Severity': []
        })
        
        for feature, results in drift_results['feature_results'].items():
            status = 'Drift Detected' if results['drift_detected'] else 'No Drift'
            severity = results['severity'].title() if results['drift_detected'] else 'None'
            
            summary_df = pd.concat([summary_df, pd.DataFrame({
                'Feature': [feature],
                'Drift Status': [status],
                'Severity': [severity]
            })], ignore_index=True)
        
        # Create color mapping
        color_map = {
            'High': '#e74c3c',
            'Medium': '#f39c12',
            'Low': '#f1c40f',
            'None': '#2ecc71'
        }
        
        fig = px.bar(
            summary_df,
            x='Feature',
            y='Drift Status',
            color='Severity',
            color_discrete_map=color_map,
            title='Data Drift Summary by Feature',
            template='plotly_white',
            height=max(400, len(summary_df) * 30)
        )
        
        fig.update_layout(
            xaxis_title='Feature',
            yaxis_title='',
            showlegend=True
        )
        
        return fig
    
    def plot_statistical_test_results(self, feature_name: str,
                                     test_results: Dict) -> go.Figure:
        """
        Visualize statistical test results for a feature
        
        Args:
            feature_name: Name of the feature
            test_results: Dictionary of test results
            
        Returns:
            Plotly figure
        """
        test_names = []
        pvalues = []
        drift_status = []
        
        for test_name, result in test_results.items():
            if 'pvalue' in result:
                test_names.append(result.get('test_name', test_name))
                pvalue = result.get('pvalue', 1.0)
                pvalues.append(pvalue)
                drift_status.append('Drift' if result.get('drift_detected', False) else 'No Drift')
            elif 'psi' in result:
                test_names.append('PSI')
                psi = result.get('psi', 0)
                # Convert PSI to a p-value-like metric for visualization
                pvalues.append(min(psi / 2, 1.0))  # Normalize
                drift_status.append('Drift' if result.get('drift_detected', False) else 'No Drift')
        
        colors = [self.color_palette['drift'] if status == 'Drift' 
                 else self.color_palette['no_drift'] 
                 for status in drift_status]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=test_names,
            y=pvalues,
            marker_color=colors,
            text=[f'p={p:.4f}' if p < 1.0 else 'PSI' for p in pvalues],
            textposition='outside'
        ))
        
        # Add significance line
        fig.add_hline(
            y=0.05,
            line_dash="dash",
            line_color="red",
            annotation_text="α = 0.05"
        )
        
        fig.update_layout(
            title=f'Statistical Test Results: {feature_name}',
            xaxis_title='Test',
            yaxis_title='P-value / Normalized Score',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def plot_feature_statistics_comparison(self, train_stats: Dict,
                                          test_stats: Dict,
                                          feature_name: str,
                                          feature_type: str) -> go.Figure:
        """
        Compare statistics between training and test data
        
        Args:
            train_stats: Training statistics
            test_stats: Test statistics
            feature_name: Name of the feature
            feature_type: Type of feature
            
        Returns:
            Plotly figure
        """
        if feature_type == 'numeric':
            stats_to_compare = ['mean', 'median', 'std', 'min', 'max']
            train_values = [train_stats.get(stat, 0) for stat in stats_to_compare]
            test_values = [test_stats.get(stat, 0) for stat in stats_to_compare]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=stats_to_compare,
                y=train_values,
                name='Training',
                marker_color=self.color_palette['train']
            ))
            
            fig.add_trace(go.Bar(
                x=stats_to_compare,
                y=test_values,
                name='Test',
                marker_color=self.color_palette['test']
            ))
            
            fig.update_layout(
                title=f'Statistics Comparison: {feature_name}',
                xaxis_title='Statistic',
                yaxis_title='Value',
                barmode='group',
                template='plotly_white',
                height=400
            )
        else:
            # For categorical, show top categories
            train_dist = train_stats.get('value_distribution', {})
            test_dist = test_stats.get('value_distribution', {})
            
            # Get top 10 categories
            all_cats = set(list(train_dist.keys())[:10]) | set(list(test_dist.keys())[:10])
            
            train_props = [train_dist.get(cat, 0) / train_stats.get('count', 1) 
                          for cat in all_cats]
            test_props = [test_dist.get(cat, 0) / test_stats.get('count', 1) 
                         for cat in all_cats]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=list(all_cats),
                y=train_props,
                name='Training',
                marker_color=self.color_palette['train']
            ))
            
            fig.add_trace(go.Bar(
                x=list(all_cats),
                y=test_props,
                name='Test',
                marker_color=self.color_palette['test']
            ))
            
            fig.update_layout(
                title=f'Top Categories Comparison: {feature_name}',
                xaxis_title='Category',
                yaxis_title='Proportion',
                barmode='group',
                template='plotly_white',
                height=400
            )
        
        return fig
    
    def create_drift_report_dashboard(self, train_df: pd.DataFrame,
                                     test_df: pd.DataFrame,
                                     drift_results: Dict) -> List[go.Figure]:
        """
        Create comprehensive dashboard with multiple visualizations
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            drift_results: Drift detection results
            
        Returns:
            List of Plotly figures
        """
        figures = []
        
        # 1. Summary plot
        figures.append(self.plot_drift_summary(drift_results))
        
        # 2. Individual feature plots for drifted features
        drifted_features = [
            f for f, r in drift_results['feature_results'].items()
            if r['drift_detected']
        ]
        
        for feature in drifted_features[:10]:  # Limit to top 10
            results = drift_results['feature_results'][feature]
            feature_type = results['feature_type']
            
            # Distribution comparison
            fig = self.plot_distribution_comparison(
                train_df[feature],
                test_df[feature],
                feature,
                feature_type
            )
            figures.append(fig)
            
            # Statistical test results
            fig = self.plot_statistical_test_results(
                feature,
                results['test_results']
            )
            figures.append(fig)
        
        return figures
