#!/usr/bin/env python3
"""
Model Evaluation and Comparison Dashboard
=========================================

This module provides comprehensive tools for evaluating and comparing
different AI models across various metrics and datasets.

Features:
- Performance metrics tracking
- Model comparison visualization
- Automated benchmarking
- Interactive dashboards
- Export capabilities

Author: AI Tutorial by AI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluationDashboard:
    """Comprehensive model evaluation and comparison dashboard"""
    
    def __init__(self, results_dir: str = "evaluation_results"):
        """
        Initialize the evaluation dashboard
        
        Args:
            results_dir: Directory to store evaluation results
        """
        self.results_dir = results_dir
        self.results = {}
        self.metrics_history = []
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Load existing results if available
        self._load_existing_results()
    
    def add_model_results(self, 
                         model_name: str,
                         model_type: str,
                         dataset: str,
                         metrics: Dict[str, float],
                         training_time: float = None,
                         model_size: int = None,
                         additional_info: Dict[str, Any] = None):
        """
        Add evaluation results for a model
        
        Args:
            model_name: Name of the model
            model_type: Type/category of the model (e.g., 'CNN', 'Transformer', 'RNN')
            dataset: Dataset used for evaluation
            metrics: Dictionary of metric names and values
            training_time: Training time in seconds
            model_size: Model size in parameters
            additional_info: Additional metadata
        """
        timestamp = datetime.now().isoformat()
        
        result = {
            'timestamp': timestamp,
            'model_name': model_name,
            'model_type': model_type,
            'dataset': dataset,
            'metrics': metrics,
            'training_time': training_time,
            'model_size': model_size,
            'additional_info': additional_info or {}
        }
        
        # Store in memory
        key = f"{model_name}_{dataset}_{timestamp}"
        self.results[key] = result
        self.metrics_history.append(result)
        
        # Save to file
        self._save_result(key, result)
        
        print(f"✅ Added results for {model_name} on {dataset}")
        print(f"   Metrics: {metrics}")
    
    def compare_models(self, 
                      metric: str = 'accuracy',
                      dataset: str = None,
                      model_type: str = None,
                      top_n: int = 10) -> pd.DataFrame:
        """
        Compare models based on a specific metric
        
        Args:
            metric: Metric to compare (e.g., 'accuracy', 'loss', 'f1_score')
            dataset: Filter by dataset (optional)
            model_type: Filter by model type (optional)
            top_n: Number of top models to show
            
        Returns:
            DataFrame with comparison results
        """
        # Filter results
        filtered_results = []
        for result in self.metrics_history:
            if metric not in result['metrics']:
                continue
            if dataset and result['dataset'] != dataset:
                continue
            if model_type and result['model_type'] != model_type:
                continue
            filtered_results.append(result)
        
        if not filtered_results:
            print(f"❌ No results found for metric '{metric}'")
            return pd.DataFrame()
        
        # Create comparison DataFrame
        comparison_data = []
        for result in filtered_results:
            comparison_data.append({
                'Model': result['model_name'],
                'Type': result['model_type'],
                'Dataset': result['dataset'],
                metric.title(): result['metrics'][metric],
                'Training Time (s)': result['training_time'],
                'Model Size': result['model_size'],
                'Timestamp': result['timestamp']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by metric (higher is better for most metrics except loss)
        ascending = metric.lower() in ['loss', 'error', 'mse', 'mae']
        df = df.sort_values(metric.title(), ascending=ascending)
        
        return df.head(top_n)
    
    def create_performance_dashboard(self, save_html: str = None) -> go.Figure:
        """
        Create an interactive performance dashboard
        
        Args:
            save_html: Path to save HTML file (optional)
            
        Returns:
            Plotly figure object
        """
        if not self.metrics_history:
            print("❌ No results available for dashboard")
            return None
        
        # Prepare data
        df_data = []
        for result in self.metrics_history:
            base_data = {
                'Model': result['model_name'],
                'Type': result['model_type'],
                'Dataset': result['dataset'],
                'Training Time': result['training_time'],
                'Model Size': result['model_size'],
                'Timestamp': result['timestamp']
            }
            
            # Add all metrics
            for metric, value in result['metrics'].items():
                row = base_data.copy()
                row['Metric'] = metric
                row['Value'] = value
                df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance by Model Type', 'Training Time vs Performance',
                          'Model Size vs Performance', 'Performance Trends'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Performance by Model Type (Box plot)
        accuracy_data = df[df['Metric'] == 'accuracy'] if 'accuracy' in df['Metric'].values else df[df['Metric'] == df['Metric'].iloc[0]]
        if not accuracy_data.empty:
            for model_type in accuracy_data['Type'].unique():
                type_data = accuracy_data[accuracy_data['Type'] == model_type]
                fig.add_trace(
                    go.Box(y=type_data['Value'], name=model_type, showlegend=False),
                    row=1, col=1
                )
        
        # 2. Training Time vs Performance (Scatter)
        if not accuracy_data.empty and accuracy_data['Training Time'].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=accuracy_data['Training Time'],
                    y=accuracy_data['Value'],
                    mode='markers',
                    text=accuracy_data['Model'],
                    name='Models',
                    showlegend=False,
                    marker=dict(
                        size=10,
                        color=accuracy_data['Value'],
                        colorscale='Viridis',
                        showscale=True
                    )
                ),
                row=1, col=2
            )
        
        # 3. Model Size vs Performance (Scatter)
        if not accuracy_data.empty and accuracy_data['Model Size'].notna().any():
            # Convert categorical types to numeric for coloring
            unique_types = accuracy_data['Type'].unique()
            type_to_color = {type_name: i for i, type_name in enumerate(unique_types)}
            colors = [type_to_color[t] for t in accuracy_data['Type']]
            
            fig.add_trace(
                go.Scatter(
                    x=accuracy_data['Model Size'],
                    y=accuracy_data['Value'],
                    mode='markers',
                    text=accuracy_data['Model'],
                    name='Models',
                    showlegend=False,
                    marker=dict(
                        size=10,
                        color=colors,
                        colorscale='Viridis',
                        showscale=False
                    )
                ),
                row=2, col=1
            )
        
        # 4. Performance Trends (Line plot)
        accuracy_data_sorted = accuracy_data.sort_values('Timestamp')
        if not accuracy_data_sorted.empty:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(accuracy_data_sorted))),
                    y=accuracy_data_sorted['Value'],
                    mode='lines+markers',
                    text=accuracy_data_sorted['Model'],
                    name='Performance Trend',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="🏁 Model Performance Dashboard",
            height=800,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Model Type", row=1, col=1)
        fig.update_yaxes(title_text="Performance", row=1, col=1)
        
        fig.update_xaxes(title_text="Training Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Performance", row=1, col=2)
        
        fig.update_xaxes(title_text="Model Size (parameters)", row=2, col=1)
        fig.update_yaxes(title_text="Performance", row=2, col=1)
        
        fig.update_xaxes(title_text="Evaluation Order", row=2, col=2)
        fig.update_yaxes(title_text="Performance", row=2, col=2)
        
        # Save HTML if requested
        if save_html:
            fig.write_html(save_html)
            print(f"📊 Dashboard saved as {save_html}")
        
        return fig
    
    def create_metrics_comparison(self, models: List[str] = None, 
                                dataset: str = None) -> go.Figure:
        """
        Create a radar chart comparing multiple metrics across models
        
        Args:
            models: List of model names to compare (optional)
            dataset: Dataset to filter by (optional)
            
        Returns:
            Plotly figure object
        """
        # Filter results
        filtered_results = []
        for result in self.metrics_history:
            if models and result['model_name'] not in models:
                continue
            if dataset and result['dataset'] != dataset:
                continue
            filtered_results.append(result)
        
        if not filtered_results:
            print("❌ No results found for comparison")
            return None
        
        # Get all unique metrics
        all_metrics = set()
        for result in filtered_results:
            all_metrics.update(result['metrics'].keys())
        all_metrics = sorted(list(all_metrics))
        
        # Create radar chart
        fig = go.Figure()
        
        for result in filtered_results:
            values = []
            for metric in all_metrics:
                values.append(result['metrics'].get(metric, 0))
            
            # Close the radar chart
            values.append(values[0])
            metrics_labels = all_metrics + [all_metrics[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_labels,
                fill='toself',
                name=f"{result['model_name']} ({result['dataset']})"
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="📊 Multi-Metric Model Comparison"
        )
        
        return fig
    
    def export_results(self, format: str = 'csv', filename: str = None) -> str:
        """
        Export all results to a file
        
        Args:
            format: Export format ('csv', 'json', 'excel')
            filename: Custom filename (optional)
            
        Returns:
            Path to exported file
        """
        if not self.metrics_history:
            print("❌ No results to export")
            return None
        
        # Prepare data for export
        export_data = []
        for result in self.metrics_history:
            base_data = {
                'Timestamp': result['timestamp'],
                'Model': result['model_name'],
                'Type': result['model_type'],
                'Dataset': result['dataset'],
                'Training Time': result['training_time'],
                'Model Size': result['model_size']
            }
            
            # Add metrics
            base_data.update(result['metrics'])
            
            # Add additional info
            if result['additional_info']:
                for key, value in result['additional_info'].items():
                    base_data[f"Info_{key}"] = value
            
            export_data.append(base_data)
        
        df = pd.DataFrame(export_data)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_evaluation_results_{timestamp}"
        
        # Export based on format
        if format.lower() == 'csv':
            filepath = os.path.join(self.results_dir, f"{filename}.csv")
            df.to_csv(filepath, index=False)
        elif format.lower() == 'json':
            filepath = os.path.join(self.results_dir, f"{filename}.json")
            with open(filepath, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        elif format.lower() == 'excel':
            filepath = os.path.join(self.results_dir, f"{filename}.xlsx")
            df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"📁 Results exported to {filepath}")
        return filepath
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of all evaluated models
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.metrics_history:
            return {}
        
        # Basic statistics
        stats = {
            'total_models': len(self.metrics_history),
            'unique_models': len(set(r['model_name'] for r in self.metrics_history)),
            'model_types': list(set(r['model_type'] for r in self.metrics_history)),
            'datasets': list(set(r['dataset'] for r in self.metrics_history)),
            'date_range': {
                'first': min(r['timestamp'] for r in self.metrics_history),
                'last': max(r['timestamp'] for r in self.metrics_history)
            }
        }
        
        # Metric statistics
        all_metrics = set()
        for result in self.metrics_history:
            all_metrics.update(result['metrics'].keys())
        
        metric_stats = {}
        for metric in all_metrics:
            values = [r['metrics'][metric] for r in self.metrics_history if metric in r['metrics']]
            if values:
                metric_stats[metric] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        stats['metrics'] = metric_stats
        
        return stats
    
    def _load_existing_results(self):
        """Load existing results from files"""
        if not os.path.exists(self.results_dir):
            return
        
        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json') and filename.startswith('result_'):
                filepath = os.path.join(self.results_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        result = json.load(f)
                        key = filename[7:-5]  # Remove 'result_' and '.json'
                        self.results[key] = result
                        self.metrics_history.append(result)
                except Exception as e:
                    print(f"⚠️ Could not load {filename}: {e}")
    
    def _save_result(self, key: str, result: Dict[str, Any]):
        """Save a single result to file"""
        filepath = os.path.join(self.results_dir, f"result_{key}.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save result: {e}")


# ============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# ============================================================================

def demonstrate_evaluation_dashboard():
    """Demonstrate the model evaluation dashboard with sample data"""
    print("🏁 Model Evaluation Dashboard Demo")
    print("=" * 60)
    
    # Create dashboard
    dashboard = ModelEvaluationDashboard()
    
    # Add some example results
    sample_results = [
        {
            'model_name': 'Simple CNN',
            'model_type': 'CNN',
            'dataset': 'CIFAR-10',
            'metrics': {'accuracy': 0.85, 'f1_score': 0.83, 'precision': 0.84, 'recall': 0.82},
            'training_time': 120.5,
            'model_size': 50000
        },
        {
            'model_name': 'ResNet-18',
            'model_type': 'CNN',
            'dataset': 'CIFAR-10',
            'metrics': {'accuracy': 0.92, 'f1_score': 0.91, 'precision': 0.92, 'recall': 0.90},
            'training_time': 300.2,
            'model_size': 11000000
        },
        {
            'model_name': 'Simple RNN',
            'model_type': 'RNN',
            'dataset': 'Text Classification',
            'metrics': {'accuracy': 0.78, 'f1_score': 0.76, 'precision': 0.79, 'recall': 0.73},
            'training_time': 85.3,
            'model_size': 25000
        },
        {
            'model_name': 'Transformer',
            'model_type': 'Transformer',
            'dataset': 'Text Classification',
            'metrics': {'accuracy': 0.94, 'f1_score': 0.93, 'precision': 0.94, 'recall': 0.92},
            'training_time': 450.8,
            'model_size': 15000000
        },
        {
            'model_name': 'Simple MLP',
            'model_type': 'MLP',
            'dataset': 'Iris',
            'metrics': {'accuracy': 0.96, 'f1_score': 0.95, 'precision': 0.96, 'recall': 0.95},
            'training_time': 15.2,
            'model_size': 5000
        }
    ]
    
    # Add results to dashboard
    for result in sample_results:
        dashboard.add_model_results(**result)
    
    print("\n📊 Model Comparison Results:")
    print("-" * 40)
    
    # Compare models by accuracy
    comparison = dashboard.compare_models(metric='accuracy', top_n=5)
    print(comparison.to_string(index=False))
    
    print("\n📈 Summary Statistics:")
    print("-" * 40)
    stats = dashboard.get_summary_statistics()
    print(f"Total evaluations: {stats['total_models']}")
    print(f"Unique models: {stats['unique_models']}")
    print(f"Model types: {', '.join(stats['model_types'])}")
    print(f"Datasets: {', '.join(stats['datasets'])}")
    
    if 'accuracy' in stats['metrics']:
        acc_stats = stats['metrics']['accuracy']
        print(f"\nAccuracy Statistics:")
        print(f"  Mean: {acc_stats['mean']:.3f}")
        print(f"  Best: {acc_stats['max']:.3f}")
        print(f"  Worst: {acc_stats['min']:.3f}")
    
    # Create and save dashboard
    print("\n🎨 Creating Performance Dashboard...")
    fig = dashboard.create_performance_dashboard(save_html="model_performance_dashboard.html")
    
    # Create metrics comparison
    print("🎯 Creating Metrics Comparison...")
    comparison_fig = dashboard.create_metrics_comparison()
    if comparison_fig:
        comparison_fig.write_html("model_metrics_comparison.html")
        print("📊 Metrics comparison saved as 'model_metrics_comparison.html'")
    
    # Export results
    print("💾 Exporting Results...")
    dashboard.export_results(format='csv')
    dashboard.export_results(format='json')
    
    print("\n✅ Dashboard demonstration completed!")
    print("📁 Check the 'evaluation_results' directory and HTML files for outputs")


if __name__ == "__main__":
    demonstrate_evaluation_dashboard()