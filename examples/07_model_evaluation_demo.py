#!/usr/bin/env python3
"""
Model Evaluation and Comparison Demo
====================================

This script demonstrates the comprehensive model evaluation dashboard
by running various ML models and tracking their performance.

Features demonstrated:
- Running multiple model types (ML, DL, LLM)
- Tracking performance metrics
- Creating interactive dashboards
- Comparing models across different metrics
- Exporting results

Author: AI Tutorial by AI
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_evaluation import ModelEvaluationDashboard
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
import warnings
warnings.filterwarnings('ignore')

def run_sklearn_experiments(dashboard):
    """Run sklearn-based experiments and add results to dashboard"""
    print("🔬 Running Scikit-Learn Experiments")
    print("=" * 50)
    
    # Iris dataset experiments
    print("🌸 Iris Dataset Classification")
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)
    
    # Models to test
    models = [
        ('Logistic Regression', LogisticRegression(random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('MLP', MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=500, random_state=42))
    ]
    
    for model_name, model in models:
        print(f"  Training {model_name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predictions and metrics
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted')
        }
        
        # Estimate model parameters
        if hasattr(model, 'coef_'):
            model_size = np.prod(model.coef_.shape)
        elif hasattr(model, 'feature_importances_'):
            model_size = len(model.feature_importances_) * 100  # Rough estimate for RF
        elif hasattr(model, 'coefs_'):
            model_size = sum(np.prod(coef.shape) for coef in model.coefs_)
        else:
            model_size = 1000  # Default estimate
        
        dashboard.add_model_results(
            model_name=model_name,
            model_type='Traditional ML',
            dataset='Iris',
            metrics=metrics,
            training_time=training_time,
            model_size=int(model_size)
        )
    
    # Synthetic classification dataset
    print("\n🎯 Synthetic Classification Dataset")
    X_synth, y_synth = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, n_classes=3, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X_synth, y_synth, test_size=0.3, random_state=42)
    
    for model_name, model in models:
        print(f"  Training {model_name} on synthetic data...")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predictions and metrics
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted')
        }
        
        # Estimate model parameters
        if hasattr(model, 'coef_'):
            model_size = np.prod(model.coef_.shape)
        elif hasattr(model, 'feature_importances_'):
            model_size = len(model.feature_importances_) * 100
        elif hasattr(model, 'coefs_'):
            model_size = sum(np.prod(coef.shape) for coef in model.coefs_)
        else:
            model_size = 1000
        
        dashboard.add_model_results(
            model_name=f"{model_name} (Complex)",
            model_type='Traditional ML',
            dataset='Synthetic Classification',
            metrics=metrics,
            training_time=training_time,
            model_size=int(model_size)
        )

def run_deep_learning_simulation(dashboard):
    """Simulate deep learning experiments with realistic results"""
    print("\n🧠 Simulating Deep Learning Experiments")
    print("=" * 50)
    
    # Simulate CNN results
    cnn_results = [
        {
            'model_name': 'Simple CNN',
            'model_type': 'CNN',
            'dataset': 'CIFAR-10',
            'metrics': {
                'accuracy': 0.78 + np.random.normal(0, 0.02),
                'f1_score': 0.76 + np.random.normal(0, 0.02),
                'precision': 0.77 + np.random.normal(0, 0.02),
                'recall': 0.75 + np.random.normal(0, 0.02)
            },
            'training_time': 180 + np.random.normal(0, 20),
            'model_size': 250000
        },
        {
            'model_name': 'ResNet-50',
            'model_type': 'CNN',
            'dataset': 'CIFAR-10',
            'metrics': {
                'accuracy': 0.94 + np.random.normal(0, 0.01),
                'f1_score': 0.93 + np.random.normal(0, 0.01),
                'precision': 0.94 + np.random.normal(0, 0.01),
                'recall': 0.92 + np.random.normal(0, 0.01)
            },
            'training_time': 600 + np.random.normal(0, 50),
            'model_size': 25000000
        },
        {
            'model_name': 'EfficientNet-B0',
            'model_type': 'CNN',
            'dataset': 'CIFAR-10',
            'metrics': {
                'accuracy': 0.96 + np.random.normal(0, 0.005),
                'f1_score': 0.95 + np.random.normal(0, 0.005),
                'precision': 0.96 + np.random.normal(0, 0.005),
                'recall': 0.95 + np.random.normal(0, 0.005)
            },
            'training_time': 420 + np.random.normal(0, 30),
            'model_size': 5300000
        }
    ]
    
    # Simulate RNN/LSTM results for text classification
    rnn_results = [
        {
            'model_name': 'Simple RNN',
            'model_type': 'RNN',
            'dataset': 'Text Sentiment',
            'metrics': {
                'accuracy': 0.72 + np.random.normal(0, 0.03),
                'f1_score': 0.70 + np.random.normal(0, 0.03),
                'precision': 0.73 + np.random.normal(0, 0.03),
                'recall': 0.68 + np.random.normal(0, 0.03)
            },
            'training_time': 150 + np.random.normal(0, 15),
            'model_size': 100000
        },
        {
            'model_name': 'LSTM',
            'model_type': 'RNN',
            'dataset': 'Text Sentiment',
            'metrics': {
                'accuracy': 0.84 + np.random.normal(0, 0.02),
                'f1_score': 0.83 + np.random.normal(0, 0.02),
                'precision': 0.85 + np.random.normal(0, 0.02),
                'recall': 0.81 + np.random.normal(0, 0.02)
            },
            'training_time': 280 + np.random.normal(0, 25),
            'model_size': 500000
        },
        {
            'model_name': 'BiLSTM',
            'model_type': 'RNN',
            'dataset': 'Text Sentiment',
            'metrics': {
                'accuracy': 0.87 + np.random.normal(0, 0.015),
                'f1_score': 0.86 + np.random.normal(0, 0.015),
                'precision': 0.88 + np.random.normal(0, 0.015),
                'recall': 0.84 + np.random.normal(0, 0.015)
            },
            'training_time': 350 + np.random.normal(0, 30),
            'model_size': 800000
        }
    ]
    
    # Simulate Transformer results
    transformer_results = [
        {
            'model_name': 'Small Transformer',
            'model_type': 'Transformer',
            'dataset': 'Text Classification',
            'metrics': {
                'accuracy': 0.89 + np.random.normal(0, 0.02),
                'f1_score': 0.88 + np.random.normal(0, 0.02),
                'precision': 0.90 + np.random.normal(0, 0.02),
                'recall': 0.86 + np.random.normal(0, 0.02)
            },
            'training_time': 480 + np.random.normal(0, 40),
            'model_size': 12000000
        },
        {
            'model_name': 'BERT-Base',
            'model_type': 'Transformer',
            'dataset': 'Text Classification',
            'metrics': {
                'accuracy': 0.93 + np.random.normal(0, 0.01),
                'f1_score': 0.92 + np.random.normal(0, 0.01),
                'precision': 0.94 + np.random.normal(0, 0.01),
                'recall': 0.90 + np.random.normal(0, 0.01)
            },
            'training_time': 900 + np.random.normal(0, 80),
            'model_size': 110000000
        },
        {
            'model_name': 'DistilBERT',
            'model_type': 'Transformer',
            'dataset': 'Text Classification',
            'metrics': {
                'accuracy': 0.91 + np.random.normal(0, 0.015),
                'f1_score': 0.90 + np.random.normal(0, 0.015),
                'precision': 0.92 + np.random.normal(0, 0.015),
                'recall': 0.88 + np.random.normal(0, 0.015)
            },
            'training_time': 520 + np.random.normal(0, 45),
            'model_size': 66000000
        }
    ]
    
    # Add all simulated results
    all_simulated = cnn_results + rnn_results + transformer_results
    
    for result in all_simulated:
        # Ensure metrics are in valid range [0, 1]
        for metric in result['metrics']:
            result['metrics'][metric] = np.clip(result['metrics'][metric], 0, 1)
        
        dashboard.add_model_results(**result)
        print(f"  Added {result['model_name']} results")

def create_comprehensive_analysis(dashboard):
    """Create comprehensive analysis and visualizations"""
    print("\n📊 Creating Comprehensive Analysis")
    print("=" * 50)
    
    # Model comparison by accuracy
    print("🏆 Top Models by Accuracy:")
    top_models = dashboard.compare_models(metric='accuracy', top_n=8)
    print(top_models[['Model', 'Type', 'Dataset', 'Accuracy']].to_string(index=False))
    
    # Model comparison by training efficiency (accuracy/time)
    print("\n⚡ Most Efficient Models (Accuracy/Training Time):")
    efficiency_df = dashboard.compare_models(metric='accuracy', top_n=20)
    if not efficiency_df.empty and 'Training Time (s)' in efficiency_df.columns:
        efficiency_df = efficiency_df.dropna(subset=['Training Time (s)'])
        efficiency_df['Efficiency'] = efficiency_df['Accuracy'] / efficiency_df['Training Time (s)']
        efficiency_df = efficiency_df.sort_values('Efficiency', ascending=False)
        print(efficiency_df[['Model', 'Type', 'Accuracy', 'Training Time (s)', 'Efficiency']].head(5).to_string(index=False))
    
    # Summary statistics
    print("\n📈 Summary Statistics:")
    stats = dashboard.get_summary_statistics()
    print(f"Total models evaluated: {stats['total_models']}")
    print(f"Unique model architectures: {stats['unique_models']}")
    print(f"Model types: {', '.join(stats['model_types'])}")
    print(f"Datasets: {', '.join(stats['datasets'])}")
    
    # Metric statistics
    if 'metrics' in stats:
        print("\n📊 Performance Metrics Overview:")
        for metric, metric_stats in stats['metrics'].items():
            print(f"  {metric.title()}:")
            print(f"    Mean: {metric_stats['mean']:.3f}")
            print(f"    Best: {metric_stats['max']:.3f}")
            print(f"    Worst: {metric_stats['min']:.3f}")
            print(f"    Std Dev: {metric_stats['std']:.3f}")
    
    # Create interactive dashboards
    print("\n🎨 Creating Interactive Dashboards...")
    
    # Main performance dashboard
    dashboard.create_performance_dashboard(save_html="comprehensive_model_dashboard.html")
    
    # Metrics comparison
    comparison_fig = dashboard.create_metrics_comparison()
    if comparison_fig:
        comparison_fig.write_html("model_metrics_radar.html")
        print("📊 Radar comparison saved as 'model_metrics_radar.html'")
    
    # Model type comparison
    cnn_comparison = dashboard.create_metrics_comparison(dataset='CIFAR-10')
    if cnn_comparison:
        cnn_comparison.write_html("cnn_models_comparison.html")
        print("🖼️ CNN models comparison saved as 'cnn_models_comparison.html'")
    
    # Export comprehensive results
    print("\n💾 Exporting Results...")
    dashboard.export_results(format='csv', filename='comprehensive_model_results')
    dashboard.export_results(format='json', filename='comprehensive_model_results')
    
    print("\n✅ Comprehensive analysis completed!")
    print("\n📁 Generated Files:")
    print("  - comprehensive_model_dashboard.html")
    print("  - model_metrics_radar.html")
    print("  - cnn_models_comparison.html")
    print("  - evaluation_results/comprehensive_model_results.csv")
    print("  - evaluation_results/comprehensive_model_results.json")

def main():
    """Main demonstration function"""
    print("🚀 Comprehensive AI Model Evaluation Dashboard")
    print("=" * 70)
    print("This demo runs multiple AI models and creates a comprehensive")
    print("evaluation dashboard for comparing their performance.")
    print("=" * 70)
    
    # Initialize dashboard
    dashboard = ModelEvaluationDashboard()
    
    # Run experiments
    run_sklearn_experiments(dashboard)
    run_deep_learning_simulation(dashboard)
    
    # Create analysis
    create_comprehensive_analysis(dashboard)
    
    print("\n🎉 Demo completed successfully!")
    print("\n🔍 Next Steps:")
    print("  1. Open the HTML files in your browser to explore interactive dashboards")
    print("  2. Check the evaluation_results directory for exported data")
    print("  3. Use the ModelEvaluationDashboard class in your own projects")
    print("  4. Integrate with existing model training scripts for automatic tracking")

if __name__ == "__main__":
    main()