#!/usr/bin/env python3
"""
Enhanced AI Tutorial - Advanced Features Demo
=============================================

This script demonstrates the new advanced features added to the AI tutorial:
- Real-time Training Progress Tracking
- Model Interpretability Tools  
- Automated Hyperparameter Tuning
- Comprehensive Model Analysis

These tools provide practical, educational insights into AI model development.

Author: AI Tutorial by AI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# Import our new utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.training_tracker import TrainingTracker
from utils.interpretability import ModelInterpreter
from utils.hyperparameter_tuning import HyperparameterTuner
from utils.model_evaluation import ModelEvaluationDashboard

class AdvancedAIDemo:
    """Comprehensive demonstration of advanced AI tutorial features"""
    
    def __init__(self):
        """Initialize the demo with sample data"""
        print("🚀 Advanced AI Tutorial Features Demo")
        print("=" * 50)
        
        # Create datasets for demonstration
        self.create_datasets()
        
        # Initialize results storage
        self.results = {}
        
    def create_datasets(self):
        """Create sample datasets for demonstration"""
        print("📊 Creating sample datasets...")
        
        # Dataset 1: Synthetic classification data
        X_synthetic, y_synthetic = make_classification(
            n_samples=2000,
            n_features=12,
            n_informative=8,
            n_redundant=2,
            n_clusters_per_class=2,
            random_state=42
        )
        
        # Dataset 2: Iris dataset
        iris = load_iris()
        X_iris, y_iris = iris.data, iris.target
        
        # Prepare datasets
        self.datasets = {
            'synthetic': {
                'X': X_synthetic,
                'y': y_synthetic,
                'feature_names': [f"Feature_{i+1}" for i in range(X_synthetic.shape[1])],
                'class_names': ['Class_A', 'Class_B'],
                'description': 'Synthetic binary classification'
            },
            'iris': {
                'X': X_iris,
                'y': y_iris,
                'feature_names': iris.feature_names,
                'class_names': iris.target_names.tolist(),
                'description': 'Iris flower classification'
            }
        }
        
        print(f"✅ Created {len(self.datasets)} datasets")
        for name, data in self.datasets.items():
            print(f"  - {name}: {data['X'].shape[0]} samples, {data['X'].shape[1]} features")
    
    def demo_training_tracker(self):
        """Demonstrate real-time training progress tracking"""
        print("\n" + "="*60)
        print("🎯 DEMO 1: Real-time Training Progress Tracking")
        print("="*60)
        
        # Use synthetic dataset
        dataset = self.datasets['synthetic']
        X, y = dataset['X'], dataset['y']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Initialize training tracker
        tracker = TrainingTracker(
            metrics=['loss', 'accuracy', 'f1_score'],
            update_frequency=5,
            save_dir="demo_outputs"
        )
        
        # Simulate iterative training with improving metrics
        print("🚀 Simulating training process with live tracking...")
        tracker.start_training(total_epochs=50)
        
        # Mock training loop
        base_loss = 1.5
        base_acc = 0.2
        base_f1 = 0.15
        
        for epoch in range(1, 51):
            # Simulate training with gradual improvement
            noise = np.random.normal(0, 0.03)
            
            # Training metrics (improving over time)
            train_loss = base_loss * np.exp(-epoch * 0.04) + abs(noise)
            train_acc = min(0.95, base_acc + epoch * 0.012 + noise)
            train_f1 = min(0.93, base_f1 + epoch * 0.011 + noise)
            
            # Validation metrics (slightly worse than training)
            val_loss = train_loss + np.random.uniform(0, 0.05)
            val_acc = max(0.1, train_acc - np.random.uniform(0, 0.08))
            val_f1 = max(0.1, train_f1 - np.random.uniform(0, 0.06))
            
            # Log metrics
            tracker.log_epoch(
                epoch=epoch,
                train_metrics={'loss': train_loss, 'accuracy': train_acc, 'f1_score': train_f1},
                val_metrics={'loss': val_loss, 'accuracy': val_acc, 'f1_score': val_f1}
            )
            
            # Small delay to simulate actual training
            time.sleep(0.02)
        
        # Finish training and generate report
        summary = tracker.finish_training()
        
        print(f"\n📈 Training Summary:")
        print(f"  - Total epochs: {summary['current_epoch']}")
        print(f"  - Total time: {summary['total_time']}")
        print(f"  - Training efficiency: {summary['training_efficiency']:.1f} epochs/hour")
        
        self.results['training_tracker'] = summary
        
        return tracker
    
    def demo_model_interpretability(self):
        """Demonstrate model interpretability tools"""
        print("\n" + "="*60)
        print("🔍 DEMO 2: Model Interpretability & Explainability")
        print("="*60)
        
        # Use iris dataset for interpretability
        dataset = self.datasets['iris']
        X, y = dataset['X'], dataset['y']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train a Random Forest model
        print("🌲 Training Random Forest for interpretability analysis...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Test model performance
        accuracy = model.score(X_test, y_test)
        print(f"✅ Model accuracy: {accuracy:.3f}")
        
        # Initialize model interpreter
        interpreter = ModelInterpreter(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=dataset['feature_names'],
            class_names=dataset['class_names'],
            results_dir="demo_outputs"
        )
        
        # Feature importance analysis
        print("📊 Analyzing feature importance...")
        interpreter.plot_feature_importance(method='builtin')
        interpreter.plot_feature_importance(method='permutation')
        
        # Decision boundary visualization (using first two features)
        print("🎯 Creating decision boundary visualization...")
        interpreter.create_decision_boundary_plot(feature_indices=(0, 1))
        
        # SHAP analysis (if available)
        print("🔬 Generating SHAP interpretability dashboard...")
        try:
            interpreter.create_shap_dashboard(max_samples=50)
        except Exception as e:
            print(f"⚠️  SHAP analysis not available: {e}")
        
        # Generate comprehensive interpretation report
        report = interpreter.generate_interpretation_report()
        
        print(f"\n📋 Interpretability Summary:")
        print(f"  - Model type: {report['model_info']['model_type']}")
        print(f"  - Features analyzed: {report['model_info']['n_features']}")
        
        if 'builtin' in report['feature_importance']:
            print("  - Top 3 important features:")
            top_features = list(report['feature_importance']['builtin'].items())[:3]
            for feature, importance in top_features:
                print(f"    • {feature}: {importance:.3f}")
        
        self.results['interpretability'] = report
        
        return interpreter
    
    def demo_hyperparameter_tuning(self):
        """Demonstrate automated hyperparameter tuning"""
        print("\n" + "="*60)
        print("🎯 DEMO 3: Automated Hyperparameter Tuning")
        print("="*60)
        
        # Use synthetic dataset for tuning
        dataset = self.datasets['synthetic']
        X, y = dataset['X'], dataset['y']
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"📊 Dataset: {X_train.shape[0]} training samples")
        
        # Define parameter space for Random Forest
        param_space = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        print(f"🔍 Parameter space: {np.prod([len(v) for v in param_space.values()])} total combinations")
        
        # Initialize hyperparameter tuner
        tuner = HyperparameterTuner(
            model_class=RandomForestClassifier,
            param_space=param_space,
            scoring='accuracy',
            cv_folds=5,
            results_dir="demo_outputs"
        )
        
        # Compare optimization methods
        print("\n🔄 Comparing optimization methods...")
        comparison_df = tuner.compare_methods(
            X_train_scaled, y_train,
            methods=['grid_search', 'random_search'],
            random_n_iter=50
        )
        
        print("\n📊 Method Comparison:")
        print(comparison_df.round(4).to_string(index=False))
        
        # Test best model
        best_method = comparison_df.loc[comparison_df['Best Score'].idxmax(), 'Method'].lower().replace(' ', '_')
        best_model = tuner.tuning_results[best_method]['best_estimator']
        
        # Evaluate on test set
        test_accuracy = best_model.score(X_test_scaled, y_test)
        
        print(f"\n🏆 Optimization Results:")
        print(f"  - Best method: {best_method.replace('_', ' ').title()}")
        print(f"  - Best CV score: {tuner.best_scores[best_method]:.4f}")
        print(f"  - Test accuracy: {test_accuracy:.4f}")
        print(f"  - Best parameters: {tuner.best_params[best_method]}")
        
        # Generate optimization visualizations
        tuner.plot_optimization_progress()
        
        # Generate tuning report
        report = tuner.generate_tuning_report()
        
        self.results['hyperparameter_tuning'] = {
            'comparison': comparison_df.to_dict(),
            'best_method': best_method,
            'best_score': tuner.best_scores[best_method],
            'test_accuracy': test_accuracy,
            'best_params': tuner.best_params[best_method]
        }
        
        return tuner
    
    def demo_comprehensive_evaluation(self):
        """Demonstrate comprehensive model evaluation with new features"""
        print("\n" + "="*60)
        print("📊 DEMO 4: Comprehensive Model Evaluation Dashboard")
        print("="*60)
        
        # Use both datasets for evaluation
        dashboard = ModelEvaluationDashboard(results_dir="demo_outputs")
        
        models_to_evaluate = [
            (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
            (LogisticRegression(random_state=42, max_iter=1000), "Logistic Regression"),
            (MLPClassifier(hidden_layer_sizes=(50,), random_state=42, max_iter=500), "Neural Network")
        ]
        
        print(f"🔬 Evaluating {len(models_to_evaluate)} models on {len(self.datasets)} datasets...")
        
        for dataset_name, dataset in self.datasets.items():
            X, y = dataset['X'], dataset['y']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            print(f"\n📈 Evaluating on {dataset_name} dataset...")
            
            for model, model_name in models_to_evaluate:
                start_time = time.time()
                
                # Train model
                model.fit(X_train_scaled, y_train)
                training_time = time.time() - start_time
                
                # Evaluate model
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                # Calculate model size (approximation)
                if hasattr(model, 'n_estimators'):
                    model_size = model.n_estimators
                elif hasattr(model, 'coefs_') and model.coefs_:
                    model_size = sum(layer.size for layer in model.coefs_)
                else:
                    model_size = 1
                
                # Add results to dashboard
                dashboard.add_model_results(
                    model_name=f"{model_name} ({dataset_name})",
                    model_type=model_name,
                    dataset=dataset_name.title(),
                    metrics={
                        'accuracy': test_score,
                        'train_accuracy': train_score,
                        'overfitting': train_score - test_score
                    },
                    training_time=training_time,
                    model_size=model_size
                )
                
                print(f"  ✅ {model_name}: {test_score:.3f} accuracy ({training_time:.2f}s)")
        
        # Create comprehensive dashboard
        print(f"\n🎨 Creating comprehensive evaluation dashboard...")
        dashboard.create_performance_dashboard(save_html="enhanced_model_dashboard.html")
        
        # Generate comparison
        comparison_df = dashboard.compare_models(metric='accuracy', top_n=10)
        print(f"\n🏆 Top Model Performances:")
        print(comparison_df[['Model', 'Type', 'Dataset', 'Accuracy']].head(5).to_string(index=False))
        
        # Export results
        dashboard.export_results(format='csv', filename='enhanced_evaluation_results')
        
        # Get summary statistics
        stats = dashboard.get_summary_statistics()
        print(f"\n📊 Evaluation Summary:")
        print(f"  - Total models evaluated: {stats['total_models']}")
        print(f"  - Model types: {', '.join(stats['model_types'])}")
        print(f"  - Datasets: {', '.join(stats['datasets'])}")
        
        self.results['comprehensive_evaluation'] = stats
        
        return dashboard
    
    def generate_final_report(self):
        """Generate final comprehensive report"""
        print("\n" + "="*60)
        print("📋 FINAL COMPREHENSIVE REPORT")
        print("="*60)
        
        print("🎯 Enhanced AI Tutorial Features Demonstration Complete!")
        print("\n📊 Summary of Demonstrations:")
        
        print("\n1️⃣ Real-time Training Progress Tracking:")
        if 'training_tracker' in self.results:
            summary = self.results['training_tracker']
            print(f"   ✅ Tracked {summary['current_epoch']} training epochs")
            print(f"   ⏱️  Training efficiency: {summary['training_efficiency']:.1f} epochs/hour")
            print(f"   📈 Generated live training dashboard")
        
        print("\n2️⃣ Model Interpretability & Explainability:")
        if 'interpretability' in self.results:
            report = self.results['interpretability']
            print(f"   ✅ Analyzed {report['model_info']['n_features']} features")
            print(f"   🔍 Generated feature importance visualizations")
            print(f"   🎯 Created decision boundary plots")
            if 'builtin' in report['feature_importance']:
                top_feature = list(report['feature_importance']['builtin'].items())[0]
                print(f"   🏆 Most important feature: {top_feature[0]} ({top_feature[1]:.3f})")
        
        print("\n3️⃣ Automated Hyperparameter Tuning:")
        if 'hyperparameter_tuning' in self.results:
            tuning = self.results['hyperparameter_tuning']
            print(f"   ✅ Compared optimization methods")
            print(f"   🏆 Best method: {tuning['best_method'].replace('_', ' ').title()}")
            print(f"   📊 Best CV score: {tuning['best_score']:.4f}")
            print(f"   🎯 Test accuracy: {tuning['test_accuracy']:.4f}")
        
        print("\n4️⃣ Comprehensive Model Evaluation:")
        if 'comprehensive_evaluation' in self.results:
            eval_stats = self.results['comprehensive_evaluation']
            print(f"   ✅ Evaluated {eval_stats['total_models']} models")
            print(f"   📊 Across {len(eval_stats['datasets'])} datasets")
            print(f"   🔧 Model types: {', '.join(eval_stats['model_types'])}")
        
        print("\n🎉 All Features Successfully Demonstrated!")
        print("\n📁 Generated Outputs:")
        print("   - 📊 Interactive training dashboards")
        print("   - 🔍 Model interpretability reports")
        print("   - 🎯 Hyperparameter optimization results")
        print("   - 📈 Comprehensive evaluation dashboards")
        print("   - 💾 Detailed analysis reports (JSON/CSV)")
        
        print(f"\n💡 These enhanced features provide:")
        print("   • Real-time insights into model training")
        print("   • Deep understanding of model decisions")
        print("   • Automated optimization capabilities")
        print("   • Comprehensive performance analysis")
        
        print(f"\n🚀 The AI tutorial now offers state-of-the-art tools for:")
        print("   • Educational model development")
        print("   • Professional AI workflows")
        print("   • Research and experimentation")
        print("   • Production model analysis")


def main():
    """Main demonstration function"""
    # Initialize demo
    demo = AdvancedAIDemo()
    
    # Run all demonstrations
    print("\n🚀 Starting comprehensive feature demonstrations...")
    
    # Demo 1: Training Progress Tracking
    demo.demo_training_tracker()
    
    # Demo 2: Model Interpretability
    demo.demo_model_interpretability()
    
    # Demo 3: Hyperparameter Tuning
    demo.demo_hyperparameter_tuning()
    
    # Demo 4: Comprehensive Evaluation
    demo.demo_comprehensive_evaluation()
    
    # Generate final report
    demo.generate_final_report()
    
    print(f"\n🎓 Enhanced AI Tutorial demonstration complete!")
    print(f"📁 Check the 'demo_outputs' directory for all generated files.")


if __name__ == "__main__":
    main()