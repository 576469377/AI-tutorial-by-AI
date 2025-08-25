#!/usr/bin/env python3
"""
Model Interpretability Tools
===========================

This module provides comprehensive tools for understanding and explaining
AI model decisions, including feature importance, SHAP values, and
model behavior analysis.

Features:
- Feature importance calculation and visualization
- Model explanation dashboards
- Decision boundary visualization
- Partial dependence plots
- Model behavior analysis

Author: AI Tutorial by AI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("💡 SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("💡 LIME not available. Install with: pip install lime")

from typing import Dict, List, Any, Optional, Union, Tuple
import os
import json
from datetime import datetime

class ModelInterpreter:
    """Comprehensive model interpretability and explanation toolkit"""
    
    def __init__(self, model, X_train, y_train, X_test=None, y_test=None, 
                 feature_names=None, class_names=None, results_dir="interpretability_results"):
        """
        Initialize the model interpreter
        
        Args:
            model: Trained ML model
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            feature_names: Names of features
            class_names: Names of classes (for classification)
            results_dir: Directory to save results
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test if X_test is not None else X_train
        self.y_test = y_test if y_test is not None else y_train
        
        # Feature and class names
        if feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
        else:
            self.feature_names = feature_names
            
        self.class_names = class_names
        self.results_dir = results_dir
        
        # Model type detection
        self.is_classifier = hasattr(model, 'predict_proba') or hasattr(model, 'decision_function')
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Cache for expensive computations
        self._feature_importance_cache = None
        self._shap_values_cache = None
    
    def calculate_feature_importance(self, method='permutation', n_repeats=10) -> Dict[str, np.ndarray]:
        """
        Calculate feature importance using various methods
        
        Args:
            method: Method to use ('permutation', 'builtin', 'shap_mean')
            n_repeats: Number of repeats for permutation importance
            
        Returns:
            Dictionary with feature names and importance scores
        """
        print(f"🔍 Calculating feature importance using {method} method...")
        
        if method == 'permutation':
            # Permutation importance
            perm_importance = permutation_importance(
                self.model, self.X_test, self.y_test, 
                n_repeats=n_repeats, random_state=42
            )
            importance_scores = perm_importance.importances_mean
            importance_std = perm_importance.importances_std
            
        elif method == 'builtin' and hasattr(self.model, 'feature_importances_'):
            # Built-in importance (for tree-based models)
            importance_scores = self.model.feature_importances_
            importance_std = np.zeros_like(importance_scores)
            
        elif method == 'shap_mean' and SHAP_AVAILABLE:
            # SHAP-based importance
            shap_values = self.get_shap_values()
            if shap_values is not None:
                if isinstance(shap_values, list):
                    # Multi-class case
                    importance_scores = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0)
                else:
                    importance_scores = np.mean(np.abs(shap_values), axis=0)
                importance_std = np.zeros_like(importance_scores)
            else:
                print("⚠️  Could not calculate SHAP values")
                return {}
        else:
            print(f"⚠️  Method '{method}' not available for this model")
            return {}
        
        # Create results dictionary
        results = {
            'importance_scores': importance_scores,
            'importance_std': importance_std,
            'feature_names': self.feature_names,
            'method': method
        }
        
        self._feature_importance_cache = results
        return results
    
    def plot_feature_importance(self, method='permutation', top_n=None, save_plot=True) -> go.Figure:
        """Create feature importance visualization"""
        
        # Calculate importance if not cached
        if self._feature_importance_cache is None or self._feature_importance_cache['method'] != method:
            importance_data = self.calculate_feature_importance(method=method)
        else:
            importance_data = self._feature_importance_cache
        
        if not importance_data:
            print("⚠️  No feature importance data available")
            return None
        
        importance_scores = importance_data['importance_scores']
        importance_std = importance_data['importance_std']
        feature_names = importance_data['feature_names']
        
        # Sort by importance
        indices = np.argsort(importance_scores)[::-1]
        
        # Limit to top_n features
        if top_n:
            indices = indices[:top_n]
        
        sorted_features = [feature_names[i] for i in indices]
        sorted_scores = importance_scores[indices]
        sorted_std = importance_std[indices]
        
        # Create interactive plot
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=sorted_scores,
            y=sorted_features,
            orientation='h',
            error_x=dict(type='data', array=sorted_std),
            marker_color='lightblue',
            text=[f"{score:.3f}" for score in sorted_scores],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"📊 Feature Importance ({method.title()})",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(sorted_features) * 25),
            template="plotly_white"
        )
        
        if save_plot:
            filename = f"feature_importance_{method}.html"
            filepath = os.path.join(self.results_dir, filename)
            fig.write_html(filepath)
            print(f"💾 Feature importance plot saved to {filepath}")
        
        return fig
    
    def get_shap_values(self, X=None):
        """Calculate SHAP values for model explanations"""
        if not SHAP_AVAILABLE:
            print("⚠️  SHAP not available. Install with: pip install shap")
            return None
        
        if X is None:
            X = self.X_test
        
        try:
            # Ensure input is proper numpy array
            if hasattr(X, 'values'):
                X = X.values
            X = np.asarray(X)
            
            # Initialize SHAP explainer if not already done
            if self.shap_explainer is None:
                # Use a smaller background dataset for efficiency
                background_size = min(100, len(self.X_train))
                if hasattr(self.X_train, 'values'):
                    background = self.X_train.values[:background_size]
                else:
                    background = np.asarray(self.X_train)[:background_size]
                
                if hasattr(self.model, 'predict_proba'):
                    # For classifiers - use TreeExplainer for tree models or Explainer for others
                    if hasattr(self.model, 'estimators_') or 'RandomForest' in str(type(self.model)):
                        try:
                            self.shap_explainer = shap.TreeExplainer(self.model)
                        except:
                            self.shap_explainer = shap.Explainer(self.model.predict_proba, background)
                    else:
                        self.shap_explainer = shap.Explainer(self.model.predict_proba, background)
                else:
                    # For regressors
                    if hasattr(self.model, 'estimators_') or 'RandomForest' in str(type(self.model)):
                        try:
                            self.shap_explainer = shap.TreeExplainer(self.model)
                        except:
                            self.shap_explainer = shap.Explainer(self.model.predict, background)
                    else:
                        self.shap_explainer = shap.Explainer(self.model.predict, background)
            
            # Calculate SHAP values
            print("🔍 Calculating SHAP values...")
            # Limit the number of samples for efficiency
            sample_size = min(50, len(X))
            if len(X) > sample_size:
                sample_indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X[sample_indices]
            else:
                X_sample = X
            
            shap_values = self.shap_explainer(X_sample)
            
            # Handle different SHAP output formats
            if hasattr(shap_values, 'values'):
                values = shap_values.values
            else:
                values = shap_values
            
            # For multi-output models, ensure consistent format
            if isinstance(values, list):
                # Multi-class case - return as is
                self._shap_values_cache = values
                return values
            elif len(values.shape) == 3:
                # Multi-class case with shape (n_samples, n_features, n_classes)
                # Convert to list format for consistency
                values_list = [values[:, :, i] for i in range(values.shape[2])]
                self._shap_values_cache = values_list
                return values_list
            else:
                # Binary classification or regression
                self._shap_values_cache = values
                return values
            
        except Exception as e:
            print(f"⚠️  Error calculating SHAP values: {e}")
            print(f"   Model type: {type(self.model)}")
            print(f"   Input shape: {X.shape if hasattr(X, 'shape') else 'Unknown'}")
            return None
    
    def create_shap_dashboard(self, X_sample=None, max_samples=100, save_html=True) -> go.Figure:
        """Create comprehensive SHAP analysis dashboard"""
        if not SHAP_AVAILABLE:
            print("⚠️  SHAP not available. Install with: pip install shap")
            return None
        
        if X_sample is None:
            # Use a sample of test data
            n_samples = min(max_samples, len(self.X_test))
            indices = np.random.choice(len(self.X_test), n_samples, replace=False)
            X_sample = self.X_test[indices]
        
        shap_values = self.get_shap_values(X_sample)
        if shap_values is None:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Feature Importance (SHAP)",
                "SHAP Summary Plot",
                "Feature Interaction",
                "Individual Prediction"
            ],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # 1. Feature importance from SHAP
        if isinstance(shap_values, list):
            # Multi-class: use mean absolute SHAP values
            mean_shap = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0)
        else:
            mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        importance_order = np.argsort(mean_shap)[::-1][:10]  # Top 10
        
        fig.add_trace(
            go.Bar(
                x=mean_shap[importance_order],
                y=[self.feature_names[i] for i in importance_order],
                orientation='h',
                marker_color='lightcoral',
                name="SHAP Importance"
            ),
            row=1, col=1
        )
        
        # 2. SHAP summary (scatter plot for top features)
        if not isinstance(shap_values, list):
            top_features = importance_order[:5]
            for i, feature_idx in enumerate(top_features):
                fig.add_trace(
                    go.Scatter(
                        x=X_sample[:, feature_idx],
                        y=shap_values[:, feature_idx],
                        mode='markers',
                        name=self.feature_names[feature_idx],
                        marker=dict(size=6, opacity=0.7)
                    ),
                    row=1, col=2
                )
        
        # 3. Feature interaction heatmap (simplified)
        correlation_matrix = np.corrcoef(X_sample.T)
        top_10_features = importance_order[:10]
        correlation_subset = correlation_matrix[np.ix_(top_10_features, top_10_features)]
        
        fig.add_trace(
            go.Heatmap(
                z=correlation_subset,
                x=[self.feature_names[i] for i in top_10_features],
                y=[self.feature_names[i] for i in top_10_features],
                colorscale='RdBu',
                zmid=0,
                showscale=True
            ),
            row=2, col=1
        )
        
        # 4. Individual prediction explanation (first sample)
        if not isinstance(shap_values, list):
            sample_shap = shap_values[0]
            sample_features = importance_order[:8]  # Top 8 for individual prediction
            
            fig.add_trace(
                go.Bar(
                    x=sample_shap[sample_features],
                    y=[self.feature_names[i] for i in sample_features],
                    orientation='h',
                    marker_color=['red' if x < 0 else 'blue' for x in sample_shap[sample_features]],
                    name="SHAP Values"
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="🔍 SHAP Model Interpretability Dashboard",
            height=800,
            showlegend=False,
            template="plotly_white"
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="SHAP Importance", row=1, col=1)
        fig.update_xaxes(title_text="Feature Value", row=1, col=2)
        fig.update_yaxes(title_text="SHAP Value", row=1, col=2)
        fig.update_xaxes(title_text="SHAP Value", row=2, col=2)
        
        if save_html:
            filename = "shap_interpretability_dashboard.html"
            filepath = os.path.join(self.results_dir, filename)
            fig.write_html(filepath)
            print(f"💾 SHAP dashboard saved to {filepath}")
        
        return fig
    
    def create_decision_boundary_plot(self, feature_indices=(0, 1), resolution=100, save_plot=True) -> go.Figure:
        """Create decision boundary visualization for 2D feature space"""
        
        if len(self.feature_names) < 2:
            print("⚠️  Need at least 2 features for decision boundary plot")
            return None
        
        if not self.is_classifier:
            print("⚠️  Decision boundaries only available for classification models")
            return None
        
        print(f"🎨 Creating decision boundary plot for features {feature_indices}")
        
        # Extract two features
        X_2d = self.X_test[:, feature_indices]
        
        # Create mesh grid
        x_min, x_max = X_2d[:, 0].min() - 0.1, X_2d[:, 0].max() + 0.1
        y_min, y_max = X_2d[:, 1].min() - 0.1, X_2d[:, 1].max() + 0.1
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        # Create prediction grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # We need to create a full feature vector for prediction
        # Use mean values for other features
        if self.X_test.shape[1] > 2:
            other_features_mean = np.mean(self.X_test, axis=0)
            full_grid = np.tile(other_features_mean, (len(grid_points), 1))
            full_grid[:, feature_indices] = grid_points
        else:
            full_grid = grid_points
        
        # Get predictions
        try:
            if hasattr(self.model, 'predict_proba'):
                Z = self.model.predict_proba(full_grid)[:, 1]  # Probability of positive class
            elif hasattr(self.model, 'decision_function'):
                Z = self.model.decision_function(full_grid)
            else:
                Z = self.model.predict(full_grid)
                
            Z = Z.reshape(xx.shape)
        except Exception as e:
            print(f"⚠️  Error creating decision boundary: {e}")
            return None
        
        # Create plot
        fig = go.Figure()
        
        # Add decision boundary contour
        fig.add_trace(go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=Z,
            showscale=True,
            colorscale='RdYlBu',
            opacity=0.7,
            contours=dict(
                showlines=True,
                showlabels=True
            ),
            name="Decision Boundary"
        ))
        
        # Add data points
        for class_label in np.unique(self.y_test):
            mask = self.y_test == class_label
            X_class = X_2d[mask]
            
            class_name = str(class_label)
            if self.class_names and class_label < len(self.class_names):
                class_name = self.class_names[class_label]
            
            fig.add_trace(go.Scatter(
                x=X_class[:, 0],
                y=X_class[:, 1],
                mode='markers',
                name=f'Class {class_name}',
                marker=dict(
                    size=8,
                    line=dict(width=1, color='black'),
                    opacity=0.8
                )
            ))
        
        # Update layout
        feature_x = self.feature_names[feature_indices[0]]
        feature_y = self.feature_names[feature_indices[1]]
        
        fig.update_layout(
            title=f"🎯 Decision Boundary: {feature_x} vs {feature_y}",
            xaxis_title=feature_x,
            yaxis_title=feature_y,
            template="plotly_white",
            height=600
        )
        
        if save_plot:
            filename = f"decision_boundary_{feature_indices[0]}_{feature_indices[1]}.html"
            filepath = os.path.join(self.results_dir, filename)
            fig.write_html(filepath)
            print(f"💾 Decision boundary plot saved to {filepath}")
        
        return fig
    
    def generate_interpretation_report(self) -> Dict[str, Any]:
        """Generate comprehensive interpretation report"""
        print("📋 Generating comprehensive interpretation report...")
        
        report = {
            "model_info": {
                "model_type": str(type(self.model).__name__),
                "is_classifier": self.is_classifier,
                "n_features": len(self.feature_names),
                "n_samples": len(self.X_test),
                "feature_names": self.feature_names
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Feature importance analysis
        methods = ['permutation']
        if hasattr(self.model, 'feature_importances_'):
            methods.append('builtin')
        if SHAP_AVAILABLE:
            methods.append('shap_mean')
        
        feature_importance_results = {}
        for method in methods:
            try:
                importance_data = self.calculate_feature_importance(method=method)
                if importance_data:
                    # Get top 10 features
                    indices = np.argsort(importance_data['importance_scores'])[::-1][:10]
                    top_features = {
                        self.feature_names[i]: float(importance_data['importance_scores'][i])
                        for i in indices
                    }
                    feature_importance_results[method] = top_features
            except Exception as e:
                print(f"⚠️  Error with {method} importance: {e}")
        
        report["feature_importance"] = feature_importance_results
        
        # Model performance on test set
        try:
            y_pred = self.model.predict(self.X_test)
            if self.is_classifier:
                accuracy = accuracy_score(self.y_test, y_pred)
                report["performance"] = {
                    "accuracy": float(accuracy),
                    "prediction_method": "classification"
                }
            else:
                mse = np.mean((self.y_test - y_pred) ** 2)
                report["performance"] = {
                    "mse": float(mse),
                    "prediction_method": "regression"
                }
        except Exception as e:
            print(f"⚠️  Error calculating performance: {e}")
        
        # Save report
        filename = f"interpretation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Interpretation report saved to {filepath}")
        
        return report


def demonstrate_model_interpreter():
    """Demonstrate the model interpretability tools"""
    print("🔬 Model Interpretability Demonstration")
    print("=" * 50)
    
    # Create synthetic dataset
    print("📊 Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
    class_names = ["Class_A", "Class_B"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train a random forest model
    print("🌲 Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Test accuracy
    accuracy = model.score(X_test, y_test)
    print(f"✅ Model accuracy: {accuracy:.3f}")
    
    # Initialize interpreter
    interpreter = ModelInterpreter(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        class_names=class_names
    )
    
    # Generate various interpretations
    print("\n🔍 Analyzing model interpretability...")
    
    # Feature importance
    interpreter.plot_feature_importance(method='builtin')
    interpreter.plot_feature_importance(method='permutation')
    
    # Decision boundary (using first two features)
    interpreter.create_decision_boundary_plot(feature_indices=(0, 1))
    
    # SHAP analysis (if available)
    if SHAP_AVAILABLE:
        interpreter.create_shap_dashboard()
    
    # Generate comprehensive report
    report = interpreter.generate_interpretation_report()
    
    print("\n📋 Interpretation Summary:")
    print(f"Model: {report['model_info']['model_type']}")
    print(f"Features: {report['model_info']['n_features']}")
    print(f"Test samples: {report['model_info']['n_samples']}")
    
    if 'builtin' in report['feature_importance']:
        print("\n🏆 Top 5 Most Important Features (Built-in):")
        top_features = list(report['feature_importance']['builtin'].items())[:5]
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.3f}")
    
    print(f"\n📁 Results saved to: {interpreter.results_dir}")
    
    return interpreter


if __name__ == "__main__":
    # Run demonstration
    demo_interpreter = demonstrate_model_interpreter()