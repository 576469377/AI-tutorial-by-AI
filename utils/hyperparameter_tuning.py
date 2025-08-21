#!/usr/bin/env python3
"""
Automated Hyperparameter Tuning
===============================

This module provides comprehensive tools for automated hyperparameter optimization
using various strategies including grid search, random search, and Bayesian optimization.

Features:
- Grid search optimization
- Random search optimization
- Bayesian optimization (optional)
- Cross-validation integration
- Performance tracking and visualization
- Automated model comparison

Author: AI Tutorial by AI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score,
    StratifiedKFold, KFold, validation_curve
)
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Optional Bayesian optimization
try:
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("💡 Scikit-optimize not available. Install with: pip install scikit-optimize")

from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import os
import json
import time
from datetime import datetime
from itertools import product

class HyperparameterTuner:
    """Comprehensive hyperparameter tuning toolkit"""
    
    def __init__(self, 
                 model_class,
                 param_space: Dict[str, List],
                 scoring: str = 'accuracy',
                 cv_folds: int = 5,
                 random_state: int = 42,
                 results_dir: str = "tuning_results"):
        """
        Initialize the hyperparameter tuner
        
        Args:
            model_class: ML model class to tune
            param_space: Dictionary defining parameter search space
            scoring: Scoring metric for optimization
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            results_dir: Directory to save results
        """
        self.model_class = model_class
        self.param_space = param_space
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results_dir = results_dir
        
        # Results storage
        self.tuning_results = {}
        self.best_params = {}
        self.best_scores = {}
        self.tuning_history = []
        
        # Performance tracking
        self.search_times = {}
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"🎯 Hyperparameter Tuner initialized for {model_class.__name__}")
        print(f"📊 Parameter space: {len(param_space)} parameters")
        print(f"🔍 Scoring metric: {scoring}")
        print(f"📁 Results will be saved to: {results_dir}")
    
    def grid_search(self, X, y, n_jobs=-1, verbose=True) -> Dict[str, Any]:
        """
        Perform grid search optimization
        
        Args:
            X: Training features
            y: Training labels
            n_jobs: Number of parallel jobs
            verbose: Whether to print progress
            
        Returns:
            Dictionary with grid search results
        """
        print(f"\n🔍 Starting Grid Search...")
        print(f"📊 Total combinations: {np.prod([len(v) for v in self.param_space.values()])}")
        
        start_time = time.time()
        
        # Create model instance
        model = self.model_class(random_state=self.random_state)
        
        # Setup cross-validation
        if 'classif' in str(self.model_class).lower():
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.param_space,
            scoring=self.scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1 if verbose else 0,
            return_train_score=True
        )
        
        grid_search.fit(X, y)
        
        search_time = time.time() - start_time
        self.search_times['grid_search'] = search_time
        
        # Store results
        results = {
            'method': 'grid_search',
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_,
            'cv_results': pd.DataFrame(grid_search.cv_results_),
            'search_time': search_time,
            'n_combinations': len(grid_search.cv_results_['params'])
        }
        
        self.tuning_results['grid_search'] = results
        self.best_params['grid_search'] = grid_search.best_params_
        self.best_scores['grid_search'] = grid_search.best_score_
        
        if verbose:
            print(f"✅ Grid Search completed in {search_time:.2f} seconds")
            print(f"🏆 Best score: {grid_search.best_score_:.4f}")
            print(f"🎯 Best parameters: {grid_search.best_params_}")
        
        return results
    
    def random_search(self, X, y, n_iter=100, n_jobs=-1, verbose=True) -> Dict[str, Any]:
        """
        Perform random search optimization
        
        Args:
            X: Training features
            y: Training labels
            n_iter: Number of random combinations to try
            n_jobs: Number of parallel jobs
            verbose: Whether to print progress
            
        Returns:
            Dictionary with random search results
        """
        print(f"\n🎲 Starting Random Search...")
        print(f"🔢 Random iterations: {n_iter}")
        
        start_time = time.time()
        
        # Create model instance
        model = self.model_class(random_state=self.random_state)
        
        # Setup cross-validation
        if 'classif' in str(self.model_class).lower():
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Perform random search
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=self.param_space,
            n_iter=n_iter,
            scoring=self.scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1 if verbose else 0,
            random_state=self.random_state,
            return_train_score=True
        )
        
        random_search.fit(X, y)
        
        search_time = time.time() - start_time
        self.search_times['random_search'] = search_time
        
        # Store results
        results = {
            'method': 'random_search',
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'best_estimator': random_search.best_estimator_,
            'cv_results': pd.DataFrame(random_search.cv_results_),
            'search_time': search_time,
            'n_iterations': n_iter
        }
        
        self.tuning_results['random_search'] = results
        self.best_params['random_search'] = random_search.best_params_
        self.best_scores['random_search'] = random_search.best_score_
        
        if verbose:
            print(f"✅ Random Search completed in {search_time:.2f} seconds")
            print(f"🏆 Best score: {random_search.best_score_:.4f}")
            print(f"🎯 Best parameters: {random_search.best_params_}")
        
        return results
    
    def bayesian_optimization(self, X, y, n_calls=50, verbose=True) -> Dict[str, Any]:
        """
        Perform Bayesian optimization (requires scikit-optimize)
        
        Args:
            X: Training features
            y: Training labels
            n_calls: Number of optimization calls
            verbose: Whether to print progress
            
        Returns:
            Dictionary with Bayesian optimization results
        """
        if not BAYESIAN_AVAILABLE:
            print("⚠️  Bayesian optimization not available. Install scikit-optimize: pip install scikit-optimize")
            return {}
        
        print(f"\n🧠 Starting Bayesian Optimization...")
        print(f"🔢 Optimization calls: {n_calls}")
        
        start_time = time.time()
        
        # Convert parameter space to skopt format
        dimensions = []
        param_names = []
        
        for param, values in self.param_space.items():
            param_names.append(param)
            
            # Determine parameter type and create appropriate dimension
            if isinstance(values[0], int):
                dimensions.append(Integer(min(values), max(values), name=param))
            elif isinstance(values[0], float):
                dimensions.append(Real(min(values), max(values), name=param))
            else:
                dimensions.append(Categorical(values, name=param))
        
        # Define objective function
        @use_named_args(dimensions)
        def objective(**params):
            # Create model with current parameters
            model = self.model_class(random_state=self.random_state, **params)
            
            # Setup cross-validation
            if 'classif' in str(self.model_class).lower():
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            else:
                cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            # Calculate cross-validation score
            scores = cross_val_score(model, X, y, cv=cv, scoring=self.scoring, n_jobs=-1)
            
            # Return negative score (since skopt minimizes)
            return -np.mean(scores)
        
        # Perform Bayesian optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=self.random_state,
            verbose=verbose
        )
        
        search_time = time.time() - start_time
        self.search_times['bayesian_optimization'] = search_time
        
        # Extract best parameters
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun  # Convert back to positive score
        
        # Create best estimator
        best_estimator = self.model_class(random_state=self.random_state, **best_params)
        best_estimator.fit(X, y)
        
        # Store results
        results = {
            'method': 'bayesian_optimization',
            'best_params': best_params,
            'best_score': best_score,
            'best_estimator': best_estimator,
            'optimization_result': result,
            'search_time': search_time,
            'n_calls': n_calls,
            'convergence': result.func_vals
        }
        
        self.tuning_results['bayesian_optimization'] = results
        self.best_params['bayesian_optimization'] = best_params
        self.best_scores['bayesian_optimization'] = best_score
        
        if verbose:
            print(f"✅ Bayesian Optimization completed in {search_time:.2f} seconds")
            print(f"🏆 Best score: {best_score:.4f}")
            print(f"🎯 Best parameters: {best_params}")
        
        return results
    
    def compare_methods(self, X, y, methods=['grid_search', 'random_search'], 
                       random_n_iter=100, bayesian_n_calls=50) -> pd.DataFrame:
        """
        Compare different optimization methods
        
        Args:
            X: Training features
            y: Training labels
            methods: List of methods to compare
            random_n_iter: Number of iterations for random search
            bayesian_n_calls: Number of calls for Bayesian optimization
            
        Returns:
            DataFrame comparing method performance
        """
        print("🔄 Comparing optimization methods...")
        
        comparison_results = []
        
        for method in methods:
            if method == 'grid_search':
                result = self.grid_search(X, y, verbose=False)
            elif method == 'random_search':
                result = self.random_search(X, y, n_iter=random_n_iter, verbose=False)
            elif method == 'bayesian_optimization':
                result = self.bayesian_optimization(X, y, n_calls=bayesian_n_calls, verbose=False)
            else:
                print(f"⚠️  Unknown method: {method}")
                continue
            
            if result:
                comparison_results.append({
                    'Method': method.replace('_', ' ').title(),
                    'Best Score': result['best_score'],
                    'Search Time (s)': result['search_time'],
                    'Evaluations': result.get('n_combinations', result.get('n_iterations', result.get('n_calls', 0))),
                    'Efficiency (Score/Time)': result['best_score'] / result['search_time']
                })
        
        comparison_df = pd.DataFrame(comparison_results)
        
        # Save comparison
        filepath = os.path.join(self.results_dir, "method_comparison.csv")
        comparison_df.to_csv(filepath, index=False)
        print(f"💾 Method comparison saved to {filepath}")
        
        return comparison_df
    
    def plot_optimization_progress(self, methods=None, save_plot=True) -> go.Figure:
        """Plot optimization progress for different methods"""
        
        if methods is None:
            methods = list(self.tuning_results.keys())
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Score vs Iteration",
                "Time Comparison", 
                "Parameter Sensitivity",
                "Best Scores Comparison"
            ]
        )
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # 1. Score vs Iteration
        for i, method in enumerate(methods):
            if method in self.tuning_results:
                result = self.tuning_results[method]
                
                if 'cv_results' in result:
                    scores = result['cv_results']['mean_test_score']
                    iterations = range(1, len(scores) + 1)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=list(iterations),
                            y=scores,
                            mode='lines+markers',
                            name=f'{method.replace("_", " ").title()}',
                            line=dict(color=colors[i % len(colors)])
                        ),
                        row=1, col=1
                    )
                elif 'convergence' in result:
                    # Bayesian optimization convergence
                    scores = [-x for x in result['convergence']]  # Convert back to positive
                    iterations = range(1, len(scores) + 1)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=list(iterations),
                            y=scores,
                            mode='lines+markers',
                            name=f'{method.replace("_", " ").title()}',
                            line=dict(color=colors[i % len(colors)])
                        ),
                        row=1, col=1
                    )
        
        # 2. Time Comparison
        method_names = []
        search_times = []
        for method in methods:
            if method in self.search_times:
                method_names.append(method.replace('_', ' ').title())
                search_times.append(self.search_times[method])
        
        if method_names:
            fig.add_trace(
                go.Bar(
                    x=method_names,
                    y=search_times,
                    name='Search Time',
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
        
        # 3. Parameter Sensitivity (for first method with detailed results)
        first_method = next((m for m in methods if m in self.tuning_results and 'cv_results' in self.tuning_results[m]), None)
        if first_method:
            cv_results = self.tuning_results[first_method]['cv_results']
            
            # Find most important parameter (most variance in scores)
            param_cols = [col for col in cv_results.columns if col.startswith('param_')]
            if param_cols:
                param_col = param_cols[0]  # Use first parameter
                unique_params = cv_results[param_col].unique()
                
                param_scores = []
                for param_val in unique_params:
                    mask = cv_results[param_col] == param_val
                    mean_score = cv_results.loc[mask, 'mean_test_score'].mean()
                    param_scores.append(mean_score)
                
                fig.add_trace(
                    go.Scatter(
                        x=unique_params,
                        y=param_scores,
                        mode='lines+markers',
                        name=f'{param_col.replace("param_", "")}',
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )
        
        # 4. Best Scores Comparison
        if self.best_scores:
            method_names = [m.replace('_', ' ').title() for m in self.best_scores.keys()]
            best_scores = list(self.best_scores.values())
            
            fig.add_trace(
                go.Bar(
                    x=method_names,
                    y=best_scores,
                    name='Best Score',
                    marker_color='lightgreen'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="🎯 Hyperparameter Optimization Progress",
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Iteration", row=1, col=1)
        fig.update_yaxes(title_text="CV Score", row=1, col=1)
        fig.update_yaxes(title_text="Time (seconds)", row=1, col=2)
        fig.update_yaxes(title_text="CV Score", row=2, col=1)
        fig.update_yaxes(title_text="Best Score", row=2, col=2)
        
        if save_plot:
            filename = "optimization_progress.html"
            filepath = os.path.join(self.results_dir, filename)
            fig.write_html(filepath)
            print(f"💾 Optimization progress plot saved to {filepath}")
        
        return fig
    
    def generate_tuning_report(self) -> Dict[str, Any]:
        """Generate comprehensive tuning report"""
        print("📋 Generating hyperparameter tuning report...")
        
        report = {
            "model_info": {
                "model_class": str(self.model_class.__name__),
                "parameter_space": self.param_space,
                "scoring_metric": self.scoring,
                "cv_folds": self.cv_folds
            },
            "methods_used": list(self.tuning_results.keys()),
            "best_params_by_method": self.best_params,
            "best_scores_by_method": self.best_scores,
            "search_times": self.search_times,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add detailed results for each method
        detailed_results = {}
        for method, results in self.tuning_results.items():
            method_summary = {
                "best_score": results['best_score'],
                "best_params": results['best_params'],
                "search_time": results['search_time']
            }
            
            if 'cv_results' in results:
                cv_results = results['cv_results']
                method_summary.update({
                    "mean_score": float(cv_results['mean_test_score'].mean()),
                    "std_score": float(cv_results['mean_test_score'].std()),
                    "total_evaluations": len(cv_results)
                })
            
            detailed_results[method] = method_summary
        
        report["detailed_results"] = detailed_results
        
        # Save report
        filename = f"tuning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Tuning report saved to {filepath}")
        
        return report


def demonstrate_hyperparameter_tuning():
    """Demonstrate the hyperparameter tuning capabilities"""
    print("🔬 Hyperparameter Tuning Demonstration")
    print("=" * 50)
    
    # Create synthetic dataset
    print("📊 Creating synthetic classification dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Dataset created: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Define parameter space for Random Forest
    param_space = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        model_class=RandomForestClassifier,
        param_space=param_space,
        scoring='accuracy',
        cv_folds=5
    )
    
    # Compare methods
    print("\n🔄 Comparing optimization methods...")
    comparison_df = tuner.compare_methods(
        X_train_scaled, y_train,
        methods=['grid_search', 'random_search'],
        random_n_iter=50
    )
    
    print("\n📊 Method Comparison Results:")
    print(comparison_df.to_string(index=False))
    
    # Test best model
    best_method = comparison_df.loc[comparison_df['Best Score'].idxmax(), 'Method'].lower().replace(' ', '_')
    best_model = tuner.tuning_results[best_method]['best_estimator']
    
    test_accuracy = best_model.score(X_test_scaled, y_test)
    print(f"\n🏆 Best model test accuracy: {test_accuracy:.4f}")
    print(f"🎯 Best parameters: {tuner.best_params[best_method]}")
    
    # Generate visualizations
    tuner.plot_optimization_progress()
    
    # Generate comprehensive report
    report = tuner.generate_tuning_report()
    
    print(f"\n📁 Results saved to: {tuner.results_dir}")
    
    return tuner


if __name__ == "__main__":
    # Run demonstration
    demo_tuner = demonstrate_hyperparameter_tuning()