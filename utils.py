"""
Utility functions for AI tutorials
Common helper functions used across examples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Confusion Matrix'):
    """
    Plot a confusion matrix with nice formatting
    
    Parameters:
    y_true: True labels
    y_pred: Predicted labels
    class_names: List of class names for labels
    title: Title for the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return cm

def plot_learning_curves(estimator, X, y, title='Learning Curves', cv=5):
    """
    Plot learning curves to diagnose overfitting/underfitting
    
    Parameters:
    estimator: The machine learning model
    X: Features
    y: Target
    title: Title for the plot
    cv: Number of cross-validation folds
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()

def compare_models(models, X_train, X_test, y_train, y_test, model_names=None):
    """
    Compare multiple models and return results
    
    Parameters:
    models: Dictionary or list of models
    X_train, X_test, y_train, y_test: Train/test data
    model_names: Optional list of model names
    
    Returns:
    Dictionary with model performance results
    """
    if isinstance(models, dict):
        model_items = models.items()
    else:
        if model_names is None:
            model_names = [f'Model_{i}' for i in range(len(models))]
        model_items = zip(model_names, models)
    
    results = {}
    predictions = {}
    
    for name, model in model_items:
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
    
    return results, predictions

def plot_feature_importance(model, feature_names, title='Feature Importance', top_n=None):
    """
    Plot feature importance for tree-based models
    
    Parameters:
    model: Trained model with feature_importances_ attribute
    feature_names: List of feature names
    title: Title for the plot
    top_n: Show only top N features
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model doesn't have feature importances")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    if top_n:
        indices = indices[:top_n]
        importances = importances[indices]
        feature_names = [feature_names[i] for i in indices]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], alpha=0.7)
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title(title)
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()

def create_synthetic_data(data_type='classification', n_samples=1000, noise=0.1, random_state=42):
    """
    Create synthetic datasets for examples
    
    Parameters:
    data_type: 'classification', 'regression', or 'clustering'
    n_samples: Number of samples to generate
    noise: Amount of noise to add
    random_state: Random seed for reproducibility
    
    Returns:
    X, y: Features and target
    """
    np.random.seed(random_state)
    
    if data_type == 'classification':
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                 n_informative=2, n_clusters_per_class=1, 
                                 random_state=random_state)
        
    elif data_type == 'regression':
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=n_samples, n_features=1, noise=noise*10,
                             random_state=random_state)
        
    elif data_type == 'clustering':
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=n_samples, centers=3, n_features=2,
                         cluster_std=1.0, random_state=random_state)
    
    else:
        raise ValueError("data_type must be 'classification', 'regression', or 'clustering'")
    
    return X, y

def plot_data_distribution(data, title='Data Distribution'):
    """
    Plot distribution of data with multiple subplots
    
    Parameters:
    data: DataFrame or array-like data
    title: Title for the overall plot
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    n_cols = len(data.columns)
    fig, axes = plt.subplots(1, min(n_cols, 4), figsize=(15, 4))
    
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(data.columns[:4]):  # Show max 4 columns
        if i < len(axes):
            data[col].hist(bins=30, alpha=0.7, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def save_results(results, filename='results.txt'):
    """
    Save model results to a text file
    
    Parameters:
    results: Dictionary with model results
    filename: Output filename
    """
    with open(filename, 'w') as f:
        f.write("AI Tutorial Results\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            for metric_name, value in metrics.items():
                f.write(f"  {metric_name}: {value:.3f}\n")
            f.write("\n")
    
    print(f"Results saved to {filename}")

# Example usage functions
def demo_utilities():
    """Demonstrate the utility functions"""
    print("AI Tutorial Utilities Demo")
    print("=" * 40)
    
    # Create sample data
    X, y = create_synthetic_data('classification', n_samples=500)
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
    df['Target'] = y
    
    print(f"Created sample dataset with shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Plot data distribution
    plot_data_distribution(df[['Feature1', 'Feature2']], 'Sample Data Distribution')
    
    # Create and compare models
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results, predictions = compare_models(models, X_train, X_test, y_train, y_test)
    
    # Save results
    save_results(results, 'demo_results.txt')
    
    print("\nUtilities demo complete!")

if __name__ == "__main__":
    demo_utilities()