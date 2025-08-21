"""
Machine Learning Examples
Comprehensive examples covering classification, regression, and clustering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, 
                           mean_squared_error, r2_score, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

def load_or_create_data():
    """Load sample data or create if not available"""
    try:
        # Try to load existing sample data
        classification_df = pd.read_csv('sample_data/classification_sample.csv')
        regression_df = pd.read_csv('sample_data/regression_sample.csv')
        return classification_df, regression_df
    except FileNotFoundError:
        print("Sample data not found. Creating sample datasets...")
        
        # Create classification data
        np.random.seed(42)
        n_samples = 1000
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.normal(0, 1, n_samples)
        y_class = (X1 + X2 > 0).astype(int)
        
        classification_df = pd.DataFrame({
            'feature1': X1,
            'feature2': X2,
            'target': y_class
        })
        
        # Create regression data
        x = np.linspace(0, 10, 200)
        y_reg = 2 * x + 1 + np.random.normal(0, 1, 200)
        
        regression_df = pd.DataFrame({
            'x': x,
            'y': y_reg
        })
        
        return classification_df, regression_df

def classification_example():
    """Demonstrate binary classification"""
    print("=== Classification Example ===")
    
    # Load data
    classification_df, _ = load_or_create_data()
    
    # Prepare features and target
    X = classification_df[['feature1', 'feature2']]
    y = classification_df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"\n{name}:")
        print(f"Accuracy: {accuracy:.3f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    # Plot data
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(X['feature1'], X['feature2'], c=y, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot confusion matrix for best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test)
    
    plt.subplot(1, 3, 2)
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix\n{best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Plot model comparison
    plt.subplot(1, 3, 3)
    names = list(results.keys())
    accuracies = list(results.values())
    plt.bar(names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('Model Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('classification_results.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"\nBest model: {best_model_name} with accuracy: {results[best_model_name]:.3f}")

def regression_example():
    """Demonstrate regression analysis"""
    print("\n=== Regression Example ===")
    
    # Load data
    _, regression_df = load_or_create_data()
    
    # Prepare features and target
    X = regression_df[['x']]
    y = regression_df['y']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MSE': mse, 'R²': r2}
        
        print(f"\n{name}:")
        print(f"Mean Squared Error: {mse:.3f}")
        print(f"R² Score: {r2:.3f}")
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    # Plot original data and predictions
    plt.subplot(1, 3, 1)
    plt.scatter(X_test, y_test, alpha=0.6, label='True values')
    
    # Sort data for line plot
    sorted_indices = np.argsort(X_test.iloc[:, 0])
    plt.plot(X_test.iloc[sorted_indices, 0], 
             predictions['Linear Regression'][sorted_indices], 
             'r-', label='Linear Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Regression Results')
    plt.legend()
    
    # Plot residuals
    plt.subplot(1, 3, 2)
    residuals = y_test - predictions['Linear Regression']
    plt.scatter(predictions['Linear Regression'], residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    # Plot model comparison
    plt.subplot(1, 3, 3)
    model_names = list(results.keys())
    r2_scores = [results[name]['R²'] for name in model_names]
    plt.bar(model_names, r2_scores, color=['skyblue', 'lightgreen'])
    plt.title('Model Comparison (R² Score)')
    plt.ylabel('R² Score')
    plt.ylim(0, 1)
    
    for i, score in enumerate(r2_scores):
        plt.text(i, score + 0.02, f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('regression_results.png', dpi=100, bbox_inches='tight')
    plt.show()

def clustering_example():
    """Demonstrate unsupervised clustering"""
    print("\n=== Clustering Example ===")
    
    # Create sample data for clustering
    np.random.seed(42)
    
    # Generate clusters
    cluster1 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], 100)
    cluster2 = np.random.multivariate_normal([6, 6], [[1, 0], [0, 1]], 100)
    cluster3 = np.random.multivariate_normal([2, 6], [[1, 0], [0, 1]], 100)
    
    # Combine data
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # Apply K-means clustering
    k_values = range(1, 11)
    inertias = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Find optimal K using elbow method
    optimal_k = 3  # We know there are 3 clusters
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X)
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    # Plot original data
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot clustered data
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    plt.scatter(kmeans_final.cluster_centers_[:, 0], 
               kmeans_final.cluster_centers_[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.title(f'K-Means Clustering (K={optimal_k})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # Plot elbow curve
    plt.subplot(1, 3, 3)
    plt.plot(k_values, inertias, 'bo-')
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('clustering_results.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"Clustering complete with K={optimal_k}")
    print(f"Final inertia: {kmeans_final.inertia_:.2f}")

def cross_validation_example():
    """Demonstrate cross-validation for model evaluation"""
    print("\n=== Cross-Validation Example ===")
    
    # Load classification data
    classification_df, _ = load_or_create_data()
    X = classification_df[['feature1', 'feature2']]
    y = classification_df['target']
    
    # Models to compare
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    cv_results = {}
    
    for name, model in models.items():
        # Perform 5-fold cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_results[name] = cv_scores
        
        print(f"\n{name}:")
        print(f"CV Scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Visualize cross-validation results
    plt.figure(figsize=(10, 6))
    
    # Box plot of CV scores
    plt.subplot(1, 2, 1)
    cv_data = [cv_results[name] for name in models.keys()]
    plt.boxplot(cv_data, labels=models.keys())
    plt.title('Cross-Validation Scores')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # Mean scores comparison
    plt.subplot(1, 2, 2)
    means = [cv_results[name].mean() for name in models.keys()]
    stds = [cv_results[name].std() for name in models.keys()]
    
    plt.bar(models.keys(), means, yerr=stds, capsize=5, 
            color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('Mean CV Scores with Std Dev')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('cross_validation_results.png', dpi=100, bbox_inches='tight')
    plt.show()

def main():
    """Run all machine learning examples"""
    print("Machine Learning Examples")
    print("=" * 50)
    
    # Classification example
    classification_example()
    
    # Regression example
    regression_example()
    
    # Clustering example
    clustering_example()
    
    # Cross-validation example
    cross_validation_example()
    
    print("\n" + "=" * 50)
    print("Machine learning examples complete!")
    print("Key takeaways:")
    print("1. Always split your data into training and testing sets")
    print("2. Use cross-validation for robust model evaluation")
    print("3. Choose appropriate metrics for your problem type")
    print("4. Consider multiple algorithms and compare their performance")
    print("5. Visualize your results to gain insights")

if __name__ == "__main__":
    main()