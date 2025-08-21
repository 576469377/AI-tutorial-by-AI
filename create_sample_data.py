"""
Sample data generation for AI tutorials
This module creates sample datasets for learning purposes
"""

import numpy as np
import pandas as pd
import os

def create_sample_datasets():
    """Create sample datasets for tutorials"""
    
    # Ensure sample_data directory exists
    os.makedirs('sample_data', exist_ok=True)
    
    # 1. Simple classification dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = np.random.normal(0, 1, n_samples)
    
    # Create target based on simple rule
    target = (feature1 + feature2 > 0).astype(int)
    
    # Add some noise
    noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    target[noise_indices] = 1 - target[noise_indices]
    
    classification_df = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'target': target
    })
    
    classification_df.to_csv('sample_data/classification_sample.csv', index=False)
    print("Created classification_sample.csv")
    
    # 2. Regression dataset
    x = np.linspace(0, 10, 200)
    y = 2 * x + 1 + np.random.normal(0, 1, 200)
    
    regression_df = pd.DataFrame({
        'x': x,
        'y': y
    })
    
    regression_df.to_csv('sample_data/regression_sample.csv', index=False)
    print("Created regression_sample.csv")
    
    # 3. Time series data
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    trend = np.linspace(100, 200, 365)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(365) / 365.25)
    noise = np.random.normal(0, 5, 365)
    values = trend + seasonal + noise
    
    timeseries_df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    timeseries_df.to_csv('sample_data/timeseries_sample.csv', index=False)
    print("Created timeseries_sample.csv")

if __name__ == "__main__":
    create_sample_datasets()
    print("All sample datasets created successfully!")