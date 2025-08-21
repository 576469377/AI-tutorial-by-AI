"""
Python Basics for Data Science
This example demonstrates fundamental concepts with NumPy and Pandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def numpy_basics():
    """Demonstrate basic NumPy operations"""
    print("=== NumPy Basics ===")
    
    # Creating arrays
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])
    
    print(f"1D Array: {arr1}")
    print(f"2D Array:\n{arr2}")
    print(f"Array shape: {arr2.shape}")
    print(f"Array type: {arr1.dtype}")
    
    # Mathematical operations
    print(f"\nMathematical operations:")
    print(f"Sum: {np.sum(arr1)}")
    print(f"Mean: {np.mean(arr1)}")
    print(f"Standard deviation: {np.std(arr1)}")
    
    # Array operations
    print(f"\nArray operations:")
    print(f"Original: {arr1}")
    print(f"Squared: {arr1 ** 2}")
    print(f"Greater than 3: {arr1 > 3}")
    
    return arr1, arr2

def pandas_basics():
    """Demonstrate basic Pandas operations"""
    print("\n=== Pandas Basics ===")
    
    # Creating DataFrames
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'age': [25, 30, 35, 28],
        'city': ['New York', 'London', 'Tokyo', 'Paris'],
        'salary': [50000, 60000, 70000, 55000]
    }
    
    df = pd.DataFrame(data)
    print("Sample DataFrame:")
    print(df)
    
    # Basic info
    print(f"\nDataFrame info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    
    # Basic operations
    print(f"\nBasic operations:")
    print(f"Average age: {df['age'].mean():.1f}")
    print(f"Maximum salary: ${df['salary'].max():,}")
    print(f"Cities: {df['city'].unique()}")
    
    # Filtering
    high_earners = df[df['salary'] > 55000]
    print(f"\nHigh earners (>$55,000):")
    print(high_earners[['name', 'salary']])
    
    return df

def load_sample_data():
    """Load and explore sample datasets"""
    print("\n=== Loading Sample Data ===")
    
    try:
        # Load sample datasets
        classification_df = pd.read_csv('sample_data/classification_sample.csv')
        regression_df = pd.read_csv('sample_data/regression_sample.csv')
        
        print("Classification dataset:")
        print(f"Shape: {classification_df.shape}")
        print(classification_df.head())
        
        print(f"\nRegression dataset:")
        print(f"Shape: {regression_df.shape}")
        print(regression_df.head())
        
        # Basic statistics
        print(f"\nClassification target distribution:")
        print(classification_df['target'].value_counts())
        
        return classification_df, regression_df
        
    except FileNotFoundError:
        print("Sample data not found. Run create_sample_data.py first.")
        return None, None

def create_basic_visualization():
    """Create simple visualizations to demonstrate basics"""
    print("\n=== Basic Visualization ===")
    
    # Simple line plot
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, y)
    plt.title('Sine Wave')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.grid(True)
    
    # Histogram
    plt.subplot(1, 2, 2)
    data = np.random.normal(0, 1, 1000)
    plt.hist(data, bins=30, alpha=0.7)
    plt.title('Random Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('basic_plots.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print("Plots saved as 'basic_plots.png'")

def main():
    """Main function to run all examples"""
    print("Python Basics for Data Science - Examples")
    print("=" * 50)
    
    # NumPy examples
    arr1, arr2 = numpy_basics()
    
    # Pandas examples
    df = pandas_basics()
    
    # Load sample data
    class_df, reg_df = load_sample_data()
    
    # Create visualizations
    create_basic_visualization()
    
    print("\n" + "=" * 50)
    print("Tutorial complete! Check out the generated plots.")
    print("Next: Try running the Jupyter notebook for interactive exploration.")

if __name__ == "__main__":
    main()