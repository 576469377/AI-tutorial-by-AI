"""
Data Visualization Examples
Comprehensive examples of plotting with matplotlib, seaborn, and plotly
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def matplotlib_examples():
    """Demonstrate matplotlib plotting capabilities"""
    print("=== Matplotlib Examples ===")
    
    # Create sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Matplotlib Examples', fontsize=16)
    
    # Line plot
    axes[0, 0].plot(x, y1, label='sin(x)', linewidth=2)
    axes[0, 0].plot(x, y2, label='cos(x)', linewidth=2)
    axes[0, 0].set_title('Line Plot')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Scatter plot
    n = 100
    x_scatter = np.random.randn(n)
    y_scatter = x_scatter + np.random.randn(n) * 0.5
    axes[0, 1].scatter(x_scatter, y_scatter, alpha=0.6)
    axes[0, 1].set_title('Scatter Plot')
    axes[0, 1].set_xlabel('X values')
    axes[0, 1].set_ylabel('Y values')
    
    # Histogram
    data = np.random.normal(0, 1, 1000)
    axes[1, 0].hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_title('Histogram')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    
    # Bar plot
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 17, 35, 29, 12]
    axes[1, 1].bar(categories, values, color='lightcoral')
    axes[1, 1].set_title('Bar Plot')
    axes[1, 1].set_xlabel('Categories')
    axes[1, 1].set_ylabel('Values')
    
    plt.tight_layout()
    plt.savefig('matplotlib_examples.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print("Matplotlib examples saved as 'matplotlib_examples.png'")

def seaborn_examples():
    """Demonstrate seaborn statistical plotting"""
    print("\n=== Seaborn Examples ===")
    
    # Load sample data or create it
    try:
        df = pd.read_csv('sample_data/classification_sample.csv')
    except FileNotFoundError:
        # Create sample data if file doesn't exist
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 500),
            'feature2': np.random.normal(0, 1, 500),
            'target': np.random.choice([0, 1], 500)
        })
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Seaborn Examples', fontsize=16)
    
    # Distribution plot
    sns.histplot(data=df, x='feature1', hue='target', kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution Plot')
    
    # Scatter plot with regression
    sns.scatterplot(data=df, x='feature1', y='feature2', hue='target', ax=axes[0, 1])
    axes[0, 1].set_title('Scatter Plot by Category')
    
    # Box plot
    df_melted = df.melt(id_vars=['target'], value_vars=['feature1', 'feature2'])
    sns.boxplot(data=df_melted, x='variable', y='value', hue='target', ax=axes[1, 0])
    axes[1, 0].set_title('Box Plot')
    
    # Correlation heatmap
    correlation_matrix = df[['feature1', 'feature2']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
    axes[1, 1].set_title('Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig('seaborn_examples.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print("Seaborn examples saved as 'seaborn_examples.png'")

def plotly_examples():
    """Demonstrate plotly interactive plotting"""
    print("\n=== Plotly Examples ===")
    
    # Create sample data
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'x': np.random.randn(n),
        'y': np.random.randn(n),
        'size': np.random.randint(10, 50, n),
        'color': np.random.choice(['A', 'B', 'C'], n),
        'value': np.random.randint(1, 100, n)
    })
    
    # Interactive scatter plot
    fig1 = px.scatter(df, x='x', y='y', size='size', color='color',
                     hover_data=['value'], title='Interactive Scatter Plot')
    fig1.write_html('plotly_scatter.html')
    fig1.show()
    
    # Time series example
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    ts_data = pd.DataFrame({
        'date': dates,
        'value': np.cumsum(np.random.randn(100)) + 100
    })
    
    fig2 = px.line(ts_data, x='date', y='value', title='Time Series Plot')
    fig2.write_html('plotly_timeseries.html')
    fig2.show()
    
    # 3D scatter plot
    fig3 = go.Figure(data=[go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=df['value'],
        mode='markers',
        marker=dict(
            size=df['size']/3,
            color=df['value'],
            colorscale='Viridis',
            showscale=True
        )
    )])
    fig3.update_layout(title='3D Scatter Plot')
    fig3.write_html('plotly_3d.html')
    fig3.show()
    
    print("Plotly examples saved as HTML files for interactive viewing")

def visualization_tips():
    """Demonstrate best practices for data visualization"""
    print("\n=== Visualization Best Practices ===")
    
    # Create sample data
    categories = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    values = [85, 92, 78, 95, 88]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Poor visualization
    axes[0].bar(categories, values, color=['red', 'green', 'blue', 'yellow', 'purple'])
    axes[0].set_title('Poor Visualization', fontsize=10)
    axes[0].tick_params(axis='x', rotation=45, labelsize=8)
    
    # Good visualization
    colors = ['#1f77b4' if v < 90 else '#ff7f0e' for v in values]
    bars = axes[1].bar(categories, values, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_title('Good Visualization: Clear Colors and Labels', fontsize=12)
    axes[1].set_ylabel('Performance Score', fontsize=11)
    axes[1].set_ylim(70, 100)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(value), ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('visualization_tips.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print("Visualization tips saved as 'visualization_tips.png'")
    
    # Print tips
    tips = [
        "1. Use consistent color schemes",
        "2. Add clear titles and labels",
        "3. Remove unnecessary elements (chartjunk)",
        "4. Choose appropriate chart types for your data",
        "5. Consider colorblind-friendly palettes",
        "6. Make text readable (size, contrast)",
        "7. Add data labels when helpful",
        "8. Use whitespace effectively"
    ]
    
    print("\nVisualization Best Practices:")
    for tip in tips:
        print(f"  {tip}")

def main():
    """Run all visualization examples"""
    print("Data Visualization Examples")
    print("=" * 50)
    
    # Matplotlib examples
    matplotlib_examples()
    
    # Seaborn examples
    seaborn_examples()
    
    # Plotly examples (interactive)
    plotly_examples()
    
    # Best practices
    visualization_tips()
    
    print("\n" + "=" * 50)
    print("All visualization examples complete!")
    print("Check the generated image files and HTML files for interactive plots.")

if __name__ == "__main__":
    main()