#!/usr/bin/env python3
"""
AI Tutorial by AI - Project Demo
Run this script to see a quick demo of all tutorial components
"""

import sys
import os
import subprocess

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def run_example(script_name, description):
    """Run an example script and report results"""
    print(f"\n🚀 Running: {description}")
    print(f"📄 Script: {script_name}")
    print("-" * 40)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Success!")
            # Show last few lines of output
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-3:]:
                if line.strip():
                    print(f"   {line}")
        else:
            print("❌ Error occurred!")
            print(f"   {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Timeout - script took too long")
    except Exception as e:
        print(f"💥 Exception: {e}")

def main():
    """Run the complete project demo"""
    print("🤖 AI Tutorial by AI - Complete Project Demo")
    print("Welcome to your comprehensive AI learning platform!")
    
    # Check if we're in the right directory
    if not os.path.exists('requirements.txt'):
        print("❌ Please run this script from the project root directory")
        return
    
    print_header("Project Overview")
    print("📚 This project includes:")
    print("   • 4 Complete Tutorial Modules")
    print("   • Runnable Python Examples")
    print("   • Interactive Jupyter Notebooks")
    print("   • Sample Datasets")
    print("   • Comprehensive Documentation")
    print("   • Visualization Examples")
    
    print_header("Tutorial Components")
    
    # List all components
    tutorials = [
        ("01_basics", "Python for Data Science"),
        ("02_data_visualization", "Creating Compelling Visualizations"),
        ("03_machine_learning", "ML Algorithms & Techniques"),
        ("04_neural_networks", "Deep Learning Fundamentals")
    ]
    
    for folder, description in tutorials:
        if os.path.exists(f"tutorials/{folder}/README.md"):
            print(f"✅ {description} - tutorials/{folder}/")
        else:
            print(f"❌ {description} - Missing!")
    
    print_header("Runnable Examples Demo")
    
    # Test each example
    examples = [
        ("examples/01_numpy_pandas_basics.py", "NumPy & Pandas Fundamentals"),
        ("examples/02_visualization_examples.py", "Data Visualization Gallery"),
        ("examples/03_ml_examples.py", "Machine Learning Models"),
        ("examples/04_neural_network_examples.py", "Neural Networks from Scratch")
    ]
    
    for script, description in examples:
        if os.path.exists(script):
            run_example(script, description)
        else:
            print(f"❌ Missing: {script}")
    
    print_header("Generated Content")
    
    # Check generated files
    generated_files = [
        "sample_data/classification_sample.csv",
        "sample_data/regression_sample.csv", 
        "sample_data/timeseries_sample.csv",
        "notebooks/ai_tutorial_complete.ipynb",
        "docs/getting_started.md"
    ]
    
    for file_path in generated_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ Missing: {file_path}")
    
    # Count visualization files
    viz_files = [f for f in os.listdir('.') if f.endswith('.png')]
    html_files = [f for f in os.listdir('.') if f.endswith('.html')]
    
    print(f"📊 Generated {len(viz_files)} visualization images")
    print(f"🌐 Generated {len(html_files)} interactive HTML plots")
    
    print_header("Quick Start Instructions")
    print("🎓 To start learning:")
    print("   1. pip install -r requirements.txt")
    print("   2. python create_sample_data.py")
    print("   3. jupyter lab  # Open notebooks/ai_tutorial_complete.ipynb")
    print("   4. Or run individual examples in examples/ folder")
    
    print_header("Next Steps")
    print("🚀 Continue your AI journey:")
    print("   • Work through the Jupyter notebook interactively")
    print("   • Modify examples to experiment with different parameters")
    print("   • Try your own datasets with the provided utilities")
    print("   • Build your own AI projects using these foundations")
    
    print("\n🎉 Demo complete! Happy learning! 🤖✨")

if __name__ == "__main__":
    main()