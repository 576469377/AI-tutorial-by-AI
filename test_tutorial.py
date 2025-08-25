#!/usr/bin/env python3
"""
Comprehensive test script for the AI Tutorial project
Tests all examples and verifies functionality
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n{'='*50}")
    print(f"Testing: {description}")
    print(f"Command: {command}")
    print('='*50)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("Output preview:")
                print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print("❌ FAILED")
            print("Error output:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT (command took too long)")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
        return False

def check_files_exist(files, description):
    """Check if files exist"""
    print(f"\n{'='*50}")
    print(f"Checking: {description}")
    print('='*50)
    
    all_exist = True
    for file_path in files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            all_exist = False
    
    return all_exist

def main():
    """Run comprehensive tests"""
    print("🤖 AI Tutorial Comprehensive Test Suite")
    print("This script tests all components of the AI tutorial project")
    
    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Check Python version
    total_tests += 1
    if run_command("python --version", "Python installation"):
        tests_passed += 1
    
    # Test 2: Check required packages
    total_tests += 1
    if run_command("python -c \"import numpy, pandas, matplotlib, sklearn, torch; print('All packages imported successfully!')\"", 
                  "Package imports"):
        tests_passed += 1
    
    # Test 3: Check file structure
    total_tests += 1
    required_files = [
        "README.md",
        "requirements.txt", 
        "create_sample_data.py",
        "tutorials/00_ai_fundamentals/README.md",
        "tutorials/01_basics/README.md",
        "tutorials/02_data_visualization/README.md", 
        "tutorials/03_machine_learning/README.md",
        "tutorials/04_neural_networks/README.md",
        "tutorials/05_pytorch/README.md",
        "examples/01_numpy_pandas_basics.py",
        "examples/02_visualization_examples.py",
        "examples/03_ml_examples.py", 
        "examples/04_neural_network_examples.py",
        "examples/05_pytorch_examples.py",
        "notebooks/ai_tutorial_complete.ipynb",
        "notebooks/05_pytorch_tutorial.ipynb",
        "docs/getting_started.md",
        "docs/setup/cross_platform_setup.md"
    ]
    if check_files_exist(required_files, "Required files and directories"):
        tests_passed += 1
    
    # Test 4: Generate sample data
    total_tests += 1
    if run_command("python create_sample_data.py", "Sample data generation"):
        tests_passed += 1
    
    # Test 5: Run basic examples
    examples_to_test = [
        ("python examples/01_numpy_pandas_basics.py", "NumPy/Pandas basics"),
        ("python examples/02_visualization_examples.py", "Visualization examples"),
        ("python examples/03_ml_examples.py", "Machine learning examples"),
        ("python examples/04_neural_network_examples.py", "Neural network examples"),
        ("python examples/05_pytorch_examples.py", "PyTorch examples")
    ]
    
    for command, description in examples_to_test:
        total_tests += 1
        if run_command(command, description):
            tests_passed += 1
    
    # Test 6: Test new model evaluation demo
    total_tests += 1
    if run_command("python examples/07_model_evaluation_demo.py", "Model evaluation dashboard demo"):
        tests_passed += 1
    
    # Test 7: Test advanced AI demos
    total_tests += 1
    if run_command("python examples/08_advanced_ai_demos.py", "Advanced AI demos"):
        tests_passed += 1
    
    # Test 8: Test enhanced features demo
    total_tests += 1
    if run_command("python examples/09_enhanced_features_demo.py", "Enhanced features demo"):
        tests_passed += 1
    
    # Test 9: Test ethical AI practices demo
    total_tests += 1
    if run_command("python examples/10_ethical_ai_practices.py", "Ethical AI practices demo"):
        tests_passed += 1
    
    # Test 10: Test model deployment demo
    total_tests += 1
    if run_command("python examples/11_model_deployment_demo.py", "Model deployment demo"):
        tests_passed += 1
    
    # Test 11: Check generated files
    total_tests += 1
    generated_files = [
        "sample_data/classification_sample.csv",
        "sample_data/regression_sample.csv", 
        "sample_data/timeseries_sample.csv",
        "basic_plots.png",
        "matplotlib_examples.png",
        "classification_results.png",
        "neural_network_results.png",
        "pytorch_classification_results.png",
        "cnn_filters.png",
        "optimization_comparison.png",
        "comprehensive_model_dashboard.html",
        "model_metrics_radar.html",
        "enhanced_model_dashboard.html",
        "ethical_ai_bias_analysis.png",
        "explainable_ai_feature_importance.png"
    ]
    # Check some files exist (optional files may not be generated in CI)
    essential_files = [f for f in generated_files if not f.endswith('.html')]
    if check_files_exist(essential_files, "Generated output files"):
        tests_passed += 1
    
    # Test 12: Check enhanced utility modules
    total_tests += 1
    test_command = "python -c \"from utils.model_evaluation import ModelEvaluationDashboard; from utils.training_tracker import TrainingTracker; from utils.interpretability import ModelInterpreter; from utils.hyperparameter_tuning import HyperparameterTuner; print('All enhanced utils modules working')\""
    if run_command(test_command, "Enhanced utility modules import"):
        tests_passed += 1
    
    # Test 13: Check deployment modules
    total_tests += 1
    test_command = "python -c \"from deployment.model_server import ModelServer; from deployment.model_registry import ModelRegistry; from deployment.deployment_utils import DeploymentHelper; print('All deployment modules working')\""
    if run_command(test_command, "Deployment modules import"):
        tests_passed += 1
    
    # Test 14: Check web interface modules
    total_tests += 1
    test_command = "python -c \"from web_interface.dashboard_app import DashboardServer; print('Web interface modules working')\""
    if run_command(test_command, "Web interface modules import"):
        tests_passed += 1
    
    # Test 15: Jupyter notebook validation
    total_tests += 1
    if run_command("jupyter --version", "Jupyter installation"):
        tests_passed += 1
    
    # Test 16: Academic paper module
    total_tests += 1
    if run_command("python test_paper_module.py", "Academic paper module"):
        tests_passed += 1
    
    # Final report
    print(f"\n{'='*60}")
    print(f"🏁 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Success rate: {tests_passed/total_tests*100:.1f}%")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! The AI tutorial is ready to use.")
        print("\n📚 Next steps:")
        print("1. Start with tutorials/00_ai_fundamentals/README.md")
        print("2. Run 'jupyter lab' to explore the interactive notebooks")
        print("3. Generate the academic paper with 'cd paper && make pdf'")
        print("4. Check docs/setup/cross_platform_setup.md for detailed setup")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("🔧 Try running 'pip install -r requirements.txt' to fix missing packages")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)