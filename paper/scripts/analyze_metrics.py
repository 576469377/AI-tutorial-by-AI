#!/usr/bin/env python3
"""
Analyze project metrics for the AI Tutorial by AI academic paper.

This script analyzes the project structure, code quality, and educational
content to generate metrics for the academic paper.
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import subprocess

# Set up paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

def analyze_file_structure():
    """Analyze project file structure and organization."""
    print("Analyzing project file structure...")
    
    analysis = {
        'total_files': 0,
        'python_files': 0,
        'notebook_files': 0,
        'markdown_files': 0,
        'tutorial_modules': 0,
        'example_scripts': 0,
        'test_files': 0,
        'documentation_files': 0
    }
    
    # Count different file types
    for file_path in PROJECT_ROOT.rglob('*'):
        if file_path.is_file():
            analysis['total_files'] += 1
            
            if file_path.suffix == '.py':
                analysis['python_files'] += 1
                if 'test' in file_path.name.lower():
                    analysis['test_files'] += 1
                elif 'examples' in str(file_path):
                    analysis['example_scripts'] += 1
            elif file_path.suffix == '.ipynb':
                analysis['notebook_files'] += 1
            elif file_path.suffix in ['.md', '.rst']:
                analysis['markdown_files'] += 1
                if 'docs' in str(file_path) or 'README' in file_path.name:
                    analysis['documentation_files'] += 1
    
    # Count tutorial modules
    tutorials_dir = PROJECT_ROOT / 'tutorials'
    if tutorials_dir.exists():
        analysis['tutorial_modules'] = len([d for d in tutorials_dir.iterdir() if d.is_dir()])
    
    return analysis

def analyze_code_metrics():
    """Analyze code quality metrics."""
    print("Analyzing code quality metrics...")
    
    metrics = {
        'lines_of_code': 0,
        'functions_count': 0,
        'classes_count': 0,
        'complexity_score': 0,
        'documentation_ratio': 0
    }
    
    python_files = list(PROJECT_ROOT.rglob('*.py'))
    total_lines = 0
    comment_lines = 0
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                total_lines += len(lines)
                
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                        comment_lines += 1
                    elif 'def ' in line:
                        metrics['functions_count'] += 1
                    elif 'class ' in line:
                        metrics['classes_count'] += 1
        except Exception:
            continue
    
    metrics['lines_of_code'] = total_lines
    metrics['documentation_ratio'] = comment_lines / total_lines if total_lines > 0 else 0
    
    return metrics

def analyze_dependencies():
    """Analyze project dependencies."""
    print("Analyzing project dependencies...")
    
    requirements_file = PROJECT_ROOT / 'requirements.txt'
    dependencies = {
        'total_dependencies': 0,
        'core_ml_libraries': 0,
        'visualization_libraries': 0,
        'deep_learning_libraries': 0,
        'development_tools': 0
    }
    
    if requirements_file.exists():
        with open(requirements_file, 'r') as f:
            lines = f.readlines()
            
        core_ml = ['numpy', 'pandas', 'scikit-learn', 'scipy']
        visualization = ['matplotlib', 'seaborn', 'plotly']
        deep_learning = ['torch', 'tensorflow', 'transformers']
        dev_tools = ['jupyter', 'pytest', 'black', 'flake8']
        
        for line in lines:
            line = line.strip().lower()
            if line and not line.startswith('#'):
                dependencies['total_dependencies'] += 1
                
                if any(lib in line for lib in core_ml):
                    dependencies['core_ml_libraries'] += 1
                elif any(lib in line for lib in visualization):
                    dependencies['visualization_libraries'] += 1
                elif any(lib in line for lib in deep_learning):
                    dependencies['deep_learning_libraries'] += 1
                elif any(lib in line for lib in dev_tools):
                    dependencies['development_tools'] += 1
    
    return dependencies

def analyze_educational_content():
    """Analyze educational content structure."""
    print("Analyzing educational content...")
    
    content = {
        'learning_tracks': 0,
        'total_examples': 0,
        'interactive_notebooks': 0,
        'documentation_pages': 0,
        'covered_topics': []
    }
    
    # Count learning tracks
    tutorials_dir = PROJECT_ROOT / 'tutorials'
    if tutorials_dir.exists():
        content['learning_tracks'] = len([d for d in tutorials_dir.iterdir() if d.is_dir()])
    
    # Count examples
    examples_dir = PROJECT_ROOT / 'examples'
    if examples_dir.exists():
        content['total_examples'] = len(list(examples_dir.glob('*.py')))
    
    # Count notebooks
    notebooks_dir = PROJECT_ROOT / 'notebooks'
    if notebooks_dir.exists():
        content['interactive_notebooks'] = len(list(notebooks_dir.glob('*.ipynb')))
    
    # Count documentation
    docs_dir = PROJECT_ROOT / 'docs'
    if docs_dir.exists():
        content['documentation_pages'] = len(list(docs_dir.rglob('*.md')))
    
    # Identify covered topics
    topic_keywords = [
        'fundamentals', 'basics', 'visualization', 'machine_learning',
        'neural_networks', 'pytorch', 'language_models', 'ethical_ai',
        'model_evaluation', 'hyperparameter_tuning', 'interpretability'
    ]
    
    for keyword in topic_keywords:
        if any(keyword in str(path).lower() for path in PROJECT_ROOT.rglob('*')):
            content['covered_topics'].append(keyword.replace('_', ' ').title())
    
    return content

def run_test_coverage():
    """Run test coverage analysis if possible."""
    print("Analyzing test coverage...")
    
    coverage = {
        'test_coverage': 'Unknown',
        'tests_passing': 'Unknown',
        'total_tests': 0
    }
    
    try:
        # Count test files
        test_files = list(PROJECT_ROOT.glob('test_*.py'))
        coverage['total_tests'] = len(test_files)
        
        # Try to run a simple test to check if they pass
        if test_files:
            result = subprocess.run(
                ['python', '-m', 'pytest', '--version'],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT
            )
            if result.returncode == 0:
                coverage['test_framework'] = 'pytest available'
            else:
                coverage['test_framework'] = 'pytest not available'
        
    except Exception as e:
        coverage['error'] = str(e)
    
    return coverage

def generate_metrics_report():
    """Generate comprehensive metrics report."""
    print("Generating comprehensive metrics report...")
    
    report = {
        'timestamp': str(Path.cwd()),
        'project_name': 'AI Tutorial by AI',
        'analysis_date': '2024',
        'file_structure': analyze_file_structure(),
        'code_metrics': analyze_code_metrics(),
        'dependencies': analyze_dependencies(),
        'educational_content': analyze_educational_content(),
        'test_coverage': run_test_coverage()
    }
    
    return report

def print_summary(report):
    """Print a summary of the analysis."""
    print("\n" + "="*60)
    print("AI TUTORIAL BY AI - PROJECT ANALYSIS SUMMARY")
    print("="*60)
    
    fs = report['file_structure']
    print(f"\n📁 FILE STRUCTURE:")
    print(f"   Total files: {fs['total_files']}")
    print(f"   Python files: {fs['python_files']}")
    print(f"   Jupyter notebooks: {fs['notebook_files']}")
    print(f"   Tutorial modules: {fs['tutorial_modules']}")
    print(f"   Example scripts: {fs['example_scripts']}")
    print(f"   Test files: {fs['test_files']}")
    
    cm = report['code_metrics']
    print(f"\n💻 CODE METRICS:")
    print(f"   Lines of code: {cm['lines_of_code']:,}")
    print(f"   Functions: {cm['functions_count']}")
    print(f"   Classes: {cm['classes_count']}")
    print(f"   Documentation ratio: {cm['documentation_ratio']:.1%}")
    
    deps = report['dependencies']
    print(f"\n📦 DEPENDENCIES:")
    print(f"   Total dependencies: {deps['total_dependencies']}")
    print(f"   Core ML libraries: {deps['core_ml_libraries']}")
    print(f"   Visualization libraries: {deps['visualization_libraries']}")
    print(f"   Deep learning libraries: {deps['deep_learning_libraries']}")
    
    ec = report['educational_content']
    print(f"\n🎓 EDUCATIONAL CONTENT:")
    print(f"   Learning tracks: {ec['learning_tracks']}")
    print(f"   Example scripts: {ec['total_examples']}")
    print(f"   Interactive notebooks: {ec['interactive_notebooks']}")
    print(f"   Documentation pages: {ec['documentation_pages']}")
    print(f"   Covered topics: {', '.join(ec['covered_topics'][:5])}...")
    
    tc = report['test_coverage']
    print(f"\n🧪 TESTING:")
    print(f"   Test files: {tc['total_tests']}")
    print(f"   Test framework: {tc.get('test_framework', 'Unknown')}")

def save_report(report):
    """Save the analysis report to JSON file."""
    output_file = SCRIPT_DIR.parent / 'tables' / 'project_metrics.json'
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved to: {output_file}")

def main():
    """Main analysis function."""
    print("AI Tutorial by AI - Project Metrics Analysis")
    print("=" * 50)
    
    try:
        report = generate_metrics_report()
        print_summary(report)
        save_report(report)
        
        print(f"\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()