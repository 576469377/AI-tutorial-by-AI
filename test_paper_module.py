#!/usr/bin/env python3
"""
Test script for the academic paper module.

This script tests the paper generation functionality including
figure generation, table creation, and LaTeX compilation.
"""

import os
import sys
import subprocess
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).parent
PAPER_DIR = PROJECT_ROOT / 'paper'
SCRIPTS_DIR = PAPER_DIR / 'scripts'

def test_figure_generation():
    """Test figure generation script."""
    print("Testing figure generation...")
    
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / 'generate_figures.py')],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print("✅ Figure generation: SUCCESS")
            return True
        else:
            print("❌ Figure generation: FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Figure generation: ERROR - {e}")
        return False

def test_table_generation():
    """Test table generation script."""
    print("Testing table generation...")
    
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / 'generate_tables.py')],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ Table generation: SUCCESS")
            return True
        else:
            print("❌ Table generation: FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Table generation: ERROR - {e}")
        return False

def test_metrics_analysis():
    """Test metrics analysis script."""
    print("Testing metrics analysis...")
    
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / 'analyze_metrics.py')],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ Metrics analysis: SUCCESS")
            return True
        else:
            print("❌ Metrics analysis: FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Metrics analysis: ERROR - {e}")
        return False

def test_latex_compilation():
    """Test LaTeX compilation."""
    print("Testing LaTeX compilation...")
    
    try:
        # Check if LaTeX is available
        result = subprocess.run(
            ['which', 'pdflatex'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("⚠️  LaTeX compilation: SKIPPED (pdflatex not found)")
            return True
        
        # Test compilation
        result = subprocess.run(
            ['pdflatex', 'simple_paper.tex'],
            cwd=PAPER_DIR,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ LaTeX compilation: SUCCESS")
            return True
        else:
            print("❌ LaTeX compilation: FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ LaTeX compilation: ERROR - {e}")
        return False

def test_file_structure():
    """Test paper module file structure."""
    print("Testing paper module file structure...")
    
    required_files = [
        PAPER_DIR / 'README.md',
        PAPER_DIR / 'main.tex',
        PAPER_DIR / 'simple_paper.tex',
        PAPER_DIR / 'bibliography.bib',
        PAPER_DIR / 'Makefile',
        SCRIPTS_DIR / 'generate_figures.py',
        SCRIPTS_DIR / 'generate_tables.py',
        SCRIPTS_DIR / 'analyze_metrics.py'
    ]
    
    required_dirs = [
        PAPER_DIR / 'sections',
        PAPER_DIR / 'figures',
        PAPER_DIR / 'tables',
        PAPER_DIR / 'scripts'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            missing_dirs.append(dir_path)
    
    if missing_files or missing_dirs:
        print("❌ File structure: FAILED")
        if missing_files:
            print(f"   Missing files: {[str(f) for f in missing_files]}")
        if missing_dirs:
            print(f"   Missing directories: {[str(d) for d in missing_dirs]}")
        return False
    else:
        print("✅ File structure: SUCCESS")
        return True

def test_generated_outputs():
    """Test that required outputs are generated."""
    print("Testing generated outputs...")
    
    # Check for generated figures
    figures_dir = PAPER_DIR / 'figures'
    expected_figures = [
        'learning_outcomes.png',
        'technical_quality_radar.png',
        'user_satisfaction.png',
        'framework_architecture.png',
        'improvement_timeline.png',
        'coverage_comparison.png'
    ]
    
    missing_figures = []
    for fig in expected_figures:
        if not (figures_dir / fig).exists():
            missing_figures.append(fig)
    
    # Check for generated tables
    tables_dir = PAPER_DIR / 'tables'
    expected_tables = [
        'technical_metrics.tex',
        'learning_outcomes.tex',
        'satisfaction.tex',
        'expert_review.tex'
    ]
    
    missing_tables = []
    for table in expected_tables:
        if not (tables_dir / table).exists():
            missing_tables.append(table)
    
    # Check for PDF
    pdf_exists = (PAPER_DIR / 'paper.pdf').exists() or (PAPER_DIR / 'simple_paper.pdf').exists()
    
    if missing_figures or missing_tables or not pdf_exists:
        print("❌ Generated outputs: INCOMPLETE")
        if missing_figures:
            print(f"   Missing figures: {missing_figures}")
        if missing_tables:
            print(f"   Missing tables: {missing_tables}")
        if not pdf_exists:
            print("   Missing PDF output")
        return False
    else:
        print("✅ Generated outputs: SUCCESS")
        return True

def main():
    """Run all paper module tests."""
    print("🤖 AI Tutorial by AI - Paper Module Test Suite")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_figure_generation,
        test_table_generation,
        test_metrics_analysis,
        test_generated_outputs,
        test_latex_compilation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ {test.__name__}: ERROR - {e}")
            results.append(False)
            print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100
    
    print("=" * 50)
    print("📊 PAPER MODULE TEST SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("✅ Paper module is working well!")
        return 0
    else:
        print("⚠️  Some paper module tests failed.")
        return 1

if __name__ == '__main__':
    sys.exit(main())