"""
Interactive Learning Dashboard for AI Tutorial

This module provides a web-based dashboard for accessing and running
AI tutorial examples interactively through a browser interface.
"""

import os
import json
import datetime
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import plotly
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np

# Import AI tutorial modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_evaluation import ModelEvaluationDashboard
from utils.training_tracker import TrainingTracker
from utils.interpretability import ModelInterpreter
from utils.hyperparameter_tuning import HyperparameterTuner


class DashboardServer:
    """
    Interactive web dashboard for AI Tutorial
    """
    
    def __init__(self, port: int = 8080, host: str = 'localhost', debug: bool = True):
        """
        Initialize dashboard server
        
        Args:
            port: Port to run server on
            host: Host to bind to
            debug: Enable debug mode
        """
        self.port = port
        self.host = host
        self.debug = debug
        
        # Initialize Flask app
        self.app = Flask(__name__,
                         template_folder='templates',
                         static_folder='static')
        self.app.config['DEBUG'] = debug
        self.app.secret_key = 'ai_tutorial_dashboard_key'
        
        # Tutorial state
        self.tutorial_state = {
            'current_module': None,
            'completed_modules': [],
            'user_progress': {},
            'examples_run': [],
            'last_results': {}
        }
        
        # Available tutorial modules
        self.tutorial_modules = self._discover_tutorial_modules()
        
        # Setup routes
        self._setup_routes()
    
    def _discover_tutorial_modules(self) -> Dict:
        """Discover available tutorial modules and examples"""
        base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        modules = {
            'examples': {
                'title': 'Interactive Examples',
                'description': 'Run and explore AI tutorial examples',
                'items': []
            },
            'tutorials': {
                'title': 'Tutorial Guides',
                'description': 'Step-by-step learning paths',
                'items': []
            },
            'tools': {
                'title': 'AI Development Tools',
                'description': 'Advanced tools for model development',
                'items': []
            }
        }
        
        # Discover examples
        examples_dir = base_dir / 'examples'
        if examples_dir.exists():
            for example_file in sorted(examples_dir.glob('*.py')):
                if not example_file.name.startswith('__'):
                    modules['examples']['items'].append({
                        'id': example_file.stem,
                        'name': example_file.stem.replace('_', ' ').title(),
                        'file': str(example_file),
                        'description': self._extract_description(example_file)
                    })
        
        # Discover tutorials
        tutorials_dir = base_dir / 'tutorials'
        if tutorials_dir.exists():
            for tutorial_dir in sorted(tutorials_dir.iterdir()):
                if tutorial_dir.is_dir() and not tutorial_dir.name.startswith('.'):
                    readme_file = tutorial_dir / 'README.md'
                    modules['tutorials']['items'].append({
                        'id': tutorial_dir.name,
                        'name': tutorial_dir.name.replace('_', ' ').title(),
                        'path': str(tutorial_dir),
                        'description': self._extract_readme_description(readme_file)
                    })
        
        # Add development tools
        modules['tools']['items'] = [
            {
                'id': 'model_evaluation',
                'name': 'Model Evaluation Dashboard',
                'description': 'Comprehensive model performance analysis'
            },
            {
                'id': 'training_tracker',
                'name': 'Training Progress Tracker',
                'description': 'Real-time training monitoring'
            },
            {
                'id': 'interpretability',
                'name': 'Model Interpretability',
                'description': 'Understand model decisions and feature importance'
            },
            {
                'id': 'hyperparameter_tuning',
                'name': 'Hyperparameter Optimization',
                'description': 'Automated parameter search and optimization'
            }
        ]
        
        return modules
    
    def _extract_description(self, file_path: Path) -> str:
        """Extract description from Python file docstring"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                # Simple docstring extraction
                if '"""' in content:
                    start = content.find('"""') + 3
                    end = content.find('"""', start)
                    if end > start:
                        docstring = content[start:end].strip()
                        return docstring.split('\n')[0][:100]
        except Exception:
            pass
        return "AI Tutorial example"
    
    def _extract_readme_description(self, readme_path: Path) -> str:
        """Extract description from README file"""
        try:
            if readme_path.exists():
                with open(readme_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            return line[:100]
        except Exception:
            pass
        return "Tutorial module"
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def home():
            """Home page with tutorial overview"""
            return render_template('dashboard_home.html',
                                 modules=self.tutorial_modules,
                                 progress=self.tutorial_state)
        
        @self.app.route('/examples')
        def examples():
            """Examples listing page"""
            return render_template('examples_list.html',
                                 examples=self.tutorial_modules['examples']['items'],
                                 completed=self.tutorial_state['examples_run'])
        
        @self.app.route('/run_example/<example_id>')
        def run_example(example_id):
            """Run a specific example"""
            # Find example
            example = None
            for item in self.tutorial_modules['examples']['items']:
                if item['id'] == example_id:
                    example = item
                    break
            
            if not example:
                return jsonify({'error': 'Example not found'}), 404
            
            try:
                # Run example in subprocess
                result = subprocess.run([
                    'python', example['file']
                ], capture_output=True, text=True, timeout=60)
                
                # Store result
                self.tutorial_state['last_results'][example_id] = {
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                # Mark as run
                if example_id not in self.tutorial_state['examples_run']:
                    self.tutorial_state['examples_run'].append(example_id)
                
                return render_template('example_result.html',
                                     example=example,
                                     result=self.tutorial_state['last_results'][example_id])
            
            except subprocess.TimeoutExpired:
                return jsonify({'error': 'Example timed out'}), 500
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/tutorials')
        def tutorials():
            """Tutorials listing page"""
            return render_template('tutorials_list.html',
                                 tutorials=self.tutorial_modules['tutorials']['items'])
        
        @self.app.route('/tools')
        def tools():
            """AI development tools page"""
            return render_template('tools_list.html',
                                 tools=self.tutorial_modules['tools']['items'])
        
        @self.app.route('/tools/<tool_id>')
        def run_tool(tool_id):
            """Run a specific development tool"""
            if tool_id == 'model_evaluation':
                return self._demo_model_evaluation()
            elif tool_id == 'training_tracker':
                return self._demo_training_tracker()
            elif tool_id == 'interpretability':
                return self._demo_interpretability()
            elif tool_id == 'hyperparameter_tuning':
                return self._demo_hyperparameter_tuning()
            else:
                return jsonify({'error': 'Tool not found'}), 404
        
        @self.app.route('/progress')
        def progress():
            """Show user progress"""
            total_examples = len(self.tutorial_modules['examples']['items'])
            completed_examples = len(self.tutorial_state['examples_run'])
            
            progress_data = {
                'examples_progress': {
                    'total': total_examples,
                    'completed': completed_examples,
                    'percentage': (completed_examples / max(total_examples, 1)) * 100
                },
                'recent_activity': self._get_recent_activity(),
                'achievements': self._calculate_achievements()
            }
            
            return render_template('progress.html', progress=progress_data)
        
        @self.app.route('/api/run_code', methods=['POST'])
        def run_code():
            """API endpoint to run custom Python code"""
            data = request.json
            code = data.get('code', '')
            
            if not code:
                return jsonify({'error': 'No code provided'}), 400
            
            try:
                # Create temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    temp_file = f.name
                
                # Run code
                result = subprocess.run([
                    'python', temp_file
                ], capture_output=True, text=True, timeout=30)
                
                # Clean up
                os.unlink(temp_file)
                
                return jsonify({
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                })
            
            except subprocess.TimeoutExpired:
                return jsonify({'error': 'Code execution timed out'}), 500
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def _demo_model_evaluation(self):
        """Demo model evaluation dashboard"""
        try:
            # Create demo evaluation
            dashboard = ModelEvaluationDashboard()
            
            # Add sample results
            sample_results = [
                {
                    'model_name': 'Random Forest',
                    'model_type': 'ensemble',
                    'dataset': 'Demo Dataset',
                    'metrics': {'accuracy': 0.89, 'f1_score': 0.87, 'precision': 0.88, 'recall': 0.86},
                    'training_time': 45.2,
                    'model_size': 150000
                },
                {
                    'model_name': 'Neural Network',
                    'model_type': 'neural_net',
                    'dataset': 'Demo Dataset',
                    'metrics': {'accuracy': 0.92, 'f1_score': 0.91, 'precision': 0.93, 'recall': 0.89},
                    'training_time': 120.5,
                    'model_size': 250000
                }
            ]
            
            for result in sample_results:
                dashboard.add_result(**result)
            
            # Generate dashboard
            dashboard_html = dashboard.create_performance_dashboard(return_html=True)
            
            return render_template('tool_result.html',
                                 tool_name='Model Evaluation Dashboard',
                                 dashboard_html=dashboard_html)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def _demo_training_tracker(self):
        """Demo training tracker"""
        try:
            # Create sample training data
            epochs = list(range(1, 21))
            loss = [1.0 - 0.04 * i + 0.01 * np.random.random() for i in epochs]
            accuracy = [0.5 + 0.02 * i + 0.005 * np.random.random() for i in epochs]
            
            # Create interactive plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=epochs,
                y=loss,
                mode='lines+markers',
                name='Training Loss',
                line=dict(color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=epochs,
                y=accuracy,
                mode='lines+markers',
                name='Accuracy',
                yaxis='y2',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title='Training Progress Demo',
                xaxis_title='Epoch',
                yaxis=dict(title='Loss', side='left'),
                yaxis2=dict(title='Accuracy', side='right', overlaying='y'),
                height=500
            )
            
            plot_html = plotly.offline.plot(fig, output_type='div', include_plotlyjs=True)
            
            return render_template('tool_result.html',
                                 tool_name='Training Progress Tracker',
                                 dashboard_html=plot_html)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def _demo_interpretability(self):
        """Demo model interpretability"""
        try:
            # Create sample feature importance data
            features = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5']
            importance = [0.25, 0.20, 0.18, 0.15, 0.12]
            
            fig = px.bar(
                x=importance,
                y=features,
                orientation='h',
                title='Feature Importance Demo',
                labels={'x': 'Importance Score', 'y': 'Features'}
            )
            
            fig.update_layout(height=400)
            
            plot_html = plotly.offline.plot(fig, output_type='div', include_plotlyjs=True)
            
            return render_template('tool_result.html',
                                 tool_name='Model Interpretability',
                                 dashboard_html=plot_html)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def _demo_hyperparameter_tuning(self):
        """Demo hyperparameter tuning"""
        try:
            # Create sample tuning results
            param_values = list(range(1, 21))
            scores = [0.7 + 0.01 * i + 0.02 * np.random.random() for i in param_values]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=param_values,
                y=scores,
                mode='lines+markers',
                name='Validation Score',
                line=dict(color='green')
            ))
            
            # Highlight best score
            best_idx = np.argmax(scores)
            fig.add_trace(go.Scatter(
                x=[param_values[best_idx]],
                y=[scores[best_idx]],
                mode='markers',
                marker=dict(size=15, color='red'),
                name='Best Score'
            ))
            
            fig.update_layout(
                title='Hyperparameter Tuning Demo',
                xaxis_title='Parameter Value',
                yaxis_title='Validation Score',
                height=400
            )
            
            plot_html = plotly.offline.plot(fig, output_type='div', include_plotlyjs=True)
            
            return render_template('tool_result.html',
                                 tool_name='Hyperparameter Tuning',
                                 dashboard_html=plot_html)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def _get_recent_activity(self) -> List[Dict]:
        """Get recent user activity"""
        activity = []
        
        for example_id in self.tutorial_state['examples_run'][-5:]:  # Last 5
            example = None
            for item in self.tutorial_modules['examples']['items']:
                if item['id'] == example_id:
                    example = item
                    break
            
            if example:
                activity.append({
                    'type': 'example',
                    'name': example['name'],
                    'timestamp': self.tutorial_state['last_results'].get(example_id, {}).get('timestamp', 'Unknown')
                })
        
        return activity
    
    def _calculate_achievements(self) -> List[Dict]:
        """Calculate user achievements"""
        achievements = []
        
        completed_count = len(self.tutorial_state['examples_run'])
        
        if completed_count >= 1:
            achievements.append({
                'name': 'First Steps',
                'description': 'Ran your first example',
                'earned': True
            })
        
        if completed_count >= 5:
            achievements.append({
                'name': 'Getting Started',
                'description': 'Completed 5 examples',
                'earned': True
            })
        
        if completed_count >= 10:
            achievements.append({
                'name': 'AI Explorer',
                'description': 'Completed 10 examples',
                'earned': True
            })
        
        # Future achievements
        if completed_count < 5:
            achievements.append({
                'name': 'Getting Started',
                'description': 'Complete 5 examples',
                'earned': False
            })
        
        if completed_count < 10:
            achievements.append({
                'name': 'AI Explorer',
                'description': 'Complete 10 examples',
                'earned': False
            })
        
        return achievements
    
    def run(self):
        """Run the dashboard server"""
        print(f"🌐 AI Tutorial Dashboard starting at http://{self.host}:{self.port}")
        print(f"📚 Access interactive tutorials and examples")
        print(f"🛑 Press Ctrl+C to stop")
        
        try:
            self.app.run(host=self.host, port=self.port, debug=self.debug, threaded=True)
        except KeyboardInterrupt:
            print(f"\n🛑 Dashboard stopped by user")


def create_dashboard_app(port: int = 8080, host: str = 'localhost', debug: bool = True) -> DashboardServer:
    """
    Create and configure dashboard app
    
    Args:
        port: Port to run on
        host: Host to bind to
        debug: Enable debug mode
    
    Returns:
        Configured DashboardServer instance
    """
    return DashboardServer(port=port, host=host, debug=debug)


if __name__ == "__main__":
    # Run dashboard
    dashboard = create_dashboard_app()
    dashboard.run()