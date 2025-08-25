"""
Model Server for AI Tutorial - REST API for Model Serving

This module provides a production-ready REST API server for serving
trained machine learning models. It supports multiple model types
and provides standardized endpoints for inference.
"""

import os
import json
import pickle
import datetime
import logging
from typing import Dict, Any, List, Optional, Union
import threading
import time

try:
    from flask import Flask, request, jsonify, render_template_string
    from werkzeug.serving import make_server
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

import numpy as np
import pandas as pd

# PyTorch imports (optional)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Scikit-learn imports
try:
    import sklearn
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ModelServer:
    """
    Production-ready model serving server with REST API
    """
    
    def __init__(self, port: int = 5000, host: str = 'localhost', debug: bool = False):
        """
        Initialize model server
        
        Args:
            port: Port to run server on
            host: Host to bind to
            debug: Enable debug mode
        """
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for model serving. Install with: pip install flask")
        
        self.port = port
        self.host = host
        self.debug = debug
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['DEBUG'] = debug
        
        # Model registry
        self.models = {}
        self.model_metadata = {}
        
        # Server statistics
        self.stats = {
            'requests_served': 0,
            'total_prediction_time': 0.0,
            'start_time': datetime.datetime.now(),
            'models_loaded': 0
        }
        
        # Setup routes
        self._setup_routes()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for the server"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ModelServer')
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def home():
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>AI Tutorial Model Server</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .header { color: #2c3e50; }
                    .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }
                    .stats { background: #e8f5e9; padding: 15px; margin: 10px 0; }
                    .model-list { background: #fff3cd; padding: 15px; margin: 10px 0; }
                    code { background: #f1f2f6; padding: 2px 4px; }
                </style>
            </head>
            <body>
                <h1 class="header">🤖 AI Tutorial Model Server</h1>
                <p>Welcome to the AI Tutorial Model Serving API!</p>
                
                <div class="stats">
                    <h3>📊 Server Statistics</h3>
                    <p><strong>Uptime:</strong> {{ uptime }}</p>
                    <p><strong>Requests Served:</strong> {{ stats.requests_served }}</p>
                    <p><strong>Models Loaded:</strong> {{ stats.models_loaded }}</p>
                    <p><strong>Average Response Time:</strong> {{ avg_response_time }}ms</p>
                </div>
                
                <div class="model-list">
                    <h3>🎯 Loaded Models</h3>
                    {% if models %}
                        <ul>
                        {% for model_id, metadata in models.items() %}
                            <li><strong>{{ model_id }}</strong>: {{ metadata.get('description', 'No description') }}</li>
                        {% endfor %}
                        </ul>
                    {% else %}
                        <p>No models loaded. Use <code>/load_model</code> to load models.</p>
                    {% endif %}
                </div>
                
                <h3>🚀 API Endpoints</h3>
                
                <div class="endpoint">
                    <h4>GET /health</h4>
                    <p>Check server health status</p>
                </div>
                
                <div class="endpoint">
                    <h4>POST /load_model</h4>
                    <p>Load a model for serving</p>
                    <p><strong>Body:</strong> <code>{"model_path": "/path/to/model", "model_id": "my_model"}</code></p>
                </div>
                
                <div class="endpoint">
                    <h4>POST /predict/{model_id}</h4>
                    <p>Make predictions using a loaded model</p>
                    <p><strong>Body:</strong> <code>{"data": [[1, 2, 3], [4, 5, 6]]}</code></p>
                </div>
                
                <div class="endpoint">
                    <h4>GET /models</h4>
                    <p>List all loaded models and their metadata</p>
                </div>
                
                <div class="endpoint">
                    <h4>DELETE /models/{model_id}</h4>
                    <p>Unload a specific model</p>
                </div>
                
                <div class="endpoint">
                    <h4>GET /stats</h4>
                    <p>Get detailed server statistics</p>
                </div>
                
            </body>
            </html>
            """, 
            models=self.model_metadata,
            stats=self.stats,
            uptime=str(datetime.datetime.now() - self.stats['start_time']).split('.')[0],
            avg_response_time=round(
                (self.stats['total_prediction_time'] / max(self.stats['requests_served'], 1)) * 1000, 2
            )
            )
        
        @self.app.route('/health')
        def health():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.datetime.now().isoformat(),
                'models_loaded': len(self.models),
                'uptime_seconds': (datetime.datetime.now() - self.stats['start_time']).total_seconds()
            })
        
        @self.app.route('/load_model', methods=['POST'])
        def load_model():
            try:
                data = request.json
                model_path = data.get('model_path')
                model_id = data.get('model_id')
                model_type = data.get('model_type', 'auto')
                
                if not model_path or not model_id:
                    return jsonify({'error': 'model_path and model_id are required'}), 400
                
                # Load the model
                success = self.load_model(model_path, model_id, model_type)
                
                if success:
                    return jsonify({
                        'message': f'Model {model_id} loaded successfully',
                        'model_id': model_id,
                        'metadata': self.model_metadata.get(model_id, {})
                    })
                else:
                    return jsonify({'error': 'Failed to load model'}), 500
                    
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/predict/<model_id>', methods=['POST'])
        def predict(model_id):
            start_time = time.time()
            
            try:
                if model_id not in self.models:
                    return jsonify({'error': f'Model {model_id} not found'}), 404
                
                data = request.json
                input_data = data.get('data')
                
                if input_data is None:
                    return jsonify({'error': 'data field is required'}), 400
                
                # Make prediction
                predictions = self._make_prediction(model_id, input_data)
                
                # Update statistics
                prediction_time = time.time() - start_time
                self.stats['requests_served'] += 1
                self.stats['total_prediction_time'] += prediction_time
                
                return jsonify({
                    'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                    'model_id': model_id,
                    'prediction_time_ms': round(prediction_time * 1000, 2),
                    'timestamp': datetime.datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Error making prediction: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/models')
        def list_models():
            return jsonify({
                'models': {
                    model_id: {
                        **metadata,
                        'loaded_at': metadata.get('loaded_at'),
                        'predictions_made': metadata.get('predictions_made', 0)
                    }
                    for model_id, metadata in self.model_metadata.items()
                },
                'total_models': len(self.models)
            })
        
        @self.app.route('/models/<model_id>', methods=['DELETE'])
        def unload_model(model_id):
            if model_id not in self.models:
                return jsonify({'error': f'Model {model_id} not found'}), 404
            
            # Remove model from registry
            del self.models[model_id]
            del self.model_metadata[model_id]
            self.stats['models_loaded'] -= 1
            
            return jsonify({'message': f'Model {model_id} unloaded successfully'})
        
        @self.app.route('/stats')
        def get_stats():
            uptime = datetime.datetime.now() - self.stats['start_time']
            return jsonify({
                'uptime_seconds': uptime.total_seconds(),
                'uptime_human': str(uptime).split('.')[0],
                'requests_served': self.stats['requests_served'],
                'models_loaded': len(self.models),
                'average_prediction_time_ms': round(
                    (self.stats['total_prediction_time'] / max(self.stats['requests_served'], 1)) * 1000, 2
                ),
                'total_prediction_time': round(self.stats['total_prediction_time'], 3),
                'requests_per_second': round(
                    self.stats['requests_served'] / max(uptime.total_seconds(), 1), 2
                )
            })
    
    def load_model(self, model_path: str, model_id: str, model_type: str = 'auto') -> bool:
        """
        Load a model for serving
        
        Args:
            model_path: Path to the model file
            model_id: Unique identifier for the model
            model_type: Type of model ('sklearn', 'pytorch', 'auto')
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return False
            
            # Determine model type automatically if needed
            if model_type == 'auto':
                if model_path.endswith('.pkl') or model_path.endswith('.pickle'):
                    model_type = 'sklearn'
                elif model_path.endswith('.pth') or model_path.endswith('.pt'):
                    model_type = 'pytorch'
                else:
                    model_type = 'pickle'  # Default fallback
            
            # Load model based on type
            if model_type in ['sklearn', 'pickle']:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            elif model_type == 'pytorch':
                if not TORCH_AVAILABLE:
                    raise ImportError("PyTorch not available")
                model = torch.load(model_path, map_location='cpu')
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Store model and metadata
            self.models[model_id] = model
            self.model_metadata[model_id] = {
                'model_type': model_type,
                'model_path': model_path,
                'loaded_at': datetime.datetime.now().isoformat(),
                'predictions_made': 0,
                'description': f"{model_type.title()} model loaded from {os.path.basename(model_path)}"
            }
            
            self.stats['models_loaded'] += 1
            self.logger.info(f"Successfully loaded model {model_id} from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            return False
    
    def _make_prediction(self, model_id: str, input_data: List) -> Union[List, np.ndarray]:
        """
        Make prediction using the specified model
        
        Args:
            model_id: ID of the model to use
            input_data: Input data for prediction
        
        Returns:
            Model predictions
        """
        model = self.models[model_id]
        metadata = self.model_metadata[model_id]
        model_type = metadata['model_type']
        
        # Convert input data to appropriate format
        X = np.array(input_data)
        
        if model_type in ['sklearn', 'pickle']:
            # Scikit-learn or pickle model
            predictions = model.predict(X)
        elif model_type == 'pytorch':
            # PyTorch model
            if hasattr(model, 'eval'):
                model.eval()
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = model(X_tensor)
                
                # Convert to numpy
                if hasattr(predictions, 'cpu'):
                    predictions = predictions.cpu().numpy()
                else:
                    predictions = predictions.detach().numpy()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Update prediction counter
        metadata['predictions_made'] += 1
        
        return predictions
    
    def run(self, threaded: bool = True):
        """
        Run the model server
        
        Args:
            threaded: Whether to run in threaded mode
        """
        self.logger.info(f"Starting Model Server on {self.host}:{self.port}")
        self.logger.info(f"Models loaded: {len(self.models)}")
        
        try:
            self.app.run(
                host=self.host,
                port=self.port,
                debug=self.debug,
                threaded=threaded
            )
        except KeyboardInterrupt:
            self.logger.info("Server stopped by user")
        except Exception as e:
            self.logger.error(f"Server error: {e}")


def create_model_api(models_dir: str = 'models', port: int = 5000, host: str = 'localhost') -> ModelServer:
    """
    Create a model API server with automatic model loading
    
    Args:
        models_dir: Directory containing model files
        port: Port to run server on
        host: Host to bind to
    
    Returns:
        Configured ModelServer instance
    """
    server = ModelServer(port=port, host=host)
    
    # Auto-load models from directory if it exists
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.endswith(('.pkl', '.pickle', '.pth', '.pt')):
                model_path = os.path.join(models_dir, filename)
                model_id = os.path.splitext(filename)[0]
                server.load_model(model_path, model_id)
    
    return server


if __name__ == "__main__":
    # Example usage
    print("🤖 AI Tutorial Model Server")
    print("=" * 50)
    
    # Create server
    server = create_model_api()
    
    # Load some example models if they exist
    example_models = [
        ('simple_model.pth', 'simple_neural_net'),
        ('complete_model.pth', 'complete_neural_net')
    ]
    
    for model_file, model_id in example_models:
        if os.path.exists(model_file):
            server.load_model(model_file, model_id)
    
    print(f"\n🚀 Server starting at http://localhost:5000")
    print("📖 Visit the URL above for API documentation")
    print("🛑 Press Ctrl+C to stop the server")
    
    # Run server
    server.run()