"""
Model Deployment and Production Examples

This example demonstrates how to deploy trained AI models in production
using the AI Tutorial deployment framework. It covers:

1. Model registry for version management
2. REST API server for model serving
3. Containerization with Docker
4. Production deployment configurations
"""

import os
import sys
import time
import threading
import requests
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import deployment modules
from deployment.model_server import ModelServer, create_model_api
from deployment.model_registry import ModelRegistry
from deployment.deployment_utils import DeploymentHelper

# Import other utilities
import numpy as np
import pandas as pd

# Optional imports for demo
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    import pickle
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def create_demo_models():
    """Create some demo models for deployment examples"""
    print("🏗️ Creating Demo Models for Deployment")
    print("=" * 50)
    
    models_dir = Path('demo_models')
    models_dir.mkdir(exist_ok=True)
    
    # Create a simple PyTorch model
    if TORCH_AVAILABLE:
        print("📦 Creating PyTorch demo model...")
        
        # Create a simple model using built-in modules to avoid pickle issues
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            nn.Softmax(dim=1)
        )
        
        # Save model state dict instead of the model object
        torch.save(model.state_dict(), models_dir / 'demo_pytorch_model.pth')
        print("  ✅ Saved demo_pytorch_model.pth")
    
    # Create a scikit-learn model
    if SKLEARN_AVAILABLE:
        print("📦 Creating Scikit-learn demo model...")
        
        # Generate demo data
        X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save model
        with open(models_dir / 'demo_sklearn_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("  ✅ Saved demo_sklearn_model.pkl")
        
        # Save test data for later use
        test_data = X[:5].tolist()  # First 5 samples for testing
        import json
        with open(models_dir / 'test_data.json', 'w') as f:
            json.dump({'test_samples': test_data}, f)
        print("  ✅ Saved test_data.json")
    
    print(f"📁 Demo models created in: {models_dir}")
    return models_dir


def demonstrate_model_registry():
    """Demonstrate model registry functionality"""
    print("\n🗂️ Model Registry Demonstration")
    print("=" * 50)
    
    # Create registry
    registry = ModelRegistry('demo_model_registry')
    
    # Register demo models
    models_dir = Path('demo_models')
    
    if (models_dir / 'demo_pytorch_model.pth').exists():
        registry.register_model(
            model_path=str(models_dir / 'demo_pytorch_model.pth'),
            model_name='demo_pytorch',
            version='v1.0',
            description='Demo PyTorch neural network for classification',
            tags=['pytorch', 'demo', 'classification'],
            metadata={'accuracy': 0.87, 'framework': 'pytorch'}
        )
        print("  ✅ Registered PyTorch model")
    
    if (models_dir / 'demo_sklearn_model.pkl').exists():
        registry.register_model(
            model_path=str(models_dir / 'demo_sklearn_model.pkl'),
            model_name='demo_sklearn',
            version='v1.0',
            description='Demo Random Forest classifier',
            tags=['sklearn', 'demo', 'random_forest'],
            metadata={'accuracy': 0.92, 'n_estimators': 100}
        )
        print("  ✅ Registered Scikit-learn model")
    
    # Show registry statistics
    stats = registry.get_registry_stats()
    print(f"\n📊 Registry Statistics:")
    print(f"  Total models: {stats['total_models']}")
    print(f"  Total versions: {stats['total_versions']}")
    print(f"  Total size: {stats['total_size_mb']} MB")
    
    # List models
    print(f"\n📋 Registered Models:")
    for model_name in registry.list_models():
        versions = registry.list_versions(model_name)
        model_info = registry.get_model(model_name)
        print(f"  📁 {model_name}:{versions[0]} - {model_info['description']}")
    
    return registry


def demonstrate_model_server():
    """Demonstrate model server functionality"""
    print("\n🖥️ Model Server Demonstration")
    print("=" * 50)
    
    # Create server
    server = ModelServer(port=5001, host='localhost')  # Use different port to avoid conflicts
    
    # Load demo models
    models_dir = Path('demo_models')
    
    if (models_dir / 'demo_pytorch_model.pth').exists():
        server.load_model(str(models_dir / 'demo_pytorch_model.pth'), 'pytorch_demo', 'pytorch')
        print("  ✅ Loaded PyTorch model")
    
    if (models_dir / 'demo_sklearn_model.pkl').exists():
        server.load_model(str(models_dir / 'demo_sklearn_model.pkl'), 'sklearn_demo', 'sklearn')
        print("  ✅ Loaded Scikit-learn model")
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=server.run, kwargs={'threaded': True})
    server_thread.daemon = True
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    print(f"🚀 Server started at http://localhost:5001")
    print("📖 Visit http://localhost:5001 for API documentation")
    
    # Test API endpoints
    try:
        # Test health endpoint
        response = requests.get('http://localhost:5001/health', timeout=5)
        if response.status_code == 200:
            print("  ✅ Health check passed")
        
        # Test models list
        response = requests.get('http://localhost:5001/models', timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            print(f"  📋 {models_data['total_models']} models loaded")
        
        # Test prediction (if we have test data)
        test_data_file = models_dir / 'test_data.json'
        if test_data_file.exists() and 'sklearn_demo' in [m for m in server.models.keys()]:
            import json
            with open(test_data_file, 'r') as f:
                test_data = json.load(f)
            
            # Make prediction request
            prediction_data = {'data': test_data['test_samples'][:2]}  # Use first 2 samples
            response = requests.post(
                'http://localhost:5001/predict/sklearn_demo',
                json=prediction_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  🎯 Prediction successful: {len(result['predictions'])} predictions made")
                print(f"     Prediction time: {result['prediction_time_ms']}ms")
            else:
                print(f"  ⚠️ Prediction failed: {response.status_code}")
    
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️ Could not test API: {e}")
    
    return server


def demonstrate_deployment_automation():
    """Demonstrate deployment automation"""
    print("\n🚀 Deployment Automation Demonstration")
    print("=" * 50)
    
    # Create deployment helper
    helper = DeploymentHelper()
    
    # Generate deployment package
    package_path = helper.generate_deployment_package(
        output_dir='demo_deployment_package',
        deployment_type='docker'
    )
    
    print(f"📦 Deployment package created: {package_path}")
    
    # List generated files
    package_dir = Path(package_path)
    print(f"📁 Generated deployment files:")
    for file_path in sorted(package_dir.iterdir()):
        if file_path.is_file():
            print(f"  📄 {file_path.name}")
    
    # Show sample Dockerfile content
    dockerfile_path = package_dir / 'Dockerfile'
    if dockerfile_path.exists():
        print(f"\n🐳 Sample Dockerfile (first 10 lines):")
        with open(dockerfile_path, 'r') as f:
            lines = f.readlines()[:10]
            for i, line in enumerate(lines, 1):
                print(f"  {i:2d}: {line.rstrip()}")
    
    # Show deployment instructions
    print(f"\n📋 Deployment Instructions:")
    print(f"  1. Navigate to: cd {package_path}")
    print(f"  2. Make script executable: chmod +x deploy.sh")
    print(f"  3. Run deployment: ./deploy.sh")
    print(f"  4. Or use Docker Compose: docker-compose up -d")
    
    return package_path


def create_production_example():
    """Create a comprehensive production deployment example"""
    print("\n🏭 Production Deployment Example")
    print("=" * 50)
    
    # Create models directory structure
    production_dir = Path('production_example')
    production_dir.mkdir(exist_ok=True)
    
    (production_dir / 'models').mkdir(exist_ok=True)
    (production_dir / 'config').mkdir(exist_ok=True)
    (production_dir / 'logs').mkdir(exist_ok=True)
    
    # Create production configuration
    production_config = {
        'server': {
            'host': '0.0.0.0',
            'port': 8000,
            'debug': False,
            'threaded': True
        },
        'models': {
            'auto_load': True,
            'models_directory': './models',
            'supported_formats': ['.pkl', '.pth', '.pt']
        },
        'logging': {
            'level': 'INFO',
            'file': './logs/api.log',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'security': {
            'rate_limiting': True,
            'max_requests_per_minute': 60,
            'api_key_required': False
        },
        'monitoring': {
            'enable_metrics': True,
            'health_check_interval': 30
        }
    }
    
    import json
    config_file = production_dir / 'config' / 'production.json'
    with open(config_file, 'w') as f:
        json.dump(production_config, f, indent=2)
    
    # Create production deployment helper
    helper = DeploymentHelper(str(production_dir))
    
    # Generate production deployment files
    deployment_files = {
        'Dockerfile': helper.generate_dockerfile(
            base_image='python:3.9-slim',
            port=8000,
            custom_requirements='requirements.txt'
        ),
        'docker-compose.yml': helper.create_docker_compose(
            service_name='ai-tutorial-prod',
            port=8000,
            volumes=['./config:/app/config', './logs:/app/logs', './models:/app/models']
        ),
        'nginx.conf': helper.create_nginx_config(
            server_name='api.ai-tutorial.com',
            api_port=8000
        ),
        'deploy.sh': helper.create_deployment_script('docker-compose')
    }
    
    # Write deployment files
    for filename, content in deployment_files.items():
        file_path = production_dir / filename
        with open(file_path, 'w') as f:
            f.write(content)
        
        if filename == 'deploy.sh':
            file_path.chmod(0o755)
    
    # Create production README
    readme_content = """# AI Tutorial Production Deployment

This directory contains a complete production deployment setup for the AI Tutorial model API.

## Quick Start

1. **Prepare your models**: Place model files (.pkl, .pth) in the `models/` directory
2. **Configure**: Edit `config/production.json` for your environment
3. **Deploy**: Run `./deploy.sh` to deploy with Docker Compose

## Structure

- `models/` - Model files for serving
- `config/` - Configuration files
- `logs/` - Application logs
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Multi-container deployment
- `nginx.conf` - Reverse proxy configuration
- `deploy.sh` - Automated deployment script

## Endpoints

- `GET /` - API documentation
- `GET /health` - Health check
- `POST /predict/{model_id}` - Make predictions
- `GET /models` - List loaded models
- `GET /stats` - Server statistics

## Monitoring

- Health checks: `/health`
- Metrics: `/stats`
- Logs: `logs/api.log`

## Security

- Rate limiting enabled (60 requests/minute)
- Security headers configured in nginx
- No debug mode in production

## Scaling

To scale the API:
```bash
docker-compose up -d --scale ai-tutorial-prod=3
```
"""
    
    with open(production_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print(f"🏭 Production example created in: {production_dir}")
    print(f"📁 Files generated:")
    for file_path in sorted(production_dir.rglob('*')):
        if file_path.is_file():
            rel_path = file_path.relative_to(production_dir)
            print(f"  📄 {rel_path}")
    
    return production_dir


def main():
    """Main demonstration function"""
    print("🚀 AI TUTORIAL MODEL DEPLOYMENT DEMONSTRATION")
    print("=" * 60)
    print("This demo shows how to deploy AI models in production using")
    print("the AI Tutorial deployment framework.")
    print()
    
    try:
        # Step 1: Create demo models
        models_dir = create_demo_models()
        
        # Step 2: Demonstrate model registry
        registry = demonstrate_model_registry()
        
        # Step 3: Demonstrate model server
        server = demonstrate_model_server()
        
        # Step 4: Demonstrate deployment automation
        package_path = demonstrate_deployment_automation()
        
        # Step 5: Create production example
        production_dir = create_production_example()
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 DEPLOYMENT DEMONSTRATION COMPLETED!")
        print("=" * 60)
        print(f"📁 Demo models: {models_dir}")
        print(f"🗂️ Model registry: demo_model_registry/")
        print(f"🖥️ Model server: Running at http://localhost:5001")
        print(f"📦 Deployment package: {package_path}")
        print(f"🏭 Production example: {production_dir}")
        
        print(f"\n🚀 Next Steps:")
        print(f"1. Visit http://localhost:5001 to see the API in action")
        print(f"2. Check {package_path}/README.md for deployment instructions")
        print(f"3. Explore {production_dir}/ for production deployment example")
        print(f"4. Run 'docker-compose up' in {production_dir}/ for production deployment")
        
        print(f"\n💡 Key Features Demonstrated:")
        print(f"  ✅ Model versioning and registry")
        print(f"  ✅ REST API for model serving")
        print(f"  ✅ Docker containerization")
        print(f"  ✅ Production deployment configurations")
        print(f"  ✅ Monitoring and health checks")
        print(f"  ✅ Automated deployment scripts")
        
        # Keep server running for a bit
        print(f"\n⏰ Server will continue running for testing...")
        print(f"   Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping server...")
    
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n✅ Demonstration finished!")


if __name__ == "__main__":
    main()