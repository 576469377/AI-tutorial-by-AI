"""
Deployment utilities for AI Tutorial - Helper functions and tools

This module provides utility functions for deployment tasks including
containerization, configuration management, and deployment automation.
"""

import os
import json
import yaml
import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import subprocess
import logging


class DeploymentHelper:
    """
    Helper class for deployment tasks and automation
    """
    
    def __init__(self, project_dir: str = '.'):
        """
        Initialize deployment helper
        
        Args:
            project_dir: Project root directory
        """
        self.project_dir = Path(project_dir)
        self.logger = logging.getLogger('DeploymentHelper')
    
    def generate_dockerfile(self, 
                          base_image: str = 'python:3.9-slim',
                          port: int = 5000,
                          model_files: List[str] = None,
                          custom_requirements: str = None) -> str:
        """
        Generate Dockerfile for model deployment
        
        Args:
            base_image: Docker base image
            port: Port to expose
            model_files: List of model files to include
            custom_requirements: Custom requirements file path
        
        Returns:
            Dockerfile content as string
        """
        requirements_file = custom_requirements or 'requirements.txt'
        
        dockerfile_content = f"""# AI Tutorial Model Deployment
FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY {requirements_file} .
RUN pip install --no-cache-dir -r {requirements_file}

# Install Flask for model serving
RUN pip install flask

# Copy application code
COPY deployment/ ./deployment/
COPY utils/ ./utils/

# Copy model files
"""
        
        if model_files:
            for model_file in model_files:
                dockerfile_content += f"COPY {model_file} ./models/\n"
        else:
            dockerfile_content += "COPY *.pth ./models/\n"
            dockerfile_content += "COPY *.pkl ./models/\n"
        
        dockerfile_content += f"""
# Create models directory
RUN mkdir -p ./models

# Expose port
EXPOSE {port}

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_ENV=production
ENV MODEL_SERVER_PORT={port}

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{port}/health || exit 1

# Run the model server
CMD ["python", "-m", "deployment.model_server"]
"""
        
        return dockerfile_content
    
    def create_docker_compose(self, 
                            service_name: str = 'ai-tutorial-api',
                            port: int = 5000,
                            volumes: List[str] = None) -> str:
        """
        Generate docker-compose.yml for easy deployment
        
        Args:
            service_name: Name of the service
            port: Port mapping
            volumes: Additional volumes to mount
        
        Returns:
            docker-compose.yml content as string
        """
        volumes_config = volumes or []
        volumes_config.extend([
            './models:/app/models',
            './logs:/app/logs'
        ])
        
        compose_content = {
            'version': '3.8',
            'services': {
                service_name: {
                    'build': '.',
                    'ports': [f'{port}:{port}'],
                    'volumes': volumes_config,
                    'environment': [
                        f'MODEL_SERVER_PORT={port}',
                        'PYTHONPATH=/app'
                    ],
                    'restart': 'unless-stopped',
                    'healthcheck': {
                        'test': f'curl -f http://localhost:{port}/health || exit 1',
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '40s'
                    }
                }
            },
            'volumes': {
                'model_data': {},
                'logs': {}
            },
            'networks': {
                'ai_tutorial_network': {
                    'driver': 'bridge'
                }
            }
        }
        
        return yaml.dump(compose_content, default_flow_style=False)
    
    def create_kubernetes_config(self, 
                                app_name: str = 'ai-tutorial-api',
                                port: int = 5000,
                                replicas: int = 2,
                                image: str = 'ai-tutorial:latest') -> Dict[str, str]:
        """
        Generate Kubernetes deployment and service configs
        
        Args:
            app_name: Application name
            port: Service port
            replicas: Number of replicas
            image: Docker image name
        
        Returns:
            Dictionary with deployment and service YAML configs
        """
        
        deployment_config = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f'{app_name}-deployment',
                'labels': {'app': app_name}
            },
            'spec': {
                'replicas': replicas,
                'selector': {'matchLabels': {'app': app_name}},
                'template': {
                    'metadata': {'labels': {'app': app_name}},
                    'spec': {
                        'containers': [{
                            'name': app_name,
                            'image': image,
                            'ports': [{'containerPort': port}],
                            'env': [
                                {'name': 'MODEL_SERVER_PORT', 'value': str(port)},
                                {'name': 'PYTHONPATH', 'value': '/app'}
                            ],
                            'livenessProbe': {
                                'httpGet': {'path': '/health', 'port': port},
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {'path': '/health', 'port': port},
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            },
                            'resources': {
                                'requests': {'memory': '512Mi', 'cpu': '250m'},
                                'limits': {'memory': '1Gi', 'cpu': '500m'}
                            }
                        }]
                    }
                }
            }
        }
        
        service_config = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f'{app_name}-service',
                'labels': {'app': app_name}
            },
            'spec': {
                'selector': {'app': app_name},
                'ports': [{
                    'protocol': 'TCP',
                    'port': 80,
                    'targetPort': port
                }],
                'type': 'LoadBalancer'
            }
        }
        
        return {
            'deployment.yaml': yaml.dump(deployment_config, default_flow_style=False),
            'service.yaml': yaml.dump(service_config, default_flow_style=False)
        }
    
    def create_nginx_config(self, 
                          server_name: str = 'localhost',
                          api_port: int = 5000,
                          static_files_path: str = '/app/static') -> str:
        """
        Generate nginx configuration for production deployment
        
        Args:
            server_name: Server name/domain
            api_port: Backend API port
            static_files_path: Path to static files
        
        Returns:
            nginx configuration as string
        """
        
        nginx_config = f"""# AI Tutorial API - Nginx Configuration
upstream api_backend {{
    server localhost:{api_port};
}}

server {{
    listen 80;
    server_name {server_name};

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied expired no-cache no-store private must-revalidate auth;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

    # Static files
    location /static/ {{
        alias {static_files_path}/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }}

    # API endpoints
    location /api/ {{
        proxy_pass http://api_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }}

    # Root location
    location / {{
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}

    # Health check endpoint
    location /health {{
        proxy_pass http://api_backend/health;
        access_log off;
    }}
}}
"""
        return nginx_config
    
    def create_systemd_service(self, 
                             service_name: str = 'ai-tutorial-api',
                             working_dir: str = '/opt/ai-tutorial',
                             user: str = 'www-data',
                             python_path: str = '/usr/bin/python3') -> str:
        """
        Generate systemd service file for Linux deployment
        
        Args:
            service_name: Service name
            working_dir: Working directory
            user: User to run service as
            python_path: Path to Python executable
        
        Returns:
            systemd service file content
        """
        
        service_config = f"""[Unit]
Description=AI Tutorial Model API Server
After=network.target

[Service]
Type=simple
User={user}
Group={user}
WorkingDirectory={working_dir}
Environment=PATH={working_dir}/venv/bin
Environment=PYTHONPATH={working_dir}
ExecStart={python_path} -m deployment.model_server
Restart=always
RestartSec=10

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths={working_dir}

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier={service_name}

[Install]
WantedBy=multi-user.target
"""
        return service_config
    
    def create_deployment_script(self, 
                                deployment_type: str = 'docker',
                                **kwargs) -> str:
        """
        Generate deployment script
        
        Args:
            deployment_type: Type of deployment ('docker', 'kubernetes', 'systemd')
            **kwargs: Additional configuration options
        
        Returns:
            Deployment script content
        """
        
        script_header = """#!/bin/bash
# AI Tutorial Model Deployment Script
set -e

echo "🚀 Starting AI Tutorial Model Deployment"
echo "========================================"
"""
        
        if deployment_type == 'docker':
            script_content = script_header + """
# Build Docker image
echo "📦 Building Docker image..."
docker build -t ai-tutorial:latest .

# Stop existing container if running
echo "🛑 Stopping existing containers..."
docker stop ai-tutorial-api || true
docker rm ai-tutorial-api || true

# Run new container
echo "🚀 Starting new container..."
docker run -d \\
    --name ai-tutorial-api \\
    -p 5000:5000 \\
    -v $(pwd)/models:/app/models \\
    -v $(pwd)/logs:/app/logs \\
    --restart unless-stopped \\
    ai-tutorial:latest

echo "✅ Deployment completed!"
echo "🌐 API available at: http://localhost:5000"
"""
        
        elif deployment_type == 'docker-compose':
            script_content = script_header + """
# Deploy with docker-compose
echo "📦 Deploying with Docker Compose..."
docker-compose down || true
docker-compose build
docker-compose up -d

echo "✅ Deployment completed!"
echo "🌐 API available at: http://localhost:5000"
"""
        
        elif deployment_type == 'kubernetes':
            script_content = script_header + """
# Deploy to Kubernetes
echo "☸️ Deploying to Kubernetes..."

# Apply configurations
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/ai-tutorial-api-deployment

echo "✅ Deployment completed!"
echo "🌐 Get service URL with: kubectl get services"
"""
        
        else:
            script_content = script_header + """
echo "❌ Unsupported deployment type: """ + deployment_type + """
exit 1
"""
        
        return script_content
    
    def generate_deployment_package(self, 
                                  output_dir: str = 'deployment_package',
                                  deployment_type: str = 'docker') -> str:
        """
        Generate complete deployment package
        
        Args:
            output_dir: Output directory for deployment files
            deployment_type: Type of deployment
        
        Returns:
            Path to generated deployment package
        """
        package_dir = Path(output_dir)
        package_dir.mkdir(exist_ok=True)
        
        # Generate Dockerfile
        dockerfile_content = self.generate_dockerfile()
        with open(package_dir / 'Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        
        # Generate docker-compose.yml
        compose_content = self.create_docker_compose()
        with open(package_dir / 'docker-compose.yml', 'w') as f:
            f.write(compose_content)
        
        # Generate Kubernetes configs
        k8s_configs = self.create_kubernetes_config()
        for filename, content in k8s_configs.items():
            with open(package_dir / filename, 'w') as f:
                f.write(content)
        
        # Generate nginx config
        nginx_config = self.create_nginx_config()
        with open(package_dir / 'nginx.conf', 'w') as f:
            f.write(nginx_config)
        
        # Generate systemd service
        systemd_service = self.create_systemd_service()
        with open(package_dir / 'ai-tutorial-api.service', 'w') as f:
            f.write(systemd_service)
        
        # Generate deployment script
        deploy_script = self.create_deployment_script(deployment_type)
        deploy_script_path = package_dir / 'deploy.sh'
        with open(deploy_script_path, 'w') as f:
            f.write(deploy_script)
        
        # Make deployment script executable
        deploy_script_path.chmod(0o755)
        
        # Create README
        readme_content = f"""# AI Tutorial Model Deployment Package

This package contains all necessary files for deploying the AI Tutorial model API.

## Files Included:

- `Dockerfile` - Docker container configuration
- `docker-compose.yml` - Docker Compose configuration
- `deployment.yaml` - Kubernetes deployment configuration
- `service.yaml` - Kubernetes service configuration
- `nginx.conf` - Nginx reverse proxy configuration
- `ai-tutorial-api.service` - Systemd service configuration
- `deploy.sh` - Automated deployment script

## Quick Start:

### Docker:
```bash
chmod +x deploy.sh
./deploy.sh
```

### Docker Compose:
```bash
docker-compose up -d
```

### Kubernetes:
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

## Generated on: {datetime.datetime.now().isoformat()}
"""
        
        with open(package_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        self.logger.info(f"Deployment package generated in: {package_dir}")
        return str(package_dir)


def demonstrate_deployment_helper():
    """Demonstrate deployment helper functionality"""
    print("🚀 Deployment Helper Demonstration")
    print("=" * 50)
    
    helper = DeploymentHelper()
    
    # Generate deployment package
    package_path = helper.generate_deployment_package('demo_deployment')
    
    print(f"✅ Deployment package created in: {package_path}")
    print(f"📁 Package contents:")
    
    package_dir = Path(package_path)
    for file_path in sorted(package_dir.iterdir()):
        if file_path.is_file():
            print(f"  📄 {file_path.name}")
    
    print(f"\n🚀 To deploy, run:")
    print(f"  cd {package_path}")
    print(f"  chmod +x deploy.sh")
    print(f"  ./deploy.sh")


if __name__ == "__main__":
    demonstrate_deployment_helper()