#!/bin/bash
# AI Tutorial Model Deployment Script
set -e

echo "🚀 Starting AI Tutorial Model Deployment"
echo "========================================"

# Build Docker image
echo "📦 Building Docker image..."
docker build -t ai-tutorial:latest .

# Stop existing container if running
echo "🛑 Stopping existing containers..."
docker stop ai-tutorial-api || true
docker rm ai-tutorial-api || true

# Run new container
echo "🚀 Starting new container..."
docker run -d \
    --name ai-tutorial-api \
    -p 5000:5000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    --restart unless-stopped \
    ai-tutorial:latest

echo "✅ Deployment completed!"
echo "🌐 API available at: http://localhost:5000"
