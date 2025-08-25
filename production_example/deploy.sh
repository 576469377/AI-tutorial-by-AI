#!/bin/bash
# AI Tutorial Model Deployment Script
set -e

echo "🚀 Starting AI Tutorial Model Deployment"
echo "========================================"

# Deploy with docker-compose
echo "📦 Deploying with Docker Compose..."
docker-compose down || true
docker-compose build
docker-compose up -d

echo "✅ Deployment completed!"
echo "🌐 API available at: http://localhost:5000"
