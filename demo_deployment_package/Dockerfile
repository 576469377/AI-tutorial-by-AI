# AI Tutorial Model Deployment
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Flask for model serving
RUN pip install flask

# Copy application code
COPY deployment/ ./deployment/
COPY utils/ ./utils/

# Copy model files
COPY *.pth ./models/
COPY *.pkl ./models/

# Create models directory
RUN mkdir -p ./models

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_ENV=production
ENV MODEL_SERVER_PORT=5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the model server
CMD ["python", "-m", "deployment.model_server"]
