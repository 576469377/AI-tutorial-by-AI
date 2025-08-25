# AI Tutorial Model Deployment Package

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

## Generated on: 2025-08-25T13:12:03.008211
