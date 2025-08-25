# AI Tutorial Production Deployment

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
