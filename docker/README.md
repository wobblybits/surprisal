# Docker Deployment for Surprisal Calculator

This directory contains all Docker-related configuration files for deploying the Surprisal Calculator.

## Quick Start

### Development
```bash
cd docker/
./run-development.sh
```
Access at: http://localhost:8001

### Production
```bash
cd docker/
./run-production.sh
```
Access at: http://localhost

## Files Overview

| File | Purpose |
|------|---------|
| `Dockerfile` | Main production image |
| `Dockerfile.minimal` | Lightweight image for faster builds |
| `docker-compose.yml` | Production deployment with Nginx + Redis |
| `docker-compose.dev.yml` | Development setup with hot reload |
| `nginx.conf` | Reverse proxy configuration |
| `run-production.sh` | Production deployment script |
| `run-development.sh` | Development deployment script |

## Configuration

### Development Features
- Hot reload enabled
- Debug mode on
- Relaxed rate limits (100/min, 1000/hour)
- CSRF protection disabled
- Direct Redis access on port 6379

### Production Features
- Nginx reverse proxy
- Rate limiting and caching
- Security headers
- Health monitoring
- Persistent Redis storage

## Manual Commands

### Development
```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f app

# Stop
docker-compose -f docker-compose.dev.yml down
```

### Production
```bash
# Start production environment
docker-compose up -d

# Scale application
docker-compose up -d --scale app=3

# View logs
docker-compose logs -f app

# Stop
docker-compose down
```

## Environment Variables

Create `.env.production` in the parent directory with:

```bash
FLASK_SECRET_KEY=your-secure-random-key-here
RATE_LIMIT_STORAGE_URL=redis://redis:6379
MAX_TEXT_LENGTH=1000
RATE_LIMIT_PER_MINUTE=10
RATE_LIMIT_PER_HOUR=100
```

## Troubleshooting

### Health Check Fails
```bash
# Check application logs
docker-compose logs app

# Check if Redis is running
docker-compose logs redis

# Test direct application access
curl http://localhost:8001/health
```

### Port Conflicts
If ports 80, 8001, or 6379 are in use:

1. **Development**: Edit `docker-compose.dev.yml` ports section
2. **Production**: Edit `docker-compose.yml` ports section

### Memory Issues
For systems with limited memory, use the minimal image:

```yaml
# In docker-compose.yml, change:
dockerfile: docker/Dockerfile
# To:
dockerfile: docker/Dockerfile.minimal
```

## Monitoring

### Application Health
- **Endpoint**: http://localhost/health (production) or http://localhost:8001/health (dev)
- **Frequency**: Every 30-60 seconds
- **Timeout**: 5-10 seconds

### Resource Usage
```bash
# View resource usage
docker stats

# View detailed container info
docker-compose ps
```
```
