# Deployment Guide

## Render.com Deployment (Recommended for Single Instance)

Render.com provides a simple, single-instance deployment that's perfect for this application.

### Environment Configuration

Set these in your Render.com service configuration:

```bash
# Core settings
FLASK_ENV=production
DEPLOYMENT_TYPE=render
PORT=8001

# Model Configuration for Memory Optimization
ENABLED_MODELS=flan
DISABLED_MODELS=smollm,gpt2,distilgpt2,nano mistral,qwen

# Rate Limiting (memory-based, no Redis needed)
RATE_LIMIT_PER_MINUTE=200
RATE_LIMIT_PER_HOUR=1000

# Security
ORIGIN_PROTECTION_ENABLED=True
MAX_TEXT_LENGTH=1000
```

**Key Benefits:**
- ✅ No Redis dependency required
- ✅ Memory-based rate limiting for single instance
- ✅ Automatic restarts clear rate limit state
- ✅ Simpler configuration

### Build & Deploy Settings

1. **Build Command**: `pip install -r requirements.txt`
2. **Start Command**: `./start.sh`
3. **Instance Type**: Starter plan works with optimized models

## Docker Deployment (For Multi-Instance or Complex Setups)

Use Docker when you need:
- Multiple app instances
- Persistent rate limiting across restarts
- More complex infrastructure

### Production Docker
```bash
cd docker/
export FLASK_ENV=docker
export DEPLOYMENT_TYPE=docker
./run-production.sh
```

### Development Docker
```bash
cd docker/
./run-development.sh
```

**Features:**
- ✅ Redis-backed rate limiting
- ✅ Persistent storage
- ✅ Nginx reverse proxy
- ✅ Multi-instance support

## Rate Limiting Details

### Render (Memory-based)
- **Storage**: Application memory
- **Persistence**: Reset on app restart
- **Limits**: 200/min, 1000/hour
- **Scaling**: Single instance only

### Docker (Redis-based)
- **Storage**: Redis database
- **Persistence**: Survives restarts
- **Limits**: 200/min, 1000/hour  
- **Scaling**: Multiple instances supported

## Migration Guide

### From Redis to Memory (Render)
No migration needed - rate limits start fresh.

### From Memory to Redis (Docker)
Rate limits will reset when switching storage backends. 