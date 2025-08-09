# Deployment Guide for Render.com

This guide explains how to deploy the Surprisal Calculator to Render.com with optimized memory usage.

## Environment Configuration

### Required Environment Variables

Set these in your Render.com service configuration:

```bash
FLASK_ENV=production
FLASK_SECRET_KEY=your-randomly-generated-secret-key-here
PORT=8001

# Model Configuration for Memory Optimization
ENABLED_MODELS=gpt2,distilgpt2
DISABLED_MODELS=smollm,nano mistral,qwen,flan

# Rate Limiting (adjust as needed)
RATE_LIMIT_PER_MINUTE=20
RATE_LIMIT_PER_HOUR=500
RATE_LIMIT_STORAGE_URL=memory://

# Security
CSRF_ENABLED=True
MAX_TEXT_LENGTH=1000
```

### Memory Optimization

The production configuration disables larger models by default:
- **Enabled**: `gpt2`, `distilgpt2` (lighter models)
- **Disabled**: `smollm`, `nano mistral`, `qwen`, `flan` (heavier models)

Disabled models will appear as grayed-out buttons in the UI with "unavailable in demo" message on hover.

## Render.com Configuration

### Build & Deploy Settings

1. **Build Command**: `pip install -r requirements.txt`
2. **Start Command**: `gunicorn -c gunicorn.conf.py app:app`
3. **Instance Type**: Choose based on your needs (Starter plan should work with optimized models)

### Auto-Deploy

Connect your GitHub repository and enable auto-deploy for automatic updates.

## Health Check

The application provides a health check endpoint at `/health` that includes:
- Model loading status
- Memory usage information
- Configuration details
- Uptime statistics

## Customizing Model Selection

To change which models are enabled/disabled, update the environment variables:

```bash
# Enable all models (requires more memory)
ENABLED_MODELS=gpt2,distilgpt2,smollm,nano mistral,qwen,flan
DISABLED_MODELS=

# Enable only the smallest models
ENABLED_MODELS=gpt2,distilgpt2
DISABLED_MODELS=smollm,nano mistral,qwen,flan

# Custom selection
ENABLED_MODELS=gpt2,smollm,flan
DISABLED_MODELS=distilgpt2,nano mistral,qwen
```

## Port Configuration

The application automatically uses the `PORT` environment variable provided by Render.com. The gunicorn configuration handles this automatically.

## Troubleshooting

### Memory Issues
- Reduce the number of enabled models
- Check the `/health` endpoint for memory usage
- Consider upgrading your Render.com plan

### Model Loading Errors
- Check the logs for specific model loading failures
- Verify that only available models are enabled
- Use the `/api/models` endpoint to check model availability

### Rate Limiting
- Adjust `RATE_LIMIT_PER_MINUTE` and `RATE_LIMIT_PER_HOUR` as needed
- Monitor usage through the health check endpoint 