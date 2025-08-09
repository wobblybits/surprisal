#!/bin/bash
# Render.com startup script with environment variables

echo "Starting Surprisal Calculator on Render.com..."

# Set default environment variables (can be overridden by Render.com settings)
export FLASK_ENV=${FLASK_ENV:-production}
export DEPLOYMENT_TYPE=${DEPLOYMENT_TYPE:-render}
export MAX_TEXT_LENGTH=${MAX_TEXT_LENGTH:-1000}

# Rate limiting configuration - optimized for Render single-instance deployment
export RATE_LIMIT_STORAGE_URL=${RATE_LIMIT_STORAGE_URL:-memory://}
export RATE_LIMIT_PER_MINUTE=${RATE_LIMIT_PER_MINUTE:-200}
export RATE_LIMIT_PER_HOUR=${RATE_LIMIT_PER_HOUR:-1000}

# Security settings
export ORIGIN_PROTECTION_ENABLED=${ORIGIN_PROTECTION_ENABLED:-True}

# Model configuration for production (memory optimization)
export ENABLED_MODELS=${ENABLED_MODELS:-flan,distilgpt2}
export DISABLED_MODELS=${DISABLED_MODELS:-smollm,gpt2,nano mistral,qwen}

# Display configuration
echo "Environment: $FLASK_ENV"
echo "Deployment type: $DEPLOYMENT_TYPE"
echo "Port: $PORT"
echo "Rate limiting storage: $RATE_LIMIT_STORAGE_URL"
echo "Enabled models: $ENABLED_MODELS"
echo "Disabled models: $DISABLED_MODELS"
echo "Rate limits: $RATE_LIMIT_PER_MINUTE/min, $RATE_LIMIT_PER_HOUR/hour"
echo "Origin protection enabled: $ORIGIN_PROTECTION_ENABLED"

# Start the application with gunicorn
exec gunicorn -c gunicorn.conf.py app:app 