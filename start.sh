#!/bin/bash
# Render.com startup script with environment variables

echo "Starting Surprisal Calculator on Render.com..."

# Set default environment variables (can be overridden by Render.com settings)
export FLASK_ENV=${FLASK_ENV:-production}
export FLASK_SECRET_KEY=${FLASK_SECRET_KEY:-$(openssl rand -hex 32)}
export MAX_TEXT_LENGTH=${MAX_TEXT_LENGTH:-1000}

# Rate limiting configuration
export RATE_LIMIT_STORAGE_URL=${RATE_LIMIT_STORAGE_URL:-memory://}
export RATE_LIMIT_PER_MINUTE=${RATE_LIMIT_PER_MINUTE:-20}
export RATE_LIMIT_PER_HOUR=${RATE_LIMIT_PER_HOUR:-500}

# Security settings
export CSRF_ENABLED=${CSRF_ENABLED:-False}

# Model configuration for production (memory optimization)
export ENABLED_MODELS=${ENABLED_MODELS:flan}
export DISABLED_MODELS=${DISABLED_MODELS:smollm,-gpt2,distilgpt2,-nano mistral,qwen}

# Display configuration
echo "Environment: $FLASK_ENV"
echo "Port: $PORT"
echo "Enabled models: $ENABLED_MODELS"
echo "Disabled models: $DISABLED_MODELS"
echo "Rate limits: $RATE_LIMIT_PER_MINUTE/min, $RATE_LIMIT_PER_HOUR/hour"
echo "CSRF enabled: $CSRF_ENABLED"

# Generate secret key if not provided (for demo purposes)
if [ -z "$FLASK_SECRET_KEY" ] || [ "$FLASK_SECRET_KEY" = "dev-key-not-for-production" ]; then
    echo "WARNING: Generating random secret key. Set FLASK_SECRET_KEY for production!"
    export FLASK_SECRET_KEY=$(openssl rand -hex 32)
fi

# Start the application with gunicorn
exec gunicorn -c gunicorn.conf.py app:app 