#!/bin/bash
# Production startup script

export FLASK_ENV=production
export ENABLED_MODELS=gpt2,distilgpt2
export DISABLED_MODELS=smollm,nano mistral,qwen,flan
export RATE_LIMIT_PER_MINUTE=200
export RATE_LIMIT_PER_HOUR=6000

exec gunicorn -c gunicorn.conf.py app:app 