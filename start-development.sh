#!/bin/bash
# Development startup script

export FLASK_ENV=development
export ENABLED_MODELS=gpt2,distilgpt2,smollm,nano mistral,qwen,flan
export DISABLED_MODELS=
export RATE_LIMIT_PER_MINUTE=1000
export RATE_LIMIT_PER_HOUR=10000

exec gunicorn -c gunicorn.conf.py app:app 