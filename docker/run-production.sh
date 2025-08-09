#!/bin/bash
echo "Starting Docker production deployment..."

# Set Docker-specific environment
export FLASK_ENV=docker
export DEPLOYMENT_TYPE=docker

# Start with docker-compose
docker-compose -f docker-compose.yml up --build -d

echo "Production deployment started!"
echo "Access the application at: http://localhost"
echo "Direct app access at: http://localhost:8001" 