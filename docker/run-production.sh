#!/bin/bash

# Production deployment script
echo "🚀 Starting Surprisal Calculator in production mode..."

# Check if we're in the right directory
if [ ! -f "../app.py" ]; then
    echo "❌ Please run this script from the docker/ directory"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Ensure we have a production environment file
if [ ! -f "../.env.production" ]; then
    echo "⚠️ No .env.production file found. Creating from template..."
    cp ../.env.example ../.env.production
    echo "🔑 Please update .env.production with production values"
    echo "   Especially FLASK_SECRET_KEY and RATE_LIMIT_STORAGE_URL"
fi

# Pull latest images
echo "📦 Pulling latest base images..."
docker-compose pull

# Build and start services
echo "🏗️ Building and starting services..."
docker-compose up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check health
echo "🔍 Checking application health..."
if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✅ Application is healthy!"
    echo ""
    echo "🌐 Application available at: http://localhost"
    echo "📊 Health check: http://localhost/health"
    echo "📖 API docs: Open ../api-docs.yaml in Swagger Editor"
    echo ""
    echo "📋 To view logs: docker-compose logs -f"
    echo "🛑 To stop: docker-compose down"
else
    echo "❌ Health check failed. Check logs with: docker-compose logs"
fi 