#!/bin/bash

# Development deployment script
echo "🛠️ Starting Surprisal Calculator in development mode..."

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

echo "🔧 Environment: Development"
echo "📝 Features:"
echo "   - Hot reload enabled"
echo "   - Debug mode on"
echo "   - Relaxed rate limits"
echo "   - CSRF disabled"
echo "   - Redis accessible on localhost:6379"
echo ""

# Build and start services
echo "🏗️ Building and starting development services..."
docker-compose -f docker-compose.dev.yml up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check health
echo "🔍 Checking application health..."
if curl -f http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ Development environment is ready!"
    echo ""
    echo "🌐 Application: http://localhost:8001"
    echo "📊 Health check: http://localhost:8001/health"
    echo "🔍 Redis: localhost:6379"
    echo "📖 API docs: Open ../api-docs.yaml in Swagger Editor"
    echo ""
    echo "📋 To view logs: docker-compose -f docker-compose.dev.yml logs -f"
    echo "🛑 To stop: docker-compose -f docker-compose.dev.yml down"
else
    echo "❌ Health check failed. Check logs with: docker-compose -f docker-compose.dev.yml logs"
fi 