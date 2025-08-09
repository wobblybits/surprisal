#!/bin/bash

# Quick development server startup
echo "🎵 Starting Surprisal Calculator development server..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run ./setup-dev.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Run ./setup-dev.sh first"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Set development defaults
export FLASK_DEBUG=True
export FLASK_ENV=development

echo "🔧 Environment: Development"
echo "🌐 URL: http://localhost:${FLASK_PORT:-8001}"
echo "📊 Health: http://localhost:${FLASK_PORT:-8001}/health"
echo ""

# Start the development server
python app.py 