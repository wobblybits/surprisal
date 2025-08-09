#!/bin/bash

# Surprisal Calculator Development Setup
echo "🎵 Setting up Surprisal Calculator development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ required. Found: Python $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment template if .env doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️ Creating .env file from template..."
    cp .env.example .env
    echo "🔑 Please update .env with your settings, especially FLASK_SECRET_KEY"
fi

# Check if Redis is available (optional)
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "✅ Redis is running (rate limiting will use Redis)"
    else
        echo "⚠️ Redis not running (will use memory-based rate limiting)"
    fi
else
    echo "⚠️ Redis not installed (will use memory-based rate limiting)"
    echo "   Install Redis for production: brew install redis (macOS) or apt-get install redis-server (Ubuntu)"
fi

# Generate secret key if not set
if grep -q "your-secret-key-here" .env; then
    echo "🔐 Generating secure secret key..."
    secret_key=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
    sed -i.bak "s/your-secret-key-here-change-in-production/$secret_key/" .env
    rm .env.bak
fi

echo ""
echo "✨ Setup complete!"
echo ""
echo "🚀 To start development:"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "🌐 Application will be available at: http://localhost:8001"
echo "📊 Health check: http://localhost:8001/health"
echo "📖 API documentation: Open api-docs.yaml in Swagger Editor"
echo ""
echo "🐳 To run with Docker:"
echo "   docker-compose up"
echo "" 