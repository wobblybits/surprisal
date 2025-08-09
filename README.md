# Surprisal Calculator WM7±2

Backend: https://surprisal.onrender.com/process/
Frontend: https://surprisal.onrender.com

## Security Features

- **Rate Limiting**: Protects against API abuse (configurable per minute/hour)
- **CSRF Protection**: Prevents cross-site request forgery attacks
- **Input Validation**: Comprehensive sanitization of all inputs
- **Health Monitoring**: `/health` endpoint for system monitoring

## Configuration

The application can be configured using environment variables:

### Flask Configuration
- `FLASK_DEBUG`: Enable debug mode (default: False)
- `FLASK_HOST`: Host to bind to (default: 0.0.0.0)
- `FLASK_PORT`: Port to run on (default: 8001)
- `FLASK_SECRET_KEY`: Secret key for sessions and CSRF (required in production)

### Application Configuration
- `MAX_TEXT_LENGTH`: Maximum input text length (default: 1000)

### Rate Limiting Configuration
- `RATE_LIMIT_STORAGE_URL`: Redis URL for rate limiting storage (default: memory://)
- `RATE_LIMIT_PER_MINUTE`: Requests per minute limit (default: 10)
- `RATE_LIMIT_PER_HOUR`: Requests per hour limit (default: 100)

### Security Configuration
- `CSRF_ENABLED`: Enable CSRF protection (default: True)

Copy `.env.example` to `.env` and modify as needed.

## Installation

```bash
pip install -r requirements.txt
```

## Development Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# Set FLASK_DEBUG=True for development

# Install Redis for rate limiting (optional, falls back to memory)
# macOS: brew install redis
# Ubuntu: sudo apt-get install redis-server

# Run the application
python app.py
```

## Production Deployment

```bash
# Set production environment variables
export FLASK_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex())')
export FLASK_DEBUG=False
export RATE_LIMIT_STORAGE_URL=redis://your-redis-host:6379

# Run with gunicorn
gunicorn app:app
```

## API Endpoints

- `GET /`: Main application interface
- `POST /process/`: Convert text to surprisal values (rate limited)
- `POST /reverse/`: Generate text from musical notes (rate limited)
- `GET /health`: Health check and system status
- `GET /debug_tokens/<model>`: Debug tokenization (rate limited)

## Monitoring

The `/health` endpoint provides:
- Application status and uptime
- Model loading status
- Configuration information
- System health metrics

Use this endpoint for:
- Load balancer health checks
- Monitoring system integration
- Debugging deployment issues

Should be available locally from 127.0.0.1:8000