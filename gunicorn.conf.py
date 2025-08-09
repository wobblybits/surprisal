# Gunicorn configuration for Render.com deployment
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '8001')}"
backlog = 2048

# Worker processes
workers = 1  # Start with 1 worker for minimal memory usage
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Memory management
max_requests = 1000  # Restart workers after 1000 requests to prevent memory leaks
max_requests_jitter = 50
preload_app = True  # Preload application for memory efficiency

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'surprisal-calculator'

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL (if needed in the future)
keyfile = None
certfile = None

# Environment-specific settings
if os.getenv('FLASK_ENV') == 'production':
    # Production optimizations
    workers = min(2, (os.cpu_count() or 1) + 1)  # Limit workers for memory
    max_requests = 500  # More frequent worker recycling in production
    timeout = 60  # Longer timeout for model loading 