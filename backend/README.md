# Flask Hello World with Gunicorn

A simple Flask "Hello World" application that can be run with gunicorn.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Option 1: Direct gunicorn command
```bash
gunicorn --bind 0.0.0.0:5000 wsgi:app
```

### Option 2: Using the run script
```bash
python run.py
```

### Option 3: Development mode (Flask built-in server)
```bash
python app.py
```

## Endpoints

- `GET /` - Returns "Hello, World!"
- `GET /health` - Returns health status

## Gunicorn Configuration

The application is configured to run with:
- 4 worker processes
- 120 second timeout
- Binding to all interfaces on port 5000

You can customize these settings by modifying the gunicorn command or the `run.py` script.

## Production Deployment

For production deployment, consider:
- Using a reverse proxy (nginx)
- Setting up proper logging
- Configuring environment variables
- Using a process manager (systemd, supervisor, etc.) 