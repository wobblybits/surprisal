import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration."""
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY') or 'dev-key-not-for-production'
    MAX_TEXT_LENGTH = int(os.getenv('MAX_TEXT_LENGTH', 1000))
    
    # Rate Limiting - Default to memory for local development
    RATE_LIMIT_STORAGE_URL = os.getenv('RATE_LIMIT_STORAGE_URL', 'memory://')
    RATE_LIMIT_PER_MINUTE = os.getenv('RATE_LIMIT_PER_MINUTE', '10')
    RATE_LIMIT_PER_HOUR = os.getenv('RATE_LIMIT_PER_HOUR', '100')
    
    # Security
    CSRF_ENABLED = os.getenv('CSRF_ENABLED', 'True').lower() == 'true'
    
    # Flask settings
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('FLASK_PORT', 8001))

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    HOST = os.getenv('FLASK_HOST', '127.0.0.1')  # localhost for dev
    
    # Force memory storage for development to avoid Redis dependency
    RATE_LIMIT_STORAGE_URL = 'memory://'
    CSRF_ENABLED = False  # Disable CSRF for easier development
    
    def __init__(self):
        super().__init__()
        # Override rate limits for development - much higher limits
        self.RATE_LIMIT_PER_MINUTE = '1000'  # 1000/min for dev
        self.RATE_LIMIT_PER_HOUR = '10000'   # 10000/hour for dev
        
        if self.SECRET_KEY == 'dev-key-not-for-production':
            print("WARNING: Using default secret key. Set FLASK_SECRET_KEY in production!")
        print(f"Development mode: Rate limits set to {self.RATE_LIMIT_PER_MINUTE}/min, {self.RATE_LIMIT_PER_HOUR}/hour")

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    
    def __init__(self):
        super().__init__()
        if not self.SECRET_KEY or self.SECRET_KEY == 'dev-key-not-for-production':
            raise ValueError("FLASK_SECRET_KEY environment variable must be set in production")

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    # Use memory storage for testing to avoid external dependencies
    RATE_LIMIT_STORAGE_URL = 'memory://'
    CSRF_ENABLED = False
    
    def __init__(self):
        super().__init__()
        # High limits for testing
        self.RATE_LIMIT_PER_MINUTE = '1000'
        self.RATE_LIMIT_PER_HOUR = '10000'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on FLASK_ENV environment variable."""
    config_name = os.getenv('FLASK_ENV', 'default')
    return config[config_name]()