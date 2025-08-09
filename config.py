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
    
    # Flask settings - Use PORT env var for Render.com compatibility
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', os.getenv('FLASK_PORT', 8001)))
    
    # Model configuration - which models to load/enable
    ENABLED_MODELS = os.getenv('ENABLED_MODELS', 'gpt2,distilgpt2,smollm,nano mistral,qwen,flan').split(',')
    DISABLED_MODELS = os.getenv('DISABLED_MODELS', '').split(',') if os.getenv('DISABLED_MODELS') else []

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
        
        # Disable qwen model in development
        self.DISABLED_MODELS = []
        self.ENABLED_MODELS = [model for model in self.ENABLED_MODELS if model not in self.DISABLED_MODELS]
        
        if self.SECRET_KEY == 'dev-key-not-for-production':
            print("WARNING: Using default secret key. Set FLASK_SECRET_KEY in production!")
        print(f"Development mode: Rate limits set to {self.RATE_LIMIT_PER_MINUTE}/min, {self.RATE_LIMIT_PER_HOUR}/hour")
        print(f"Development mode: Enabled models: {self.ENABLED_MODELS}")
        print(f"Development mode: Disabled models: {self.DISABLED_MODELS}")

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    
    # Production model configuration - disable larger models to save memory
    DISABLED_MODELS = os.getenv('DISABLED_MODELS', 'smollm,nano mistral,qwen,flan').split(',')
    
    def __init__(self):
        super().__init__()
        if not self.SECRET_KEY or self.SECRET_KEY == 'dev-key-not-for-production':
            raise ValueError("FLASK_SECRET_KEY environment variable must be set in production")
        
        # Filter out disabled models from enabled models
        self.ENABLED_MODELS = [model for model in self.ENABLED_MODELS if model not in self.DISABLED_MODELS]
        print(f"Production mode: Enabled models: {self.ENABLED_MODELS}")
        print(f"Production mode: Disabled models: {self.DISABLED_MODELS}")

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