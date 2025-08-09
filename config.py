import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration."""
    MAX_TEXT_LENGTH = int(os.getenv('MAX_TEXT_LENGTH', 1000))
    
    # Rate Limiting - Default to memory for local development
    RATE_LIMIT_STORAGE_URL = os.getenv('RATE_LIMIT_STORAGE_URL', 'memory://')
    RATE_LIMIT_PER_MINUTE = os.getenv('RATE_LIMIT_PER_MINUTE', '10')
    RATE_LIMIT_PER_HOUR = os.getenv('RATE_LIMIT_PER_HOUR', '100')
    
    # Security - Origin-based protection
    ORIGIN_PROTECTION_ENABLED = os.getenv('ORIGIN_PROTECTION_ENABLED', 'True').lower() == 'true'
    
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
    ORIGIN_PROTECTION_ENABLED = False  # Disable origin protection for easier development
    
    def __init__(self):
        super().__init__()
        # Override rate limits for development - much higher limits
        self.RATE_LIMIT_PER_MINUTE = '1000'  # 1000/min for dev
        self.RATE_LIMIT_PER_HOUR = '10000'   # 10000/hour for dev
        
        # Disable qwen model in development
        self.DISABLED_MODELS = []
        self.ENABLED_MODELS = [model for model in self.ENABLED_MODELS if model not in self.DISABLED_MODELS]
        
        print(f"Development mode: Rate limits set to {self.RATE_LIMIT_PER_MINUTE}/min, {self.RATE_LIMIT_PER_HOUR}/hour")
        print(f"Development mode: Enabled models: {self.ENABLED_MODELS}")
        print(f"Development mode: Disabled models: {self.DISABLED_MODELS}")

class ProductionConfig(Config):
    """Production configuration - optimized for single-instance cloud deployment (Render)."""
    DEBUG = False
    
    def __init__(self):
        super().__init__()
        
        # Determine deployment type
        deployment_type = os.getenv('DEPLOYMENT_TYPE', 'render').lower()
        
        if deployment_type == 'docker':
            # Docker deployment - use Redis if available
            self.RATE_LIMIT_STORAGE_URL = os.getenv('RATE_LIMIT_STORAGE_URL', 'redis://redis:6379')
            self.RATE_LIMIT_PER_MINUTE = os.getenv('RATE_LIMIT_PER_MINUTE', '200')
            self.RATE_LIMIT_PER_HOUR = os.getenv('RATE_LIMIT_PER_HOUR', '1000')
            print(f"Docker deployment: Using Redis storage at {self.RATE_LIMIT_STORAGE_URL}")
        else:
            # Render/cloud deployment - use memory storage for single instance
            self.RATE_LIMIT_STORAGE_URL = 'memory://'
            self.RATE_LIMIT_PER_MINUTE = os.getenv('RATE_LIMIT_PER_MINUTE', '200')
            self.RATE_LIMIT_PER_HOUR = os.getenv('RATE_LIMIT_PER_HOUR', '1000')
            print("Cloud deployment: Using memory storage (single instance)")
        
        # Production model configuration - disable larger models to save memory
        self.DISABLED_MODELS = os.getenv('DISABLED_MODELS', 'smollm,nano mistral,qwen,flan').split(',')
        self.ENABLED_MODELS = [model for model in self.ENABLED_MODELS if model not in self.DISABLED_MODELS]

        print(f"Production mode: Rate limits set to {self.RATE_LIMIT_PER_MINUTE}/min, {self.RATE_LIMIT_PER_HOUR}/hour")
        print(f"Production mode: Storage type: {self.RATE_LIMIT_STORAGE_URL}")
        print(f"Production mode: Disabled models: {self.DISABLED_MODELS}")

class DockerProductionConfig(Config):
    """Docker-specific production configuration - uses Redis for multi-instance deployments."""
    DEBUG = False
    
    def __init__(self):
        super().__init__()
        
        # Force Redis for Docker deployments
        self.RATE_LIMIT_STORAGE_URL = os.getenv('RATE_LIMIT_STORAGE_URL', 'redis://redis:6379')
        self.RATE_LIMIT_PER_MINUTE = os.getenv('RATE_LIMIT_PER_MINUTE', '200')
        self.RATE_LIMIT_PER_HOUR = os.getenv('RATE_LIMIT_PER_HOUR', '1000')
        
        # Production model configuration
        self.DISABLED_MODELS = os.getenv('DISABLED_MODELS', 'smollm,nano mistral,qwen,flan').split(',')
        self.ENABLED_MODELS = [model for model in self.ENABLED_MODELS if model not in self.DISABLED_MODELS]

        print(f"Docker production: Rate limits set to {self.RATE_LIMIT_PER_MINUTE}/min, {self.RATE_LIMIT_PER_HOUR}/hour")
        print(f"Docker production: Using Redis storage at {self.RATE_LIMIT_STORAGE_URL}")
        print(f"Docker production: Disabled models: {self.DISABLED_MODELS}")

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    # Use memory storage for testing to avoid external dependencies
    RATE_LIMIT_STORAGE_URL = 'memory://'
    ORIGIN_PROTECTION_ENABLED = False
    
    def __init__(self):
        super().__init__()
        # High limits for testing
        self.RATE_LIMIT_PER_MINUTE = '1000'
        self.RATE_LIMIT_PER_HOUR = '10000'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'docker': DockerProductionConfig,  # Explicit Docker configuration
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on FLASK_ENV environment variable."""
    config_name = os.getenv('FLASK_ENV', 'default')
    return config[config_name]()
