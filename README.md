# Surprisal Calculator WM7±2

> Converting the surprisingness of language into musical compositions

**Live Demo**: https://surprisal.onrender.com  
**API**: https://surprisal.onrender.com/process/

An innovative web application that bridges linguistics and music by calculating text "surprisal" values using large language models and translating them into musical compositions. Built by [David Feil](https://github.com/wobblybits) and [Elise Kim](https://github.com/elisekim) during a batch at the [Recurse Center](https://www.recurse.com/).

![Surprisal Calculator Interface](https://via.placeholder.com/800x400?text=Surprisal+Calculator+Screenshot)

## 🎵 What is Surprisal?

**Surprisal theory** suggests that the more surprising a word is in context, the longer it takes the human brain to process. Consider these sentences:

- "The man fed the **cat** some tuna." *(low surprisal)*
- "The lawyer presented the **cat** with a lawsuit." *(high surprisal)*

The word "cat" is far more surprising in the legal context! This "surprisingness" can be quantified using Claude Shannon's information theory formula:

```Surprisal(x) = -log₂ P(x | context)```

## 🎹 How It Works

1. **Text → Music**: Input text → Calculate word surprisal → Map to musical pitches → Generate melody
2. **Music → Text**: Play musical notes → Find words with matching surprisal values → Generate coherent text

The application uses multiple language models (GPT-2, SmolLM, Qwen, etc.) to calculate probability distributions and supports various musical scales and instruments.

## ✨ Features

### Core Functionality
- **Multi-Model Support**: 6 different language models for surprisal calculation
- **Musical Mapping**: Convert surprisal values to multiple musical scales
- **Reverse Generation**: Generate text from musical input
- **Real-time Audio**: Play compositions with realistic instrument sounds

### Technical Features
- **Production Ready**: Rate limiting, CSRF protection, health monitoring
- **Configurable**: Environment-based configuration for all settings
- **Scalable**: Redis-backed rate limiting, containerized deployment
- **Accessible**: Screen reader support, keyboard navigation
- **Developer Friendly**: Comprehensive API documentation, Docker setup

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js (for development tools, optional)
- Redis (optional, falls back to memory-based rate limiting)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/wobblybits/surprisal.git
cd surprisal

# Run setup script
chmod +x setup-dev.sh
./setup-dev.sh

# Start development server
./run-dev.sh
```

Visit http://localhost:8001 to use the application.

### Docker Setup (Recommended for Production)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

## 🛠️ Configuration

Configure the application using environment variables. Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

### Key Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_SECRET_KEY` | Secret key for sessions (required in production) | - |
| `MAX_TEXT_LENGTH` | Maximum input text length | 1000 |
| `RATE_LIMIT_PER_MINUTE` | API requests per minute | 10 |
| `RATE_LIMIT_PER_HOUR` | API requests per hour | 100 |
| `RATE_LIMIT_STORAGE_URL` | Redis URL for rate limiting | `memory://` |
| `CSRF_ENABLED` | Enable CSRF protection | `True` |

See `.env.example` for all available options.

## 📖 API Documentation

The application provides a RESTful API for programmatic access:

### Core Endpoints

- `POST /process/` - Convert text to surprisal values
- `POST /reverse/` - Generate text from musical notes  
- `GET /health` - Application health and status
- `GET /debug_tokens/{model}` - Debug tokenization

### Example API Usage

```bash
# Convert text to musical data
curl -X POST http://localhost:8001/process/ \
  -H "Content-Type: application/json" \
  -d '{"text": "The cat sat on the mat", "model": "gpt2"}'

# Generate text from musical note
curl -X POST http://localhost:8001/reverse/ \
  -H "Content-Type: application/json" \
  -d '{"text": "The cat", "note": 5, "model": "gpt2"}'
```

**Full API Documentation**: Open `api-docs.yaml` in [Swagger Editor](https://editor.swagger.io/) for interactive documentation.

## 🏗️ Architecture

### Backend (Python/Flask)
- **Language Models**: Multiple HuggingFace transformers for surprisal calculation
- **Audio Processing**: Tone.js integration for real-time audio synthesis
- **Security**: Rate limiting, CSRF protection, input validation
- **Monitoring**: Health checks, error handling, logging

### Frontend (JavaScript ES6)
- **Modular Design**: Separate configs, utilities, and main application
- **Audio Engine**: Tone.js for synthesis with multiple instrument presets  
- **Real-time UI**: Dynamic keyboard highlighting, visual feedback
- **Accessibility**: Screen reader support, keyboard navigation

### Key Files

├── app.py # Main Flask application
├── assets/js/
│ ├── config.js # Configuration and presets
│ ├── surprisal-app.js # Main application logic
│ └── utilities.js # Helper functions and error handling
├── templates/wireframe.html # Main UI template
├── requirements.txt # Python dependencies
├── Dockerfile # Container configuration
└── docker-compose.yml # Multi-service setup


## 🎼 Musical Features

### Supported Instruments
- **Synthesized**: Piano, Violin, Flute, Theremin, Xylophone
- **Sampled**: Cat sounds (for fun!)

### Musical Scales
- **Heptatonic**: Major 7th, Minor 7th
- **Pentatonic**: Major 5th, Minor 5th  
- **Atonal**: Chromatic, Continuous pitch

### Audio Processing
- **Dynamic Volume**: Based on word frequency
- **Note Duration**: Based on word length
- **Real-time Playback**: Visual highlighting synchronized with audio

## 🔬 Language Models

| Model | Size | Description |
|-------|------|-------------|
| **GPT-2** | 124M | OpenAI's foundational model |
| **SmolLM** | 135M | Hugging Face's optimized small model |
| **Nano Mistral** | 170M | Compact Mistral variant |
| **Qwen 2.5** | 494M | Multilingual question-answering model |
| **Flan T5** | 74M | Google's text-to-text transformer |

Each model has different tokenization and surprisal characteristics, leading to unique musical interpretations.

## 🚦 Development

### Local Development
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start development server
python app.py
```

### Testing
```bash
# Run health check
curl http://localhost:8001/health

# Test text processing
curl -X POST http://localhost:8001/process/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "model": "gpt2"}'
```

### Adding New Models
1. Add model configuration to `app.py`
2. Update frontend model list in `assets/js/config.js`
3. Test tokenization with `/debug_tokens/{model}` endpoint

## 🌐 Deployment

### Production Deployment with Docker
```bash
# Build and start
docker-compose -f docker-compose.yml up -d

# Scale application
docker-compose up --scale app=3

# Monitor logs
docker-compose logs -f
```

### Environment Setup
- Set `FLASK_SECRET_KEY` to a secure random value
- Configure Redis for rate limiting
- Set appropriate rate limits for your use case
- Enable CSRF protection

## 📊 Monitoring

### Health Monitoring
- **Endpoint**: `GET /health`
- **Metrics**: Uptime, model status, configuration
- **Use Cases**: Load balancer health checks, monitoring systems

### Rate Limiting
- **Default**: 10 requests/minute, 100 requests/hour
- **Storage**: Redis (production) or memory (development)
- **Headers**: Rate limit information in response headers

## 🔗 Links and References

### Academic Sources
- [Testing the Predictions of Surprisal Theory in 11 Languages](https://aclanthology.org/2023.tacl-1.82/) (Wilcox et al., TACL 2023)
- [Expectation-based syntactic comprehension](https://doi.org/10.1016/j.cognition.2007.05.006) (Levy, Cognition 2008)
- [A mathematical theory of communication](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x) (Shannon, 1948)

### Models and Assets
- Language models from [Hugging Face](https://huggingface.co/)
- Audio synthesis with [Tone.js](https://tonejs.github.io/)
- Icons from [Flaticon](https://www.flaticon.com/)
- Fonts from [Google Fonts](https://fonts.google.com/) and [Old School PC Fonts](https://int10h.org/oldschool-pc-fonts/)

## 📄 License

This project is open source. Built with ❤️ at the [Recurse Center](https://recurse.com/).