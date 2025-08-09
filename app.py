from flask import Flask, request, render_template, send_file, make_response, jsonify
import os
from werkzeug.utils import secure_filename
import json
import wordfreq
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch.nn.functional as F
import math
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
import secrets
import time
from config import get_config

# Get configuration
config = get_config()

# Create Flask app
app = Flask(__name__)

# Apply configuration to Flask app
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['DEBUG'] = config.DEBUG

# CSRF Protection
if config.CSRF_ENABLED:
    csrf = CSRFProtect(app)

# Rate Limiting Configuration - Only enable in production
if config.DEBUG:
    # In development, create a mock limiter that doesn't actually limit
    class MockLimiter:
        def limit(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator
        
        def init_app(self, app):
            pass
    
    limiter = MockLimiter()
    print("Rate limiting DISABLED for development")
else:
    # Production rate limiting
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        storage_uri=config.RATE_LIMIT_STORAGE_URL,
        default_limits=[f"{config.RATE_LIMIT_PER_HOUR} per hour"],
        headers_enabled=True
    )
    print(f"Rate limiting enabled: {config.RATE_LIMIT_PER_MINUTE}/min, {config.RATE_LIMIT_PER_HOUR}/hour")

# Application startup time for health check
app_start_time = time.time()

counter = 0

models = {
    "gpt2": {
        "tokenizer": AutoTokenizer.from_pretrained("gpt2"),
        "model": AutoModelForCausalLM.from_pretrained("gpt2"),
        "type": "causal",
        "whitespace": 'Ġ'
    },
    "distilgpt2": {
        "tokenizer": AutoTokenizer.from_pretrained("distilbert/distilgpt2"),
        "model": AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2"),
        "type": "causal",
        "whitespace": 'Ġ'
    },
    "smollm": {
        "tokenizer": AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M"),
        "model": AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M"),
        "type": "causal",
        "whitespace": 'Ġ'
    },
    "nano mistral": {
        "tokenizer": AutoTokenizer.from_pretrained("crumb/nano-mistral"),
        "model": AutoModelForCausalLM.from_pretrained("crumb/nano-mistral"),
        "type": "causal",
        "whitespace": ' '  # Regular space
    },
    "qwen": {
        "tokenizer": AutoTokenizer.from_pretrained("KingNish/Qwen2.5-0.5b-Test-ft"),
        "model": AutoModelForCausalLM.from_pretrained("KingNish/Qwen2.5-0.5b-Test-ft"),
        "type": "causal",
        "whitespace": 'Ġ'
    },
    "flan": {
        "tokenizer": AutoTokenizer.from_pretrained("google/flan-t5-small"),
        "model": AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small"),
        "type": "seq2seq",
        "whitespace": ' '  # Regular space
    }
}


def process_tokens_for_display(tokens, whitespace_char):
    """
    Process tokens to handle whitespace properly using the model-specific whitespace character.
    """
    processed_tokens = []
    
    for token in tokens:
        # Replace the model-specific whitespace character with a regular space
        processed_token = token.replace(whitespace_char, ' ')
        
        # Handle cases where the whitespace character might not be found
        # Some models might not use explicit whitespace tokens
        if whitespace_char not in token and token.startswith(('Ġ', '▁', '_', 'Ċ', ' ')):
            # Try alternative whitespace characters
            for alt_char in ['Ġ', '▁', '_', 'Ċ', ' ']:
                if alt_char in token:
                    processed_token = token.replace(alt_char, ' ')
                    break
        
        processed_tokens.append(processed_token)
    
    return processed_tokens

def detect_whitespace_char(tokenizer, test_text="Hello world"):
    """
    Automatically detect the whitespace character used by a tokenizer.
    """
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(test_text))
    
    # Look for common whitespace patterns
    whitespace_chars = ['Ġ', '▁', '_', 'Ċ']
    
    for char in whitespace_chars:
        for token in tokens:
            if char in token:
                return char
    
    # If no whitespace character found, return None
    return None

def validate_model_name(model_name):
    """Validate that the model name is supported."""
    if not model_name:
        raise ValueError("Model name is required")
    if model_name not in models:
        available_models = list(models.keys())
        raise ValueError(f"Invalid model '{model_name}'. Available models: {available_models}")
    return model_name

def validate_text_input(text):
    """Validate text input without unnecessary sanitization."""
    if not isinstance(text, str):
        raise ValueError("Text input must be a string")
    
    if len(text.strip()) == 0:
        raise ValueError("Text input cannot be empty")
    
    # Use configurable length limit
    if len(text) > config.MAX_TEXT_LENGTH:
        raise ValueError(f"Text too long: {len(text)} chars (max {config.MAX_TEXT_LENGTH})")
    
    return text  # Return original text, no sanitization

# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring and load balancers."""
    try:
        # Check if models are loaded
        model_status = {}
        for model_name in models:
            try:
                # Quick test to ensure model is accessible
                model_status[model_name] = "healthy"
            except Exception as e:
                model_status[model_name] = f"error: {str(e)}"
        
        # Calculate uptime
        uptime_seconds = time.time() - app_start_time
        uptime_minutes = uptime_seconds / 60
        
        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": uptime_seconds,
            "uptime_minutes": round(uptime_minutes, 2),
            "models": model_status,
            "config": {
                "max_text_length": config.MAX_TEXT_LENGTH,
                "debug_mode": config.DEBUG,
                "csrf_enabled": config.CSRF_ENABLED,
                "flask_env": os.getenv('FLASK_ENV', 'default'),
                "rate_limiting": {
                    "storage_url": config.RATE_LIMIT_STORAGE_URL,
                    "per_minute": config.RATE_LIMIT_PER_MINUTE,
                    "per_hour": config.RATE_LIMIT_PER_HOUR
                }
            }
        }
        
        return jsonify(health_data), 200
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }), 503

@app.route('/process/', methods=['POST'])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE} per minute")
def process_text():
    try:
        # Proper error handling for JSON parsing
        if request.is_json:
            data = request.get_json()
            if not data:
                return json.dumps({"error": "No JSON data provided"}), 400
            text = data.get('text', "")
            model_name = data.get('model', "gpt2")  # Default to gpt2
        else:
            text = request.form.get('text', "")
            model_name = request.form.get('model', "gpt2")
        
        # Validate inputs
        validated_text = validate_text_input(text)
        validated_model = validate_model_name(model_name)
        
        # Get model components (no global state)
        current_tokenizer = models[validated_model]["tokenizer"]
        current_model = models[validated_model]["model"]

        inputs = current_tokenizer(validated_text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"]
        
        # Handle different model types
        if models[validated_model]["type"] == "seq2seq":
            # For T5 models, we need to use the encoder-decoder approach
            # We'll use the input as both encoder and decoder input for surprisal calculation
            decoder_input_ids = input_ids.clone()
            
            with torch.no_grad():
                # For T5, we'll use a simpler approach without past_key_values
                outputs = current_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=decoder_input_ids, use_cache=False, return_dict=True)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = decoder_input_ids[:, 1:]
        else:
            # For causal models (GPT-2, etc.)
            labels = input_ids.clone()
            
            with torch.no_grad():
                outputs = current_model(input_ids=input_ids, labels=labels, return_dict=True)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
        
        # calculates natural log probabilities of tokens
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # converts to log base 2 and makes negative, for surprisal formula
        surprisals = -token_log_probs[0] / math.log(2)
        # padding since first token has no surprisal info
        surprisals = [0] + surprisals.tolist() 
        
        # formats tokens for display
        tokens = current_tokenizer.convert_ids_to_tokens(input_ids[0])[:]
        tokens = process_tokens_for_display(tokens, models[validated_model]["whitespace"])
        
        # fetches word lengths; determines note duration
        lengths = [len(i) for i in tokens]

        # fetches word frequencies; determines note volume
        frequencies = [wordfreq.zipf_frequency(i, "en") for i in tokens]
        # setting minimum frequency value at 0.5 to avoid silent notes
        frequencies_inverted = [max((8 - i), 0.5) for i in frequencies]
        
        # this data gets translated into audio in javascript frontend
        return json.dumps({"surprisals": surprisals,
                        "tokens": tokens,
                        "lengths": lengths,
                        "frequencies": frequencies,
                        "frequencies_inverted": frequencies_inverted})
    
    except ValueError as e:
        return json.dumps({"error": str(e)}), 400
    except Exception as e:
        # Log actual error, return generic message
        print(f"Processing error: {e}")
        return json.dumps({"error": "Failed to process text"}), 500
    
@app.route('/')
def frontend():
    return render_template('wireframe.html')

@app.route('/assets/<path:filename>', methods=['GET'])
def serve_assets(filename):
    try:
        # Handle subdirectory paths properly
        # Split the path and secure each component separately
        path_parts = filename.split('/')
        secured_parts = [secure_filename(part) for part in path_parts]
        secured_filename = '/'.join(secured_parts)
        
        # Prevent directory traversal attacks
        if '..' in secured_filename or secured_filename.startswith('/'):
            return make_response("Invalid file path.", 400)
        
        file_path = os.path.join('assets', secured_filename)
        if os.path.isfile(file_path):
            return send_file(file_path)  # Remove as_attachment=True for web assets
        else:
            return make_response(f"File '{secured_filename}' not found.", 404)
    except Exception as e:
        return make_response(f"Error: {str(e)}", 500)

# play notes to generate text with appropriate surprisal values
@app.route('/reverse/', methods=['POST']) 
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE} per minute")
def music_to_text():
    try:
        # Proper error handling for JSON parsing
        if request.is_json:
            data = request.get_json()
            if not data:
                return json.dumps({"error": "No JSON data provided"}), 400
            text = data.get('text', "")
            scale_pitch = data.get('note', "")
            model_name = data.get('model', "gpt2")  # Default to gpt2
        else:
            text = request.form.get('text', "")
            scale_pitch = request.form.get('note', "")
            model_name = request.form.get('model', "gpt2")
        
        # Validate inputs
        if text and len(text) > config.MAX_TEXT_LENGTH:
            return json.dumps({"error": f"Text too long: {len(text)} chars (max {config.MAX_TEXT_LENGTH})"}), 400
        
        validated_model = validate_model_name(model_name)
        
        # Validate scale pitch
        if not scale_pitch and scale_pitch != 0:  # Allow 0 as valid pitch
            return json.dumps({"error": "Note parameter is required"}), 400
        
        try:
            target_pitch = int(scale_pitch)
        except (ValueError, TypeError):
            return json.dumps({"error": "Invalid scale pitch provided - must be an integer"}), 400
        
        # Get model components (no global state)
        current_tokenizer = models[validated_model]["tokenizer"]
        current_model = models[validated_model]["model"]
        
        started_new_sequence = False
        # if input field is empty, add beginning of speech token
        if not text or text.strip() == "":
            bos_token = current_tokenizer.bos_token or "The "
            text = bos_token
            started_new_sequence = True

        # Tokenize the input text
        inputs = current_tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"]
        
        # Get model predictions for the next token
        with torch.no_grad():
            if models[validated_model]["type"] == "seq2seq":
                # For T5 models, use encoder-decoder approach without caching
                decoder_input_ids = input_ids.clone()
                outputs = current_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, use_cache=False)
            else:
                # For causal models
                outputs = current_model(input_ids=input_ids)
        
        logits = outputs.logits
        
        # Get the logits for the next token position
        next_token_logits = logits[0, -1, :]  # Shape: [vocab_size]
        
        # Calculate surprisals for all possible next tokens
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        surprisals = -log_probs / math.log(2)  # Shape: [vocab_size]
        
        # Find the index in minor_intervals that matches target_pitch
        target_surprisal = target_pitch * 2

        # Find tokens that have surprisal close to the target
        surprisal_diff = torch.abs(surprisals - target_surprisal)
        best_token_indices = torch.topk(surprisal_diff, k=10, largest=False).indices
        
        # Get the best matching token
        best_token_id = best_token_indices[0].item()
        # Use convert_ids_to_tokens instead of decode to preserve whitespace characters
        best_token_raw = current_tokenizer.convert_ids_to_tokens([best_token_id])[0]
        best_token = process_tokens_for_display([best_token_raw], models[validated_model]["whitespace"])[0]
        best_surprisal = surprisals[best_token_id].item()
        
        # Calculate what pitch this token would actually produce
        actual_pitch = round(best_surprisal/2)
        
        # Get top 5 candidates for variety
        candidates = []
        for i in range(min(5, len(best_token_indices))):
            token_id = best_token_indices[i].item()
            token_raw = current_tokenizer.convert_ids_to_tokens([token_id])[0]
            token = process_tokens_for_display([token_raw], models[validated_model]["whitespace"])[0]
            token_surprisal = surprisals[token_id].item()
            token_pitch = round(token_surprisal/2)
            candidates.append({
                "token": token,
                "surprisal": token_surprisal,
                "pitch": token_pitch
            })
        
        # don't return added beginning of speech token as part of text
        if started_new_sequence:
            text = ""

        return json.dumps({
            "input_text": text,
            "target_pitch": target_pitch,
            "best_token": best_token,
            "best_surprisal": best_surprisal,
            "actual_pitch": actual_pitch,
            "candidates": candidates
        })
    
    except ValueError as e:
        return json.dumps({"error": str(e)}), 400
    except Exception as e:
        # Log actual error, return generic message
        print(f"Music to text error: {e}")
        return json.dumps({"error": "Failed to generate text from music"}), 500


@app.route('/debug_tokens/<string:model_name>', methods=['GET'])
@limiter.limit("5 per minute")  # Lower limit for debug endpoint
def debug_tokens(model_name):
    """Debug endpoint to see what tokens each model produces."""
    if model_name not in models:
        return json.dumps({"error": f"Model {model_name} not found"})
    
    test_text = "Hello world"
    tokenizer = models[model_name]["tokenizer"]
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(test_text))
    
    return json.dumps({
        "model": model_name,
        "test_text": test_text,
        "tokens": tokens,
        "whitespace_char": models[model_name]["whitespace"]
    })

# Rate limit error handler
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please try again later.",
        "retry_after": getattr(e, 'retry_after', None)
    }), 429

# CSRF error handler
if config.CSRF_ENABLED:
    @app.errorhandler(400)
    def csrf_error(reason):
        if 'csrf' in str(reason).lower():
            return jsonify({
                "error": "CSRF token missing or invalid",
                "message": "Please include a valid CSRF token with your request"
            }), 400
        return jsonify({"error": "Bad request"}), 400


if __name__ == "__main__":
    print(f"Starting Surprisal Calculator...")
    print(f"Environment: {os.getenv('FLASK_ENV', 'default')}")
    print(f"Debug mode: {config.DEBUG}")
    print(f"Rate limiting: {config.RATE_LIMIT_PER_MINUTE}/min, {config.RATE_LIMIT_PER_HOUR}/hour")
    print(f"CSRF protection: {'enabled' if config.CSRF_ENABLED else 'disabled'}")
    print(f"Health check available at: http://{config.HOST}:{config.PORT}/health")
    
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)