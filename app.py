from flask import Flask, request, render_template, send_file, make_response
import os
from werkzeug.utils import secure_filename
import json
import wordfreq
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch.nn.functional as F
import math
app = Flask(__name__)

counter = 0

models = {
    "gpt2": {
        "tokenizer": AutoTokenizer.from_pretrained("gpt2"),
        "model": AutoModelForCausalLM.from_pretrained("gpt2"),
        "type": "causal",
        "whitespace": 'Ġ'
    },
    # "r1-1776": {
    #     "tokenizer": AutoTokenizer.from_pretrained("perplexity-ai/r1-1776"),
    #     "model": AutoModelForCausalLM.from_pretrained("perplexity-ai/r1-1776")
    # },
    "smolLM-135M": {
        "tokenizer": AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M"),
        "model": AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M"),
        "type": "causal",
        "whitespace": 'Ġ'
    },
    "nano-mistral": {
        "tokenizer": AutoTokenizer.from_pretrained("crumb/nano-mistral"),
        "model": AutoModelForCausalLM.from_pretrained("crumb/nano-mistral"),
        "type": "causal",
        "whitespace": ' '  # Regular space
    },
    "smol_llama-101M": {
        "tokenizer": AutoTokenizer.from_pretrained("BEE-spoke-data/smol_llama-101M-GQA"),
        "model": AutoModelForCausalLM.from_pretrained("BEE-spoke-data/smol_llama-101M-GQA"),
        "type": "causal",
        "whitespace": ' '  # Regular space
    }, 
    "qwen-2.5-0.5b": {
        "tokenizer": AutoTokenizer.from_pretrained("KingNish/Qwen2.5-0.5b-Test-ft"),
        "model": AutoModelForCausalLM.from_pretrained("KingNish/Qwen2.5-0.5b-Test-ft"),
        "type": "causal",
        "whitespace": 'Ġ'
    },
    "flan-t5-small": {
        "tokenizer": AutoTokenizer.from_pretrained("google/flan-t5-small"),
        "model": AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small"),
        "type": "seq2seq",
        "whitespace": ' '  # Regular space
    }
}

current_model = "gpt2"
tokenizer = models[current_model]["tokenizer"]
model = models[current_model]["model"]
model.eval()


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

@app.route('/process/', methods=['POST'])
def backend():
    # Get text from POST request body
    if request.is_json:
        data = request.get_json()
        text = data.get('text', "No text provided")
    else:
        text = request.form.get('text', "No text provided")

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    # Handle different model types
    if models[current_model]["type"] == "seq2seq":
        # For T5 models, we need to use the encoder-decoder approach
        # We'll use the input as both encoder and decoder input for surprisal calculation
        decoder_input_ids = input_ids.clone()
        
        with torch.no_grad():
            # For T5, we'll use a simpler approach without past_key_values
            outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=decoder_input_ids, use_cache=False)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = decoder_input_ids[:, 1:]
    else:
        # For causal models (GPT-2, etc.)
        labels = input_ids.clone()
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    surprisals = -token_log_probs[0] / math.log(2)
    surprisals = [0] + surprisals.tolist() # padding since first token has no surprisal info

    converted_pitches = convert_to_scale(surprisals)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])[:]
    tokens = process_tokens_for_display(tokens, models[current_model]["whitespace"])
    
    lengths = [len(i) for i in tokens]

    frequencies = [wordfreq.zipf_frequency(i, "en") for i in tokens]
    # setting minimum frequency value at 0.5 to avoid silent notes
    frequencies_inverted = [max((8 - i), 0.5) for i in frequencies]
    return json.dumps({"surprisals": surprisals,
                    "scale_pitches": converted_pitches,
                    "tokens": tokens,
                    "lengths": lengths,
                    "frequencies": frequencies,
                    "frequencies_inverted": frequencies_inverted})
    

@app.route('/')
def frontend():
    return render_template('wireframe.html')

@app.route('/model/<string:model_name>', methods=['GET'])
def change_model(model_name):
    global current_model
    global tokenizer
    global model
    current_model = model_name
    tokenizer = models[current_model]["tokenizer"]
    model = models[current_model]["model"]
    model.eval()
    return json.dumps({"message": f"Model changed to {model_name}"})

@app.route('/assets/<string:filename>', methods=['GET'])
def assets(filename):
    try:
        filename = secure_filename(filename)  # Sanitize the filename
        file_path = os.path.join('assets', filename)
        if os.path.isfile(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return make_response(f"File '{filename}' not found.", 404)
    except Exception as e:
        return make_response(f"Error: {str(e)}", 500)

@app.route('/test/')
def test():
    g = surprisal.AutoHuggingFaceModel.from_pretrained(model_id="gpt2")
    surps = [*g.surprise(["Hello, World!"])]
    return json.dumps({"surprisals": surps[0].surprisals, "tokens": surps[0].tokens})


@app.route('/reverse/', methods=['POST']) 
def music2text():
    # Get text and note from POST request body
    if request.is_json:
        data = request.get_json()
        text = data.get('text', "No text provided")
        scale_pitch = data.get('note', "No note provided")
    else:
        text = request.form.get('text', "No text provided")
        scale_pitch = request.form.get('note', "No note provided")
    
    try:
        target_pitch = int(scale_pitch)
    except ValueError:
        return json.dumps({"error": "Invalid scale pitch provided"})
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    # Get model predictions for the next token
    with torch.no_grad():
        if models[current_model]["type"] == "seq2seq":
            # For T5 models, use encoder-decoder approach without caching
            decoder_input_ids = input_ids.clone()
            outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, use_cache=False)
        else:
            # For causal models
            outputs = model(input_ids=input_ids)
    
    logits = outputs.logits
    
    # Get the logits for the next token position
    next_token_logits = logits[0, -1, :]  # Shape: [vocab_size]
    
    # Calculate surprisals for all possible next tokens
    log_probs = F.log_softmax(next_token_logits, dim=-1)
    surprisals = -log_probs / math.log(2)  # Shape: [vocab_size]
    
    # Find which surprisal value would produce the target pitch
    # From convert_to_scale: pitch = minor_intervals[int(surprisal/2)]
    # So: surprisal = 2 * index where minor_intervals[index] == target_pitch
    # minor_intervals = [0,2,3,5,7,8,10,12,14,15,17,18,19,20,22]
    # scale_intervals = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    
    # Find the index in minor_intervals that matches target_pitch
    target_surprisal = target_pitch * 2

    # Find tokens that have surprisal close to the target
    surprisal_diff = torch.abs(surprisals - target_surprisal)
    best_token_indices = torch.topk(surprisal_diff, k=10, largest=False).indices
    
    # Get the best matching token
    best_token_id = best_token_indices[0].item()
    # Use convert_ids_to_tokens instead of decode to preserve whitespace characters
    best_token_raw = tokenizer.convert_ids_to_tokens([best_token_id])[0]
    best_token = process_tokens_for_display([best_token_raw], models[current_model]["whitespace"])[0]
    best_surprisal = surprisals[best_token_id].item()
    
    # Calculate what pitch this token would actually produce
    actual_pitch = int(best_surprisal/2)
    
    # Get top 5 candidates for variety
    candidates = []
    for i in range(min(5, len(best_token_indices))):
        token_id = best_token_indices[i].item()
        token_raw = tokenizer.convert_ids_to_tokens([token_id])[0]
        token = process_tokens_for_display([token_raw], models[current_model]["whitespace"])[0]
        token_surprisal = surprisals[token_id].item()
        token_pitch = int(token_surprisal/2)
        candidates.append({
            "token": token,
            "surprisal": token_surprisal,
            "pitch": token_pitch
        })
    
    return json.dumps({
        "input_text": text,
        "target_pitch": target_pitch,
        "best_token": best_token,
        "best_surprisal": best_surprisal,
        "actual_pitch": actual_pitch,
        "candidates": candidates
    })

def convert_to_scale(surprisals_list):
    # rounds surprisal values and converts them to intervals of a scale
    converted_pitches = []
    scale = "minor"
    # major_intervals = [0, 2, 4,5,7,9,11,12,14,16,17]
    # minor_intervals = [0,2,3,5,7,8,10,12,14,15,17,18,19,20,22]
    if scale == "minor":
        for i in range(len(surprisals_list)):
            # arbitrarily squished surprisal values in half to make it sound better
            converted_pitches.append(int(i/2))

    return converted_pitches

@app.route('/debug_tokens/<string:model_name>', methods=['GET'])
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


if __name__ == "__main__":
    app.run(debug=True)