import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import json

# Define the models to test
models = {
    "gpt2": {
        "tokenizer": AutoTokenizer.from_pretrained("gpt2"),
        "model": AutoModelForCausalLM.from_pretrained("gpt2"),
        "type": "causal",
        "whitespace": 'Ġ'
    },
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
        "whitespace": ' '
    },
    "smol_llama-101M": {
        "tokenizer": AutoTokenizer.from_pretrained("BEE-spoke-data/smol_llama-101M-GQA"),
        "model": AutoModelForCausalLM.from_pretrained("BEE-spoke-data/smol_llama-101M-GQA"),
        "type": "causal",
        "whitespace": ' '
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
        "whitespace": ' '
    }
}

def test_model_whitespace(model_name, model_config, test_texts):
    """Test how a model handles whitespace during text generation."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    tokenizer = model_config["tokenizer"]
    model = model_config["model"]
    model.eval()
    
    for test_text in test_texts:
        print(f"\nTest text: '{test_text}'")
        print(f"Text length: {len(test_text)}")
        
        # Tokenize the input text
        inputs = tokenizer(test_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        # Get raw tokens
        raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        print(f"Raw tokens: {raw_tokens}")
        
        # Test text generation (similar to music2text)
        with torch.no_grad():
            if model_config["type"] == "seq2seq":
                # For T5 models, use encoder-decoder approach
                decoder_input_ids = input_ids.clone()
                outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, use_cache=False)
            else:
                # For causal models
                outputs = model(input_ids=input_ids)
        
        logits = outputs.logits
        next_token_logits = logits[0, -1, :]  # Shape: [vocab_size]
        
        # Get top 5 predicted tokens
        top_logits, top_indices = torch.topk(next_token_logits, k=5)
        
        print(f"Top 5 predicted tokens:")
        for i, (logit, token_id) in enumerate(zip(top_logits, top_indices)):
            token = tokenizer.decode([token_id])
            print(f"  {i+1}. '{token}' (logit: {logit:.4f})")
        
        # Test whitespace handling
        print(f"\nWhitespace analysis:")
        print(f"  Expected whitespace char: '{model_config['whitespace']}'")
        
        # Check if any tokens contain whitespace characters
        whitespace_chars = ['Ġ', '▁', '_', 'Ċ', ' ']
        found_whitespace = []
        for token in raw_tokens:
            for char in whitespace_chars:
                if char in token:
                    found_whitespace.append((token, char))
        
        if found_whitespace:
            print(f"  Found whitespace in tokens:")
            for token, char in found_whitespace:
                print(f"    '{token}' contains '{char}'")
        else:
            print(f"  No explicit whitespace characters found")
        
        # Test what happens when we add a space
        test_with_space = test_text + " "
        inputs_with_space = tokenizer(test_with_space, return_tensors="pt")
        tokens_with_space = tokenizer.convert_ids_to_tokens(inputs_with_space["input_ids"][0])
        print(f"\nTest with trailing space: '{test_with_space}'")
        print(f"Tokens with space: {tokens_with_space}")

def main():
    """Main function to test all models."""
    test_texts = [
        "Hello",
        "Hello world",
        "Hello world ",
        " Hello world",
        "Hello  world",  # Double space
        "Hello\nworld",   # Newline
        "Hello\tworld",   # Tab
    ]
    
    print("Testing whitespace handling for all models...")
    print("This will help understand how each model tokenizes whitespace.")
    
    for model_name, model_config in models.items():
        try:
            test_model_whitespace(model_name, model_config, test_texts)
        except Exception as e:
            print(f"\nError testing {model_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 