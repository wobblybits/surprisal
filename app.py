from flask import Flask, request, render_template, send_file, make_response
import os
from werkzeug.utils import secure_filename
import json
import wordfreq
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import math
app = Flask(__name__)

counter = 0

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()




@app.route('/process/', methods=['POST','GET'])
def backend():
    text = request.args.get('text', "No text provided")

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
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
    tokens = [token.replace('Ġ', ' ') for token in tokens]
    
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

def convert_to_scale(surprisals_list):
    # rounds surprisal values and converts them to intervals of a scale
    converted_pitches = []
    scale = "minor"
    major_intervals = [0, 2, 4,5,7,9,11,12,14,16,17]
    minor_intervals = [0,2,3,5,7,8,10,12,14,15,17,18,19,20,22]
    if scale == "minor":
        for i in range(len(surprisals_list)):
            # arbitrarily squished surprisal values in half to make it sound better
            converted_pitches.append(minor_intervals[int(surprisals_list[i]/2)])

    return converted_pitches



if __name__ == "__main__":
    app.run(debug=True)