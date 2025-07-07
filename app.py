from flask import Flask, request, render_template
import json
import surprisal

app = Flask(__name__)

counter = 0

@app.route('/process/', methods=['POST','GET'])
def backend():
    text = request.args.get('text', "No text provided")
    global counter
    counter += 1
    json_response = {
        "received": str(text),
        "response": "Hello, World!",
        "counter": counter
    }
    return json.dumps(json_response)

@app.route('/')
def frontend():
    return render_template('index.html')

@app.route('/test/')
def test():
    g = surprisal.AutoHuggingFaceModel.from_pretrained(model_id="gpt2")
    surps = [*g.surprise(["Hello, World!"])]
    return json.dumps(surps)