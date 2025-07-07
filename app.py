from flask import Flask, request, render_template
import json
import surprisal

app = Flask(__name__)

counter = 0
g = surprisal.AutoHuggingFaceModel.from_pretrained(model_id="gpt2")

@app.route('/process/', methods=['POST','GET'])
def backend():
    text = request.args.get('text', "No text provided")
    surps = [*g.surprise([text])]
    return json.dumps({
        surprisals: [*surps[0].surprisals],
        tokens: [*surps[0].tokens],
    })

@app.route('/')
def frontend():
    return render_template('index.html')

@app.route('/test/')
def test():
    g = surprisal.AutoHuggingFaceModel.from_pretrained(model_id="gpt2")
    surps = [*g.surprise(["Hello, World!"])]
    return json.dumps({"surprisals": surps[0].surprisals, "tokens": surps[0].tokens})