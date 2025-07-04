from flask import Flask, request, render_template
import json

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