from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/process/', methods=['POST'])
def backend():
    text = request.args.get('text', "No text provided")
    return "{\"received\": " + text + ", \"response\": \"Hello, World!\"}";

@app.route('/')
def frontend():
    return render_template('index.html')