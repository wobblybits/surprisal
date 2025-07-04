from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/process/')
def backend():
    text = request.args.get('text')
    return "{\"received\": " + text + ", \"response\": \"Hello, World!\"}";

@app.route('/')
def frontend():
    return render_template('index.html')