from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/process/')
def app():
    text = request.args.get('text')
    return "{\"received\": " + text + ", \"response\": \"Hello, World!\"}";

@app.route('/')
def index():
    return render_template('../frontend/index.html')