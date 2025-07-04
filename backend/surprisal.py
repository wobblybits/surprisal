from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    text = request.args.get('text')
    return "{\"received\": " + text + ", \"response\": \"Hello, World!\"}";

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 