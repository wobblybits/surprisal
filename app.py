from flask import Flask, request, render_template
import json
import surprisal
import wordfreq
app = Flask(__name__)

counter = 0
g = surprisal.AutoHuggingFaceModel.from_pretrained(model_id="gpt2")

@app.route('/process/', methods=['POST','GET'])
def backend():
    text = request.args.get('text', "No text provided")
    surps = [*g.surprise([text])]
    surprisals = [round(i) for i in [*surps[0].surprisals]]

    tokens = [*surps[0].tokens]
    tokens_clean = [i.lstrip("Ġ") for i in tokens]

    lengths = [len(i) for i in tokens_clean]

    frequencies = [wordfreq.zipf_frequency(i, "en") for i in tokens_clean]
    # setting minimum frequency value at 0.5 to avoid silent notes
    frequencies_inverted = [max((8 - i), 0.5) for i in frequencies]
    return json.dumps({"surprisals": surprisals,
                    "tokens": tokens_clean,
                    "lengths": lengths,
                    "frequencies": frequencies,
                    "frequencies_inverted": frequencies_inverted})
    

@app.route('/')
def frontend():
    return render_template('index.html')

@app.route('/test/')
def test():
    g = surprisal.AutoHuggingFaceModel.from_pretrained(model_id="gpt2")
    surps = [*g.surprise(["Hello, World!"])]
    return json.dumps({"surprisals": surps[0].surprisals, "tokens": surps[0].tokens})

if __name__ == "__main__":
    app.run(debug=True)