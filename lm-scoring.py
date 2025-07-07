"""
a test script to test MWEs in the `surprisal` module
"""
from matplotlib import pyplot as plt

import surprisal

g = surprisal.AutoHuggingFaceModel.from_pretrained(model_id="gpt2")
# b = surprisal.AutoHuggingFaceModel.from_pretrained(model_id="bert-base-uncased")


stims = [
    "I am a cat on the mat",
    "The cat sat on the mat.",
    "I was wearing a purple cloud.",
    "It was the best of times, it was the worst of times."
]

surps = [*g.surprise(stims)]
local_variances = []
UID_values = []
for i in range(len(stims)):
    sentence = surps[i]
    # TODO: for some reason, length of the surps[i] vector matches length of longest sentence in stimulus set.
    sent_len = len([x for x in sentence.tokens if x != '<|endoftext|>'])

    print("------------------")
    print("sentence: ", sentence)

    local_var = []
    print(sentence.tokens[:sent_len])
    print(sentence.surprisals[:sent_len])


    for word in range(1, sent_len):
        local_var.append((sentence.surprisals[word] - sentence.surprisals[word - 1]) ** 2)
    
    uid = sum(local_var) / (sent_len - 1)

    UID_values.append(uid)
    local_variances.append(local_var)

    print("UID: ", uid)
    print("local variances: ", local_var)
