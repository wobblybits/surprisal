# Surprisal Calculator WM7±2
![https://www.recurse.com/](https://img.shields.io/badge/created%20at-recurse%20center-white)

> Converting the surprisingness of language into musical compositions

**Live Demo**: https://surprisal.onrender.com  

![Surprisal Calculator Interface](./preview/interface.png)

[Surprisal Calculator Demo Video](https://github.com/user-attachments/assets/70932955-181f-4263-9641-2505d6971241)

## 🎵 What is Surprisal?

**Surprisal theory** suggests that the more surprising a word is in context, the longer it takes the human brain to process. Consider these sentences:

- "The man fed the **cat** some tuna." *(low surprisal)*
- "The lawyer presented the **cat** with a lawsuit." *(high surprisal)*

The word "cat" is far more surprising in the legal context! This "surprisingness" can be quantified using Claude Shannon's information theory formula:

```
Surprisal(x) = -log₂ P(x | context)
```

## 🎹 How It Works

1. **Text → Music**: Input text → Calculate word surprisal → Map the numeric values to musical pitches → Generate melody
2. **Music → Text**: Play musical notes → Find words that would have a similar surprisal value in the given context → Generate text

This is meant as a fun experiment to help build intuition about how humans process natural language as well as how LLMs model the compositional features of communication. The surprisal data of a sentence could be abstracted and presented in many different ways, but we thought musical melody would be a form where the abstraction actually uses some similar properties of perception and processing.

The results change on the model used, both due to the underlying tokenization process that each uses as well as the statistical models that develop during their training. Making these differences audible (and interactive) has been a fun way to build new intuition and make the "black box" of the models' inner workings more accessible.

We have chosen to focus on small models, partly to lower the computational overhead required, but also to get a sense of how these little guys are trying to squeeze as much coherence as possible out of their training. The [live](https://surprisal.onrender.com) demo only exposes one model, but cloning the repo and running it locally would allow you to experiment with the other models we have selected or to choose your own!

## 🚀 Quick Start

### Option 1: Local Development

```bash
# Clone and setup
git clone https://github.com/wobblybits/surprisal.git
cd surprisal
pip install -r requirements.txt
python app.py
```

The first time you run it, the transformers library will download and cache the model tensors, which all combined is ~3GB. You can disable certain models in `config.py`. If you want to add your own models, you will need to edit `app.py` to provide the configuration details as well as enable them in `config.py`.

### Repository Structure
```
├── app.py                     # Main Flask application
├── assets/js/
│   ├── config.js             # Configuration and presets
│   ├── surprisal-app.js      # Main application logic
│   └── utilities.js          # Helper functions and error handling
├── templates/wireframe.html   # Main UI template
├── requirements.txt          # Python dependencies
└── .env.example
```

## 🔬 Language Models

| Model | Size | Description |
|-------|------|-------------|
| **GPT-2** | 124M | OpenAI's foundational model |
| **DistilGPT-2** | 88M | A distilled version of GPT2 |
| **SmolLM** | 135M | Hugging Face's optimized small model |
| **Nano Mistral** | 170M | Compact Mistral variant |
| **Qwen 2.5** | 494M | Multilingual question-answering model |
| **Flan T5** | 74M | Google's text-to-text transformer |

Each model has different tokenization and surprisal characteristics, leading to unique musical interpretations.

## 🔗 Links and References

### Academic Sources
- [Testing the Predictions of Surprisal Theory in 11 Languages](https://aclanthology.org/2023.tacl-1.82/) (Wilcox et al., TACL 2023)
- [Expectation-based syntactic comprehension](https://doi.org/10.1016/j.cognition.2007.05.006) (Levy, Cognition 2008)
- [A mathematical theory of communication](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x) (Shannon, 1948)

### Models and Assets
- Language models from [Hugging Face](https://huggingface.co/)
- Audio synthesis with [Tone.js](https://tonejs.github.io/)
- Icons from [Flaticon](https://www.flaticon.com/)
- Fonts from [Google Fonts](https://fonts.google.com/) and [Old School PC Fonts](https://int10h.org/oldschool-pc-fonts/)
- Sound effects from [Pixabay](https://www.pixabay.com/)

More detailed attributions are included at the bottom of the main html file.

## ❤️ Appreciation

Built with ❤️ at the [Recurse Center](https://www.recurse.com/).