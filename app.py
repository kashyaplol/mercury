from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from googletrans import Translator
import nltk

# Ensure punkt is available for tokenization
nltk.download("punkt", quiet=True)

app = Flask(__name__, template_folder="templates")
CORS(app)  # allow frontend requests

translator = Translator()

# === Simple chatbot logic ===
def simple_chatbot_response(text):
    tokens = nltk.word_tokenize(text.lower())
    if "hello" in tokens or "hi" in tokens:
        return "Hello! How can I help you today?"
    elif "help" in tokens:
        return "Sure, tell me what you need help with."
    elif "language" in tokens or "translate" in tokens:
        return "I can translate messages between different languages for you."
    else:
        return "I'm not sure about that, but I'm learning every day!"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    input_lang = data.get("input_lang", "en")
    output_lang = data.get("output_lang", "en")

    # Translate user input to English
    translated_input = translator.translate(user_message, src=input_lang, dest="en").text

    # Get chatbot reply in English
    bot_reply_en = simple_chatbot_response(translated_input)

    # Translate bot reply to desired language
    translated_reply = translator.translate(bot_reply_en, src="en", dest=output_lang).text

    return jsonify({"reply": translated_reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
