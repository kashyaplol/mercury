import random
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from deep_translator import GoogleTranslator
import nltk

nltk.download("punkt", quiet=True)

app = Flask(__name__, template_folder="templates")
CORS(app)

# --- Load jokes from file ---
def load_jokes(filename="jokes.txt"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            jokes = [line.strip() for line in f if line.strip()]
        return jokes
    except Exception as e:
        print("Error loading jokes:", e)
        return []

jokes_list = load_jokes()

# --- Local chatbot responses ---
def local_chatbot_response(text):
    tokens = nltk.word_tokenize(text.lower())

    if "hello" in tokens or "hi" in tokens:
        return "Hello! How can I help you today?"
    elif "help" in tokens:
        return "Sure! You can ask me about programming, languages, or just chat."
    elif "language" in tokens or "translate" in tokens:
        return "I can translate messages between different languages."
    elif "name" in tokens:
        return "My name is Mercury, your friendly chatbot."
    elif "weather" in tokens:
        return "I cannot check live weather, but I hope it's sunny wherever you are!"
    elif "joke" in tokens:
        # Pick a random joke from the text file
        if jokes_list:
            return random.choice(jokes_list)
        else:
            return "I couldn't find any jokes right now."
    elif "bye" in tokens or "goodbye" in tokens:
        return "Goodbye! Have a great day!"
    else:
        return "I'm not sure about that, but I'm learning every day!"

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    input_lang = data.get("input_lang", "en")
    output_lang = data.get("output_lang", "en")

    # Translate input to English
    try:
        translated_input = GoogleTranslator(source=input_lang, target="en").translate(user_message)
    except:
        translated_input = user_message

    # Get bot reply
    bot_reply_en = local_chatbot_response(translated_input)

    # Translate reply back to user's language
    try:
        translated_reply = GoogleTranslator(source="en", target=output_lang).translate(bot_reply_en)
    except:
        translated_reply = bot_reply_en

    return jsonify({"reply": translated_reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
