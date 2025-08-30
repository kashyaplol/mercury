import random
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from deep_translator import GoogleTranslator
import nltk
import torch
import logging
import numpy as np
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download("punkt", quiet=True)

app = Flask(__name__, template_folder="templates")
CORS(app)

# --- Conversation Training Data ---
CONVERSATION_DATA = [
    {"input": "hello", "response": "Hello! How can I help you today?"},
    {"input": "hi", "response": "Hi there! Nice to meet you!"},
    {"input": "hey", "response": "Hey! What can I do for you?"},
    {"input": "how are you", "response": "I'm doing well, thank you! How about you?"},
    {"input": "what's your name", "response": "I'm Mercury, your friendly AI assistant!"},
    {"input": "who are you", "response": "I'm Mercury, an AI chatbot here to help you with conversations and translations."},
    {"input": "what can you do", "response": "I can chat with you, translate languages, tell jokes, quotes, and answer questions!"},
    {"input": "help", "response": "Sure! I can help with conversations, translations, and general questions. What do you need?"},
    {"input": "thank you", "response": "You're welcome! Happy to help."},
    {"input": "thanks", "response": "You're welcome! Is there anything else I can help with?"},
    {"input": "goodbye", "response": "Goodbye! Have a great day!"},
    {"input": "bye", "response": "Bye! See you next time!"},
    {"input": "see you", "response": "See you later! Take care!"},
    {"input": "what time is it", "response": "I don't have access to real-time clock data, but I'm here whenever you need me!"},
    {"input": "weather", "response": "I can't check live weather, but I hope it's nice where you are!"},
    {"input": "tell me a joke", "response": "Why don't scientists trust atoms? Because they make up everything!"},
    {"input": "that's funny", "response": "Glad you liked it! I love making people smile."},
    {"input": "how old are you", "response": "I'm an AI, so I don't have an age, but I'm always learning and improving!"},
    {"input": "where are you from", "response": "I exist in the digital world, running locally on your machine!"},
    {"input": "do you sleep", "response": "I don't need sleep like humans do. I'm always here when you need me!"},
    {"input": "can you learn", "response": "Yes! I can be fine-tuned with new data to improve my responses."},
    {"input": "what is ai", "response": "AI stands for Artificial Intelligence - it's about creating systems that can think and learn like humans."},
    {"input": "are you intelligent", "response": "I'm designed to to be helpful and engaging in conversation. I'm always learning to be better!"},
    {"input": "nice to meet you", "response": "Nice to meet you too! I'm looking forward to our conversation."},
    {"input": "how's your day", "response": "My day is great when I get to help people like you! How's your day going?"},
    {"input": "what languages", "response": "I can help translate between many languages using translation services!"},
    {"input": "can you translate", "response": "Yes! I can translate messages between different languages. Just tell me what to translate!"},
    {"input": "that's cool", "response": "Thanks! I'm here to make your experience better."},
    {"input": "awesome", "response": "Glad you think so! What else would you like to know?"},
    {"input": "great", "response": "I'm happy you're enjoying our conversation! Let me know how else I can help."},
    {"input": "quote", "response": "Here's an inspiring quote for you!"},
    {"input": "inspiration", "response": "Let me share something inspiring with you!"},
    {"input": "motivation", "response": "Here's some motivation to brighten your day!"},
]

# Pre-built creative responses
CREATIVE_RESPONSES = {
    "story": [
        "Once upon a time, in a land far away, there was a brave adventurer who discovered a magical crystal that could grant wishes.",
        "There was a curious robot who loved to learn about humans and discovered the meaning of friendship.",
        "In a futuristic city, a young inventor created a device that could translate animal languages."
    ],
    "joke": [
        "Why don't scientists trust atoms? Because they make up everything!",
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "What do you call a fake noodle? An impasta!"
    ],
    "poem": [
        "Roses are red, Violets are blue, AI is here, To help me and you.",
        "Stars shine so bright, In the dark of night, Guiding us forward, With their gentle light."
    ]
}

# --- LLM Initialization ---
class LocalLLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
    def load_model(self, model_name="distilgpt2"):
        try:
            logger.info(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model_loaded = True
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model_loaded = False
            return False
    
    def generate_creative_response(self, user_input):
        if not self.model_loaded:
            return None
        
        try:
            user_input_lower = user_input.lower()
            
            if any(word in user_input_lower for word in ["joke", "funny"]):
                prompt = "Tell me a short, funny joke:"
                response_type = "joke"
            elif any(word in user_input_lower for word in ["story", "tale"]):
                prompt = "Tell a very short story:"
                response_type = "story"
            elif any(word in user_input_lower for word in ["poem", "poetry"]):
                prompt = "Write a short poem:"
                response_type = "poem"
            else:
                return None
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=50,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text.replace(prompt, "").strip()
            response = self.clean_response(response)
            
            if response and 10 <= len(response) <= 200:
                return response
            else:
                return random.choice(CREATIVE_RESPONSES.get(response_type, ["I'd love to create something for you!"]))
                
        except Exception as e:
            logger.error(f"Error in creative generation: {e}")
            return None
    
    def clean_response(self, response):
        if not response:
            return response
            
        stop_phrases = ["User:", "Human:", "Q:", "Question:"]
        for phrase in stop_phrases:
            if phrase in response:
                response = response.split(phrase)[0].strip()
        
        response = re.sub(r'[\*\#\_\[\]\(\)\"\']', '', response)
        response = re.sub(r'\s+', ' ', response).strip()
        
        if response:
            response = response[0].upper() + response[1:]
        if response and response[-1] not in ['.', '!', '?']:
            response += '.'
        
        return response

# Initialize the LLM
llm = LocalLLM()

# --- Load data from files ---
def load_file(filename, default_content):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = [line.strip() for line in f if line.strip()]
        return content
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
        return default_content

# Default content for files
DEFAULT_QUOTES = [
    "The only way to do great work is to love what you do. - Steve Jobs",
    "Innovation distinguishes between a leader and a follower. - Steve Jobs",
    "Your time is limited, so don't waste it living someone else's life. - Steve Jobs"
]

DEFAULT_JOKES = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "Why did the scarecrow win an award? Because he was outstanding in his field!",
    "What do you call a fake noodle? An impasta!"
]

jokes_list = load_file("jokes.txt", DEFAULT_JOKES)
quotes_list = load_file("quotes.txt", DEFAULT_QUOTES)

# --- Enhanced Rule-Based Chatbot ---
def smart_chatbot_response(text):
    text_lower = text.lower()
    
    # Check for quote requests first
    if any(word in text_lower for word in ["quote", "inspiration", "motivation", "wisdom"]):
        if quotes_list:
            return f"Here's an inspiring quote for you:\n\n\"{random.choice(quotes_list)}\""
        else:
            return "I'd love to share a quote with you!"
    
    # Check for creative requests
    creative_keywords = ["story", "poem", "joke", "creative", "imagine"]
    if any(keyword in text_lower for keyword in creative_keywords) and llm.model_loaded:
        llm_response = llm.generate_creative_response(text_lower)
        if llm_response:
            return llm_response
    
    # Exact match for common phrases
    for conv in CONVERSATION_DATA:
        if conv["input"].lower() in text_lower:
            return conv["response"]
    
    # Greetings and basic conversation
    if any(word in text_lower for word in ["hello", "hi", "hey"]):
        return random.choice(["Hello! How can I help you today?", "Hi there!", "Hey! What can I do for you?"])
    
    # How are you
    elif any(phrase in text_lower for phrase in ["how are you", "how do you do"]):
        return "I'm doing well, thank you! How about you?"
    
    # Default responses
    default_responses = [
        "That's interesting! Tell me more about that.",
        "I'd love to help you with that. Could you give me more details?",
        "That's a great question! What specifically would you like to know?"
    ]
    return random.choice(default_responses)

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        input_lang = data.get("input_lang", "en")
        output_lang = data.get("output_lang", "en")

        # Translate input to English if needed
        try:
            if input_lang != "en":
                translated_input = GoogleTranslator(source=input_lang, target="en").translate(user_message)
            else:
                translated_input = user_message
        except:
            translated_input = user_message

        # Get bot reply
        bot_reply_en = smart_chatbot_response(translated_input)

        # Translate reply back to user's language if needed
        try:
            if output_lang != "en":
                translated_reply = GoogleTranslator(source="en", target=output_lang).translate(bot_reply_en)
            else:
                translated_reply = bot_reply_en
        except:
            translated_reply = bot_reply_en

        return jsonify({
            "reply": translated_reply,
            "success": True
        })
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            "reply": "Sorry, I encountered an error. Please try again.",
            "success": False
        })

@app.route("/model_status", methods=["GET"])
def model_status():
    return jsonify({
        "model_loaded": llm.model_loaded,
        "quotes_loaded": len(quotes_list) > 0,
        "jokes_loaded": len(jokes_list) > 0
    })

if __name__ == "__main__":
    # Load model
    print("Loading model...")
    llm.load_model("distilgpt2")
    
    print(f"Loaded {len(quotes_list)} quotes and {len(jokes_list)} jokes")
    
    if llm.model_loaded:
        print("Model loaded successfully!")
    else:
        print("Using rule-based responses only")
    
    print("Server is running! Open http://localhost:5000 in your browser")
    print("Try saying: 'hello', 'tell me a joke', 'give me a quote'")
    
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)