import openai
from openai import OpenAI

client = OpenAI(api_key="OPENAI_API_KEY")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or try other models
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )
    print(f"Model used: {response.model}")
    print("API key is valid and has access to this model")
except Exception as e:
    print(f"Error: {e}")