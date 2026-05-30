import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Groq
groq_key = os.getenv("GROQ_API_KEY")
if groq_key:
    headers = {"Authorization": f"Bearer {groq_key}"}
    r = requests.get("https://api.groq.com/openai/v1/models", headers=headers)
    if r.status_code == 200:
        models = r.json().get("data", [])
        print("Groq Models:")
        for m in models:
            print("-", m["id"])
    else:
        print("Groq API error:", r.text)

# Gemini
gemini_key = os.getenv("GEMINI_API_KEY_1") or os.getenv("GOOGLE_API_KEY")
if gemini_key:
    r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key}")
    if r.status_code == 200:
        models = r.json().get("models", [])
        print("\nGemini Models:")
        for m in models:
            print("-", m["name"])
    else:
        print("Gemini API error:", r.text)
