import google.generativeai as genai
import os

# Read from .env manually if not in environment
if "GEMINI_API_KEY" not in os.environ and os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "=" in line:
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip().strip("'\"")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
