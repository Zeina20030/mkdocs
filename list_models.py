# list_models.py (or add this temporarily in rag.py)
import os
from google import genai

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Set GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

for m in client.models.list():
    # Only print models that support generateContent
    if "generateContent" in m.supported_actions:
        print(m.name)
