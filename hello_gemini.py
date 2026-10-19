import os
from dotenv import load_dotenv
from google import genai

load_dotenv() # reads .env

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

if not api_key:
	raise SystemExit("GEMINI_API_KEY is missing in .env")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
	model=model,
	contents="in one sentence: what is a VAT invoice?"
)
print(f"[{model}]", response.text)
print("\n\n")
print(response)