import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
PROVIDER = os.getenv("PROVIDER", "groq").strip().lower()
BASE_URL = "https://api.groq.com/openai/v1"
API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")
if not API_KEY:
    raise SystemExit("No API key found. Check your .env file.")
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def banner(name):
    print(f"\n=== {name} | model: {MODEL} ===\n")