"""Day 5 task 3.3 (c): call the custom model through the OpenAI-compatible endpoint
and show whether the Modelfile SYSTEM prompt or the program's own prompt wins."""
import requests

URL = "http://localhost:11434/v1/chat/completions"
QUESTION = "Explain what a neural network is."


def ask(system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": QUESTION})
    r = requests.post(URL, json={"model": "study-buddy", "messages": messages}, timeout=600)
    return r.json()["choices"][0]["message"]["content"].strip()


print("1) No system prompt from the program (Modelfile default applies):")
print(ask(), "\n")

print("2) Program sends its own system prompt (pirate):")
print(ask("You are a pirate. Answer in pirate speech, in two sentences."))