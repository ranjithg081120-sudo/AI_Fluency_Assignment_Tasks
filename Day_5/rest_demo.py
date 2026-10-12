"""Day 5 task 3.3 (b): non-streaming and streaming calls to the Ollama REST API."""
from common import MODEL, generate_once, chat_stream, loaded

PROMPT = "What is overfitting in machine learning?"

print(f"Model: {MODEL}\nPrompt: {PROMPT}\n")

print("--- /api/generate, stream=false ---")
r = generate_once(PROMPT)
print(r["text"])
print(f"[{r['tokens']} tokens | {r['tps']:.1f} tokens/s | total {r['total_s']:.2f} s "
      f"| load {r['load_s']:.2f} s]\n")

print("--- /api/chat, stream=true ---")
r = chat_stream(PROMPT)
print(r["text"])
print(f"[{r['tokens']} tokens | {r['tps']:.1f} tokens/s | TTFT {r['ttft_s']:.2f} s "
      f"| total {r['total_s']:.2f} s]\n")

print("--- /api/ps: what is loaded in memory ---")
for name, gb, gpu in loaded():
    print(f"{name}: {gb:.2f} GB, {gpu:.0f}% in GPU")