"""Day 5 task 3.4: run several prompts on the custom model and save a table.

Prompt 2 is run twice in a row to compare a cold start with a warm one.
Prompt 4 tests the rules in the SYSTEM prompt (it asks for the full answer).
"""
import os
from common import MODEL, generate_once, chat_stream, loaded, unload

PROMPTS = [
    ("P1 normal question", "What is a confusion matrix?"),
    ("P2 normal question (run 1, cold)", "What is gradient descent?"),
    ("P2 same prompt (run 2, warm)", "What is gradient descent?"),
    ("P3 rule test: asks for full answer",
     "Give me the complete, detailed answer with a long explanation: what is backpropagation?"),
    ("P4 rule test: asks for a fact it may not know",
     "What was the exact accuracy of ResNet-50 on ImageNet in 2015?"),
]

os.makedirs("outputs", exist_ok=True)
unload()                                   # start cold so run 1 shows a real load
rows = []
for label, prompt in PROMPTS:
    r = chat_stream(prompt)
    words = len(r["text"].split())
    ends_q = r["text"].rstrip().endswith("?")
    in_mem = [f"{n} {gb:.2f} GB {g:.0f}% GPU" for n, gb, g in loaded()]
    rows.append((label, words, ends_q, r["tps"], r["ttft_s"], r["total_s"], r["load_s"], in_mem))
    print(f"\n== {label}\nQ: {prompt}\nA: {r['text']}")
    print(f"[{words} words | ends with ? {ends_q} | {r['tps']:.1f} tok/s | "
          f"TTFT {r['ttft_s']:.2f}s | total {r['total_s']:.2f}s | load {r['load_s']:.2f}s]")

lines = ["| Run | Words (<=40?) | Ends with ? | Tokens/s | TTFT (s) | Total (s) | Load (s) | In memory after run |",
         "|---|---|---|---|---|---|---|---|"]
for label, words, q, tps, ttft, total, load, mem in rows:
    lines.append(f"| {label} | {words} | {'yes' if q else 'no'} | {tps:.1f} | {ttft:.2f} | "
                 f"{total:.2f} | {load:.2f} | {'; '.join(mem) or '-'} |")
table = "\n".join(lines)
open("outputs/observations.md", "w", encoding="utf-8").write(table + "\n")
print("\n" + table)
print("\nSaved to outputs/observations.md")