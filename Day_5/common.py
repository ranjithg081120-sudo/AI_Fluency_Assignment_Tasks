"""Shared helpers: talk to the Ollama REST API with plain `requests`."""
import json
import time
import requests

BASE = "http://localhost:11434"
MODEL = "study-buddy"          # the custom model built from the Modelfile


def generate_once(prompt, model=MODEL, system=None):
    """POST /api/generate, stream=false. Returns a dict of measurements."""
    body = {"model": model, "prompt": prompt, "stream": False}
    if system:
        body["system"] = system
    start = time.time()
    r = requests.post(f"{BASE}/api/generate", json=body, timeout=600).json()
    total = time.time() - start
    tokens = r.get("eval_count", 0)
    eval_s = r.get("eval_duration", 0) / 1e9
    return {
        "text": r.get("response", "").strip(),
        "tokens": tokens,
        "tps": tokens / eval_s if eval_s else 0.0,
        "load_s": r.get("load_duration", 0) / 1e9,
        "total_s": total,
    }


def chat_stream(prompt, model=MODEL, system=None):
    """POST /api/chat, stream=true. Measures TTFT, total time and tokens/s."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    start = time.time()
    first = None
    pieces = []
    final = {}
    with requests.post(f"{BASE}/api/chat",
                       json={"model": model, "messages": messages, "stream": True},
                       stream=True, timeout=600) as r:
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            piece = chunk.get("message", {}).get("content", "")
            if piece and first is None:
                first = time.time() - start
            pieces.append(piece)
            if chunk.get("done"):
                final = chunk
    total = time.time() - start
    tokens = final.get("eval_count", 0)
    eval_s = final.get("eval_duration", 0) / 1e9
    return {
        "text": "".join(pieces).strip(),
        "tokens": tokens,
        "tps": tokens / eval_s if eval_s else 0.0,
        "ttft_s": first if first is not None else total,
        "load_s": final.get("load_duration", 0) / 1e9,
        "total_s": total,
    }


def loaded():
    """GET /api/ps: what is in memory right now. Returns list of (name, GB, gpu%)."""
    out = []
    for m in requests.get(f"{BASE}/api/ps", timeout=30).json().get("models", []):
        size = m.get("size", 0)
        vram = m.get("size_vram", 0)
        out.append((m["name"], size / 1e9, 100 * vram / size if size else 0))
    return out


def unload(model=MODEL):
    """keep_alive=0 tells Ollama to drop the model from memory now."""
    requests.post(f"{BASE}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)