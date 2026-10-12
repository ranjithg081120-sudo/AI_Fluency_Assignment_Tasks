# Day 5 Task: Serving a Study Buddy with Ollama, a Modelfile and the REST API

**Name:** Ranjith | **Date written:** 12 October 2026

## Scenario

I want a small **study buddy** for my AI and data science classes. I want it to behave the same way every time: give one short hint instead of the full answer, use at most 40 words, always end with a question that makes me think, and say so when it is not sure. I built it on `qwen2.5:1.5b` (the small model I chose on Day 4, which fits easily in my 6 GB GPU) and called it `study-buddy`. Prompts used: "What is a confusion matrix?", "What is gradient descent?" (run twice), a rule test that demands a long full answer about backpropagation, and a rule test that asks for an exact accuracy figure the model may not know.

## 3.1 Explanation of concepts

### What is Ollama?

Ollama is a local model server. It has three parts. The **command-line client** (`ollama run`, `ollama pull`, `ollama list`) is only a front end that sends requests. The **background server** listens on `localhost:11434`, exposes the REST API, manages which models are loaded and routes each request. The **inference engine** (a llama.cpp-based runner) does the real work of loading the weights and generating tokens. Model files are stored on disk in the `.ollama/models` folder under my user profile as content-addressed blobs plus small manifest files. When I send a request, the server checks whether the model is already in memory. If it is not, it loads the weights from disk into VRAM or RAM. Then it applies the model's chat template to my messages, the engine generates tokens one at a time, and the server sends them back, either all at once or as a stream. For my study buddy, `ollama run study-buddy "What is overfitting?"` went through exactly this path, and the first call included a load step that the second call did not.

### What is a Modelfile?

A Modelfile is a small recipe that packages behaviour on top of an existing model. I used `FROM qwen2.5:1.5b` for the base, `PARAMETER temperature 0.2` so hints stay consistent and not creative, `PARAMETER num_ctx 4096` because a short study chat does not need more and a smaller window saves KV-cache memory, `PARAMETER repeat_penalty 1.1` to reduce repeated phrases, and a `SYSTEM` prompt with my four rules. A custom model is made of a manifest plus a few tiny layers (the parameters, the system text and the template) that point to the **same weights blob** as the base model. That is why `ollama create` finished instantly and `ollama list` shows `study-buddy` at about the same size as `qwen2.5:1.5b`: no new weights were downloaded or copied.

### Which system prompt wins?

When my program sends its own system prompt to `study-buddy`, **the program's prompt wins** over the one in the Modelfile. In my test (`override_demo.py`), the call without a system prompt followed the study-buddy rules (__ describe what you saw), and the call with the pirate system prompt answered in pirate speech (__ confirm what you saw). This is the right way round for anyone building an agent. The Modelfile is a default that makes the model sensible when nothing else is said, while the program knows the current task and must have the final word. If the Modelfile won, no program could change the behaviour of a packaged model without rebuilding it.

### /api/generate versus /api/chat, and the OpenAI-compatible endpoint

`/api/generate` takes one prompt string (and an optional system string) and returns one completion. It is simple but has no notion of a conversation. `/api/chat` takes a list of messages with roles (system, user, assistant), so the whole history is sent each time and the server applies the chat template, which is what an agent needs. The OpenAI-compatible endpoint, `/v1/chat/completions`, accepts the same request shape as OpenAI's API. It matters because my code only needs a different base URL and model name to move to another server such as vLLM, which exposes the same endpoint. I do not have to rewrite my application when the deployment changes.

### Streaming, TTFT and total time

With streaming, the server sends each token as soon as it is generated instead of waiting for the full answer. **TTFT** (time to first token) measures the wait from sending the request until the first token arrives, and includes any model load and prompt processing. **Total time** measures until the last token. Streaming does not change how long generation takes. The total is the same, but the reader starts reading after the TTFT instead of staring at a blank screen until the end. In my runs the TTFT was __ s against a total of __ s (see the table in 3.4), so the answer felt much faster than the total suggests.

### num_ctx, OLLAMA_KEEP_ALIVE and OLLAMA_NUM_PARALLEL

`num_ctx` sets the context window in tokens, meaning how much conversation the model can hold at once. `OLLAMA_KEEP_ALIVE` controls how long a model stays loaded in memory after its last request (the default is a few minutes). That is why my second run of the same prompt was faster to start: the model was still loaded. `OLLAMA_NUM_PARALLEL` sets how many requests one loaded model handles at the same time. `num_ctx` connects directly to the KV-cache estimate from Day 4: the cache grows with context, so with the Day 4 formula a 1.5B model at 4K needs about 1.5 × 4 × 0.02 = 0.12 GB of cache on top of about 0.85 GB of weights. Parallel requests each need their own context space, so raising `OLLAMA_NUM_PARALLEL` multiplies the KV-cache memory.

### Why simple servers waste KV-cache memory, and PagedAttention

A simple server reserves one contiguous block of KV-cache memory for each request, sized for the maximum possible length, even if the answer turns out short. The unused part of the reservation is wasted, and the gaps between blocks cause fragmentation, so fewer requests fit on the GPU. **PagedAttention** (used by vLLM) splits the KV cache into small fixed-size blocks and uses a **block table** to map a request's tokens to blocks, like virtual memory in an operating system. Blocks are allocated only when needed, so almost nothing is wasted. This also allows **prefix caching**: if many requests share the same beginning, for example my study-buddy system prompt that an agent resends every time, the blocks for that prefix are computed once and reused. This saves GPU work and lowers TTFT.

### Static versus continuous batching, and the metrics

With **static batching** the server collects a group of requests and runs them together, and the whole batch waits for the slowest request before new ones start, leaving the GPU idle for the finished ones. With **continuous batching** the server schedules at the level of single tokens: when a request finishes, a waiting request takes its slot immediately. The GPU stays busy. **Throughput** is total tokens per second across all users. **TTFT** is the wait for the first token. **TPOT** is the time per output token after the first, which controls how smooth the stream feels. **P95 latency** is the time that 95% of requests beat, which shows the slow tail that users complain about. Raising one usually hurts another: packing more requests into a batch raises throughput but makes each request wait longer (higher TTFT, TPOT and P95), while serving few requests at once gives each user speed but wastes the GPU.

## 3.2 Comparison table

| Basis for comparison | Ollama | vLLM |
|---|---|---|
| Built for (who and how many users) | One person or a small team on a laptop or workstation | Many concurrent users on a shared service |
| Hardware it needs | Runs on CPU, or on a consumer GPU; works on a normal laptop | Needs a server-class NVIDIA GPU (a datacentre card), usually Linux |
| How it handles several requests at once | Limited parallel slots (`OLLAMA_NUM_PARALLEL`); extra requests queue | Continuous batching: requests join and leave the running batch token by token |
| How it manages memory | Reserves KV cache per request and per context size; can waste memory | PagedAttention: fixed-size blocks and a block table, plus prefix caching |
| Setup effort and model format | One installer and `ollama pull`; quantized GGUF models | More setup (Python/GPU stack, server flags); mostly full-precision Hugging Face weights, some quantized formats |
| Your choice if only you use the scenario, and why | **Ollama.** It runs on my laptop, setup takes minutes, and one user never needs batching | Not needed for one person |
| Your choice if 100 people use it at once, and why | Would queue requests and waste memory | **vLLM.** Continuous batching and PagedAttention keep the GPU busy and let many users share one card with an acceptable P95 latency |

## 3.3 Minimal implementation

- `Modelfile`: base `qwen2.5:1.5b`, three `PARAMETER` values (temperature 0.2, num_ctx 4096, repeat_penalty 1.1) and a four-rule `SYSTEM` prompt. Built with `ollama create study-buddy -f Modelfile`.
- `common.py` and `rest_demo.py`: call `/api/generate` (non-streaming) and `/api/chat` (streaming) with `requests`, and print tokens per second and TTFT.
- `override_demo.py`: calls the custom model through `/v1/chat/completions`, once with no system prompt and once with a pirate system prompt.
- `observe.py`: runs the prompts for the observation table and saves `outputs/observations.md`.

Screenshots of `ollama list`, `ollama show`, `ollama ps` and each script's output are in `screenshots/`.

## 3.4 Observation

*(Paste the table from `outputs/observations.md` here.)*

| Run | Words (<=40?) | Ends with ? | Tokens/s | TTFT (s) | Total (s) | Load (s) | In memory after run |
|---|---|---|---|---|---|---|---|
| __ | __ | __ | __ | __ | __ | __ | __ |

**Behaviour.** *(Say whether the model followed the rules on each prompt. For the full-answer prompt, say whether it stayed under 40 words and refused to give the whole answer. For the exact-accuracy prompt, say whether it admitted uncertainty or guessed a number. A 1.5B model may break rules, so report what actually happened.)* With the program's pirate prompt in `override_demo.py`, the reply __ (followed the pirate prompt, not the Modelfile).

**Speed and memory.** *(Compare TTFT with total time. Compare the first run of "What is gradient descent?" with the second. State the load time on each, and whether `ollama ps` showed the model already loaded on the second run. Say whether it reported 100% GPU and how its memory figure compares with the Day 4 estimate.)*

**Surprises.** *(Add your own: for example whether the 40-word limit held, how quickly a 1.5B model replied, or that the warm run had about zero load time.)*

## 3.5 Suitability and conclusion

For my scenario, Ollama on a single machine was good enough. I am the only user, the 1.5B model fits comfortably in my 6 GB GPU, requests arrive one at a time, and the warm model answers quickly. A shared deployment would need vLLM or a similar server if, say, a whole class of 100 students used the study buddy at the same time. Ollama would queue them or run slots with wasted KV cache, and the last student would wait a long time. Continuous batching and PagedAttention would let one GPU serve everyone with acceptable TTFT and P95 latency.

In general, a local Ollama setup is sufficient when one person or a very small group uses a model, when privacy or offline use matters, and when hardware is a laptop or a single desktop GPU. A Modelfile adds a fixed, shareable behaviour (role, rules, parameters) on top of any base model without copying weights, and the REST API (with its OpenAI-compatible endpoint) lets my own programs use the model, override the defaults and measure speed. The need to move to a server built for many concurrent users appears when requests overlap: when GPU memory is wasted on the KV cache, when queues grow, when TTFT and P95 latency matter to users, or when throughput per GPU becomes a cost question. Because the OpenAI-compatible endpoint is the same, that move can be made by changing a base URL, not by rewriting the application.