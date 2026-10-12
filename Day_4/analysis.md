# Day 4 Task: Will It Fit, and May I Use It?

**Name:** Ranjith | **Date written:** 12 October 2026

## Scenario

I want a private study-helper agent that runs on my own laptop. It reads my college notes, calls small tools (calculator, lookup) and keeps a conversation going for a whole study session. Only I use it, so there are no customers and no public release for now.

- **Machine:** laptop with an NVIDIA GPU, **6 GB VRAM**.
- **Memory budget:** 6 GB, because the model must sit in VRAM. If it spills into system RAM it runs split across CPU and GPU, which is much slower.
- **Licensing situation:** personal use now, but I may put the project on GitHub and show it in my capstone. I want a licence that allows redistribution and commercial use without asking anyone.
- **Estimator:** `vram_estimate.py` (weights, KV cache, total, fit verdict). `sweep.py` varies context and quantization.

Formula from the Day 4 lab:

- weights (GB) = parameters (B) × bytes per parameter
- KV cache (GB) = parameters (B) × context (K tokens) × 0.02
- total (GB) = (weights + KV cache) × 1.10

## 3.1 Explanation of each concept

### Model weights

The weights are the learned numbers that make up the model. Their size depends only on how many parameters there are and how many bytes each one takes. They are the biggest fixed cost and do not change while I chat. In my scenario I chose Qwen2.5 1.5B, which has 1.5 billion parameters. At Q4_K_M (0.57 bytes per parameter) the weights are 1.5 × 0.57 = **0.85 GB**. Llama 3.1 8B at the same precision is 4.56 GB, which is 76% of my 6 GB budget before any conversation exists.

If I judged a model by its name alone, I might download a 5 GB file that can never run. The limitation is that the formula assumes a flat bytes-per-parameter rate. Real builds mix precisions across layers, so the true file differs a little.

### Quantization

Quantization stores each weight with fewer bits, so the weights shrink at some cost in quality. This controls the *fit* side of the decision. I tried Qwen2.5 1.5B at several precisions with an 8K context. The weights were 0.65 GB at Q3_K_M, 0.85 GB at Q4_K_M, 1.02 GB at Q5_K_M, 1.50 GB at Q8_0 and 3.00 GB at FP16. The totals were 0.97, 1.20, 1.39, 1.91 and 3.56 GB. Because the model is so small, even FP16 fits my 6 GB card.

If I misjudge this, I either choose too high a precision and waste memory, or too low a precision and answers get worse. Small models lose quality faster when compressed, and my agent needs exact JSON for tool arguments. The formula cannot tell me how much quality I lose, so memory alone should not push me to Q3.

### KV cache and context length

While the model reads and writes, it keeps a key/value entry for every token in the conversation. That is the KV cache. It grows with context length while the weights stay fixed. For Qwen2.5 1.5B at Q4_K_M the cache is 0.12 GB at 4K, 0.24 GB at 8K, 0.96 GB at 32K and 3.84 GB at 128K. The total goes from 1.07 GB to 5.16 GB with the same weights each time.

This matters for my agent because every tool result is added to the conversation, so the context keeps growing. By the formula, memory would allow about 153K tokens at Q4_K_M. But the model's own context window is about 32K, so the model limits the session before memory does. The limitation of the formula is that the 0.02 constant assumes a modern grouped-query-attention model and a 16-bit cache. A different architecture could need more.

### Model card

A model card is the publisher's page for a model. It states the parameter count, context window, licence, intended use, limitations and whether tool calling is supported. It controls the *permission* side of the decision and gives me the inputs for the fit side. I used the cards and Ollama library pages to fill my comparison table.

If I ignore the card, I might build on a model with no tool calling, or one whose licence forbids what I plan. The limitation is that cards change, and the Ollama page does not always repeat the licence, so I recorded the date I looked.

### Open-weight versus open-source licensing

Open-weight means the trained weights can be downloaded. It does not mean the licence is open-source. Apache 2.0 allows commercial use and redistribution with few conditions. Community licences add conditions, for instance on very large user counts, on attribution and naming, or on acceptable use. The exact licence name decides what I may do.

In my scenario this matters because I may publish the project. Qwen2.5 1.5B and Granite 4 are on Apache 2.0 (confirmed on the cards on __ Oct 2026). Llama 3.1 uses Meta's Llama 3.1 Community License, which has extra conditions. Not every Qwen2.5 size shares the same licence, so I checked the 1.5B card specifically. The limitation is that I am not a lawyer, and a commercial product would need a proper legal check.

## 3.2 Tables

### (a) Memory estimate table (available memory: 6 GB)

| Model | Params (B) | Precision | Context (K) | Weights (GB) | KV cache (GB) | Total (GB) | Fits in 6 GB? |
|---|---|---|---|---|---|---|---|
| Qwen2.5 1.5B | 1.5 | Q4_K_M | 8 | 0.85 | 0.24 | 1.20 | Yes, comfortably |
| Qwen2.5 1.5B | 1.5 | Q4_K_M | 32 | 0.85 | 0.96 | 2.00 | Yes, comfortably |
| Granite 4 micro 3B | 3.0 | Q4_K_M | 8 | 1.71 | 0.48 | 2.41 | Yes, comfortably |
| Llama 3.1 8B | 8.0 | Q4_K_M | 4 | 4.56 | 0.64 | 5.72 | Yes, but tight |
| Llama 3.1 8B | 8.0 | Q4_K_M | 8 | 4.56 | 1.28 | 6.42 | No |
| Qwen3 8B | 8.0 | Q4_K_M | 8 | 4.56 | 1.28 | 6.42 | No |
| Qwen3 8B | 8.0 | Q3_K_M | 8 | 3.44 | 1.28 | 5.19 | Yes, but tight |

### (b) Model comparison table

Date I checked the cards: **__ October 2026**

| Basis for comparison | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| Full model name and version | Qwen2.5 1.5B (`qwen2.5:1.5b`) | Granite 4 micro 3B (`granite4:3b`) | Llama 3.1 8B (`llama3.1:8b`) |
| Publisher | Alibaba (Qwen team) | IBM | Meta |
| Total / active parameters (MoE?) | 1.5B / 1.5B, dense | 3B / 3B, dense | 8B / 8B, dense |
| Context window | __ (about 32K, check card) | 128K | 128K |
| Licence (exact name) | __ (from card; expected Apache 2.0) | Apache 2.0 | __ (from HF card) |
| Commercial use allowed? | __ | Yes | __ |
| Any extra conditions? | __ | None beyond Apache 2.0 notice terms | __ (quote the clause) |
| Tool calling stated on the card? | __ (check Ollama tag) | Yes | Yes |
| GGUF / Ollama build available? | Yes | Yes | Yes |
| Download size at Q4 | 986 MB (from `ollama list`) | 2.1 GB | 4.9 GB |
| Your memory estimate (total) | 1.20 GB at 8K | 2.41 GB at 8K | 6.42 GB at 8K (5.72 GB at 4K) |
| Fits your scenario's machine? | Yes | Yes | No at 8K, tight at 4K |
| Date you checked the card | __ Oct 2026 | __ Oct 2026 | __ Oct 2026 |

## 3.3 Context length and quantization observation

I chose **Qwen2.5 1.5B**.

| Setting changed | Value used | Weights (GB) | KV cache (GB) | Total (GB) | Fits? |
|---|---|---|---|---|---|
| Context length (Q4_K_M) | 4K | 0.85 | 0.12 | 1.07 | Yes, comfortably |
| Context length (Q4_K_M) | 8K | 0.85 | 0.24 | 1.20 | Yes, comfortably |
| Context length (Q4_K_M) | 32K | 0.85 | 0.96 | 2.00 | Yes, comfortably |
| Context length (Q4_K_M) | 128K | 0.85 | 3.84 | 5.16 | Yes, but tight |
| Quantization (8K) | Q3_K_M | 0.65 | 0.24 | 0.97 | Yes, comfortably |
| Quantization (8K) | Q4_K_M | 0.85 | 0.24 | 1.20 | Yes, comfortably |
| Quantization (8K) | Q5_K_M | 1.02 | 0.24 | 1.39 | Yes, comfortably |
| Quantization (8K) | Q8_0 | 1.50 | 0.24 | 1.91 | Yes, comfortably |
| Quantization (8K) | FP16 | 3.00 | 0.24 | 3.56 | Yes, comfortably |

When the context grew, only the KV cache changed (0.12 GB to 3.84 GB) and the weights stayed at 0.85 GB. When the quantization changed, the weights changed (0.65 GB to 3.00 GB) while the KV cache stayed at 0.24 GB. By the formula the largest context that fits at Q4_K_M is about 153K tokens, and at Q8_0 about 132K. The model's own window (about 32K) is the real limit. I would choose **Q4_K_M** as the default, because it is what Ollama pulls and it leaves most of the 6 GB free. Since memory is so plentiful, **Q8_0** (1.91 GB at 8K) is a sensible upgrade if tool arguments turn out unreliable. What I give up at Q4 is a little accuracy.

## 3.4 Estimate versus reality

| Model | ollama list size | ollama ps size | Processor | Your estimate (weights / total) |
|---|---|---|---|---|
| qwen2.5:1.5b | 986 MB | __ GB | __ | 0.85 GB / 1.20 GB (8K) |

The `ollama list` size is the file on disk, so it should be close to the 0.85 GB weights estimate. It showed 986 MB (about 0.99 GB), slightly above the estimate. The gap comes from the build's quantization mix, embeddings and block metadata, which matter more for a small model. `ollama ps` showed __ GB against my 1.20 GB total. *(Say whether it was close. Ollama uses a limited default context, so the figure may differ from my 8K estimate. The runtime also reserves extra buffers.)* The Processor column showed __. "100% GPU" means the whole model is in VRAM and runs at full speed. A split such as "30%/70% CPU/GPU" would mean it spilled over and would run much slower.

## 3.5 Suitability analysis

I recommend **Qwen2.5 1.5B at Q4_K_M with an 8K context (up to its 32K window when needed), under Apache 2.0 (confirmed on the card)**. It needs about 1.2 GB at 8K and 2.0 GB at 32K, so it fits comfortably inside 6 GB and leaves room for the context to grow as the agent collects tool results. It also leaves memory free for other apps and for the notes the agent reads.

The runner-up is **Granite 4 micro 3B** (about 2.4 GB at 8K, Apache 2.0, tool calling listed). It is a stronger model with a longer 128K window and still fits, so it is the first one I would try if 1.5B proves too weak. Llama 3.1 8B is rejected: it does not fit at 8K (6.42 GB), is tight even at 4K, and carries a community licence with conditions.

The honest weakness of this choice is capability. A 1.5B model reasons less well and may make mistakes in tool-call JSON. My recommendation would change if: tool calling proved unreliable in testing (then I would move to Granite 4 3B or use Q8_0); I had **more memory** (12 GB or more makes an 8B model comfortable); I needed a **context longer than 32K** (then a 128K-window model); or I released the project **commercially** (then I would re-read the licence in full).

## 3.6 Conclusion

Each factor carries the most weight in different situations. **Memory (size and quantization)** decides when the hardware is fixed and small. A student with a 6 GB card cannot run an 8B model at 8K whatever its quality, so the choice is a smaller model or lower quantization. **Context length** decides for anything that keeps adding text, such as agents, document readers and long chats, because the KV cache can make a model that fits at 4K fail at 32K. For very small models such as this one, the model's own context window can matter more than memory. **Quantization** is the tool for squeezing in a model that is almost too big. A chatbot tolerates Q4 well, but exact-JSON tool calling or code may justify Q5 or Q8 when memory allows. **Licence** decides whenever the model leaves my laptop: a company or a public demo should pick Apache 2.0 or MIT, even if a community-licensed model is bigger or scores better. In short, memory decides whether a model can run, and the licence decides whether it may be used. A good choice passes both, and since the formula is only an estimate, I confirm it with `ollama ps`.
