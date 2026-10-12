"""Day 4 task 3.3: vary context (quantization fixed), then quantization (context fixed)."""
import sys
from vram_estimate import report, max_context_k, AVAILABLE_GB

avail = float(sys.argv[1]) if len(sys.argv) > 1 else AVAILABLE_GB
MODEL, PARAMS = "Qwen2.5 1.5B", 1.5
print(f"Memory available: {avail} GB\n")

print(f"{MODEL}: context varies, quantization fixed at Q4_K_M")
for ctx in (4, 8, 32, 128):
    report(MODEL, PARAMS, "Q4_K_M", ctx, avail)

print(f"\n{MODEL}: quantization varies, context fixed at 8K")
for q in ("Q3_K_M", "Q4_K_M", "Q5_K_M", "Q8_0", "FP16"):
    report(MODEL, PARAMS, q, 8, avail)

print("\nLargest context that fits (estimate):")
for q in ("Q4_K_M", "Q5_K_M", "Q8_0"):
    print(f"  {MODEL} {q}: about {max_context_k(PARAMS, q, avail):.0f}K tokens")