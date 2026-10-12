"""Day 4 task: estimate the memory an open model needs and whether it fits.

Formula (from the Day 4 lab):
    weights (GB) = params (B) x bytes per parameter
    KV cache (GB) = params (B) x context (K tokens) x 0.02
    total (GB)   = (weights + KV cache) x 1.10  (10% runtime overhead)

It is an ESTIMATE used to decide fit, not to predict memory to the megabyte.
"""
import sys

BYTES_PER_PARAM = {"FP16": 2.00, "Q8_0": 1.00, "Q6_K": 0.81,
                   "Q5_K_M": 0.68, "Q4_K_M": 0.57, "Q3_K_M": 0.43}
KV_GB_PER_B_PER_1K = 0.02
OVERHEAD = 1.10

# ---- YOUR SCENARIO: change this to your real VRAM (run `nvidia-smi` to check) ----
AVAILABLE_GB = 6.0


def estimate(params_b, precision="Q4_K_M", context_k=8):
    """Return (weights_gb, kv_gb, total_gb)."""
    if precision not in BYTES_PER_PARAM:
        raise ValueError(f"Unknown precision {precision}. Choose from {list(BYTES_PER_PARAM)}")
    weights = params_b * BYTES_PER_PARAM[precision]
    kv = params_b * context_k * KV_GB_PER_B_PER_1K
    total = (weights + kv) * OVERHEAD
    return weights, kv, total


def verdict(total, available=AVAILABLE_GB):
    if total <= available * 0.7:
        return "fits comfortably"
    if total <= available:
        return "fits, but tight"
    return "does NOT fit"


def max_context_k(params_b, precision, available=AVAILABLE_GB):
    """Largest context (K tokens) whose total still fits in `available` GB."""
    room = available / OVERHEAD - params_b * BYTES_PER_PARAM[precision]
    return max(0.0, room / (params_b * KV_GB_PER_B_PER_1K))


def report(name, params_b, precision, context_k, available=AVAILABLE_GB):
    w, kv, t = estimate(params_b, precision, context_k)