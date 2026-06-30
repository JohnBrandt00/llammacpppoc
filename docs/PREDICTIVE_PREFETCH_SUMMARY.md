# Predictive Expert Prefetch — Summary

Concise overview of the project on branch `predictive-prefetch-stage0`. The blow-by-blow
data is in [RESULTS.md](RESULTS.md); this file answers two questions: **what we
accomplished** and **what we changed**.

---

## 1. What we fully accomplished

**Goal:** run an MoE model whose experts don't fit in VRAM (Qwen3-30B-A3B) at GPU
speed, by keeping the *hot* experts resident in VRAM and streaming the rest from RAM —
instead of falling back to slow CPU expert compute.

**Result: proven, lossless, and a big win.**

| Machine | Config | Decode: cache OFF → ON | Hit rate |
|---|---|---|---|
| RTX 3070 8 GB (fast CPU) | Qwen3-30B-A3B Q4, 4 GiB pool | ~9.9 → 12.9 t/s (**+30%**) | 91.7% |
| **Tesla P40 24 GB (weak CPU)** | Qwen3-30B-A3B Q8, 16 GiB pool | **2.60 → 14.39 t/s (5.5×)** | **~97%** |

What this means: a 30B model that **does not fit in 24 GB of VRAM** went from *unusably
slow* (2.6 t/s, CPU-bound) to *perfectly usable* (14.4 t/s, GPU-speed). That is the
original premise, demonstrated end-to-end. The bigger the gap between the GPU and the
host CPU (and the bigger the VRAM cache), the bigger the win — which is why the P40 box
(weak CPU, 24 GB) crushed it.

**Properties we verified:**
- **Lossless** — cache-on output is bit-identical to cache-off (it changes *where* the
  expert weights live, not the math).
- **Cross-platform with zero source changes** — same binary path on CUDA 13.3/sm_86
  (3070) and CUDA 12.4/sm_61 (P40); a version guard handles the batched-copy API.
- **Prefill is net-neutral** — the cache only helps decode; it neither helps nor
  measurably hurts prefill.
- **Reactive LRU is enough** — replaying real routing traces, plain LRU hits 83–88% at
  a 33% budget, within ~8% of the Belady optimum, beating LFU / LRU-K / static-freq.

**Ideas we tested cheaply and *killed* (don't re-explore for this model):**
- **Expert weight synthesis** (reconstruct experts from a small basis) — NO-GO: experts
  are near full-rank (107/128 dims for 85% variance); ~23% fidelity at K=32.
- **Causal commitment windows** (lock routing for N tokens to raise hit rate) — NO-GO:
  ~40% routing fidelity / ~60% wrong experts on locked tokens; routing too unstable.
- **History / Markov / NN predictors** (the original "predictive" Stages 2–4) —
  unnecessary: reactive LRU is already within ~8% of optimal.

**One idea that passed Stage-0 but isn't built:** popularity-tiered precision (hot
experts at Q4, cold at Q2/IQ1) — 21–32% smaller with only ~9% low-precision exposure.
Viable, needs requantizing stacked expert tensors + a perplexity check before shipping.

---

## 2. What we changed

All changes live on `predictive-prefetch-stage0` (19 files, ~1,970 insertions vs
`master`). Nothing is upstreamed — this is exploratory research on our own repo.

### The shipped feature — the LRU VRAM expert cache

| File | What it does |
|---|---|
| `ggml/src/ggml-cuda/expert-cache.h` | `expert_cache_lru` — O(1) LRU slot manager (hash map + intrusive doubly-linked recency list); `access(key)` returns `{slot, hit, evicted, evicted_key}`. |
| `ggml/src/ggml-cuda/expert-cache.cu` | `cuda_expert_cache` — the dedicated VRAM slot pool + the copy hook `ggml_cuda_moe_expert_cache_cpy`. **Hit** → D2D copy slot→destination; **miss** → H2D host→slot, then D2D slot→destination. Pool sized from `cudaMemGetInfo` free VRAM minus 1.5 GiB headroom. Batched path via `cudaMemcpyBatchAsync` (CUDA 12.8+, version-guarded; older CUDA falls back to per-expert). The GEMM itself is unchanged. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` (+4), `common.cuh` (+3) | Register the CUDA cache hook during `ggml_backend_cuda_init`. |
| `ggml/src/ggml-backend-impl.h` (+22) | Backend-agnostic hook typedef `ggml_moe_expert_cache_cpy_t` + the `set`/`enabled` declarations, so the scheduler can call into CUDA without depending on it. |
| `ggml/src/ggml-backend.cpp` (+220) | Wire the hook into the scheduler's expert-copy path: instead of the stock "copy only the used experts into a transient buffer," call the cache (copy only *misses*, keep *hits* resident). Env-gated by `GGML_MOE_EXPERT_CACHE`. Also holds the `GGML_MOE_PREFETCH_METRICS` instrumentation. |

**The key mechanism:** experts are keyed by their **stable host address**. ggml builds a
fresh static graph each forward pass, so the cache can't live in the graph — it lives in
the scheduler's copy path and persists across passes. That persistence is the whole
point: an expert copied to VRAM on one token stays there for the next.

### Research instrumentation (not the feature, the evidence)

| File | What it does |
|---|---|
| `ggml/src/ggml-cpu/ggml-cpu.c` (+46) | `ggml_moe_routing_trace()` in `mul_mat_id` — dumps which experts each token routes to (env `GGML_MOE_ROUTING_TRACE`), the raw data behind the LRU/skew analysis. |
| `tools/predictive-prefetch/*.py` | `analyze_routing.py`, `analyze_metrics.py`, `simulate_cache.py` (policy comparison), `simulate_commitment.py` + `validate_synthesis.py` + `simulate_mixed_precision.py` (the three Stage-0 go/no-go checks). |
| `tools/predictive-prefetch/test_expert_cache.cpp` | Standalone unit test for the LRU slot manager (validated against a Python reference sim). |
| `tools/predictive-prefetch/baseline_bench.{ps1,sh}`, `run_stage0.ps1` | Benchmark / experiment runners. |

### How to turn it on

```bash
# required
GGML_MOE_EXPERT_CACHE=1          # enable the cache
GGML_OP_OFFLOAD_MIN_BATCH=1      # force batch-1 decode onto the GPU
                                 # (default 32 keeps decode on CPU → cache never engages)
# optional
GGML_MOE_EXPERT_CACHE_BYTES=N    # pool budget override (default 2 GiB, else free-VRAM sized)
GGML_MOE_EXPERT_CACHE_BATCH=0    # disable batched copies (force per-expert)
GGML_MOE_EXPERT_CACHE_STATS=1    # print accesses / hit_rate / miss_GiB
```
Run with experts on CPU/RAM and everything else on GPU, e.g.
`-ngl 999 -ot ".ffn_.*_exps.=CPU" --no-mmap`.

### Known limitation / clear next step

During **prefill** (large batches) every expert fires, so the working set exceeds the
pool → the cache thrashes (~1% hit, hundreds of GiB of useless H2D). It costs no
prefill *time* (prefill is GEMM-bound, copies overlap), but it's wasted PCIe traffic.
**Next improvement:** bypass the cache for large batches and engage it only for
small/decode batches — pure upside, since prefill gets no benefit from it anyway.
