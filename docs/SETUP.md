# Setup: MoE expert-cache fork on a new machine

This fork of llama.cpp adds a lossless three-tier expert system for MoE models
(Qwen3-30B, GLM-4.5-Air, ...) that don't fit in VRAM — or even in RAM:

- **VRAM tier**: persistent LRU expert cache on the GPU (hot experts stay
  resident; measured 5.5× decode on a Tesla P40, +30% on an RTX 3070)
- **Heat**: per-expert usage persisted across runs; restarts pre-warm the cache
- **Disk tier** (optional): heat-ranked experts pinned in RAM, cold tail
  streamed as whole-expert reads from the GGUF (for models bigger than RAM)

Output is bit-identical to stock llama.cpp — the tiers change where weights
live and how they travel, never the math.

## 1. Clone + build

Requirements: CUDA toolkit (12.8+ or 13.x recommended — enables the batched
copy fast path; 12.0–12.7 works with automatic fallback), cmake, a C++17
compiler.

```bash
git clone -b predictive-prefetch-stage0 https://github.com/JohnBrandt00/llammacpppoc.git
cd llammacpppoc
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89   # 89=RTX 4090/Ada; 86=RTX 30xx; 61=P40
cmake --build build -j --target llama-server
```

## 2. The env-var cheat sheet

| var | what | when |
|---|---|---|
| `GGML_MOE_EXPERT_CACHE=1` | enable the VRAM expert cache | always |
| `GGML_OP_OFFLOAD_MIN_BATCH=1` | run batch-1 decode MoE on the GPU (required for the cache to engage on decode) | always |
| `GGML_MOE_EXPERT_CACHE_BYTES` | VRAM pool size in bytes | always — see sizing below |
| `GGML_MOE_EXPERT_CACHE_STATS=1` | print hit-rate / traffic stats | recommended |
| `GGML_MOE_EXPERT_CACHE_HEAT=<file>` | persist expert heat + pre-warm on restart | recommended; one file per model+quant |
| `GGML_MOE_EXPERT_CACHE_GATHER=1` | single-kernel serve (instead of batched copies) | only on CUDA < 12.8 (e.g. P40) |
| `GGML_MOE_EXPERT_DISK=1` | disk tier (whole-expert streaming; **requires mmap**, i.e. no `--no-mmap`) | only when model > RAM |
| `GGML_MOE_EXPERT_RAM_BYTES` | mlock budget for hot experts (needs a heat file) | disk tier only |
| `GGML_MOE_EXPERT_DISK_CACHE_BYTES` | RAM LRU for streamed cold experts (default 2 GiB) | disk tier only |
| `GGML_MOE_EXPERT_DISK_THREADS` | parallel expert reads (default 4) | disk tier only |

**Pool sizing (`CACHE_BYTES`)**: free VRAM minus model layers/KV minus ~2 GB
headroom. Too small THRASHES to ~0% hit (measured cliff: a pool below the
per-layer working set is useless). The startup line
`ggml_moe_expert_cache: ... pool N slots` + the stats hit-rate tell you if
you're above the cliff. On the GPU driving the display, leave extra headroom
(VRAM oversubscription = WDDM paging "lockup").

**mlock budget (`RAM_BYTES`)**: size from what the HOST can spare, never from
container-reported free memory — mlock'd RAM is unreclaimable and can starve
the host (learned the hard way on an Unraid LXC).

**Heat workflow**: run 1 tracks and saves on CLEAN exit (Ctrl-C once, not
kill -9); run 2+ loads, pins (disk tier), and pre-warms. Heat sharpens with
decode-heavy use.

## 3. Recipes (example: 4090 24 GB + 96 GB RAM + NVMe)

### A. Model fits in RAM (Qwen3-Coder-30B Q8 ~33 GB, GLM-4.5-Air Q4_K_XL ~73 GB)

Experts live in RAM (`--no-mmap`), non-experts on GPU, VRAM pool caches hot
experts. No disk tier.

```bash
GGML_MOE_EXPERT_CACHE=1 GGML_OP_OFFLOAD_MIN_BATCH=1 GGML_MOE_EXPERT_CACHE_STATS=1 \
GGML_MOE_EXPERT_CACHE_HEAT=/path/model.heat \
GGML_MOE_EXPERT_CACHE_BYTES=$((14*1024**3)) \
./build/bin/llama-server -m /path/model.gguf \
  -ngl 999 -ot ".ffn_.*_exps.=CPU" --no-mmap -c 32768 \
  --host 0.0.0.0 --port 8000
```

GLM-4.5-Air Q4_K_XL fully in 96 GB RAM is the sweet spot of this machine:
big-model quality with the cache serving hot experts from VRAM at 4090 speed.

### B. Model bigger than RAM (GLM-355B-class Q2/Q3, 130–160 GB) — the experiment

mmap + disk tier: pin the hot ~60 GB in RAM, stream the cold tail from NVMe.

```bash
ulimit -l unlimited
GGML_MOE_EXPERT_CACHE=1 GGML_OP_OFFLOAD_MIN_BATCH=1 GGML_MOE_EXPERT_CACHE_STATS=1 \
GGML_MOE_EXPERT_CACHE_HEAT=/path/bigmodel.heat \
GGML_MOE_EXPERT_CACHE_BYTES=$((14*1024**3)) \
GGML_MOE_EXPERT_DISK=1 \
GGML_MOE_EXPERT_RAM_BYTES=$((60*1024**3)) \
GGML_MOE_EXPERT_DISK_CACHE_BYTES=$((8*1024**3)) \
./build/bin/llama-server -m /path/bigmodel-00001-of-000NN.gguf \
  -ngl 999 -ot ".ffn_.*_exps.=CPU" -c 16384 \
  --host 0.0.0.0 --port 8000
```

Expectations, honestly: decode a few t/s once heat is warm (skewed expert
traffic + NVMe latency); long-prompt prefill is the hard part (a prefill batch
touches every expert — NVMe makes it minutes, not hours). First run has no
heat → no pinning → slow; clean-exit and relaunch.

## 4. What to look for

- `ggml_moe_expert_cache_stats: ... hit_rate=...` — decode hit should be 90%+
  in recipe A after warmup. ~0% = pool too small.
- `ggml_moe_expert_disk_stats: ... read_fail=0` — pager health (recipe B).
- `heat saved ...` on exit — the learning loop is closing.
- A/B the cache itself: same request with `GGML_MOE_EXPERT_CACHE=0` and no
  `GGML_OP_OFFLOAD_MIN_BATCH` → compare `eval time` t/s.

Full research log: `docs/RESULTS.md`. Architecture: `docs/DISK_TIER_DESIGN.md`
and `docs/PREDICTIVE_PREFETCH_SUMMARY.md`.
