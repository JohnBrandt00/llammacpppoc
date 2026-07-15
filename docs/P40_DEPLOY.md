# P40 (guanshiyin) deploy — gather + heat/pre-warm

Gets the two new cache upgrades onto the Tesla P40 box:
**Stage A gather** (single-kernel expert serve — targets exactly this box, whose
CUDA 12.4 has no `cudaMemcpyBatchAsync`) and **heat + pre-warm** (server restarts
start with a warm pool instead of the 1%-hit cold grind).

## 1. Push from the dev box (Windows)

```powershell
git push origin predictive-prefetch-stage0
```

## 2. Pull + rebuild on guanshiyin

```bash
cd <your llama.cpp clone>   # the one cloned from JohnBrandt00/llammacpppoc
git pull

export CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc
cmake --build build -j --target llama-server
```

(A pull onto an existing configured build tree only needs the last line; the
full `cmake -B build ...` is there in case the tree is fresh.)

## 3. Server command — two new env vars

```bash
GGML_MOE_EXPERT_CACHE=1 \
GGML_OP_OFFLOAD_MIN_BATCH=1 \
GGML_MOE_EXPERT_CACHE_STATS=1 \
GGML_MOE_EXPERT_CACHE_GATHER=1 \
GGML_MOE_EXPERT_CACHE_HEAT=/root/qwen30b-q8.heat \
./build/bin/llama-server \
  -m /root/models/Qwen3-30B-A3B-Q8_0.gguf \
  -ngl 999 -ot ".ffn_.*_exps.=CPU" --no-mmap -c 65536 \
  --host 0.0.0.0 --port 8000
```

New vs the last working command:
- `GGML_MOE_EXPERT_CACHE_GATHER=1` — serve cache hits with one kernel per MoE op
  instead of one D2D memcpy per expert. This box runs `mode=per-expert` today
  (CUDA < 12.8), so it has the most launch overhead to reclaim.
- `GGML_MOE_EXPERT_CACHE_HEAT=<file>` — per-expert traffic counts persisted
  across runs; when the file exists, the pool is pre-warmed with the hottest
  experts at startup.

## 4. What to expect

- **First run**: `heat file ... not found; tracking, will create on exit` —
  behaves like before. On shutdown it prints
  `heat saved ... run: accesses=... hit_rate=... prewarmed=0`.
- **Second run onward**: `heat loaded from ... (N tensors) -> pre-warm enabled`,
  and the stats lines show `mode=gather prewarm=~10277` (the 16 GiB pool holds
  ~56% of experts, so pre-warm fills it with the top ~56% by traffic). Decode
  hits should be high from the *first* request; the prefill warmup grind on a
  fresh server should shrink noticeably.

## 5. Gotchas

- **Heat saves on clean exit only** (atexit): stop the server with Ctrl-C /
  SIGINT/SIGTERM. `kill -9` or an OOM loses that run's counts (the previous
  file survives — counts just don't accumulate).
- **One heat file per model+quant.** Keys are tensor names + expert ids, so a
  mismatched file can't corrupt anything (offsets always come from the running
  model), but rankings from a different model pollute the counts.
- **Pool size is make-or-break** (measured on the 3070): a pool below the
  per-layer working set thrashes to ~0% hit. The P40's default free-VRAM
  sizing (16 GiB) is comfortably past the cliff for Qwen3-30B — but check the
  `hit_rate` in the stats line whenever you change model or `-c`.

## 6. A/B tests worth pasting back

Same short-prompt request via curl, read the server's `eval time =` line:

1. **Gather win** (Stage A's target is this box): `GGML_MOE_EXPERT_CACHE_GATHER=1`
   vs `0`, same heat file. Expect a bump from collapsing ~1,150 copy launches
   per token into ~150.
2. **Pre-warm win**: first run (cold, `prewarmed=0`) vs restart (warm). Compare
   time-to-first-token and the hit_rate line.
