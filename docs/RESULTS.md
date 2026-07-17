# Predictive Expert Prefetch Results

This file tracks the Qwen3-30B-A3B expert-prefetch experiment.

## Test machine

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3070, 8 GB, compute 8.6 |
| CPU | AMD Ryzen 9 7900X (12C/24T) |
| Driver | 591.86 |
| CUDA toolkit | 13.3.33 (the pre-existing 11.7 install has no `nvcc`) |
| Host compiler | MSVC 19.44 (VS 2022) |
| Model | Qwen3-30B-A3B Q4_K_M, 18.6 GB (17.35 GiB), 30.5B params, 48 layers, 128 experts, top-8 |

## Build

CUDA 13.x keeps its runtime DLLs (`cudart64_13.dll`, `cublas64_13.dll`, ...) in
`bin\x64`, and the VS generator's CUDA targets read the toolkit path from the
`CUDA_PATH_V13_3` environment variable. Both must be set:

```powershell
$env:CUDA_PATH_V13_3 = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
cmake -B build -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 `
  "-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe"
cmake --build build --config Release --target llama-bench
```

To run the binary, put the CUDA runtime DLLs on `PATH`:

```powershell
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64;$env:PATH"
```

## Model

```powershell
huggingface-cli download bartowski/Qwen_Qwen3-30B-A3B-GGUF --include "Qwen_Qwen3-30B-A3B-Q4_K_M.gguf" --local-dir .\models
```

## Benchmark

A full `-ngl 0..48` sweep is not meaningful on an 8 GB card: the 18.6 GB model
cannot be fully offloaded (high `-ngl` OOMs), and a plain `-ngl` offload places
experts *on the GPU*, which never exercises the instrumented host->device
expert-copy path. The two feasible, informative configurations are:

- **`-ngl 0`** — pure-CPU baseline (the t/s to beat).
- **`-ngl 99 -ncmoe 48`** — all attention/norms on the GPU, all expert tensors
  kept in host RAM (the realistic 8 GB deployment). This is the only config that
  triggers the instrumented expert-copy path.

```powershell
.\tools\predictive-prefetch\run_stage0.ps1
python .\tools\predictive-prefetch\analyze_metrics.py .\metrics\predictive-prefetch\stage0-*-ngl99-ncmoe48.err.log
```

## Stage 0 Measurements

Commit `predictive-prefetch-stage0` (instrumentation), `-p 512 -n 128`, 8 threads.

| Config | pp t/s | tg t/s | Notes |
|---|---:|---:|---|
| `-ngl 0` (CPU only) | 84.6 | 13.0 | speed to beat |
| `-ngl 99 -ncmoe 48` (experts in RAM) | 241.7 | 26.1 | GPU attn + CPU experts |
| `-ngl 99 -ncmoe 48`, `-p 0 -n 512` | – | 20.1 | sustained long decode |

Generation is **~13 t/s CPU → ~26 t/s** with GPU attention and experts streamed
from RAM — already well above the original plan's pessimistic 2–4 t/s estimate
(Qwen3-30B-A3B activates only ~3B params/token).

### MoE expert-copy behavior (the key Stage 0 finding)

The instrumented host->device expert-copy path is a **prefill** phenomenon, not a
decode one:

| Phase (probe) | copy-ops | expert payload | experts copied |
|---|---:|---:|---:|
| Prefill only (`-p 512 -n 0 -r 1`, +warmup) | 240 | 20.2 GB | 21,237 |
| Decode only (`-p 0 -n 512 -r 1`, +warmup) | <48 (no dump) | ~0 | ~0 |

So a single 512-token prefill streams **~10 GB** of expert weights to the GPU
(the union of experts touched by the batch, ~74% of the pool), while 512
single-token decode steps trigger essentially **no** expert offload — llama.cpp
keeps the batch-1 expert GEMM on the CPU. Padding overhead in the copy path is
negligible (<0.1%).

**Implication for the plan.** The prefetch architecture assumes experts stream
into VRAM *per decode token*; by default llama.cpp does not do this under
`-ncmoe` (decode experts compute on CPU). Stage 1 must therefore either force the
decode-time expert GEMM onto the GPU (so a persistent VRAM expert cache has
something to serve) or target the prefill copies — which today are issued as
**synchronous** `cudaMemcpyAsync` from *pageable* host memory (no overlap). The
pinned-memory + dual-stream work in Stage 1 is what makes those copies
genuinely async and hideable.

## Stage 1 premise check — resident-expert ceiling (zero code)

The plan's Stage 1 assumes streaming experts to VRAM per decode token beats the
baseline. But PCIe (~28 GB/s) is *slower* than this box's DDR5 (~60-80 GB/s), so
streaming a **cold** expert per token is slower than the CPU GEMM llama.cpp
already does. The only win is from experts that are **resident** in VRAM (zero
transfer). The existing `-ncmoe N` flag measures that ceiling for free: lowering
N keeps more expert *layers* permanently on the GPU.

Decode (`-p 0 -n 128 -r 2`), `-ngl 99`, single invocation (consistent cache):

| `-ncmoe` | expert layers resident | decode t/s |
|---:|---:|---:|
| 48 | 0  | 7.2 |
| 44 | 4  | 9.0 |
| 40 | 8  | 12.0 |
| 36 | 12 | 16.6 |
| 34 | 14 | 14.7 |
| 32 | 16 | 15.4 |
| 30 | 18 | 18.8 |

Moving ~25% of experts into VRAM roughly **doubles** decode throughput. Absolute
t/s drifts run-to-run with OS file-cache warmth (the 36 row read 16.6 then 13.1
in two sweeps), but the monotonic slope is stable and steep. `-ncmoe 30`
(18 layers, ~6.3 GB of experts) fits 8 GB without OOM.

**Conclusions for Stage 1:**
- Reactive per-token streaming of cold experts is a *non-starter* on this
  hardware (transfer slower than CPU compute) — drop it.
- Expert **residency** is the lever, and it is strong. The win the project must
  chase is keeping the *hottest* experts resident.
- `-ncmoe` already delivers the *static, layer-granularity* version of this for
  free. The project's only value-add over the stock flag is **dynamic,
  expert-granularity** residency (hottest experts across all layers), which is
  worthwhile only if expert activation is skewed/temporally local enough that a
  dynamic cache beats the static layer split at equal VRAM. That is the next
  thing to measure.

## Stage 1 go/no-go — routing skew & locality

To decide whether a *dynamic, expert-granularity* cache can beat the *static,
layer-granularity* residency that `-ncmoe` already provides, the CPU
`mul_mat_id` op was instrumented (`GGML_MOE_ROUTING_TRACE=<file>`) to log the
experts selected per layer per token. Trace captured with `llama-cli`,
`-ngl 99 -ncmoe 48` (all 48 layers route on CPU), 256 generated tokens.

Two domains, ~256 generated tokens each (× 48 layers):

| Domain | distinct experts/layer | Gini | lag-1 overlap |
|---|---:|---:|---:|
| code (red-black tree in C) | ~101 of 128 | 0.184 | 41.2% |
| prose (jazz history essay) | ~86 of 128 | 0.209 | 45.2% |

Cache hit rate at a 33% VRAM budget (cache top-42 of 128 experts per layer):

| Strategy | code | prose |
|---|---:|---:|
| Static `-ncmoe` (layer granularity) | ~33% | ~33% |
| Dynamic, oracle top-42 | 81.3% | 89.1% |
| Dynamic, warm top-42 (learn 1st half, test 2nd) | **73.4%** | **87.3%** |

Oracle hit-rate curve (code / prose): 12% budget -> 51% / 58%, 25% -> 72% / 81%,
33% -> 81% / 89%, 50% -> 93% / 98%.

**Verdict: build the dynamic cache.** Across both domains, with broad and only
modestly-skewed routing (Gini ~0.2, ~90-100 of 128 experts used per layer), a
frequency cache that keeps each layer's *hottest* experts resident hits
**73-87% (warm)** vs ~33% (static `-ncmoe`) at equal VRAM — a >2x reduction in
expert misses, and prose is even more cacheable than code. Budget is far better
spent on the popular experts of *every* layer than on *all* experts of a third
of the layers. The ~41-45% lag-1 overlap means a naive reactive (previous-token)
predictor tops out there, so a frequency/LFU-style resident cache is the better
mechanism than per-token reactive prefetch.

Caveat: two domains (code, prose); both confirm. A topic-switching worst-case
trace is still worth capturing but is unlikely to change the verdict.

## Cache policy selection (offline simulation)

Before writing the CUDA cache, the routing traces were replayed through a
per-layer expert cache (M = 42 of 128 slots, 33% budget) under several eviction
policies to find the realized *online* hit rate and how close each gets to the
Belady optimal (`simulate_cache.py`):

| Policy | code | prose | of optimal |
|---|---:|---:|---:|
| static frequency (warmup top-M) | 73.4% | 87.3% | 81-93% |
| **LRU** | **83.1%** | **87.8%** | **92-94%** |
| LRU-2 (K=2) | 81.9% | 87.7% | 90-94% |
| LFU | 78.3% | 86.5% | 86-93% |
| Belady (optimal) | 90.6% | 93.5% | — |

**Design decision: plain LRU.** It beats static placement, LFU, and even the
LRU-K(2) the original plan proposed, and reaches 92-94% of the Belady ceiling.
The simplest policy is the best, so v1 needs no frequency table, no K-history,
and no predictor.

**Roadmap simplification.** Reactive LRU already lands within ~8% of the
theoretical optimum, so the plan's Stages 2-4 (history / Markov / neural-net
predictors, ~5+ weeks) chase a marginal gap and are very likely unnecessary on
this model. The project reduces to **one stage**: a per-layer LRU expert cache in
VRAM, with the GPU computing resident (cached) experts and the CPU handling
misses. Target realized hit rate ~83-88% at a 33% VRAM budget.

## Keystone experiment — forcing decode experts onto the GPU (no cache)

llama.cpp keeps batch-1 (decode) expert GEMMs on the CPU because the CUDA
backend only offloads a host-weight op when `batch_size >=
op_offload_min_batch_size` (default 32, env `GGML_OP_OFFLOAD_MIN_BATCH`;
`ggml_backend_cuda_device_offload_op` in ggml-cuda.cu). Setting it to 1 forces
decode MoE onto the GPU — with the existing per-token expert copy and **no
persistent cache**:

| Decode (`-ncmoe 48`, `-p 0 -n 128`) | tg t/s |
|---|---:|
| CPU experts (default `min_batch=32`) | 9.88 |
| GPU experts forced (`min_batch=1`, per-token copy, no cache) | 4.61 |

Forcing experts to the GPU naively is **2.1x slower** than the CPU baseline,
confirming the core thesis: a cold per-token copy loses to CPU compute. The win
exists only if experts are **resident** so the copy is skipped on cache hits.

## Build plan (the one remaining stage)

Everything above reduces the project to a single, well-specified build:

1. **Force decode MoE onto the GPU** — `GGML_OP_OFFLOAD_MIN_BATCH=1` (no code).
2. **Persistent VRAM expert cache** with plain **LRU** eviction. Today the
   scheduler's "copy only used experts" path (ggml-backend.cpp) copies *all*
   routed experts into a transient per-graph buffer every forward pass. Replace
   that with a persistent VRAM slot pool keyed by (layer, expert): on a hit,
   reuse the resident slot (no copy); on a miss, copy into an LRU-evicted slot
   and keep it resident across tokens. The `mul_mat_id` GEMM must read each
   expert from its slot (gather by cached pointer/offset).
3. **Leave misses' copies synchronous for v1** — at the measured ~83-88% LRU hit
   rate, per-token copy volume drops from 8 experts/layer to ~1-1.4, which alone
   should lift the GPU path back above the 9.9 t/s CPU baseline toward the
   resident-expert ceiling (16-19 t/s seen in the `-ncmoe` sweep). A later
   refinement can route misses to CPU compute instead of copying.

Expected v1 outcome at 33% VRAM budget: ~83-88% of expert GEMMs served resident
on GPU, decode in the high-teens t/s vs 9.9 baseline. Stages 2-4 predictors are
deferred/dropped (reactive LRU is already ~92-94% of Belady).

### Implementation progress

- **Cache core — done & validated.** `ggml/src/ggml-cuda/expert-cache.h` is an
  O(1) `expert_cache_lru` (hash map + intrusive recency list) mapping packed
  keys to a fixed slot pool. The standalone `test_expert_cache.cpp` replays the
  routing traces and reproduces `simulate_cache.py` exactly (83.1% code / 87.8%
  prose per-layer), so the policy logic is correct independent of any GPU run. A
  single **global pool** edges out per-layer partitioning (84.1% / 89.2%) and
  is the chosen layout.

- **Key architectural constraint (found in code).** The scheduler's `input_cpy`
  copy tensors are persistent per backend, but the graph allocator *reuses their
  VRAM across layers* (that is exactly why experts are re-copied every pass). So
  the cache pool must be a **dedicated cudaMalloc the allocator never touches**;
  the copy path then sources hits VRAM->VRAM from the pool and misses host->VRAM,
  while `input_cpy` is still populated at natural offsets so the GEMM is
  unchanged.

- **CUDA cache — built & working.** `expert-cache.cu` owns a dedicated VRAM
  slot pool, registered via the `ggml_backend_set_moe_expert_cache_cpy` hook
  (ggml-backend-impl.h) and invoked per expert from the MoE copy path in
  ggml-backend.cpp. Key = the expert's stable host address; hit -> D2D
  slot->`input_cpy`, miss -> H2D host->slot then D2D. Enable with
  `GGML_MOE_EXPERT_CACHE=1` (plus `GGML_OP_OFFLOAD_MIN_BATCH=1` to force decode
  onto the GPU); size with `GGML_MOE_EXPERT_CACHE_BYTES`.

  **Validation (Qwen3-30B-A3B, `-ncmoe 48`, decode):**

  | Config | tg t/s |
  |---|---:|
  | CPU experts (baseline) | 9.9 |
  | GPU-forced, no cache | ~7-8 |
  | GPU-forced, cache 2 GiB pool | 9.3 |
  | GPU-forced, cache 4 GiB pool | **12.2** |

  Correctness: cache-on vs cache-off generate **bit-identical** tokens (the cache
  is lossless), confirmed with a seeded `llama-cli` run. The cache scales with
  pool size and at 4 GiB beats the CPU baseline by ~23%.

  **Measured runtime hit rate** (`GGML_MOE_EXPERT_CACHE_STATS=1`, `-n 256`):

  | Pool | budget | hit rate | tg t/s |
  |---|---|---:|---:|
  | 2 GiB | ~13% | 75.3% | 8.7 |
  | 4 GiB | ~26% | 91.7% | 13.2 |

  At 4 GiB the realized hit rate (91.7%) exceeds the offline estimate — the
  global pool reuses hot experts heavily over a long generation. Misses are
  already ~8% (~85 MB/token), so the remaining gap to the residency ceiling
  (16-19 t/s) is dominated by the **1,152 individual D2D memcpy launches per
  token** (48 layers x 3 projections x 8 experts), i.e. launch overhead, not
  transfer bytes. Removing those (have the GEMM gather weights directly from
  their slots instead of copying into `input_cpy`) is the next real lever.

  **Gotcha:** the pool is dedicated VRAM that competes with the KV cache and
  compute buffers. It is sized from *free* VRAM (`cudaMemGetInfo`) with a 1.5 GiB
  headroom; a too-large fixed pool OOM-crashes the model (observed at a blind
  3 GiB pool with a large context). Default budget is a conservative 2 GiB.

- **Batched copies (`GGML_MOE_EXPERT_CACHE_BATCH=1`).** The per-token cost was
  dominated by ~1,152 individual `cudaMemcpyAsync` launches (48 layers x 3 proj x
  8 experts), not bytes. Replacing them with `cudaMemcpyBatchAsync` (CUDA 12.8+;
  version-guarded, falls back to per-expert) collapses each MoE input to two
  batched calls: a *populate* batch (misses host->slot) then a *serve* batch
  (all experts slot->`input_cpy`, D2D), stream-ordered so serve sees the loaded
  slots. Controlled A/B (4 GiB pool, same 91.8% hit): **per-expert 11.3 -> batch
  12.9 t/s (+14%)**, and the batch output is bit-identical to cache-off (still
  lossless). Opt-in for now; the default stays per-expert (portable to any CUDA).

- **Remaining.** Pin host memory so the miss H2D is truly async; have the GEMM
  read resident experts directly from their slots to drop the serve D2D entirely
  (closes the rest of the gap to the `-ncmoe` residency ceiling, 16-19 t/s).

## Speculative bets — Stage 0 go/no-go (both negative)

Two ideas from the second plan were evaluated with cheap, model-free checks
before any build. Both are no-gos for Qwen3-30B-A3B.

**Expert weight synthesis** (`validate_synthesis.py`) — could experts be stored
as a small latent code + a shared decoder? SVD across the 128 experts of a layer
(Q4_K dequantized to f32):

| layer | dims for 85% var (want <64) | fidelity>0.92 @K=32 (want >60%) |
|---|---:|---:|
| 4 | 107 | 22.7% |
| 24 | 107 | 23.4% |
| 40 | 107 | 23.4% |

The expert matrices are nearly full-rank (need 107/128 dims for 85% variance) and
a 32-float code reconstructs only ~23% of experts acceptably. No shared
manifold -> **abandon synthesis**. (Measured on dequantized Q4_K, but 107 >> 64
leaves no doubt.)

**Causal commitment windows** (`simulate_commitment.py`) — lock routing for N
tokens to raise the hit rate. Simulated on the routing traces:

| N | forced hit (code/prose) | routing fidelity (code/prose) |
|---|---|---|
| 2 | 89% / 92% | 40% / 44% |
| 8 | 95% / 97% | 33% / 37% |

The hit rate is already 83-88% without locking, so the gain is small, while
fidelity is brutal: even at N=2 only ~40% of forced experts match the natural
choice (~60% are wrong on locked tokens). Routing is too unstable (lag-1 overlap
~41%) to lock without wrecking quality -> **not worth building**.

## Popularity-tiered precision -- Stage 0 (GO)

Synthesis failed because experts are incompressible in the *weights*. But the
*traffic* is highly skewed (top 25% of experts handle 73-81% of activations;
bottom 50% handle 2.5-6.7%), so spend bits by popularity: hot experts at high
precision, cold experts at low precision. Simulated per-layer split
(`simulate_mixed_precision.py`, aggregated code+prose traces):

| keep hot (Q4_K) | cold | footprint vs all-Q4 | low-prec exposure |
|---|---|---:|---:|
| 50% | Q2_K | 79% | 8.9% |
| 50% | IQ1  | 68% | 8.9% |
| 40% | IQ1  | 61% | 15.4% |
| 25% | IQ1  | 52% | 32.2% |

At keep-50%, only ~9% of token-expert activations touch a low-precision expert
while the experts shrink 21-32%. Three wins: smaller RAM footprint (bigger models
fit), more experts fit the VRAM cache (higher hit rate), fewer bytes per cold
miss. Verdict: viable -- the skew synthesis couldn't find in the weights is real
in the bits.

Implementation caveat: GGUF stacks all experts of a layer into one tensor with a
single quant type, so per-expert precision needs reorganizing the stacked expert
tensors (hot/cold sub-tensors) -- a moderate build, not a flag. Exposure is a
proxy; confirm real quality with perplexity before shipping.

## Caveats / methodology notes

- `GGML_MOE_PREFETCH_METRICS=1` enables the counters; they are written directly
  to **stderr** (not via the ggml log) because `llama-bench` installs a null log
  callback unless `-v` is passed. `GGML_MOE_PREFETCH_METRICS_INTERVAL` (default
  128, set to 48 by the runner) controls dump frequency; the analyzer reads the
  last cumulative sample, so a run shorter than one interval reports nothing.
- The counters are **cumulative per process** and span warmup + all pp/tg
  repetitions; they do not separate prefill from decode within one run (hence the
  separate single-phase probes above).
- `schedule_us` conflates the synchronous copy time with **cold mmap page-faults**
  on the warmup pass (first touch of the 18.6 GB file from disk), so it is not a
  clean PCIe figure. Treat the per-repetition `t/s` as the reliable performance
  numbers and `payload_bytes` / `expert_slots` as the reliable traffic numbers.

## Tesla P40 (guanshiyin) — cross-platform validation + headline result

Second machine: **NVIDIA Tesla P40** (Pascal sm_61, 24 GB GDDR5, PCIe 3.0) + ~80 GB
RAM (only ~40 GB usable under load), CUDA 12.4, gcc-12. Built with
`-DCMAKE_CUDA_ARCHITECTURES=61`. Model: **Qwen3-30B-A3B Q8_0** (32.5 GB, experts in
RAM via `-ot ".ffn_.*_exps.=CPU" --no-mmap`, non-experts on GPU `-ngl 999`,
ctx 65536). Expert cache pool sized from free VRAM: 16.00 GiB = 10,277 slots
(~56% of the 18,432 expert-slots resident). `mode=per-expert` (CUDA 12.4 < 12.8,
so the `cudaMemcpyBatchAsync` batched path falls back — no +14% here).

Same ~20k-token prompt, decode measured by the server's `eval time` line:

| phase | cache OFF (decode on CPU) | cache ON (GPU + LRU cache) | speedup |
|---|---|---|---|
| **decode (eval)** | **2.60 t/s** | **14.39 t/s** | **5.5×** |
| prefill (prompt eval) | ~88–117 t/s | ~91–114 t/s | ~unchanged |

Decode hit rate (isolating the decode slice between cumulative stat dumps, since
prefill misses dominate the running total): **~97%** — e.g. accesses 600k→800k
went 5.0%→28.1% cumulative = 194.8k hits / 200k accesses on the decode tokens.
Miss traffic over a 200k-access decode window was only ~7 GiB (~42 MiB/token over
PCIe — negligible).

Why the P40 win (5.5×) dwarfs the 3070 win (+30%):
1. **Weak host CPU** — 2.6 t/s is the entire CPU decode ceiling on this box, a low
   bar for the GPU to clear.
2. **24 GB VRAM** holds 56% of experts → 97% decode hit → nearly every expert GEMM
   runs on GPU from resident VRAM.
3. **GPU GEMM ≈ 5.5× the CPU's** for Q8 expert matmuls.

Prefill is **net-neutral**: the per-expert "thrashing" during prefill (working set
= all experts > pool → ~1% hit, ~900 GiB of wasted H2D over a long run) costs no
measurable prefill t/s, because prefill is GEMM-bound and the H2D overlaps. So the
cache is pure upside on this box.

**This is the project's headline result**: the LRU expert cache turns a 30B MoE
that does not fit in 24 GB VRAM from *unusably slow* (2.6 t/s, CPU-bound) into
*perfectly usable* (14.4 t/s, GPU-speed) by keeping hot experts resident and
streaming the ~3% of misses from RAM — exactly the original goal of running models
too large for VRAM at GPU speed. (Cross-platform: identical code path, no source
changes; CUDA-version guard handles the batched-copy difference.)

A clear next improvement: **bypass the cache for large (prefill) batches** and
engage it only for small (decode) batches — eliminates the ~900 GiB of pointless
prefill H2D with zero downside, since prefill gets no benefit from it.

## Optimization pass 2 (2026-06-29/30): what the metrics killed

After Stage A (gather), measured where offloaded-MoE decode time actually goes
(Qwen3-30B-A3B Q4_K_M, RTX 3070, 2 GiB pool, `GGML_MOE_PREFETCH_METRICS`). Three
findings, two of them negative -- all data-driven:

**Finding 1 -- the per-layer host sync is negligible (Stage B is dead).**
`ids_us` (the device->host routing readback + synchronize the scheduler does per
MoE layer) totalled **~64 ms over a 27 s run = 0.25%**. So moving routing
on-device to "kill the sync" (Stage B) and CUDA-graph-capturing the result
(Stage C) would buy ~nothing. Also learned: CUDA graphs are **already enabled**
for quantized batch-1 MoE decode (upstream `[TAG_MUL_MAT_ID_CUDA_GRAPHS]`, PR
18958) -- the premise that MoE can't be captured was outdated. Graph/heterogeneous
rewrite abandoned on evidence.

**Finding 2 -- `--no-mmap` is the real lever (config, not code).** The dominant
cost looked like expert copies (`schedule_us` = 25.1 s, decode 6.64 t/s) but that
was **mmap cold-page DISK faults**, not PCIe bandwidth. With `--no-mmap` (model
fully in RAM): `schedule_us` 25.1 s -> 5.4 s, decode **6.64 -> 15.43 t/s (2.3x)**.

**Finding 3 -- pinned host memory is a NO-GO.** Hypothesis: page-lock the expert
source so miss H2D is async + 2x faster. Built it (`GGML_MOE_EXPERT_CACHE_PIN`,
`cudaHostRegister` the source buffer). Result with the model already RAM-resident:
PIN=1 was **slower** (14.66 vs 15.43 t/s), `schedule_us` rose 5.4 -> 6.7 s. The
registration overhead is real and the benefit nil -- once experts are in RAM the
copies aren't PCIe-bound; decode is bound by GEMM + the D2D slot->input_cpy serve.
Lossless (bit-identical) but useless -> reverted.

**Net:** the LRU cache is near its practical ceiling for the RAM-resident case.
Remaining real levers: **speculative decoding** (amortize the per-token copy over
a batch-K verify), **residency** (bigger pool / lower-precision experts), and
**gather-from-slots** (delete the D2D serve -- the one deep code lever the data
doesn't rule out). The sync/graph/pinning ideas are closed.

## Expert heat persistence + startup pre-warm (colibri-inspired, 2026-07-14)

Adopted the "learned usage" idea from JustVugg/colibri (which streams a 744B MoE
from disk using persisted expert-heat stats to pin hot experts): our cache now
tracks per-(tensor, expert) access counts, persists them across runs, and
pre-warms the VRAM pool with the known-hot experts at startup. Enable with
`GGML_MOE_EXPERT_CACHE_HEAT=<file>` (off by default). First run tracks + saves;
later runs load, pre-warm (pool divided evenly across tensors in the file), and
keep accumulating.

RTX 3070, Qwen3-30B-A3B Q4_K_M, 3 GiB pool (3,638 slots), 128-token greedy:

| metric | no heat | pre-warmed | delta |
|---|---|---|---|
| decode | 14.39 t/s | 15.68 t/s | +9% |
| prompt eval | 5.82 t/s | 9.54 t/s | **+64%** |
| hit rate (run) | -- | 69.6% | prewarmed=3600 |
| output | -- | -- | bit-identical |

Pre-warm filled ~99% of the pool before the first token, so cold-start misses
(concentrated in prefill) mostly vanish. Two operational notes: (1) the default
2 GiB pool THRASHES on this model (13% of experts < per-layer working set ->
hit_rate 0.0%); 3 GiB is past the cliff. Pool size is make-or-break and worth a
stats check per model/GPU. (2) a 4 GiB pool on the 8 GB display GPU
oversubscribes WDDM and "locks up" (VRAM paging) -- headroom matters on the
card driving the desktop. (3) Q4_K_M's down_exps use a different quant/expert
size than gate/up, so they fall back to the stock copy path (120/144 tensors
cached); a per-size pool is a possible follow-up.

Value: biggest on the P40 server (every restart starts warm) and a prerequisite
for the disk tier -- the heat file IS the policy for which experts deserve RAM.

## Disk tier: first real-world test (P40 + GLM-4.5-Air, 2026-07-16)

Setup: guanshiyin (P40 24GB, ~40GB usable host RAM, LXC on Unraid, storage
measured ~110 MB/s effective), GLM-4.5-Air IQ4_XS (58GB, experts ~47GB), mmap +
GGML_MOE_EXPERT_DISK=1, pin 24GiB, cold LRU 4GiB, VRAM pool 12GiB (4204 slots),
22.5k-token coding prompt (real workload).

Results:
- prefill 22,535 tok: **12.02 t/s (31 min)** -- vs 2.14 t/s for Q4_K_XL (73GB)
- decode: **0.92 t/s**, follow-up prefill via prompt cache 2.1k suffix in 2.8min
- pager: **pread_GiB=838.63, read_fail=0** over ~294k parallel whole-expert
  reads; host stable with pin capped at 24GiB. Machinery correct.
- pin_hit ~= pread (~50/50): the heat file was warmup-only (uniform), so the
  pinned 24GiB was effectively a random sample. Cold LRU (4GiB) useless at this
  working set (ram_hit=994 of ~294k). VRAM hit 0.3-0.7% during prefill
  (expected; prefill = full expert sweep), 11.9% cumulative after decode.

Verdict: the disk tier is CORRECT but this box's storage (110 MB/s) cannot
carry a model whose expert set exceeds RAM: one 22.5k prefill = 838GB of I/O.
Lessons: (1) never size the mlock budget from container-reported free RAM --
50GiB pin starved the Unraid host (mlock is unreclaimable; 24GiB cap is safe);
(2) heat must come from decode traffic, warmup/prefill heat is uniform junk;
(3) GLM down_exps (mismatched expert size) bypass the VRAM pool -> multi-size
pool is the top code lever if this workload returns; (4) the LXC "80GB free"
is the host's RAM, not the container's to lock.

Path forward (hardware, not software): NVMe (~27x storage) + cheap DDR3 ECC
(80->256GB, model fully RAM-resident) turns this same code into Qwen-class
speeds for GLM-Air Q4 and makes 500GB-class models colibri-viable for decode.
Until then: Qwen3-30B Q8 (fits in RAM, 14.4 t/s proven) is the coding driver.
