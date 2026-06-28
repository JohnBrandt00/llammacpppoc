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
