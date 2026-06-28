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
