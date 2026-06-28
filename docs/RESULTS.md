# Predictive Expert Prefetch Results

This file tracks the Qwen3-30B-A3B expert-prefetch experiment.

## Stage 0 Baseline

Build:

```powershell
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release --target llama-bench
```

Model:

```powershell
huggingface-cli download bartowski/Qwen_Qwen3-30B-A3B-GGUF --include "Qwen_Qwen3-30B-A3B-Q4_K_M.gguf" --local-dir .\models
```

Benchmark:

```powershell
.\tools\predictive-prefetch\baseline_bench.ps1 -EnableMoeMetrics
```

For Linux or WSL:

```bash
ENABLE_MOE_METRICS=1 ./tools/predictive-prefetch/baseline_bench.sh
```

## Measurements

| Date | Commit | Model | GPU | ngl | pp t/s | tg t/s | MoE copy GB/token | Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| TBD | TBD | Qwen3-30B-A3B Q4_K_M | RTX 3070 8GB | 0 | TBD | TBD | TBD | Baseline |
| TBD | TBD | Qwen3-30B-A3B Q4_K_M | RTX 3070 8GB | 10 | TBD | TBD | TBD | Baseline |
| TBD | TBD | Qwen3-30B-A3B Q4_K_M | RTX 3070 8GB | 20 | TBD | TBD | TBD | Baseline |
| TBD | TBD | Qwen3-30B-A3B Q4_K_M | RTX 3070 8GB | 30 | TBD | TBD | TBD | Baseline |
| TBD | TBD | Qwen3-30B-A3B Q4_K_M | RTX 3070 8GB | 40 | TBD | TBD | TBD | Baseline |
| TBD | TBD | Qwen3-30B-A3B Q4_K_M | RTX 3070 8GB | 48 | TBD | TBD | TBD | Baseline |

## Notes

Set `GGML_MOE_PREFETCH_METRICS=1` to emit scheduler-side JSON-ish metric lines for the existing MoE expert slice copy path. These counters measure the current upstream behavior before any persistent VRAM cache is introduced.

Summarize those lines after a run with:

```powershell
python .\tools\predictive-prefetch\analyze_metrics.py .\metrics\predictive-prefetch\baseline-*-ngl-*.log
```
