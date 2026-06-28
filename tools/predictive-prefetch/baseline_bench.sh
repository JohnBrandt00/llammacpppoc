#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-./models/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf}"
BUILD_DIR="${BUILD_DIR:-./build}"
GPU_LAYERS="${GPU_LAYERS:-0 10 20 30 40 48}"
THREADS="${THREADS:-8}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
GEN_TOKENS="${GEN_TOKENS:-128}"
REPETITIONS="${REPETITIONS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-./metrics/predictive-prefetch}"

if [[ -x "$BUILD_DIR/bin/llama-bench" ]]; then
    BENCH="$BUILD_DIR/bin/llama-bench"
elif [[ -x "$BUILD_DIR/bin/Release/llama-bench" ]]; then
    BENCH="$BUILD_DIR/bin/Release/llama-bench"
else
    echo "Could not find llama-bench under $BUILD_DIR" >&2
    echo "Build with: cmake --build $BUILD_DIR --config Release --target llama-bench" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Model not found: $MODEL" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
stamp="$(date +%Y%m%d-%H%M%S)"
summary="$OUTPUT_DIR/baseline-$stamp.jsonl"

if [[ "${ENABLE_MOE_METRICS:-0}" != "0" ]]; then
    export GGML_MOE_PREFETCH_METRICS=1
    export GGML_MOE_PREFETCH_METRICS_INTERVAL="${GGML_MOE_PREFETCH_METRICS_INTERVAL:-48}"
fi

for ngl in $GPU_LAYERS; do
    log="$OUTPUT_DIR/baseline-$stamp-ngl-$ngl.log"
    echo "=== ngl=$ngl ==="
    echo "log: $log"

    "$BENCH" \
        -m "$MODEL" \
        -ngl "$ngl" \
        -t "$THREADS" \
        -p "$PROMPT_TOKENS" \
        -n "$GEN_TOKENS" \
        -r "$REPETITIONS" \
        -o jsonl \
        2>&1 | tee "$log"

    grep -E '^\{' "$log" >> "$summary" || true
done

echo "summary: $summary"
