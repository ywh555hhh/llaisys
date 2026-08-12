#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-/data/src/vllm-mrv100-probe}
MODEL=${MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}
LOCAL_DIR=${LOCAL_DIR:-/data/models/Qwen2.5-32B-Instruct-AWQ}
QUANTIZATION=${QUANTIZATION:-awq}
MODE=${MODE:-eager}
SLUG=$(basename "$LOCAL_DIR")

mkdir -p "$WORK/logs" "$WORK/artifacts" /data/models

export PATH=/usr/local/corex/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:${PYTHONPATH:-}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/data/hf-cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/data/hf-cache/hub}
export VLLM_TARGET_DEVICE=cuda
export VLLM_WORKER_MULTIPROC_METHOD=spawn

PYFILE=$WORK/scripts/run_model_scale_probe.py
LOG=$WORK/logs/10_${QUANTIZATION}_${MODE}_${SLUG}.txt
OUT=$WORK/artifacts/10_${QUANTIZATION}_${MODE}_${SLUG}.json
EXTRA=()
if [ "$MODE" = "cudagraph" ]; then
  EXTRA+=(--cuda-graph)
elif [ "$MODE" = "eager" ]; then
  EXTRA+=(--enforce-eager)
else
  echo "Unknown MODE=$MODE; expected eager or cudagraph" >&2
  exit 2
fi

{
  echo "===== $MODEL $(date -Is) ====="
  /usr/local/corex/bin/ixsmi || true
  if [ ! -f "$LOCAL_DIR/config.json" ] || ! find "$LOCAL_DIR" -maxdepth 1 -type f -name '*.safetensors' | grep -q .; then
    mkdir -p "$LOCAL_DIR"
    echo "Downloading/resuming $MODEL to $LOCAL_DIR"
    hf download "$MODEL" --local-dir "$LOCAL_DIR"
  fi
  du -sh "$LOCAL_DIR"
  timeout 1200 python3 "$PYFILE" \
    --model-id "$MODEL" \
    --model-dir "$LOCAL_DIR" \
    --out "$OUT" \
    --max-model-len 512 \
    --gpu-memory-utilization 0.95 \
    --batch-size 1 \
    --max-tokens 16 \
    --dtype float16 \
    --quantization "$QUANTIZATION" \
    "${EXTRA[@]}"
  /usr/local/corex/bin/ixsmi || true
} 2>&1 | tee "$LOG"
