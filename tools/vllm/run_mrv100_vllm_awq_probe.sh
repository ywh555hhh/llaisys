#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-/data/src/vllm-mrv100-probe}
MODEL=${MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}
LOCAL_DIR=${LOCAL_DIR:-/data/models/Qwen2.5-32B-Instruct-AWQ}
QUANTIZATION=${QUANTIZATION:-awq}
MODE=${MODE:-eager}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-512}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_TOKENS=${MAX_TOKENS:-16}
DTYPE=${DTYPE:-float16}
TIMEOUT_S=${TIMEOUT_S:-1200}
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
    if ! hf download "$MODEL" --local-dir "$LOCAL_DIR"; then
      python3 - <<PY
import json
from pathlib import Path
Path("$OUT").write_text(json.dumps({
  "model_id": "$MODEL",
  "model_dir": "$LOCAL_DIR",
  "ok": False,
  "stage": "download",
  "error": "hf download failed",
}, indent=2))
PY
      exit 1
    fi
  fi
  du -sh "$LOCAL_DIR"
  set +e
  timeout "$TIMEOUT_S" python3 "$PYFILE" \
    --model-id "$MODEL" \
    --model-dir "$LOCAL_DIR" \
    --out "$OUT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --batch-size "$BATCH_SIZE" \
    --max-tokens "$MAX_TOKENS" \
    --dtype "$DTYPE" \
    --quantization "$QUANTIZATION" \
    "${EXTRA[@]}"
  rc=$?
  set -e
  if [ "$rc" -ne 0 ] && [ ! -f "$OUT" ]; then
    python3 - <<PY
import json
from pathlib import Path
Path("$OUT").write_text(json.dumps({
  "model_id": "$MODEL",
  "model_dir": "$LOCAL_DIR",
  "ok": False,
  "stage": "run",
  "exit_code": $rc,
}, indent=2))
PY
  fi
  /usr/local/corex/bin/ixsmi || true
  exit "$rc"
} 2>&1 | tee "$LOG"
