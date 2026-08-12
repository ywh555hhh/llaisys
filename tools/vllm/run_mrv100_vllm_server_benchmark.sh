#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-/data/src/vllm-mrv100-probe}
MODEL_DIR=${MODEL_DIR:-/data/models/Qwen2.5-32B-Instruct-AWQ}
MODEL_NAME=${MODEL_NAME:-qwen32b-awq}
PORT=${PORT:-8000}
HOST=${HOST:-127.0.0.1}
QUANTIZATION=${QUANTIZATION:-awq_marlin}
MODE=${MODE:-cudagraph}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-512}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
PROMPT_TOKENS=${PROMPT_TOKENS:-64}
MAX_TOKENS=${MAX_TOKENS:-32}
PROMPT_MODE=${PROMPT_MODE:-shared}
SEED=${SEED:-20260812}
CONCURRENCY_LIST=${CONCURRENCY_LIST:-1,2,4}
REQUESTS_PER_CONCURRENCY=${REQUESTS_PER_CONCURRENCY:-4}
LABEL=${LABEL:-qwen32b_awq_marlin_cudagraph_server}

mkdir -p "$WORK/logs" "$WORK/artifacts" "$WORK/scripts"

export PATH=/usr/local/corex/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:${PYTHONPATH:-}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/data/hf-cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/data/hf-cache/hub}
export VLLM_TARGET_DEVICE=cuda
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [ "$MODE" = "cudagraph" ]; then
  export VLLM_ENFORCE_CUDA_GRAPH=1
  MODE_ARGS=()
elif [ "$MODE" = "eager" ]; then
  MODE_ARGS=(--enforce-eager)
else
  echo "Unknown MODE=$MODE; expected cudagraph or eager" >&2
  exit 2
fi

SERVER_LOG="$WORK/logs/15_server_${LABEL}.log"
BENCH_LOG="$WORK/logs/15_server_benchmark_${LABEL}.log"
OUT="$WORK/artifacts/15_server_benchmark_${LABEL}.json"
PID_FILE="$WORK/artifacts/15_server_${LABEL}.pid"
BENCH="$WORK/scripts/run_mrv100_vllm_server_benchmark.py"

cleanup() {
  if [ -f "$PID_FILE" ]; then
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      sleep 5
      kill -KILL "-$pid" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT

echo "===== starting vLLM server $(date -Is) =====" | tee "$SERVER_LOG"
/usr/local/corex/bin/ixsmi | tee -a "$SERVER_LOG" || true
setsid vllm serve "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$MODEL_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --quantization "$QUANTIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  "${MODE_ARGS[@]}" >> "$SERVER_LOG" 2>&1 &
echo $! > "$PID_FILE"

echo "server pid $(cat "$PID_FILE")" | tee -a "$SERVER_LOG"
for i in $(seq 1 240); do
  if curl -fsS "http://$HOST:$PORT/v1/models" >/tmp/vllm_models_${PORT}.json 2>/dev/null; then
    echo "server ready after ${i}s" | tee -a "$SERVER_LOG"
    cat /tmp/vllm_models_${PORT}.json | tee -a "$SERVER_LOG"
    break
  fi
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "server exited before readiness" | tee -a "$SERVER_LOG"
    tail -200 "$SERVER_LOG"
    exit 1
  fi
  sleep 1
  if [ "$i" = "240" ]; then
    echo "server readiness timeout" | tee -a "$SERVER_LOG"
    tail -200 "$SERVER_LOG"
    exit 1
  fi
done

python3 "$BENCH" \
  --base-url "http://$HOST:$PORT/v1" \
  --model "$MODEL_NAME" \
  --tokenizer-dir "$MODEL_DIR" \
  --out "$OUT" \
  --prompt-tokens "$PROMPT_TOKENS" \
  --max-tokens "$MAX_TOKENS" \
  --prompt-mode "$PROMPT_MODE" \
  --seed "$SEED" \
  --concurrency-list "$CONCURRENCY_LIST" \
  --requests-per-concurrency "$REQUESTS_PER_CONCURRENCY" \
  --label "$LABEL" 2>&1 | tee "$BENCH_LOG"

/usr/local/corex/bin/ixsmi | tee -a "$BENCH_LOG" || true
echo "wrote $OUT"
