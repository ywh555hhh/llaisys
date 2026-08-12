#!/usr/bin/env bash
set -euo pipefail
WORK=${WORK:-/data/src/vllm-mrv100-probe}
mkdir -p "$WORK/logs" "$WORK/artifacts" "$WORK/reports" /data/models
export PATH=/usr/local/corex/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:${PYTHONPATH:-}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/data/hf-cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/data/hf-cache/hub}
export VLLM_TARGET_DEVICE=cuda
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PYFILE=$WORK/scripts/run_model_scale_probe.py
# model_id|local_dir|max_model_len|gpu_mem|batch|max_tokens|mode
MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct|/data/models/Qwen2.5-0.5B-Instruct|1024|0.45|4|64|cudagraph"
  "Qwen/Qwen2.5-1.5B-Instruct|/data/models/Qwen2.5-1.5B-Instruct|2048|0.80|4|64|cudagraph"
  "Qwen/Qwen2.5-3B-Instruct|/data/models/Qwen2.5-3B-Instruct|2048|0.86|2|64|cudagraph"
  "Qwen/Qwen2.5-7B-Instruct|/data/models/Qwen2.5-7B-Instruct|2048|0.90|1|64|cudagraph"
)
for item in "${MODELS[@]}"; do
  IFS='|' read -r model local_dir max_len gpu_mem batch max_tokens mode <<< "$item"
  slug=$(basename "$local_dir")
  log="$WORK/logs/08_scale_${slug}.txt"
  out="$WORK/artifacts/08_scale_${slug}.json"
  echo "===== $model $(date -Is) =====" | tee "$log"
  /usr/local/corex/bin/ixsmi | tee -a "$log" || true
  if [ ! -f "$local_dir/config.json" ] || ! find "$local_dir" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) | grep -q .; then
    mkdir -p "$local_dir"
    echo "Downloading $model to $local_dir" | tee -a "$log"
    if ! hf download "$model" --local-dir "$local_dir" 2>&1 | tee -a "$log"; then
      echo "DOWNLOAD_FAILED $model" | tee -a "$log"
      python3 - <<PY
import json
open('$out','w').write(json.dumps({'model_id':'$model','model_dir':'$local_dir','ok':False,'stage':'download','error':'download failed'}, indent=2))
PY
      continue
    fi
  fi
  du -sh "$local_dir" | tee -a "$log"
  extra=()
  if [ "$mode" = "cudagraph" ]; then extra+=(--cuda-graph); else extra+=(--enforce-eager); fi
  set +e
  timeout 900 python3 "$PYFILE" \
    --model-id "$model" \
    --model-dir "$local_dir" \
    --out "$out" \
    --max-model-len "$max_len" \
    --gpu-memory-utilization "$gpu_mem" \
    --batch-size "$batch" \
    --max-tokens "$max_tokens" \
    "${extra[@]}" 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "RUN_EXIT_CODE=$rc" | tee -a "$log"
    if [ ! -f "$out" ]; then
      python3 - <<PY
import json
open('$out','w').write(json.dumps({'model_id':'$model','model_dir':'$local_dir','ok':False,'stage':'run','exit_code':$rc}, indent=2))
PY
    fi
  fi
  /usr/local/corex/bin/ixsmi | tee -a "$log" || true
  python3 - <<'PY' || true
import gc, torch
try:
    torch.cuda.empty_cache(); gc.collect()
except Exception as e:
    print('cleanup error', repr(e))
PY
  sleep 5
done
python3 - <<'PY'
import json, pathlib
root=pathlib.Path('/data/src/vllm-mrv100-probe/artifacts')
summary=[]
for p in sorted(root.glob('08_scale_*.json')):
    try: obj=json.load(open(p))
    except Exception as e: obj={'ok':False,'error':repr(e)}
    row={'file':str(p),'model_id':obj.get('model_id'),'ok':obj.get('ok'),'stage':obj.get('stage'),'error':obj.get('error')}
    cfg=obj.get('model_config') or {}
    row.update({k:cfg.get(k) for k in ['hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads','torch_dtype']})
    if obj.get('ok'):
        row['load_s']=obj.get('timing_s',{}).get('load')
        for name, run in (obj.get('runs') or {}).items():
            row[f'{name}_output_tok_s']=run.get('output_tok_s')
            row[f'{name}_elapsed_s']=run.get('elapsed_s')
            row[f'{name}_input_tokens']=run.get('input_tokens')
            row[f'{name}_output_tokens']=run.get('output_tokens')
    summary.append(row)
print(json.dumps(summary, indent=2, ensure_ascii=False))
(root/'08_scale_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
PY
