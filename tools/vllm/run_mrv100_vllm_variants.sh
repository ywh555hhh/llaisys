#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORK=${WORK:-/data/src/vllm-mrv100-probe}
MODEL=${MODEL:-/data/models/Qwen2.5-0.5B-Instruct}
PYFILE=${PYFILE:-$SCRIPT_DIR/run_mrv100_vllm_variant.py}

mkdir -p "$WORK/logs" "$WORK/artifacts" "$WORK/scripts" "$WORK/reports"

export PATH=/usr/local/corex/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:${PYTHONPATH:-}
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/hf-cache
export VLLM_TARGET_DEVICE=cuda
export VLLM_WORKER_MULTIPROC_METHOD=spawn

run_variant() {
  name=$1; shift
  log=$WORK/logs/07_variant_${name}.txt
  out=$WORK/artifacts/07_variant_${name}.json
  echo "===== $name $(date -Is) =====" | tee "$log"
  /usr/local/corex/bin/ixsmi | tee -a "$log" || true
  python3 "$PYFILE" --model "$MODEL" --variant "$name" --out "$out" "$@" 2>&1 | tee -a "$log"
  /usr/local/corex/bin/ixsmi | tee -a "$log" || true
}
run_variant eager_prefix_on --enforce-eager --prefix-cache on
run_variant eager_prefix_off --enforce-eager --prefix-cache off
run_variant cudagraph_prefix_on --cuda-graph --prefix-cache on
WORK="$WORK" python3 - <<'PY'
import json, os, pathlib
root=pathlib.Path(os.environ['WORK']) / 'artifacts'
summary=[]
for p in sorted(root.glob('07_variant_*.json')):
    obj=json.load(open(p))
    row={'file': str(p), 'variant': obj.get('variant'), 'ok': obj.get('ok'), 'error': obj.get('error')}
    if obj.get('ok'):
        row['load_s']=obj.get('timing_s',{}).get('load')
        for k,v in obj.get('runs',{}).items():
            row[k+'_s']=v.get('elapsed_s')
            row[k+'_out_tok_s']=v.get('output_tok_s')
            row[k+'_in_toks']=v.get('input_tokens')
            row[k+'_out_toks']=v.get('output_tokens')
    summary.append(row)
print(json.dumps(summary, indent=2, ensure_ascii=False))
(root / '07_variants_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
PY
