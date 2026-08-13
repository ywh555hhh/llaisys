#!/usr/bin/env bash
set -euo pipefail

SRC=${SRC:-/data/src/vllm-mrv100-probe}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
SNAP=${SNAP:-/tmp/mrv100_vllm_probe_snapshot_${STAMP}}
OUT=${OUT:-/tmp/mrv100_vllm_probe_full_snapshot_${STAMP}.tgz}

mkdir -p "$SNAP"
cd "$SRC"

cp -a scripts artifacts logs "$SNAP/"

{
  echo "# MR-V100 vLLM probe remote snapshot"
  date -Iseconds
  echo
  echo "## source"
  pwd
  echo
  echo "## counts"
  find scripts artifacts logs -type f | wc -l
  du -sh scripts artifacts logs
  echo
  echo "## file tree"
  find scripts artifacts logs -type f | sort
} > "$SNAP/MANIFEST.txt"

{
  echo "# Selected environment"
  date -Iseconds
  echo
  echo "## OS"
  cat /etc/os-release 2>/dev/null || true
  echo
  echo "## kernel"
  uname -a
  echo
  echo "## CPU"
  lscpu 2>/dev/null || true
  echo
  echo "## disks"
  df -h / /data 2>/dev/null || df -h
  echo
  echo "## GPU"
  export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
  /usr/local/corex/bin/ixsmi 2>&1 || true
  echo
  /usr/local/corex/bin/ixsmi \
    --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,gpu.power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory \
    --format=csv 2>&1 || true
  echo
  echo "## Python/vLLM"
  python3 --version 2>&1 || true
  python3 -c 'import sys; print(sys.executable)' 2>&1 || true
  vllm --version 2>&1 || true
  python3 -m pip show vllm torch transformers xformers 2>/dev/null || true
} > "$SNAP/ENVIRONMENT.txt"

(
  cd "$SNAP"
  find . -type f | sort | xargs sha256sum > SHA256SUMS.txt
)

tar czf "$OUT" -C "$(dirname "$SNAP")" "$(basename "$SNAP")"
echo "$OUT"
ls -lh "$OUT"
tar tzf "$OUT" | wc -l
