#!/usr/bin/env bash
set -euo pipefail

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
SNAP=${SNAP:-/tmp/mrv100_misc_probe_snapshot_${STAMP}}
OUT=${OUT:-/tmp/mrv100_misc_probe_snapshot_${STAMP}.tgz}

mkdir -p "$SNAP/top-level-scripts" "$SNAP/data-src"
cd /data/src

for f in \
  deep_mrv100_repo_recon.sh \
  probe_mrv100_interfaces_step1.sh \
  probe_mrv100_interfaces_step2.sh \
  probe_mrv100_interfaces_step3.sh \
  write_mrv100_research_to_llaisys.sh; do
  if [ -f "$f" ]; then
    cp -a "$f" "$SNAP/top-level-scripts/"
  fi
done

if [ -d mrv100-hw-probe ]; then
  cp -a mrv100-hw-probe "$SNAP/data-src/"
fi

if [ -d sglang-omni-corex-probe ]; then
  mkdir -p "$SNAP/data-src/sglang-omni-corex-probe"
  for d in logs reports; do
    if [ -d "sglang-omni-corex-probe/$d" ]; then
      cp -a "sglang-omni-corex-probe/$d" "$SNAP/data-src/sglang-omni-corex-probe/"
    fi
  done
fi

if [ -d llaisys ]; then
  mkdir -p "$SNAP/data-src/llaisys"
  for d in docs reports tools scripts; do
    if [ -d "llaisys/$d" ]; then
      cp -a "llaisys/$d" "$SNAP/data-src/llaisys/"
    fi
  done
fi

{
  echo "# MR-V100 misc remote snapshot"
  date -Iseconds
  echo
  echo "## source roots"
  echo "/data/src/mrv100-hw-probe"
  echo "/data/src/sglang-omni-corex-probe/{logs,reports}"
  echo "/data/src/llaisys/{docs,reports,tools,scripts}"
  echo "/data/src/*.sh probe scripts"
  echo
  echo "## counts"
  find "$SNAP" -type f | wc -l
  du -sh "$SNAP"
  echo
  echo "## file tree"
  (cd "$SNAP" && find . -type f | sort)
} > "$SNAP/MANIFEST.txt"

(
  cd "$SNAP"
  find . -type f | sort | xargs sha256sum > SHA256SUMS.txt
)

tar czf "$OUT" -C "$(dirname "$SNAP")" "$(basename "$SNAP")"
echo "$OUT"
ls -lh "$OUT"
tar tzf "$OUT" | wc -l
