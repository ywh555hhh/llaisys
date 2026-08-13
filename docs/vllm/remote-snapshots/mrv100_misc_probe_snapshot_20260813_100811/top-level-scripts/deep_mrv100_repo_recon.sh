set -u
for d in /data/src/llaisys /data/src/sglang-omni-corex-probe/repo /data/src/mrv100-hw-probe; do
  echo "=== $d ==="
  if [ -d "$d/.git" ]; then
    cd "$d"
    git status --short || true
    git branch --show-current || true
    git remote -v || true
    git log --oneline -5 || true
  else
    echo "no git repo"
    ls -la "$d" 2>/dev/null | head || true
  fi
done
