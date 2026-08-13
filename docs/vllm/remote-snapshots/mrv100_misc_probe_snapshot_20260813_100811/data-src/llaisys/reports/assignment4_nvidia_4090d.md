# Assignment 4 NVIDIA acceptance - RTX 4090D

Date: 2026-08-11
Host: suanjiayun-nvidia-node-0001 / JIJEAARNPC0SEINK
Repo: /data/src/llaisys
GPU: NVIDIA GeForce RTX 4090 D, 24564 MiB, driver 570.124.06
CUDA: 12.8, nvcc Build cuda_12.8.r12.8/compiler.35404655_0
xmake: v2.8.7+20240401
Python venv: /data/venvs/ai-infra
Model: /data/models/DeepSeek-R1-Distill-Qwen-1.5B

## Build

Command:
- xmake f --nv-gpu=y -cv
- xmake -r
- xmake install
- python -m pip install -e ./python --no-build-isolation

Result: PASS. Final NVIDIA build completed and installed.

## Runtime

Command: python test/test_runtime.py --device nvidia
Result: PASS. Found 1 nvidia device.

## NVIDIA op tests

All commands used --device nvidia.

| Test | Result |
| --- | --- |
| test/ops/add.py | PASS |
| test/ops/argmax.py | PASS |
| test/ops/embedding.py | PASS |
| test/ops/swiglu.py | PASS |
| test/ops/rms_norm.py | PASS |
| test/ops/rope.py | PASS |
| test/ops/self_attention.py | PASS |
| test/ops/linear.py | PASS |

## Inference smoke

Command: python test/test_infer.py --model /data/models/DeepSeek-R1-Distill-Qwen-1.5B --test --device nvidia
Result: PASS. Answer tokens and local result tokens matched exactly.
Elapsed wall time from shell TIMEFORMAT: 12.966 sec.
Reported generation times: reference 2.00s, LLAISYS nvidia path 2.70s.

## CPU regression

Command set: xmake f --nv-gpu=n -cv; xmake -r; xmake install; runtime + all op tests on --device cpu.
Result: PASS for runtime, add, argmax, embedding, swiglu, rms_norm, rope, self_attention, linear.

## Notes

- Upstream has no GitHub Actions GPU CI for assignment 4, so this report is the local 4090D acceptance record.
- CUDA kernels are correctness-first, not performance-optimized.
- test/ops/self_attention.py needed a device-consistency fix for the PyTorch reference mask when running on CUDA.
