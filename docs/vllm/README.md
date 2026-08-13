# MR-V100 vLLM/CoreX Experiment Index

This folder documents a small inference-systems investigation on one
Iluvatar MR-V100 32 GiB card using a CoreX vLLM image. It is meant to be read as
an engineering trace: what ran, what failed, which data supports the claim, and
where to reproduce or inspect the raw files.

## Read First

| File | Role | Read when you want to know... |
|---|---|---|
| [`mrv100-vllm-initial-exploration.md`](mrv100-vllm-initial-exploration.md) | Environment and first smoke test | Whether vLLM, Torch, model download, and basic generation worked on MR-V100. |
| [`mrv100-vllm-scale-limit.md`](mrv100-vllm-scale-limit.md) | Model scale ladder | The largest Qwen2.5 models proven runnable on one 32 GiB MR-V100 card, including 14B fp16/bf16 and 32B-AWQ. |
| [`mrv100-vllm-serving-benchmark.md`](mrv100-vllm-serving-benchmark.md) | OpenAI-compatible server benchmark | TTFT, TPOT, total latency, success rate, and output throughput for the 32B-AWQ serving baseline. |
| [`mrv100-prefix-distribution-decode128.md`](mrv100-prefix-distribution-decode128.md) | Prefix distribution benchmark | How shared-prefix versus random-prefix traffic changes serving behavior for 32B-AWQ. |
| [`mrv100-speculative-decoding-feasibility.md`](mrv100-speculative-decoding-feasibility.md) | First speculative decoding probe | Why DSpark-style speculative serving was not immediately usable on this CoreX/vLLM stack. |
| [`mrv100-qwen-small-speculative-decode.md`](mrv100-qwen-small-speculative-decode.md) | Smaller target/draft speculative probe | Whether Qwen2.5-3B plus Qwen2.5-0.5B avoided the earlier speculative-decoding blockers. |
| [`mrv100-prefix-cache-radix-workload.md`](mrv100-prefix-cache-radix-workload.md) | RadixAttention-style workload | The first focused prefix-cache experiment using grouped, random, and fully shared prompt traffic. |
| [`mrv100-prefix-cache-sweeps.md`](mrv100-prefix-cache-sweeps.md) | Main polished result | The best summary of the prefix-cache sweet spot across output length, prompt length, and prefix locality. |

## Folder Map

| Path | What it contains |
|---|---|
| [`artifacts/`](artifacts/) | Raw JSON results and benchmark logs copied into the repo. Start with [`artifacts/README.md`](artifacts/README.md). |
| [`remote-snapshots/`](remote-snapshots/) | A fuller shutdown-time copy of selected remote experiment folders, excluding model weights, caches, virtualenvs, credentials, and SSH material. |
| [`../../tools/vllm/`](../../tools/vllm/) | Reproduction and packaging scripts used for vLLM smoke tests, scale probes, server benchmarks, and remote snapshots. |

## Artifact Naming

Most artifact names start with a run number because the experiment grew in
stages:

| Prefix | Stage |
|---|---|
| `05` / `06` | Offline vLLM smoke tests. |
| `07` | vLLM variants: eager vs CUDA Graph and prefix cache on/off. |
| `08` / `09` | Scale ladder and larger-model probes. |
| `10` / `11` / `12` / `13` / `14` | AWQ, AWQ-Marlin, and 72B boundary/OOM probes. |
| `15` / `16` | OpenAI-compatible server benchmarks. |
| `17` | Prefix distribution benchmark for decode-128 workloads. |
| `18` / `19` | Speculative decoding feasibility probes. |
| `20` | Prefix-cache / RadixAttention-style workload summary. |
| `21` / `22` / `23` | Prefix-cache sweep logs for output length, prompt length, and prefix groups. |

## What Is Worth Quoting

| Claim | Best source |
|---|---|
| One MR-V100 32 GiB card can run Qwen2.5-14B and Qwen2.5-32B-AWQ with vLLM/CoreX. | [`mrv100-vllm-scale-limit.md`](mrv100-vllm-scale-limit.md) |
| 32B-AWQ serving works through the OpenAI-compatible vLLM server and has measurable TTFT/TPOT curves. | [`mrv100-vllm-serving-benchmark.md`](mrv100-vllm-serving-benchmark.md) |
| Speculative decoding is blocked by local backend/runtime issues, not by lack of curiosity. | [`mrv100-speculative-decoding-feasibility.md`](mrv100-speculative-decoding-feasibility.md), [`mrv100-qwen-small-speculative-decode.md`](mrv100-qwen-small-speculative-decode.md) |
| Prefix caching helps most when decode is short-to-medium, prompt is medium-to-long, and traffic has 2-8 prefix groups. | [`mrv100-prefix-cache-sweeps.md`](mrv100-prefix-cache-sweeps.md) |
| Fully shared-prefix traffic can improve TTFT but hurt TPOT enough to be a bad tradeoff. | [`mrv100-prefix-cache-sweeps.md`](mrv100-prefix-cache-sweeps.md) |

## Reproduce Direction

The scripts assume a CoreX/vLLM machine with local model paths under
`/data/models` and a working directory like `/data/src/vllm-mrv100-probe`.
Use [`../../tools/vllm/README.md`](../../tools/vllm/README.md) for the exact
script purpose map.

