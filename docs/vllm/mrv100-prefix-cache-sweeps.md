# MR-V100 prefix-cache sweep study

This note extends the MR-V100 prefix-cache / RadixAttention-style workload with
three c16 sweeps:

1. output length: `max_tokens=16/32/64/96`
2. prompt length: `prompt_tokens=256/512/1024`
3. prefix locality: `prefix_groups=1/2/4/8/16`

All runs use Qwen2.5-3B-Instruct on one Iluvatar MR-V100/CoreX card with a
grouped true-prefix workload. Every point compares vLLM prefix caching on vs
off.

## Setup

Common setup:

```text
device: Iluvatar MR-V100 32 GiB
CoreX driver/runtime: 4.4.0
vLLM: 0.17.0-compatible image
model: Qwen2.5-3B-Instruct
mode: eager
concurrency: 16
requests per concurrency level: 4x concurrency
max_num_seqs: 16
max_num_batched_tokens: 8192
prompt mode: grouped true-prefix
```

The benchmark simulates session-style traffic: requests are assigned to a
prefix group, share a long group prefix, and append a short unique request tail.
This is closer to agent/RAG/multi-turn serving than a fully shared synthetic
prompt.

## Output-length sweep

Fixed: prompt target 384 tokens, 4 prefix groups, `max_model_len=768`.

| Max output | Cache | Success | Output tok/s | TTFT p50 | TTFT p95 | TPOT p50 | GPU free after |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | on | 64/64 | 375.0 | 0.284s | 0.666s | 20.85ms | 752 MiB |
| 16 | off | 64/64 | 245.4 | 0.619s | 0.956s | 22.14ms | 342 MiB |
| 32 | on | 64/64 | 499.6 | 0.293s | 0.655s | 20.71ms | 748 MiB |
| 32 | off | 64/64 | 366.7 | 0.617s | 0.995s | 21.47ms | 342 MiB |
| 64 | on | 64/64 | 566.4 | 0.275s | 0.666s | 20.89ms | 748 MiB |
| 64 | off | 64/64 | 476.1 | 0.603s | 0.947s | 21.29ms | 216 MiB |
| 96 | on | 64/64 | 655.7 | 0.287s | 0.645s | 20.68ms | 752 MiB |
| 96 | off | 64/64 | 540.1 | 0.602s | 0.950s | 21.34ms | 220 MiB |

| Max output | Output tok/s delta | TTFT p50 delta | TTFT p95 delta | TPOT p50 delta |
|---:|---:|---:|---:|---:|
| 16 | +52.8% | -54.1% | -30.4% | -5.8% |
| 32 | +36.3% | -52.5% | -34.2% | -3.5% |
| 64 | +19.0% | -54.4% | -29.7% | -1.9% |
| 96 | +21.4% | -52.4% | -32.2% | -3.1% |

Interpretation:

- Prefix caching is most valuable when output is short. At 16 output tokens,
  total work is prefill-heavy, so output throughput improves by 52.8%.
- TTFT p50 improvement is stable across output lengths, about 52-54%.
- TPOT changes only slightly. This supports the expected model: prefix caching
  saves repeated prefill, not decode.

## Prompt-length sweep

Fixed: max output 32 tokens, 4 prefix groups, `max_model_len=1536`.

| Prompt target | Cache | Success | Output tok/s | TTFT p50 | TTFT p95 | TPOT p50 | GPU free after |
|---:|---|---:|---:|---:|---:|---:|---:|
| 256 | on | 64/64 | 510.7 | 0.252s | 0.656s | 20.68ms | 768 MiB |
| 256 | off | 64/64 | 431.5 | 0.433s | 0.810s | 21.13ms | 554 MiB |
| 512 | on | 64/64 | 418.5 | 0.302s | 1.356s | 21.21ms | 736 MiB |
| 512 | off | 64/64 | 317.4 | 0.719s | 1.552s | 21.98ms | 42 MiB |
| 1024 | on | 64/64 | 402.0 | 0.368s | 1.301s | 21.02ms | 580 MiB |
| 1024 | off | 64/64 | 230.0 | 1.314s | 2.170s | 21.85ms | 42 MiB |

| Prompt target | Output tok/s delta | TTFT p50 delta | TTFT p95 delta | TPOT p50 delta |
|---:|---:|---:|---:|---:|
| 256 | +18.4% | -41.8% | -19.1% | -2.1% |
| 512 | +31.9% | -58.0% | -12.6% | -3.5% |
| 1024 | +74.8% | -72.0% | -40.1% | -3.8% |

Interpretation:

- Longer prompts make prefix caching more valuable. At 1024 prompt tokens,
  cache-on is 74.8% higher in output tok/s and 72.0% lower in TTFT p50.
- Cache-off becomes memory-dangerous for long prompts: both 512 and 1024 prompt
  runs ended with only 42 MiB free.
- Cache-on leaves materially more headroom because repeated prefix state is not
  duplicated in the same way.

## Prefix-locality sweep

Fixed: prompt target 512 tokens, max output 32 tokens, `max_model_len=1024`.

| Prefix groups | Cache | Success | Output tok/s | TTFT p50 | TTFT p95 | TPOT p50 | GPU free after |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | on | 64/64 | 263.4 | 0.458s | 0.804s | 45.35ms | 852 MiB |
| 1 | off | 64/64 | 319.1 | 0.703s | 1.558s | 21.97ms | 50 MiB |
| 2 | on | 64/64 | 449.3 | 0.283s | 1.136s | 20.78ms | 776 MiB |
| 2 | off | 64/64 | 323.5 | 0.677s | 1.595s | 21.33ms | 44 MiB |
| 4 | on | 64/64 | 418.5 | 0.302s | 1.356s | 21.21ms | 736 MiB |
| 4 | off | 64/64 | 317.4 | 0.719s | 1.552s | 21.98ms | 42 MiB |
| 8 | on | 64/64 | 427.4 | 0.300s | 1.312s | 20.79ms | 546 MiB |
| 8 | off | 64/64 | 301.7 | 0.732s | 1.726s | 22.42ms | 166 MiB |
| 16 | on | 64/64 | 387.0 | 0.295s | 1.737s | 21.49ms | 56 MiB |
| 16 | off | 64/64 | 325.3 | 0.709s | 1.541s | 20.95ms | 38 MiB |

| Prefix groups | Output tok/s delta | TTFT p50 delta | TTFT p95 delta | TPOT p50 delta |
|---:|---:|---:|---:|---:|
| 1 | -17.4% | -34.8% | -48.4% | +106.4% |
| 2 | +38.9% | -58.2% | -28.8% | -2.6% |
| 4 | +31.9% | -58.0% | -12.6% | -3.5% |
| 8 | +41.7% | -59.0% | -24.0% | -7.3% |
| 16 | +19.0% | -58.4% | +12.8% | +2.6% |

Interpretation:

- `prefix_groups=1` is the pathological case. It improves TTFT but doubles
  TPOT, dropping aggregate output throughput by 17.4%. This reproduces the
  earlier "fully shared prefix" warning in a cleaner grouped workload.
- `prefix_groups=2/4/8` is the practical sweet zone. These runs preserve the
  TTFT benefit without blowing up TPOT, and output throughput improves by about
  31.9-41.7%.
- `prefix_groups=16` still improves TTFT p50, but p95 worsens and memory
  headroom collapses to 56 MiB with cache-on. Too many groups reduce locality
  and increase pressure.

## Overall answer

Prefix caching is worth it when the workload has enough prefix reuse but is not
fully synchronized into one prefix group. The useful regime on this MR-V100
setup is:

```text
short-to-medium decode
medium-to-long prompt
2-8 reusable prefix groups
```

The bad regime is:

```text
all requests share one prefix
high concurrency
decode remains long enough to dominate
```

In that bad regime, prefix caching moves the bottleneck from prefill to decode
and the serving loop pays for it in TPOT.

## Metrics limitation

The captured vLLM logs did not expose structured prefix-cache hit rate or KV
block-hit metrics. This report therefore infers cache effectiveness from TTFT,
TPOT, throughput, and `ixsmi` memory readings. A follow-up should scrape
Prometheus `/metrics` during the run, or enable a vLLM logging path that reports
prefix-cache hit/block statistics directly.

## Artifacts

```text
docs/vllm/artifacts/21_prefix_cache_sweep_summary.json
docs/vllm/artifacts/15_server_benchmark_qwen3b_g4_trueprefix_decode*_c16_r4_cache_*.json
docs/vllm/artifacts/15_server_benchmark_qwen3b_g4_trueprefix_prompt*_decode32_c16_r4_cache_*.json
docs/vllm/artifacts/15_server_benchmark_qwen3b_g*_trueprefix_prompt512_decode32_c16_r4_cache_*.json
docs/vllm/artifacts/21_sweep_qwen3b_g4_trueprefix_decode*_c16_r4_cache_*.log
docs/vllm/artifacts/22_sweep_qwen3b_g4_trueprefix_prompt*_decode32_c16_r4_cache_*.log
docs/vllm/artifacts/23_sweep_qwen3b_g*_trueprefix_prompt512_decode32_c16_r4_cache_*.log
```

## Resume framing

```text
Designed and ran a 22-point prefix-cache parameter sweep on Iluvatar MR-V100
for vLLM/Qwen2.5-3B, showing that prefix caching improves TTFT and throughput
most for short-decode, long-prompt, 2-8 session-group workloads, while an
over-synchronized single-prefix workload regresses TPOT by 106% and reduces
throughput by 17%.
```
