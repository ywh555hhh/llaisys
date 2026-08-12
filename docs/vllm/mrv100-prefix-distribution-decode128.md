# MR-V100 prefix-distribution serving benchmark

This note follows up on the MR-V100 vLLM serving benchmark by testing a more
inference-systems-oriented question: how does the same 32B-AWQ server behave
when requests share a large prefix versus when every request has a mostly
unique prefix?

The immediate motivation is the DSpark paper:

```text
Load is Not What You Should Balance: Introducing Prequal to Balance LLM Serving
```

The paper's useful framing for this project is not "copy DSpark as-is." DSpark
targets draft-model speculative decoding and shows that decode-time serving can
waste batch capacity when verification work is not balanced. Our current
MR-V100 setup serves Qwen2.5-32B-Instruct-AWQ with standard vLLM, so the first
practical step is to measure the workload knobs that determine whether scheduler,
KV cache, and decode batching behavior are being tested honestly.

## Setup

Server:

```bash
vllm serve /data/models/Qwen2.5-32B-Instruct-AWQ \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name qwen32b-awq \
  --max-model-len 512 \
  --gpu-memory-utilization 0.95 \
  --quantization awq_marlin \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192
```

Runtime:

```text
device: Iluvatar MR-V100 32 GiB
CoreX driver: 4.4.0
vLLM: 0.17.0
model: Qwen2.5-32B-Instruct-AWQ
quantization: awq_marlin
CUDA Graph: enabled
max_model_len: 512
```

Workload:

```text
endpoint: /v1/completions, streaming
prompt target: 256 tokens
observed prompt length: about 271-335 tokens
max output tokens: 128
concurrency: 1, 4, 8, 16
requests per concurrency level: 1x concurrency
```

`shared` mode repeats the same benchmark prefix across requests. `random` mode
injects per-request SHA-256 fingerprints and request-specific filler, reducing
prefix reuse across concurrent requests.

## Results

### Random-prefix workload

| Concurrency | Success | Output tok/s | TTFT p50 | TTFT p95 | TPOT p50 | Total p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 20.9 | 0.633s | 0.633s | 43.1ms | 6.106s |
| 4 | 4/4 | 68.5 | 1.058s | 1.058s | 50.4ms | 7.461s |
| 8 | 8/8 | 130.5 | 1.297s | 1.298s | 51.5ms | 7.833s |
| 16 | 16/16 | 220.2 | 2.414s | 2.415s | 54.1ms | 9.284s |

### Shared-prefix workload

| Concurrency | Success | Output tok/s | TTFT p50 | TTFT p95 | TPOT p50 | Total p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 20.4 | 0.635s | 0.635s | 43.9ms | 5.721s |
| 4 | 4/4 | 75.4 | 0.262s | 0.262s | 50.2ms | 6.634s |
| 8 | 8/8 | 79.4 | 0.535s | 0.535s | 96.2ms | 12.750s |
| 16 | 16/16 | 139.9 | 0.569s | 0.571s | 107.7ms | 14.244s |

## Interpretation

- Both workloads completed successfully through concurrency 16 on one MR-V100
  32 GiB card.
- Random-prefix long-decode kept aggregate throughput scaling through
  concurrency 16, reaching about 220.2 output tok/s, but TTFT rose to about
  2.414s.
- Shared-prefix long-decode had much lower TTFT at concurrency 16, about
  0.569s p50, but TPOT degraded to about 107.7ms and aggregate output throughput
  reached only about 139.9 output tok/s.
- This is a useful warning for future "acceleration" claims: prefix reuse can
  improve first-token behavior while changing the decode scheduling shape. A
  single aggregate tok/s number hides the trade-off.
- Peak memory is tight. The random-prefix run ended at about 32,240 MiB used
  with only about 528 MiB free; this makes larger max context, higher
  `max_num_seqs`, or speculative draft-model hosting risky without changing the
  model mix.

## DSpark applicability

DSpark/Prequal is most relevant as a next project direction if we can provide a
compatible draft model and expose speculative decoding in vLLM on this CoreX
runtime. The direct implementation path is not to port the paper wholesale into
this repository. A realistic staged plan is:

1. Establish prefix/random and short/long decode baselines, which this note does.
2. Add a same-server multi-workload runner so server cold-start cost does not
   dominate experiment time.
3. Try vLLM speculative decoding with a small Qwen draft model if the CoreX
   build supports the needed `--speculative-config` path.
4. Measure acceptance rate, TTFT, TPOT, p95/p99, and memory headroom against the
   non-speculative baseline.
5. Only after that, evaluate whether a Prequal-like verification scheduler can
   be prototyped or simulated on top of available vLLM hooks.

## Reproduce

Random-prefix:

```bash
LABEL=qwen32b_awq_marlin_random_prefix_decode128_c1_4_8_16 \
CONCURRENCY_LIST=1,4,8,16 \
REQUESTS_PER_CONCURRENCY=1 \
PROMPT_TOKENS=256 \
MAX_TOKENS=128 \
PROMPT_MODE=random \
MAX_NUM_SEQS=16 \
MAX_NUM_BATCHED_TOKENS=8192 \
MODE=cudagraph \
QUANTIZATION=awq_marlin \
  bash tools/vllm/run_mrv100_vllm_server_benchmark.sh
```

Shared-prefix:

```bash
LABEL=qwen32b_awq_marlin_shared_prefix_decode128_c1_4_8_16 \
CONCURRENCY_LIST=1,4,8,16 \
REQUESTS_PER_CONCURRENCY=1 \
PROMPT_TOKENS=256 \
MAX_TOKENS=128 \
PROMPT_MODE=shared \
MAX_NUM_SEQS=16 \
MAX_NUM_BATCHED_TOKENS=8192 \
MODE=cudagraph \
QUANTIZATION=awq_marlin \
  bash tools/vllm/run_mrv100_vllm_server_benchmark.sh
```

Artifacts:

```text
docs/vllm/artifacts/15_server_benchmark_qwen32b_awq_marlin_random_prefix_decode128_c1_4_8_16.json
docs/vllm/artifacts/15_server_benchmark_qwen32b_awq_marlin_shared_prefix_decode128_c1_4_8_16.json
docs/vllm/artifacts/17_prefix_distribution_decode128_summary.json
docs/vllm/artifacts/15_server_qwen32b_awq_marlin_random_prefix_decode128_c1_4_8_16.log
docs/vllm/artifacts/15_server_qwen32b_awq_marlin_shared_prefix_decode128_c1_4_8_16.log
```

## Resume framing

```text
Extended a single-card MR-V100/CoreX vLLM benchmark from short shared-prefix
serving into prefix-distribution and long-decode analysis for Qwen2.5-32B-AWQ,
showing that random-prefix decode scales to 220.2 output tok/s at concurrency
16 while shared-prefix serving sharply reduces TTFT but changes TPOT and
aggregate throughput behavior.
```
