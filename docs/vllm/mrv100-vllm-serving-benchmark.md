# MR-V100 vLLM serving benchmark

This note records an OpenAI-compatible serving benchmark for the largest model
that was proven runnable on one MR-V100 32 GiB card without CPU offload:
`Qwen/Qwen2.5-32B-Instruct-AWQ`.

The goal is to move beyond offline `LLM.generate` smoke tests and collect
request-level serving numbers: TTFT, total latency, TPOT, success rate, and
aggregate output throughput under increasing concurrency.

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

Runtime environment:

```bash
export PATH=/usr/local/corex/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:${PYTHONPATH:-}
export VLLM_TARGET_DEVICE=cuda
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENFORCE_CUDA_GRAPH=1
```

Server-side facts from the run:

```text
vLLM: 0.17.0
attention backend: FlashAttention
quantization: awq_marlin
max model length: 512
model loading memory: 18.14 GiB
available KV cache memory: about 10.15 GiB
GPU KV cache size: about 41,584 tokens
CUDA Graph decode capture: enabled
peak observed memory during benchmark: about 31,704 MiB / 32,768 MiB
```

## Workload

Client endpoint:

```text
POST /v1/completions
stream: true
```

Workload shape:

```text
prompt target: 64 tokens
observed prompt length: about 99-100 tokens
max output tokens: 32
concurrency: 1, 2, 4, 8, 16
requests per concurrency level: 2x concurrency
```

The prompts intentionally share a repeated benchmark prefix. This makes the
workload closer to repeated-system-prompt serving than worst-case random prompt
serving. The server logs reported prefix-cache activity during the run, so the
numbers should be interpreted as a shared-prefix serving baseline.

## Results

| Concurrency | Requests | Success | Output tok/s | TTFT p50 | TTFT p95 | Total p50 | TPOT p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 2/2 | 19.8 | 0.309s | 0.469s | 1.616s | 42.2ms |
| 2 | 4 | 4/4 | 38.0 | 0.161s | 0.194s | 1.683s | 49.1ms |
| 4 | 8 | 8/8 | 73.9 | 0.187s | 0.238s | 1.729s | 49.0ms |
| 8 | 16 | 16/16 | 138.4 | 0.282s | 0.391s | 1.842s | 49.8ms |
| 16 | 32 | 32/32 | 255.8 | 0.389s | 0.529s | 1.989s | 51.2ms |

The strongest result is stability at concurrency 16: all 32 requests succeeded,
the server stayed within the 32 GiB card, and aggregate output throughput reached
about 255.8 tokens/s for this short-output shared-prefix workload.

## Interpretation

- The serving path works end to end: OpenAI-compatible HTTP server, streaming
  completions, FlashAttention, AWQ Marlin, CUDA Graph decode capture, and request
  concurrency all function on MR-V100/CoreX.
- The card runs very close to capacity. During the benchmark, `ixsmi` reported
  about 31.7 GiB used, leaving only about 1.0 GiB free.
- TPOT stayed roughly flat around 49-51 ms from concurrency 2 through 16, while
  aggregate output throughput scaled with concurrency.
- TTFT rose with concurrency, especially at 8 and 16, which is expected as the
  scheduler batches more concurrent work.
- The workload benefits from repeated prefix structure. A random-prefix workload
  should be measured separately before making broader serving claims.

## Reproduce

Run the full curve:

```bash
LABEL=qwen32b_awq_marlin_cg_full_curve_r2 \
CONCURRENCY_LIST=1,2,4,8,16 \
REQUESTS_PER_CONCURRENCY=2 \
PROMPT_TOKENS=64 \
MAX_TOKENS=32 \
MAX_NUM_SEQS=16 \
MAX_NUM_BATCHED_TOKENS=8192 \
MODE=cudagraph \
QUANTIZATION=awq_marlin \
  bash tools/vllm/run_mrv100_vllm_server_benchmark.sh
```

Benchmark scripts:

```text
tools/vllm/run_mrv100_vllm_server_benchmark.sh
tools/vllm/run_mrv100_vllm_server_benchmark.py
```

Raw artifacts:

```text
docs/vllm/artifacts/15_server_benchmark_qwen32b_awq_marlin_cg_full_curve_r2.json
docs/vllm/artifacts/16_server_benchmark_qwen32b_awq_marlin_full_curve_summary.json
docs/vllm/artifacts/15_server_qwen32b_awq_marlin_cg_full_curve_r2.log
docs/vllm/artifacts/15_server_benchmark_qwen32b_awq_marlin_cg_full_curve_r2.log
```

## Resume framing

```text
Built a reproducible vLLM serving benchmark on Iluvatar MR-V100/CoreX,
serving Qwen2.5-32B-Instruct-AWQ through the OpenAI-compatible API with AWQ
Marlin and CUDA Graph enabled; measured concurrency 1-16 with 100% success and
up to 255.8 output tok/s on one 32 GiB card.
```

## Next steps

1. Add a random-prefix workload to separate scheduler throughput from prefix
   cache effects.
2. Add longer decode tests, such as 128 or 256 output tokens, to reduce fixed
   request overhead and better measure steady-state decode.
3. Add a 14B fp16/bfloat16 serving comparison to contrast quantized 32B vs
   smaller dense fp16 serving.
4. Capture `/metrics` snapshots during load to add queueing and scheduler
   counters to the JSON artifacts.
