# MR-V100 prefix-cache / RadixAttention-style workload

This note turns the earlier MR-V100 vLLM serving probe into a more
RadixAttention-style experiment: repeated requests are arranged so that some
traffic shares a long reusable prompt prefix, while control traffic has mostly
unique request-specific prefixes.

The goal is not to claim that vLLM implements SGLang's RadixAttention data
structure on this CoreX runtime. The practical question is narrower and more
useful: when a serving workload has session-like prefix reuse, does enabling
vLLM prefix caching improve first-token behavior and throughput on one
Iluvatar MR-V100 card?

## Setup

Runtime:

```text
device: Iluvatar MR-V100 32 GiB
CoreX driver/runtime: 4.4.0
vLLM: 0.17.0-compatible image
model: Qwen2.5-3B-Instruct
mode: eager
max_model_len: 768
max_num_seqs: 16
max_num_batched_tokens: 8192
gpu_memory_utilization: 0.95
```

Workload:

```text
endpoint: /v1/completions, streaming
prompt target: 384 tokens
max output tokens: 96
concurrency: 4, 8, 16
requests per concurrency level: 4x concurrency
```

Prompt modes:

- `grouped`: four session groups. Requests in the same group share a stable
  long prefix and add a unique tail. This is the closest approximation of a
  multi-turn agent/RAG session workload.
- `random`: each request has request-specific prompt material. This is the
  negative control for prefix reuse.
- `shared`: all requests share the same long prefix. This is an upper-bound
  probe, not a realistic production traffic model.

## Results

### Grouped session workload

| Prefix cache | Concurrency | Success | Output tok/s | TTFT p50 | TTFT p95 | TPOT p50 | Total p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| on | 4 | 16/16 | 175.8 | 0.201s | 0.512s | 18.90ms | 2.000s |
| off | 4 | 16/16 | 174.0 | 0.214s | 0.506s | 18.89ms | 2.000s |
| on | 8 | 32/32 | 378.8 | 0.144s | 0.182s | 19.56ms | 2.007s |
| off | 8 | 32/32 | 345.4 | 0.369s | 0.375s | 19.34ms | 2.207s |
| on | 16 | 64/64 | 702.1 | 0.181s | 0.240s | 20.68ms | 2.185s |
| off | 16 | 64/64 | 588.8 | 0.607s | 0.669s | 20.88ms | 2.569s |

Prefix caching helped most on first-token latency:

| Concurrency | Output tok/s delta | TTFT p50 delta | TTFT p95 delta | TPOT p50 delta |
|---:|---:|---:|---:|---:|
| 4 | +1.1% | -6.3% | +1.0% | +0.1% |
| 8 | +9.7% | -61.1% | -51.5% | +1.1% |
| 16 | +19.3% | -70.1% | -64.1% | -1.0% |

### Random-prefix control

| Prefix cache | Concurrency | Success | Output tok/s | TTFT p50 | TTFT p95 | TPOT p50 | Total p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| on | 4 | 16/16 | 165.3 | 0.219s | 0.558s | 19.64ms | 2.072s |
| off | 4 | 16/16 | 173.5 | 0.212s | 0.526s | 18.88ms | 1.999s |
| on | 8 | 32/32 | 355.8 | 0.268s | 0.286s | 20.18ms | 2.207s |
| off | 8 | 32/32 | 345.6 | 0.348s | 0.371s | 19.59ms | 2.217s |
| on | 16 | 64/64 | 628.6 | 0.183s | 0.593s | 21.80ms | 2.645s |
| off | 16 | 64/64 | 586.3 | 0.608s | 0.660s | 20.95ms | 2.640s |

The random control is noisier than a perfect no-reuse workload because the
prompt template still contains some repeated scaffolding. It is still useful:
cache-on does not produce a uniformly better curve here, and TPOT is slightly
worse with cache-on at all three concurrency levels.

### Fully shared upper bound

| Prefix cache | Concurrency | Success | Output tok/s | TTFT p50 | TTFT p95 | TPOT p50 | Total p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| on | 4 | 16/16 | 192.6 | 0.103s | 0.537s | 18.69ms | 1.874s |
| on | 8 | 32/32 | 376.2 | 0.158s | 0.236s | 19.41ms | 2.021s |
| on | 16 | 64/64 | 338.6 | 0.179s | 0.232s | 45.43ms | 4.496s |

The fully shared c16 case is the warning label. Prefix reuse keeps TTFT low,
but decode TPOT degrades sharply. This suggests that on this MR-V100/CoreX
stack, extreme shared-prefix traffic can shift the bottleneck into decode
scheduling, memory pressure, or another runtime path. It is useful evidence
against reporting only the best TTFT number.

## Interpretation

- For the realistic grouped/session workload, prefix caching improved aggregate
  output throughput by about 1.1-19.3% and reduced TTFT p50 by about 6.3-70.1%.
- TPOT barely moved in the grouped A/B. That matches the expected mental model:
  prefix caching saves repeated prefill work, while decode remains the
  token-by-token memory-bandwidth-bound phase.
- The c16 grouped result is especially resume-worthy: 64/64 requests completed,
  output throughput rose from 588.8 to 702.1 output tok/s, and TTFT p50 dropped
  from 0.607s to 0.181s.
- Memory headroom is extremely tight. The grouped cache-on run ended around
  31,974 MiB used, while grouped cache-off ended around 32,546 MiB used on a
  32,768 MiB card. Larger contexts or higher `max_num_seqs` need a new memory
  plan, not just a larger benchmark sweep.
- This is a better community artifact than a generic vLLM smoke test because it
  isolates a real inference-serving question: how session-prefix locality
  changes TTFT/TPOT/throughput trade-offs on a non-NVIDIA CUDA-compatible stack.

## Reproduce

Grouped prefix-cache on:

```bash
LABEL=qwen3b_grouped4_trueprefix_on_decode96_c4_8_16_r4 \
MODEL_DIR=/data/models/Qwen2.5-3B-Instruct \
MODEL_NAME=qwen3b \
CONCURRENCY_LIST=4,8,16 \
REQUESTS_PER_CONCURRENCY=4 \
PROMPT_TOKENS=384 \
MAX_TOKENS=96 \
PROMPT_MODE=grouped \
PREFIX_GROUPS=4 \
PREFIX_CACHING=on \
MAX_MODEL_LEN=768 \
MAX_NUM_SEQS=16 \
MAX_NUM_BATCHED_TOKENS=8192 \
MODE=eager \
QUANTIZATION=none \
bash tools/vllm/run_mrv100_vllm_server_benchmark.sh
```

Grouped prefix-cache off:

```bash
LABEL=qwen3b_grouped4_trueprefix_off_decode96_c4_8_16_r4 \
MODEL_DIR=/data/models/Qwen2.5-3B-Instruct \
MODEL_NAME=qwen3b \
CONCURRENCY_LIST=4,8,16 \
REQUESTS_PER_CONCURRENCY=4 \
PROMPT_TOKENS=384 \
MAX_TOKENS=96 \
PROMPT_MODE=grouped \
PREFIX_GROUPS=4 \
PREFIX_CACHING=off \
MAX_MODEL_LEN=768 \
MAX_NUM_SEQS=16 \
MAX_NUM_BATCHED_TOKENS=8192 \
MODE=eager \
QUANTIZATION=none \
bash tools/vllm/run_mrv100_vllm_server_benchmark.sh
```

For random/shared controls, change `PROMPT_MODE=random` or
`PROMPT_MODE=shared` and set `PREFIX_CACHING=on/off` as needed.

## Artifacts

```text
docs/vllm/artifacts/15_server_benchmark_qwen3b_grouped4_trueprefix_on_decode96_c4_8_16_r4.json
docs/vllm/artifacts/15_server_benchmark_qwen3b_grouped4_trueprefix_off_decode96_c4_8_16_r4.json
docs/vllm/artifacts/15_server_benchmark_qwen3b_random_prefix_on_decode96_c4_8_16_r4.json
docs/vllm/artifacts/15_server_benchmark_qwen3b_random_prefix_off_decode96_c4_8_16_r4.json
docs/vllm/artifacts/15_server_benchmark_qwen3b_shared_prefix_on_decode96_c4_8_16_r4.json
docs/vllm/artifacts/20_prefix_cache_radix_workload_summary.json
```

## Next steps

1. Sweep prefix groups: 1, 2, 4, 8, 16.
2. Sweep prompt length: 256, 512, 1024 tokens.
3. Add chunked-prefill on/off if the CoreX vLLM build exposes a stable flag.
4. Parse server logs or vLLM metrics for prefix-cache hit rate instead of
   inferring it only from TTFT.
5. Repeat the grouped workload on SGLang/SGLang-Omni if the MR-V100 image can
   run the corresponding serving stack.

## Resume framing

```text
Built a RadixAttention-style prefix-locality benchmark for vLLM on a single
Iluvatar MR-V100/CoreX card, adding grouped session-prefix workloads and
prefix-cache A/B controls. On Qwen2.5-3B-Instruct, prefix caching improved
grouped-workload output throughput by up to 19.3% and reduced TTFT p50 from
0.607s to 0.181s at concurrency 16, while showing that TPOT remains decode-bound
and can regress under extreme fully-shared-prefix traffic.
```
