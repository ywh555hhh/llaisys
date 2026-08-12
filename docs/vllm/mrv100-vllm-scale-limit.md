# MR-V100 vLLM scale-limit study

This note records a scale ladder on the Iluvatar MR-V100 / CoreX vLLM image.
The goal is to find the largest Qwen2.5 model that can be proven to run on one
32 GiB MR-V100 card, then use that evidence to decide where optimization work is
worth spending time.

This is not a vendor benchmark. It is a reproducible engineering probe for
learning vLLM, model memory pressure, KV cache budgeting, CUDA Graph behavior,
and non-NVIDIA accelerator compatibility.

## Environment

```text
Host: bbdc80d04b9c
OS: Ubuntu 24.04.4
GPU: Iluvatar MR-V100, 32 GiB
CoreX / IX-ML / driver: 4.4.0
CUDA compatibility reported by ixsmi: 10.2
vLLM: 0.17.0
```

The usable runtime environment was:

```bash
export PATH=/usr/local/corex/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:${PYTHONPATH:-}
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/hf-cache
export HUGGINGFACE_HUB_CACHE=/data/hf-cache/hub
export VLLM_TARGET_DEVICE=cuda
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

One practical footnote: `ixsmi` needs the CoreX library path. Running it without
`LD_LIBRARY_PATH=/usr/local/corex/lib64:...` can fail with `libixml.so` missing.

## Method

The ladder used offline `vllm.LLM.generate`, not the OpenAI-compatible HTTP
server. The probe records:

- model config and local path
- vLLM load time and total runtime
- warm and repeat generation throughput
- `ixsmi` snapshots before and after the run
- exceptions or failed stages when a model cannot run

Small models were tested with longer outputs. Larger models used smaller batch
sizes and shorter outputs because the immediate question was "can this card load
and generate at all?"

## Results

| Model | Context | Batch | Mode | Load time | Repeat output tok/s | Result |
|---|---:|---:|---|---:|---:|---|
| Qwen2.5-0.5B-Instruct | 1024 | 4 | CUDA Graph | 33.2s | 621.5 | Pass |
| Qwen2.5-1.5B-Instruct | 2048 | 4 | CUDA Graph | 35.6s | 363.0 | Pass |
| Qwen2.5-3B-Instruct | 2048 | 2 | CUDA Graph | 40.1s | 116.8 | Pass |
| Qwen2.5-7B-Instruct | 2048 | 1 | CUDA Graph | 51.6s | 39.2 | Pass |
| Qwen2.5-14B-Instruct | 1024 | 1 | CUDA Graph | 73.5s | 19.5 | Pass |
| Qwen2.5-32B-Instruct-AWQ | 512 | 1 | AWQ eager | 52.2s | 21.9 | Pass |
| Qwen2.5-32B-Instruct-AWQ | 512 | 1 | AWQ Marlin + CUDA Graph | 59.2s | 22.9 | Pass |

The strongest finding is that one MR-V100 32 GiB card can run
`Qwen/Qwen2.5-14B-Instruct` through vLLM V1 with CUDA Graph enabled at 1024
context, and can also run `Qwen/Qwen2.5-32B-Instruct-AWQ` at 512 context using
quantized weights. The largest proven fp16/bfloat16 model is 14B. The largest
proven quantized model is 32B-AWQ.

The 14B run used:

```text
max_model_len: 1024
gpu_memory_utilization: 0.95
batch_size: 1
max_tokens: 32
cuda_graph: true
torch_dtype from config: bfloat16
vLLM dtype: float16
```

The throughput numbers are not directly comparable across every row because the
larger-model probes intentionally used smaller batches and shorter decode
lengths. The meaningful comparison is the scaling trend and the fact that the
runtime can still initialize, capture graphs, and generate tokens at 14B
fp16/bfloat16 and 32B-AWQ.

## Interpretation

The scale ladder gives a useful mental model for this card:

- 0.5B and 1.5B are suitable for quick vLLM feature experiments such as CUDA
  Graph, prefix caching, scheduler behavior, and OpenAI server smoke tests.
- 3B and 7B are better for measuring realistic decode bottlenecks. Kernel time,
  memory bandwidth, and KV cache behavior start to dominate more visibly.
- 14B is the practical fp16/bfloat16 boundary probe. It proves the stack can run
  a larger instruction model, but leaves little room for serving-style
  concurrency at comfortable context lengths.
- 32B is viable as a quantization experiment. AWQ crossed the fp16 memory wall,
  and AWQ Marlin plus CUDA Graph provided a small repeat-decode improvement in
  this tiny probe.

## 32B quantization result

The vLLM/CoreX image exposes quantization methods including `awq`, `gptq`,
`bitsandbytes`, and related kernels. The AWQ probe used:

```text
Qwen/Qwen2.5-32B-Instruct-AWQ
max_model_len: 512
gpu_memory_utilization: 0.95
batch_size: 1
max_tokens: 16
quantization: awq
mode: eager
```

The baseline AWQ eager run succeeded:

```text
local model size: 19 GiB
model loading memory: 18.14 GiB
available KV cache memory: 10.16 GiB
GPU KV cache size: 41,600 tokens
repeat output throughput: 21.9 tok/s
```

vLLM reported that the model could use `awq_marlin`, so a second run used
`QUANTIZATION=awq_marlin MODE=cudagraph`. That also succeeded:

```text
AWQ Marlin runtime conversion: enabled
CUDA Graph decode capture: 35/35 capture sizes
graph capture memory: 0.66 GiB
repeat output throughput: 22.9 tok/s
```

The observed throughput lift from this tiny probe was only about 4.6%, so it is
not yet a strong performance claim. The stronger result is compatibility:
MR-V100/CoreX vLLM can load a 32B AWQ model, use FlashAttention, capture decode
graphs, and generate tokens on one 32 GiB card.

## Reproduce

Run the fp16/bfloat16 scale ladder:

```bash
bash tools/vllm/run_mrv100_vllm_scale_ladder.sh
```

Run the AWQ boundary probe:

```bash
bash tools/vllm/run_mrv100_vllm_awq_probe.sh
```

Run the AWQ Marlin + CUDA Graph variant:

```bash
QUANTIZATION=awq_marlin MODE=cudagraph \
  bash tools/vllm/run_mrv100_vllm_awq_probe.sh
```

The scale probe implementation is:

```text
tools/vllm/run_mrv100_vllm_scale_probe.py
```

Raw artifacts are stored under:

```text
docs/vllm/artifacts/
```

Important artifacts:

```text
08_scale_Qwen2.5-0.5B-Instruct.json
08_scale_Qwen2.5-1.5B-Instruct.json
09_extreme_Qwen2.5-3B-Instruct.json
09_extreme_Qwen2.5-7B-Instruct.json
09_extreme_Qwen2.5-14B-Instruct.json
09_scale_ladder_summary.json
10_awq_Qwen2.5-32B-Instruct-AWQ.json
10_awq_marlin_cudagraph_Qwen2.5-32B-Instruct-AWQ.json
10_scale_and_awq_summary.json
10_awq_Qwen2.5-32B-Instruct-AWQ.log
10_awq_marlin_cudagraph_Qwen2.5-32B-Instruct-AWQ.log
```

## Resume framing

A compact project description:

```text
Ran a vLLM scale-limit study on Iluvatar MR-V100/CoreX, validating Qwen2.5
0.5B/1.5B/3B/7B/14B offline inference on one 32 GiB card with CUDA Graph enabled,
then crossing the fp16 memory wall with Qwen2.5-32B-Instruct-AWQ and comparing
AWQ eager vs AWQ Marlin + CUDA Graph decode.
```

Stronger next steps:

1. Add OpenAI-compatible server benchmarks with TTFT, TPOT, ITL, p50/p95/p99,
   and concurrency.
2. Compare eager vs CUDA Graph on 3B/7B/14B with fixed prompt/output lengths.
3. Add prefix-cache workloads with repeated long system prompts.
4. Benchmark 32B-AWQ with longer decode lengths and controlled prompt lengths.
5. Investigate cleanup warnings from vLLM/CoreX worker shutdown.
