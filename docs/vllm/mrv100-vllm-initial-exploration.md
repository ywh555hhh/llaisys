# MR-V100 vLLM initial exploration

This note records a first vLLM exploration on the Iluvatar MR-V100 / CoreX
machine after switching to the vLLM image. The goal is not to claim vendor-grade
performance numbers. The goal is to establish a reproducible baseline and find
small optimization levers that are worth deeper AI inference infra work.

## Machine and software

Remote instance:

```text
Host: bbdc80d04b9c
OS: Ubuntu 24.04.4
GPU: Iluvatar MR-V100, 32 GiB
CoreX / IX-ML / driver: 4.4.0
CUDA compatibility reported by ixsmi: 10.2
PCIe: Gen4 x16
```

Python stack from the vLLM image:

```text
Python: 3.12.3
vLLM: 0.17.0+corex.20260515140109
torch: 2.7.1, CUDA compatibility 10.2
transformers: 5.0.0
triton: 3.1.0
flash_attn: 2.6.3+corex.4.4.0
xformers: 0.0.26.post1
```

Torch sees the device and can execute FP16 matmul:

```text
torch.cuda.is_available(): True
device: Iluvatar MR-V100
compute capability: 7.1
SM count: 16
warp size: 64
L2 cache: 16 MiB
shared memory per block: 128 KiB
```

## Network and model

Direct Hugging Face access failed with:

```text
URLError [Errno 99] Cannot assign requested address
```

The usable route was:

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir /data/models/Qwen2.5-0.5B-Instruct
```

Downloaded model:

```text
Qwen/Qwen2.5-0.5B-Instruct
Local path: /data/models/Qwen2.5-0.5B-Instruct
Size: 954 MiB
Weights: model.safetensors, 988,097,824 bytes
```

## Smoke test

The first attempt failed because the Python program was fed through stdin while
vLLM used `spawn` multiprocessing:

```text
FileNotFoundError: /root/<stdin>
```

Running the same code from a real `.py` file fixed that issue. vLLM then loaded
the model and generated text successfully.

Key engine facts from the successful run:

```text
Resolved architecture: Qwen2ForCausalLM
dtype: bf16 weights cast to fp16
max_model_len: 1024
vLLM engine: V1
attention backend: FLASH_ATTN
chunked prefill: enabled
asynchronous scheduling: enabled
model memory: 0.93 GiB
available KV cache memory: 12.0 GiB
GPU KV cache size: about 1,048,944 tokens
max concurrency at 1024 tokens/request: about 1024x
```

This confirms that the vLLM image is not merely installable on MR-V100. It can
run a real Hugging Face causal LM through vLLM V1, CoreX PyTorch, FlashAttention,
Paged KV cache, and the vLLM scheduler.

## Optimization experiments

Workload:

- Model: `Qwen2.5-0.5B-Instruct`
- Max model length: 1024
- GPU memory utilization: 0.45
- Dtype: fp16
- Batch: 8 prompts
- Short decode workload: 64 input tokens total, 512 output tokens total
- Prefix workload: 3456 input tokens total, 256 output tokens total

Variants:

| Variant | Prefix cache | CUDA Graph | Eager | Result |
|---|---:|---:|---:|---|
| `eager_prefix_on` | on | off | on | Success |
| `eager_prefix_off` | off | off | on | Success |
| `cudagraph_prefix_on` | on | on | off | Success |

Results:

| Variant | Short decode tok/s | Prefix round 1 tok/s | Prefix round 2 tok/s |
|---|---:|---:|---:|
| `eager_prefix_on` | 436.4 | 540.2 | 591.2 |
| `eager_prefix_off` | 442.5 | 518.9 | 563.1 |
| `cudagraph_prefix_on` | 1006.7 | 815.1 | 973.1 |

Observed improvements:

```text
CUDA Graph vs eager, short decode: 2.31x
CUDA Graph vs eager, prefix round 1: 1.51x
CUDA Graph vs eager, prefix round 2: 1.65x
Prefix cache on vs off, prefix round 2: 1.05x
```

The strongest initial lever is CUDA Graph decode capture. The vLLM/CoreX build
successfully captured decode graphs:

```text
Capturing CUDA graphs (decode, FULL): 35/35
Graph capturing finished in 2 sec, took 0.16 GiB
cudagraph_mode: FULL_DECODE_ONLY
max_cudagraph_capture_size: 512
```

## Interpretation

The most important result is that CUDA Graph is not just a theoretical feature
on this CoreX vLLM image. It captures and materially improves a decode-heavy
small-model workload.

This matches the inference-systems mental model:

- Decode emits many repeated per-token kernels.
- Host launch overhead becomes visible, especially for small models and small
  batches.
- CUDA Graph amortizes that launch overhead by replaying a captured graph.

Prefix caching did not show a dramatic isolated win in this tiny synthetic
workload. That does not mean prefix caching is unimportant. It means the current
test is too small and too deterministic to stress TTFT in a realistic shared
prefix serving workload. A better next test should measure request-level TTFT
through the OpenAI-compatible server with repeated system prompts and mixed
arrival times.

## Caveats

- These are smoke benchmarks, not rigorous serving benchmarks.
- The model is small, so graph-launch overhead is unusually visible.
- The timings include vLLM Python/API overhead around `LLM.generate`.
- The runs use offline generation, not HTTP serving.
- The child engine logs show SIGTERM during normal parent shutdown, plus
  resource-tracker warnings. The runs completed and wrote valid JSON, but the
  cleanup path deserves a follow-up look.
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` requires real script files and a proper
  `if __name__ == "__main__"` guard.

## Reproduce

Environment:

```bash
export PATH=/usr/local/corex/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/hf-cache
export VLLM_TARGET_DEVICE=cuda
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

Download model:

```bash
mkdir -p /data/models
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir /data/models/Qwen2.5-0.5B-Instruct
```

Run smoke test:

```bash
python3 tools/vllm/mrv100_offline_smoke.py
```

Run variant benchmark:

```bash
bash tools/vllm/run_mrv100_vllm_variants.sh
```

Raw JSON artifacts are stored under `docs/vllm/artifacts/`.

## Resume-project direction

A compact resume framing could be:

```text
Benchmarked vLLM V1 on Iluvatar MR-V100/CoreX, validating FlashAttention-backed
Paged KV inference for Qwen2.5-0.5B and showing CUDA Graph decode capture
improved short decode throughput by 2.3x over eager execution.
```

The next stronger version should add:

1. OpenAI-compatible `vllm serve` benchmark with TTFT, ITL/TPOT, throughput, and
   p50/p95/p99 latency.
2. Prefix-caching workload with repeated long system prompts and varied request
   inter-arrival times.
3. `ixsys` or vLLM hidden metrics to separate prefill, decode, graph replay, and
   attention backend time.
4. Larger model, such as Qwen2.5-1.5B or 3B, to see when the graph speedup
   shrinks as GEMM and memory bandwidth dominate more of the step time.
5. A cleanup/patch investigation for vLLM/CoreX resource shutdown warnings.
