# Compatibility probe: SGLang-Omni on Iluvatar MR-V100 / CoreX 4.4.0 reaches Qwen3-ASR stage startup, blocked by PyTorch CUDA allocator API mismatch

## Summary

I ran a compatibility bring-up of `sgl-project/sglang-omni` on an Iluvatar MR-V100 32GB machine with CoreX 4.4.0. This is not a benchmark report; it is a compatibility trace for a non-NVIDIA domestic GPU environment.

The good news: SGLang-Omni can be installed from source, the CLI and GPU diagnostics run, the MR-V100 is visible through CoreX PyTorch, and a Qwen3-ASR server attempt reaches coordinator startup and ASR stage worker construction.

The current hard blocker: `sglang==0.5.16` imports `torch.cuda.memory._cuda_beginAllocateCurrentThreadToPool`, which is not present in CoreX `torch 2.7.1+corex.4.4.0`.

## Environment

```text
GPU: Iluvatar MR-V100 32GB
CoreX: 4.4.0
IX-ML / Driver: 4.4.0
CUDA compatibility shown by ixsmi: 10.2
Python: 3.10.18
CoreX torch: 2.7.1+corex.4.4.0
Torch CUDA: 10.2
```

## What worked

```text
sglang-omni editable install: OK
sgl-omni --help: OK
sgl-omni config --help: OK
sgl-omni serve --help: OK
sgl-omni check-gpu: OK
sglang.srt.configs.qwen3_asr import: OK after using project-pinned transformers/tokenizers in an isolated venv
Qwen3-ASR serve attempt: reaches coordinator + stage worker startup
```

`sgl-omni check-gpu` reports:

```text
GPU: Iluvatar MR-V100, 32.00GiB
attention/triton: available after import-only install
attention/torch-sdpa: available; version=2.7.1+corex.4.4.0
flash-attn-4 / flashinfer / sglang-kernel / sgl-deep-gemm / nixl / mooncake: not installed
NVML: not available
```

## Reproduction notes

The mixed venv/CoreX path ordering matters. Putting CoreX `dist-packages` at the front of `PYTHONPATH` causes CoreX/global packages to shadow venv packages. The working pattern was:

```bash
python3 -m venv /data/src/sglang-omni-corex-probe/venv-transformers512
. /data/src/sglang-omni-corex-probe/venv-transformers512/bin/activate

# Install sglang-omni and sglang without replacing CoreX torch.
pip install -e /data/src/sglang-omni-corex-probe/repo --no-deps
pip install 'sglang==0.5.16' --no-deps

# Use project-compatible Python deps, but keep torch from CoreX.
pip install 'transformers==5.12.1' 'tokenizers==0.22.2' 'pydantic>=2.8,<3' 'pydantic-core==2.46.4' triton ...

# Add CoreX packages as a fallback path, not front-of-PYTHONPATH.
SITE=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
echo '/usr/local/corex/lib64/python3/dist-packages' > "$SITE/corex_after_venv.pth"

export PYTHONPATH=/data/src/sglang-omni-corex-probe/repo
export LD_LIBRARY_PATH=/usr/local/corex/lib64:$LD_LIBRARY_PATH
```

With that order:

```text
torch OK 2.7.1 from /usr/local/corex/...
huggingface_hub OK 1.27.0 from venv
transformers OK 5.12.1 from venv
tokenizers OK 0.22.2 from venv
sglang OK 0.5.16 from venv
sglang_omni OK 0.1.1 from source checkout
```

## Current failure

Command:

```bash
sgl-omni serve  --model-path /mnt/moark-models/Qwen3-ASR-1.7B  --model-name qwen3-asr  --port 18015  --log-level debug
```

It reaches:

```text
Coordinator control plane started
Coordinator started
StageGroup asr: spawned 1 process(es)
Set current device to 0 for stage asr
Building scheduler for asr (tp_rank=0/1) ...
Acquired GPU startup lock for stage asr
```

Then fails at:

```text
ImportError: cannot import name '_cuda_beginAllocateCurrentThreadToPool' from 'torch.cuda.memory'
(/usr/local/corex/lib64/python3/dist-packages/torch/cuda/memory.py)
```

Relevant import path:

```text
sglang_omni.models.qwen3_asr.engine_builder
-> sglang.srt.managers.mm_utils
-> sglang.srt.managers.schedule_batch
-> sglang.srt.configs.model_config
-> sglang.srt.layers.quantization
-> sglang.kernels.ops.quantization.fp8_kernel
-> sglang.srt.layers.deep_gemm_wrapper.compile_utils
-> sglang.srt.distributed.device_communicators.pynccl_allocator
-> torch.cuda.memory._cuda_beginAllocateCurrentThreadToPool
```

## Interpretation

This looks like a SGLang/CoreX compatibility boundary rather than an SGLang-Omni model integration issue. SGLang 0.5.16 assumes PyTorch 2.11-era CUDA allocator internals and several CUDA-13/NVIDIA-oriented optimized backend packages. CoreX currently provides torch 2.7.1 with CUDA compatibility 10.2 and lacks the private allocator API above.

## Other friction points observed

```text
Remote GitHub/ghproxy clone was unstable; local codeload tarball + scp was reliable.
NVML is absent, so diagnostics cannot report normal NVIDIA driver/runtime metadata.
NumPy 2.x triggers warnings with CoreX torch modules compiled against NumPy 1.x.
mooncake/nixl/flash-attn/flashinfer/sglang-kernel/deep-gemm CUDA stack is unavailable or incompatible.
```

## Possible next step

Check whether `pynccl_allocator.py` and deep-gemm/quantization imports can be guarded or bypassed when running on a non-NVIDIA/CoreX backend, especially for a torch-SDPA-only exploratory path.
