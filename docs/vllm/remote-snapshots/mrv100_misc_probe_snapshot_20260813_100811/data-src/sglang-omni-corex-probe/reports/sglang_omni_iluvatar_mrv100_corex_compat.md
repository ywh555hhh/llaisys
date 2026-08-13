# SGLang-Omni on Iluvatar MR-V100 / CoreX 4.4.0 Compatibility Probe

Date: 2026-08-11
Machine: 天数 / Iluvatar MR-V100 32GB
Workdir: `/data/src/sglang-omni-corex-probe`
Repo source: `sgl-project/sglang-omni`, main branch tarball copied from local Mac because remote GitHub/ghproxy clone was unstable.

## Executive summary

This machine is useful for a community compatibility bring-up report, but not for an immediate out-of-the-box SGLang-Omni performance benchmark.

What worked:

- Source checkout was established under `/data/src/sglang-omni-corex-probe/repo` after bypassing remote GitHub download instability.
- CoreX PyTorch is visible and reports one CUDA-style GPU: `Iluvatar MR-V100`, 32 GiB.
- `sglang-omni` editable install with `--no-deps` succeeded.
- After installing selected pure/control-plane dependencies, `sgl-omni --help`, `sgl-omni config --help`, `sgl-omni serve --help`, and `sgl-omni check-gpu` all ran.
- A real Qwen3-ASR server startup attempt reached coordinator startup, spawned the `asr` stage process, selected GPU 0, and began scheduler construction.

What did not work:

- The declared dependency stack is NVIDIA/CUDA-13 oriented: `torch==2.11.0`, `flash-attn-4`, `flashinfer_python[cu13]`, `kernels`, `nixl-cu13`, `mooncake-transfer-engine-cuda13`, `sglang-kernel`, `nvidia-cutlass-dsl[cu13]`, etc.
- The current machine provides CoreX torch `2.7.1+corex.4.4.0` with CUDA compatibility `10.2`, not NVIDIA CUDA 13.
- `sgl-omni check-gpu` reports SGLang-Omni can see the MR-V100 via torch, but only `torch-sdpa` is available; flash-attn, flashinfer, triton initially, sglang-kernel, nixl, mooncake are absent.
- After installing `triton` only as an import probe, server startup failed at a software compatibility issue before reaching model kernels: duplicate Transformers config registration for `qwen3_asr`.

## Baseline

From `logs/00_baseline.txt`:

```text
GPU: Iluvatar MR-V100
Memory: 32768 MiB
IX-ML / Driver: 4.4.0
CUDA compatibility shown by ixsmi: 10.2
CoreX clang: clang version 18.1.8 (CoreX 4.4.0)
/usr/local/corex/bin/nvcc: 194-byte compatibility script
Python: 3.10.18
Torch: 2.7.1
Torch CUDA: 10.2
CUDA available: True
Device name: Iluvatar MR-V100
```

## Source acquisition friction

Remote GitHub access was the first non-code blocker:

- `git clone` through `ghproxy.net` was extremely slow/stalled.
- Remote tarball download through `ghproxy.net` timed out around 7.7 MiB and produced an incomplete archive.
- Workaround: local Mac downloaded GitHub codeload tarball successfully, then `scp` transferred the 19 MiB archive to the remote machine.

This is worth reporting because domestic GPU boxes often fail before the actual compatibility question begins.

## Dependency shape

From `pyproject.toml`, `sglang-omni==0.1.1` pins or requires:

```text
torch==2.11.0
torchvision==0.26.0
torchaudio==2.11.0
transformers==5.12.1
sglang==0.5.16
flash-attn-4>=4.0.0b18
kernels>=0.14.1,<0.15
nixl-cu13>=1.1.0
mooncake-transfer-engine-cuda13>=0.3.10
torchcodec==0.11.1
```

`sglang==0.5.16` further expects CUDA/NVIDIA-oriented packages such as:

```text
cuda-python>=13.0
flashinfer_python[cu13]==0.6.14
humming-kernels[cu13]==0.1.10
nvidia-cutlass-dsl[cu13]==4.6.0
sglang-kernel==0.4.5
sgl-deep-gemm==0.1.4.post1
tilelang==0.1.11
```

This is the main reason a native CoreX/MR-V100 bring-up should be treated as compatibility research, not normal installation.

## Probes and results

### 1. Editable install without dependencies

Command:

```bash
cd /data/src/sglang-omni-corex-probe/repo
python3 -m pip install -e . --no-deps
```

Result:

```text
Successfully built sglang-omni
Successfully installed sglang-omni-0.1.1
```

Top-level import worked:

```text
sglang_omni OK /data/src/sglang-omni-corex-probe/repo/sglang_omni/__init__.py
```

Initial CLI import failed on old `pydantic 1.x`; after upgrading selected control-plane dependencies, CLI import succeeded.

### 2. CLI / diagnostics

After installing selected pure Python/control-plane dependencies:

```text
sglang_omni.cli OK
sglang_omni_router.serve OK
```

`sgl-omni --help` exposed:

```text
Commands:
  check-gpu  Report GPU mapping, runtime versions, and installed backends.
  serve      Serve the pipeline.
  config     View and export the pipeline configuration for local editing
```

`sgl-omni check-gpu` result:

```text
SGLang-Omni GPU diagnostics (no model loaded)
CUDA_VISIBLE_DEVICES: <unset>
Driver: unavailable
CUDA driver/runtime: unavailable / unavailable
PyTorch/CUDA build: 2.7.1 / 10.2
GPUs:
  logical 0 -> physical None visible=0 name=Iluvatar MR-V100 cc=7.1 memory=None/32.00GiB uuid=75401a56-2224-5257-8e2c-8f6dac7c37bc pci=unknown
Backends:
  attention/flash-attn-4: not installed
  attention/flashinfer: not installed
  attention/triton: not installed initially
  attention/torch-sdpa: available; version=2.7.1+corex.4.4.0
  gemm/sgl-deep-gemm: not installed
  gemm/sglang-kernel: not installed
  communication/nixl: not installed
  communication/mooncake: not installed
```

### 3. SGLang no-deps import

Command:

```bash
python3 -m pip install 'sglang==0.5.16' --no-deps
```

Result:

- Install succeeded.
- Initial `sglang` import failed on missing `orjson`.
- After selected pure Python deps, `sgl-omni serve --help` worked.
- Targeted import showed missing `triton` was the next blocking import.

After installing `triton` as an import-only probe:

```text
triton OK 3.7.1
sglang OK 0.5.16
sglang.srt.platforms OK
sglang.srt.platforms.interface OK
```

This does not mean Triton kernels are usable on CoreX; it only confirms the Python import layer can move forward.

### 4. Real Qwen3-ASR server startup attempt

Local model found:

```text
/mnt/moark-models/Qwen3-ASR-1.7B
```

Command:

```bash
sgl-omni serve  --model-path /mnt/moark-models/Qwen3-ASR-1.7B  --model-name qwen3-asr  --port 18003  --log-level debug
```

The startup got this far:

```text
Coordinator control plane started
Coordinator started
StageGroup asr: spawned 1 process(es)
Set current device to 0 for stage asr
Building scheduler for asr (tp_rank=0/1) ...
Acquired GPU startup lock for stage asr
```

Then failed with:

```text
ValueError: 'qwen3_asr' is already used by a Transformers config, pick another name.
RuntimeError: Process asr died during startup (exit code 1)
```

Interpretation:

- We reached actual multi-process stage construction.
- Failure occurred before model kernels or attention backend execution.
- Immediate next software issue is version/registration mismatch between installed `transformers 5.15.0` and `sglang 0.5.16` / SGLang's bundled Qwen3-ASR config registration.
- The environment also remains backend-incomplete for full serving because CUDA-13/NVIDIA-specific wheels are absent or incompatible with CoreX.

## Community value

This is meaningful community data if framed correctly:

1. It is not a benchmark result.
2. It is a compatibility bring-up trace for a domestic GPU/CoreX environment.
3. It identifies the exact layers crossed: source acquisition -> editable install -> CLI -> GPU diagnostics -> SGLang import -> pipeline config -> coordinator/stage worker startup.
4. It identifies remaining blockers without pretending they are solved:
   - domestic network / GitHub acquisition friction;
   - CUDA 13 / NVIDIA wheel assumptions;
   - CoreX torch version mismatch;
   - missing or incompatible optimized backend packages;
   - Transformers/SGLang duplicate `qwen3_asr` registration before kernel execution.

## Recommended next experiments

1. Re-run in an isolated venv with exact `transformers==5.12.1` while still preserving CoreX torch via `PYTHONPATH`, to see whether the duplicate `qwen3_asr` registration disappears.
2. Run `sgl-omni check-gpu --json` before and after dependency changes and attach both outputs to an issue/discussion.
3. Try a CPU-only or torch-SDPA-only path if SGLang-Omni exposes a backend override; otherwise record that no documented CoreX backend selection exists.
4. Do not install full `sglang-omni` dependencies normally on this machine unless the goal is to deliberately test conflict behavior; normal pip resolution will try to replace CoreX torch with NVIDIA-oriented torch 2.11/CUDA 13 packages.
5. For a community post, title it as a compatibility probe, e.g. `Compatibility probe: SGLang-Omni on Iluvatar MR-V100 / CoreX 4.4.0 reaches ASR stage startup, blocked by CUDA13 backend assumptions and qwen3_asr config registration`.

## Key log files

```text
/data/src/sglang-omni-corex-probe/logs/00_baseline.txt
/data/src/sglang-omni-corex-probe/logs/03_nodeps_import_probe.txt
/data/src/sglang-omni-corex-probe/logs/04_selected_deps_probe.txt
/data/src/sglang-omni-corex-probe/logs/05_check_gpu_sglang_probe.txt
/data/src/sglang-omni-corex-probe/logs/06_sglang_python_deps_probe.txt
/data/src/sglang-omni-corex-probe/logs/07_serve_attempt_probe.txt
/data/src/sglang-omni-corex-probe/logs/08_serve_attempt_probe2.txt
/data/src/sglang-omni-corex-probe/logs/09_targeted_import_probe.txt
/data/src/sglang-omni-corex-probe/logs/10_triton_serve_probe.txt
```

## Addendum: isolated venv probe with project-pinned Transformers

A second pass used an isolated venv at:

```text
/data/src/sglang-omni-corex-probe/venv-transformers512
```

Goal: test whether the earlier `qwen3_asr` duplicate registration was caused by the global environment's newer `transformers 5.15.0`.

Important environment lesson:

- Putting `/usr/local/corex/lib64/python3/dist-packages` at the front of `PYTHONPATH` is wrong for this mixed setup, because it lets CoreX/global packages shadow venv packages such as `huggingface-hub`.
- The better approach is to put CoreX's package directory in a venv `.pth` file so venv packages win, while `torch` still falls back to CoreX:

```bash
SITE=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
echo '/usr/local/corex/lib64/python3/dist-packages' > "$SITE/corex_after_venv.pth"
export PYTHONPATH=/data/src/sglang-omni-corex-probe/repo
export LD_LIBRARY_PATH=/usr/local/corex/lib64:$LD_LIBRARY_PATH
```

With that ordering, these imports succeeded:

```text
torch OK 2.7.1 /usr/local/corex/lib64/python3/dist-packages/torch/__init__.py
huggingface_hub OK 1.27.0 from venv
transformers OK 5.12.1 from venv
tokenizers OK 0.22.2 from venv
pydantic OK 2.13.4 from venv
sglang OK 0.5.16 from venv
sglang_omni OK 0.1.1 from repo
triton OK 3.7.1 from venv
```

The previous `qwen3_asr` registration conflict disappeared:

```text
sglang.srt.configs.qwen3_asr OK
```

The next real blocker became a PyTorch/CUDA allocator API mismatch:

```text
ImportError: cannot import name '_cuda_beginAllocateCurrentThreadToPool' from 'torch.cuda.memory'
(/usr/local/corex/lib64/python3/dist-packages/torch/cuda/memory.py)
```

Observed path to this failure:

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

Interpretation:

- The duplicate `qwen3_asr` registration was not the fundamental blocker; it was a dependency/version/path-order artifact.
- After aligning `transformers==5.12.1`, `tokenizers==0.22.2`, `pydantic-core==2.46.4`, and making CoreX torch a fallback path, SGLang-Omni again reached Qwen3-ASR stage worker startup.
- The current hard compatibility blocker is that `sglang==0.5.16` expects PyTorch 2.11-era CUDA memory allocator private symbols, while CoreX provides `torch 2.7.1+corex.4.4.0`.
- This is a stronger community data point than the earlier duplicate-config failure because it identifies an actual SGLang/CoreX API boundary.

Additional friction found in the venv pass:

```text
NumPy 2.2.6 warns against CoreX torch modules compiled with NumPy 1.x.
NVML is absent: NVML Shared Library Not Found.
mooncake and nixl are absent and remain CUDA/NVIDIA-stack dependencies.
flash-attn-4, flashinfer, sglang-kernel, sgl-deep-gemm, and CUDA-13 relay wheels remain unavailable/incompatible.
```

Recommended next step if continuing beyond compatibility reporting:

1. Downgrade venv `numpy` to `<2`, e.g. `numpy==1.26.4`, to remove the CoreX torch NumPy ABI warning.
2. Inspect SGLang's `pynccl_allocator.py` and determine whether `_cuda_beginAllocateCurrentThreadToPool` is optional or can be guarded for non-NVIDIA/CoreX torch.
3. Search SGLang for a flag to disable deep-gemm / pynccl allocator import paths for a torch-SDPA-only exploratory run.
4. If no such flag exists, a CoreX compatibility patch would likely need to begin at SGLang's platform/backend detection layer, not inside SGLang-Omni model code.

Updated key log files:

```text
/data/src/sglang-omni-corex-probe/logs/11_venv_transformers512_probe.txt
/data/src/sglang-omni-corex-probe/logs/12_venv_transformers512_fixdeps_probe.txt
/data/src/sglang-omni-corex-probe/logs/13_venv_transformers512_pinfix_probe.txt
/data/src/sglang-omni-corex-probe/logs/14_venv_transformers512_splitfix_probe.txt
/data/src/sglang-omni-corex-probe/logs/15_venv_transformers512_pathfix_probe.txt
```
