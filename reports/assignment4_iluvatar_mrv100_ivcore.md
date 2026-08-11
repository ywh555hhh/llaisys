# LLAISYS Assignment 4 - Iluvatar MR-V100 ivcore backend

Date: 2026-08-11
Host: 120.220.95.189:12222
Repo: /data/src/llaisys
GPU: Iluvatar MR-V100, 32 GB
CoreX: /usr/local/corex-4.4.0
IX-ML / Driver: 4.4.0
CUDA compatibility reported by ixsmi: 10.2

## Key finding

The usable kernel compiler on this image is not nvcc. /usr/local/corex/bin/nvcc is a 194-byte compatibility script that only prints CUDA 10.2 version text.

The real compiler path is /usr/local/corex/bin/clang++ -x ivcore.

A standalone kernel launch probe compiled and ran successfully:

```cpp
__global__ void add1(int *x) { x[0] += 1; }
```

Result:

```text
result=42
run_rc=0
```

## Build integration

xmake/nvidia.lua now defines a custom ivcore.build rule for .cu files. It compiles existing LLAISYS CUDA-style assignment-four kernel files with CoreX clang:

```bash
/usr/local/corex/bin/clang++ -x ivcore -std=c++17 -fPIC -O3 -DNDEBUG -Wno-unknown-pragmas -Iinclude -Isrc -I/usr/local/corex/include -c <source>.cu -o <object>.o
```

The xmake CUDA devlink path is intentionally bypassed because it would call the fake nvcc -dlink. Since these kernels do not use RDC, no real device runtime link step is required. A local empty build/ivcore_stub/libcudadevrt.a is generated only to satisfy the automatic CUDA link flag added by xmake.

The real ivcore build uses these existing device kernel files:

- src/device/nvidia/nvidia_runtime_api.cu
- src/device/nvidia/nvidia_resource.cu
- src/ops/add/nvidia/add_nvidia.cu
- src/ops/argmax/nvidia/argmax_nvidia.cu
- src/ops/embedding/nvidia/embedding_nvidia.cu
- src/ops/swiglu/nvidia/swiglu_nvidia.cu
- src/ops/rms_norm/nvidia/rms_norm_nvidia.cu
- src/ops/rope/nvidia/rope_nvidia.cu
- src/ops/self_attention/nvidia/self_attention_nvidia.cu
- src/ops/linear/nvidia/linear_nvidia.cu

The previous staged CPU fallback files were removed from the build path.

## Environment setup

```bash
export PATH=/root/.local/bin:$PATH
export XMAKE_ROOT=y
export LD_LIBRARY_PATH=/usr/local/corex/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/data/src/llaisys/python:/usr/local/lib/python3.10/site-packages:/usr/local/corex/lib64/python3/dist-packages:$PYTHONPATH
```

Python packages:

- CoreX torch: 2.7.1, CUDA available on Iluvatar MR-V100
- transformers: 5.15.0
- tokenizers: 0.22.2
- huggingface_hub: 1.27.0

## Build commands

```bash
xmake f --nv-gpu=y -cv
xmake -rv
xmake install
```

Result: passed. python/llaisys/libllaisys/libllaisys.so links against /usr/local/corex/lib64/libcudart.so.10.2.

## Runtime and op tests

Command:

```bash
for t in test/test_runtime.py test/ops/add.py test/ops/argmax.py test/ops/embedding.py test/ops/swiglu.py test/ops/rms_norm.py test/ops/rope.py test/ops/self_attention.py test/ops/linear.py; do
  python3 "$t" --device nvidia || exit $?
done
```

Result: exit code 0.

| Test | Result |
|---|---:|
| test/test_runtime.py --device nvidia | PASS |
| test/ops/add.py --device nvidia | PASS |
| test/ops/argmax.py --device nvidia | PASS |
| test/ops/embedding.py --device nvidia | PASS |
| test/ops/swiglu.py --device nvidia | PASS |
| test/ops/rms_norm.py --device nvidia | PASS |
| test/ops/rope.py --device nvidia | PASS |
| test/ops/self_attention.py --device nvidia | PASS |
| test/ops/linear.py --device nvidia | PASS |

## Inference smoke

Command:

```bash
python3 test/test_infer.py --model /mnt/moark-models/Qwen3-0.6B --device nvidia --test --max_steps 1 --prompt "Who are you?"
```

Result: passed. This is a short HuggingFace-wrapper smoke test; it validates the Python/device environment, but the core assignment-four evidence is the real ivcore runtime/op test matrix above.

## Conclusion

The Iluvatar / MR-V100 part of assignment four is now implemented as a real CoreX ivcore kernel build path, not a staged CPU fallback. Existing LLAISYS CUDA-style kernels compile with /usr/local/corex/bin/clang++ -x ivcore, link into libllaisys.so, and pass the runtime/op tests on MR-V100.
