set -euo pipefail
REPO=/data/src/llaisys
PROBE=/data/src/mrv100-hw-probe
cd "$REPO"
mkdir -p docs/hardware tools
cp "$PROBE/artifacts/cuda_device_attributes.json" docs/hardware/mrv100_cuda_device_attributes.json
cat > tools/probe_mrv100_hardware.py <<'PY'
#!/usr/bin/env python3
"""Probe Iluvatar MR-V100 / CoreX hardware properties.

This script is intentionally dependency-light. It uses:
- ixsmi for board/telemetry/PCIe information
- sysfs/lspci for Linux PCI placement
- cuda-python for low-level device attributes
- PyTorch, if present, for runtime properties and optional microbenchmarks

Typical CoreX environment:
  export LD_LIBRARY_PATH=/usr/local/corex/lib64:$LD_LIBRARY_PATH
  export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:$PYTHONPATH
  python3 tools/probe_mrv100_hardware.py --microbench --output mrv100_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run(cmd: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as exc:
        return {"cmd": cmd, "error": repr(exc)}


def read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def collect_ixsmi(ixsmi: str) -> dict[str, Any]:
    fields = ",".join(
        [
            "index",
            "name",
            "uuid",
            "serial",
            "vbios_version",
            "part_number",
            "pci.bus_id",
            "pci.device_id",
            "pci.sub_device_id",
            "driver_version",
            "cuda_version",
            "memory.total",
            "memory.used",
            "memory.free",
            "temperature.gpu",
            "temperature.memory",
            "gpu.power.draw",
            "gpu.current.power.limit",
            "gpu.default.power.limit",
            "clocks.current.sm",
            "clocks.current.memory",
            "clocks.max.sm",
            "clocks.max.memory",
            "pcie.link.gen.current",
            "pcie.link.gen.max",
            "pcie.link.width.current",
            "pcie.link.width.max",
            "pcie.tx.throughput",
            "pcie.rx.throughput",
            "utilization.gpu",
            "utilization.memory",
            "ecc.mode.current",
            "compute_mode",
            "pstate",
        ]
    )
    return {
        "summary": run([ixsmi]),
        "query": run([ixsmi, "-q"]),
        "list": run([ixsmi, "-L"]),
        "csv": run([ixsmi, f"--query-gpu={fields}", "--format=csv"]),
        "topology": run([ixsmi, "topo", "-m"]),
        "ixlink": run([ixsmi, "ixlink", "-s"]),
    }


def collect_sysfs(pci: str) -> dict[str, Any]:
    base = f"/sys/bus/pci/devices/{pci}"
    files = [
        "vendor",
        "device",
        "subsystem_vendor",
        "subsystem_device",
        "class",
        "numa_node",
        "local_cpulist",
        "current_link_speed",
        "current_link_width",
        "max_link_speed",
        "max_link_width",
        "resource",
    ]
    out = {name: read(f"{base}/{name}") for name in files}
    out["driver"] = None
    try:
        out["driver"] = str(Path(f"{base}/driver").resolve())
    except Exception:
        pass
    out["lspci"] = run(["lspci", "-s", pci.replace("0000:", ""), "-vvv"])
    return out


def collect_cuda_python() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        from cuda import cuda, cudart  # type: ignore

        out["cuInit"] = str(cuda.cuInit(0))
        err, dev = cuda.cuDeviceGet(0)
        out["cuDeviceGet"] = {"err": str(err), "dev": str(dev)}
        attrs = {}
        for name in dir(cuda.CUdevice_attribute):
            if not name.startswith("CU_DEVICE_ATTRIBUTE_"):
                continue
            try:
                ret = cuda.cuDeviceGetAttribute(getattr(cuda.CUdevice_attribute, name), dev)
                attrs[name] = {"err": str(ret[0]), "value": ret[1] if len(ret) > 1 else None}
            except Exception as exc:
                attrs[name] = {"error": repr(exc)}
        out["driver_attributes"] = attrs
        err, count = cudart.cudaGetDeviceCount()
        out["runtime_count"] = {"err": str(err), "count": count}
        if count:
            err, prop = cudart.cudaGetDeviceProperties(0)
            props = {}
            for key in dir(prop):
                if key.startswith("_"):
                    continue
                try:
                    val = getattr(prop, key)
                except Exception:
                    continue
                if isinstance(val, bytes):
                    val = val.decode(errors="ignore").rstrip("\x00")
                if isinstance(val, (int, float, str, bool)) or val is None:
                    props[key] = val
            out["runtime_properties"] = props
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def collect_torch(microbench: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import torch  # type: ignore

        out["torch_version"] = torch.__version__
        out["torch_cuda"] = torch.version.cuda
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["device_count"] = int(torch.cuda.device_count())
        if not torch.cuda.is_available():
            return out
        prop = torch.cuda.get_device_properties(0)
        out["device"] = {
            "name": prop.name,
            "major": prop.major,
            "minor": prop.minor,
            "multi_processor_count": prop.multi_processor_count,
            "total_memory": prop.total_memory,
            "l2_cache_size": getattr(prop, "l2_cache_size", None),
            "memory_bus_width": getattr(prop, "memory_bus_width", None),
            "memory_clock_rate": getattr(prop, "memory_clock_rate", None),
        }
        out["mem_get_info"] = tuple(int(x) for x in torch.cuda.mem_get_info(0))
        if microbench:
            out["microbench"] = run_torch_microbench(torch)
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def run_torch_microbench(torch: Any) -> dict[str, Any]:
    torch.set_grad_enabled(False)
    dev = "cuda:0"
    torch.cuda.set_device(0)

    def sync() -> None:
        torch.cuda.synchronize()

    def bench(label: str, func: Any, bytes_moved: int, warmup: int = 3, iters: int = 8) -> dict[str, Any]:
        for _ in range(warmup):
            func(); sync()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter(); func(); sync(); t1 = time.perf_counter()
            times.append(t1 - t0)
        best = min(times)
        med = statistics.median(times)
        return {"label": label, "best_s": best, "median_s": med, "best_GBps": bytes_moved / best / 1e9, "median_GBps": bytes_moved / med / 1e9}

    results: dict[str, Any] = {"transfers": [], "device_memory": [], "gemm": []}
    for mib in [256, 1024]:
        nbytes = mib * 1024 * 1024
        nelem = nbytes // 4
        cpu = torch.empty(nelem, dtype=torch.float32)
        gpu = torch.empty(nelem, dtype=torch.float32, device=dev)
        results["transfers"].append(bench(f"H2D pageable {mib}MiB", lambda: gpu.copy_(cpu), nbytes))
        results["transfers"].append(bench(f"D2H pageable {mib}MiB", lambda: cpu.copy_(gpu), nbytes))
        try:
            pcpu = torch.empty(nelem, dtype=torch.float32, pin_memory=True)
            results["transfers"].append(bench(f"H2D pinned {mib}MiB", lambda: gpu.copy_(pcpu, non_blocking=True), nbytes))
            results["transfers"].append(bench(f"D2H pinned {mib}MiB", lambda: pcpu.copy_(gpu, non_blocking=True), nbytes))
        except Exception as exc:
            results["transfers"].append({"label": f"pinned {mib}MiB", "error": repr(exc)})
        del cpu, gpu
        torch.cuda.empty_cache()
    for mib in [512, 2048]:
        nbytes = mib * 1024 * 1024
        nelem = nbytes // 4
        a = torch.empty(nelem, dtype=torch.float32, device=dev)
        b = torch.empty(nelem, dtype=torch.float32, device=dev)
        c = torch.empty(nelem, dtype=torch.float32, device=dev)
        a.fill_(1.0); b.fill_(2.0); sync()
        results["device_memory"].append(bench(f"D2D copy {mib}MiB", lambda: c.copy_(a), nbytes))
        results["device_memory"].append(bench(f"vector add {mib}MiB read2write1", lambda: torch.add(a, b, out=c), nbytes * 3))
        del a, b, c
        torch.cuda.empty_cache()
    for dtype_name in ["float32", "float16", "bfloat16"]:
        dtype = getattr(torch, dtype_name)
        n = 4096
        try:
            a = torch.randn((n, n), device=dev, dtype=dtype)
            b = torch.randn((n, n), device=dev, dtype=dtype)
            for _ in range(3):
                c = a @ b; sync()
            times = []
            for _ in range(6):
                t0 = time.perf_counter(); c = a @ b; sync(); t1 = time.perf_counter()
                times.append(t1 - t0)
            best = min(times); med = statistics.median(times); flops = 2 * n * n * n
            results["gemm"].append({"dtype": dtype_name, "n": n, "best_s": best, "median_s": med, "best_TFLOPS": flops / best / 1e12, "median_TFLOPS": flops / med / 1e12})
            del a, b, c
            torch.cuda.empty_cache()
        except Exception as exc:
            results["gemm"].append({"dtype": dtype_name, "n": n, "error": repr(exc)})
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ixsmi", default="/usr/local/corex/bin/ixsmi")
    ap.add_argument("--pci", default="0000:15:00.0")
    ap.add_argument("--microbench", action="store_true")
    ap.add_argument("--output", default="mrv100_probe.json")
    args = ap.parse_args()
    result = {
        "ixsmi": collect_ixsmi(args.ixsmi),
        "sysfs": collect_sysfs(args.pci),
        "cuda_python": collect_cuda_python(),
        "torch": collect_torch(args.microbench),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
PY
chmod +x tools/probe_mrv100_hardware.py
cat > docs/hardware/iluvatar-mrv100-corex-hardware-research.md <<'MD'
# Iluvatar MR-V100 / CoreX hardware research notes

This note records what the MR-V100 card exposes through actual programming and system interfaces on the rented 天数/CoreX machine. It is meant to support LLAISYS backend work and future AI inference infra experiments, not to be a vendor datasheet.

Probe machine:

```text
GPU: Iluvatar MR-V100, single card, 32 GiB
CoreX / IX-ML / Driver: 4.4.0
CUDA compatibility reported by ixsmi: 10.2
PCI bus id: 00000000:15:00.0
Device node: /dev/iluvatar3
Kernel driver: iluvatar
```

Raw probe logs were collected under `/data/src/mrv100-hw-probe/logs/` on the remote machine. A reusable probe script is committed at `tools/probe_mrv100_hardware.py`.

## Interface map

The useful interfaces split into five layers:

| Layer | Interface | Best for | Notes |
|---|---|---|---|
| Ops telemetry | `ixsmi`, `ixsmi -q`, `ixsmi --query-gpu=... --format=csv` | clocks, power, memory use, temp, ECC, PCIe link, process list | Scriptable and closest to `nvidia-smi` style monitoring. |
| Topology | `ixsmi topo -m`, `ixsmi ixlink -s` | GPU/NIC/CPU affinity, IXLink/P2P visibility | Single-card instance shows no active IXLink. |
| Linux device | `lspci`, `/sys/bus/pci/devices/0000:15:00.0/*`, `/dev/iluvatar3` | PCI IDs, BARs, NUMA affinity, negotiated PCIe width/speed, driver binding | Needed for placement and host/device path reasoning. |
| Programming limits | `cuda-python`: `cuda.cuDeviceGetAttribute`, `cudart.cudaGetDeviceProperties` | SM count, warp size, shared memory, registers, L2, memory bus width, feature flags | Richest interface for kernel/backend engineering. |
| Framework runtime | CoreX PyTorch | sanity checks, memory info, H2D/D2H/D2D/GEMM smoke benchmarks | Convenient, but less complete than CUDA driver attributes. |

## `ixsmi` fields worth scripting

`ixsmi --help-query-gpu` exposes many fields. The most useful for our workflow are:

```text
name, uuid, serial, vbios_version, part_number,
pci.bus_id, pci.device_id, pci.sub_device_id,
pcie.link.gen.current, pcie.link.gen.max,
pcie.link.width.current, pcie.link.width.max,
pcie.tx.throughput, pcie.rx.throughput,
memory.total, memory.used, memory.free,
utilization.gpu, utilization.memory,
temperature.gpu, temperature.memory,
gpu.power.draw, gpu.current.power.limit,
gpu.default.power.limit,
clocks.current.sm, clocks.current.memory,
clocks.max.sm, clocks.max.memory,
ecc.mode.current, compute_mode, pstate
```

Example command:

```bash
/usr/local/corex/bin/ixsmi --query-gpu=index,name,uuid,serial,pci.bus_id,memory.total,clocks.current.sm,clocks.current.memory,pcie.link.gen.current,pcie.link.width.current,gpu.power.draw --format=csv
```

Observed output summary:

```text
name: Iluvatar MR-V100
uuid: GPU-75401a56-2224-5257-8e2c-8f6dac7c37bc
serial: 23230261115496
vbios: 2.0.17
part number: MR-V100-00
memory.total: 32768 MiB
power limit: 150 W
current clocks: SM 1500 MHz, memory 1600 MHz
max clocks: SM 1600 MHz, memory 1600 MHz
ECC: enabled
compute mode: Default
```

## Linux PCI / NUMA view

`lspci` identifies the card as a processing accelerator:

```text
15:00.0 Processing accelerators [1200]: Device [1e3e:0002]
Kernel driver in use: iluvatar
Physical Slot: 14
NUMA node: 0
```

Sysfs gives the negotiated PCIe state:

```text
/sys/bus/pci/devices/0000:15:00.0/current_link_speed = 16.0 GT/s PCIe
/sys/bus/pci/devices/0000:15:00.0/current_link_width = 16
/sys/bus/pci/devices/0000:15:00.0/max_link_speed = 16.0 GT/s PCIe
/sys/bus/pci/devices/0000:15:00.0/max_link_width = 16
/sys/bus/pci/devices/0000:15:00.0/local_cpulist = 0-27,56-83
```

BAR/resource map:

```text
BAR0: 0x214000000000-0x2147ffffffff, 32 GiB, 64-bit prefetchable
BAR2: 0xb0100000-0xb013ffff, 256 KiB, non-prefetchable
```

Interpretation:

- The card is on PCIe Gen4 x16.
- One-direction theoretical PCIe payload is roughly 31.5 GB/s before software overhead.
- NUMA-local CPU list is node0: `0-27,56-83`.

## CUDA-compatible device attributes

The CUDA runtime/driver compatibility layer reports:

```text
Compute capability: 7.1
SM / multiprocessor count: 16
Warp size: 64
Global memory: 32 GiB
L2 cache: 16 MiB
Memory clock: 1,600,000 kHz
Global memory bus width: 2048 bits
ECC enabled: 1
Concurrent kernels: 1
Async engine count: 1
Unified addressing: 1
Managed memory: 0
GPUDirect RDMA supported: 1
Memory pools supported: 1
Virtual address management supported: 1
```

Kernel resource limits:

```text
Max threads per block: 4096
Max threads per SM: 8192
Max block dim: 4096 x 4096 x 256
Max grid dim: 2147483647 x 65535 x 65535
Shared memory per block: 128 KiB
Shared memory per SM: 128 KiB
Registers per block: 262144
Registers per SM: 262144
Total constant memory: 16 KiB
```

A full attribute dump is stored at `docs/hardware/mrv100_cuda_device_attributes.json`.

## Bandwidth and compute estimates

The CUDA-style memory fields imply this upper-bound estimate:

```text
2 * 1,600,000 kHz * (2048 bits / 8) / 1e6 ~= 819.2 GB/s
```

This is an inferred upper bound from runtime fields, not a vendor-published guarantee.

PyTorch/CoreX smoke benchmark results from this container:

| Probe | Best observed |
|---|---:|
| H2D pageable, 1 GiB | ~12.2 GB/s |
| D2H pageable, 1 GiB | ~12.8 GB/s |
| H2D pinned, 1 GiB | ~28.0 GB/s |
| D2H pinned, 1 GiB | ~28.5 GB/s |
| D2D copy, one-way count, 2 GiB | ~292 GB/s |
| Vector add, read2/write1 traffic count | ~585 GB/s |
| FP32 GEMM, n=8192 | ~24.4 TFLOPS |
| FP16 GEMM, n=8192 | ~96.1 TFLOPS |
| BF16 GEMM, n=8192 | ~96.1 TFLOPS |

Interpretation:

- Pinned H2D/D2H is close to practical PCIe Gen4 x16 bandwidth.
- Device memory bandwidth observed through simple PyTorch ops is below the 819 GB/s theoretical estimate, but plausible for framework-level kernels.
- FP16/BF16 GEMM looks around 4x FP32 in this CoreX/PyTorch path.

## CoreX software surface discovered

CoreX Python packages exposed on this image include:

```text
cuda, torch, torchvision, torchaudio, ixrt, pyixstream, tvm, tensorrt,
onnx, onnxruntime, pyarrow, datasets, quiver, torch_quiver,
torch_scatter, torch_sparse, torch_cluster
```

CoreX command-line tools include:

```text
ixsmi      # device telemetry / topology, like nvidia-smi
ixsys      # profiler/tracer; strings/help mention CUDA API, CUDA device, kernels, memcpy/memset, GPU summary
ixgdb      # GPU debugger
ixkn-cli   # kernel/tooling CLI
ixobjdump  # object dump
ixAssembler
ucx_info / ucx_perftest
```

`strings /usr/local/corex/bin/ixsmi` shows it links/uses `libixml.so` and contains many NVML-compatible and IX-specific symbols, including:

```text
ixmlDeviceGetComputeAllProcesses
ixmlDeviceGetIxLinkInfo
ixmlDeviceGetIxLinkMaxPortNum
ixmlDeviceGetMemPhy
ixmlDeviceGetBoardPowerUsage
ixmlDeviceGetHBMTemperature
nvmlDeviceGetCudaComputeCapability
nvmlDeviceGetPcieThroughput
nvmlDeviceGetMemoryInfo
nvmlDeviceGetClockInfo
```

So the monitoring stack seems intentionally NVML-like, but with Iluvatar-specific `ixml*` extensions.

## UCX / communication surface

`ucx_info -d` shows CoreX UCX was built with CUDA and gdrcopy support:

```text
UCX version: 1.18.0
Configured with --with-cuda=/home/corex/sw_home/local/corex --with-gdrcopy=/home/corex/sw_home/local/corex
Memory domains include cuda_cpy, cuda_ipc, gdr_copy
cuda_copy bandwidth model: 10000 MB/s, latency 8000 ns
cuda_ipc bandwidth model: 250000 MB/s, latency 1000 ns
gdr_copy memory type: cuda
```

This matters for inference infra because SGLang/NIXL/Mooncake-style transfer paths eventually care about whether GPU memory can be registered, exported, and moved efficiently between processes/devices. On this single-card box it is mostly a capability inventory, not a multi-GPU validation.

## Negative findings / caveats

- No active IXLink on this single-card instance.
- NVML shared library is not exposed under the usual NVIDIA path; SGLang diagnostics report `NVML Shared Library Not Found`.
- `/usr/local/corex/bin/nvcc` is only a small compatibility script, not a real NVIDIA nvcc toolchain.
- `modinfo iluvatar` failed inside the container even though `lsmod` shows the `iluvatar` kernel module loaded.
- `udevadm` is not installed in the container.
- Some CUDA texture/surface attributes report zero; treat those as compatibility-layer limitations unless validated by vendor docs.
- PyTorch microbenchmarks are smoke tests. They are useful for shape-of-machine intuition, not a rigorous peak performance study.

## Why this matters for LLAISYS / AI infra work

For custom backend work, the most important facts are:

1. Warp size is 64, not NVIDIA's common 32. Kernel assumptions around warp-level reductions/shuffles must be checked.
2. SM count is 16, so occupancy and launch geometry should be reasoned differently from a large NVIDIA GPU.
3. Shared memory and registers are generous per SM/block, but actual compiler mapping must be measured with CoreX tools.
4. PCIe is Gen4 x16 and pinned transfers reach ~28 GB/s, so host-device copies are not the first bottleneck for small coursework kernels but matter for serving pipelines.
5. The monitoring API is `ixsmi`/IX-ML-like, while framework compatibility exposes CUDA-style device attributes. Code should prefer CUDA runtime attributes for portable limits and `ixsmi` for operational telemetry.
6. For SGLang-Omni compatibility, the hard blocker found earlier is not raw hardware capability; it is SGLang's assumption of PyTorch 2.11/NVIDIA CUDA allocator internals and CUDA-13 backend wheels.

## Reproduce

```bash
export LD_LIBRARY_PATH=/usr/local/corex/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:$PYTHONPATH
python3 tools/probe_mrv100_hardware.py --microbench --output /tmp/mrv100_probe.json
```

The original remote logs are not committed because they are large and machine-specific. The committed JSON attribute dump is the compact, useful subset.
MD

git status --short
