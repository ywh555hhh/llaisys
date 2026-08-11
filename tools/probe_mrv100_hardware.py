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
