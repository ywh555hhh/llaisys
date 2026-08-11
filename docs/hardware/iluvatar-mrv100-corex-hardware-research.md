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
## Research questions for the next LLAISYS pass

This machine is useful precisely because it forces the backend work to separate three questions that are often blurred together on NVIDIA:

| Question | What this MR-V100 evidence tells us | What still needs a real kernel/profiler pass |
|---|---|---|
| Can CUDA-style code enumerate and allocate the device? | Yes. `cuda-python`, CoreX PyTorch, and `ixsmi` all see one CUDA-compatible device. | Whether every CUDA API used by LLAISYS runtime paths is implemented with identical semantics. |
| What should a launch geometry assume? | Warp size is 64, SM count is 16, max threads/block is 4096, and shared memory/block is 128 KiB. | Actual compiler lowering, occupancy behavior, and whether warp-level intrinsics map cleanly through CoreX. |
| Where is the first likely bottleneck? | Simple pinned PCIe copies are near Gen4 x16 practical bandwidth; simple device vector add reaches hundreds of GB/s. | Coursework ops should first optimize correctness, coalescing, and launch shape; serving workloads need profiler-backed prefill/decode separation. |
| Is it a good SGLang-Omni target today? | Hardware and CoreX runtime expose CUDA-like execution, PyTorch, UCX, and gdrcopy surfaces. | The current SGLang/NVIDIA ecosystem assumes newer PyTorch/CUDA/NVML behavior than this image exposes. Treat it as a porting research target, not a drop-in runtime. |

## Suggested acceptance data for community notes

If we want this to be useful to the community, the artifact should avoid claiming vendor-certified peak numbers. A good public note should include reproducible, machine-local evidence:

1. `ixsmi` board snapshot: card name, driver/CoreX version, PCIe link, clocks, power, ECC, memory.
2. CUDA attribute dump: compute capability, warp size, SM count, shared memory, registers, L2, memory bus width, managed-memory and GPUDirect flags.
3. Host/device path: pageable vs pinned H2D/D2H transfer numbers, plus NUMA affinity.
4. Device path: D2D copy, vector add, and at least one GEMM size for FP32/FP16/BF16.
5. Toolchain reality check: whether `nvcc`, `ixsys`, `ixgdb`, `ixobjdump`, PyTorch, ONNX Runtime, TVM, TensorRT/IXRT, UCX, and gdrcopy are actually present.
6. Negative results: NVML gaps, missing headers/tools, single-card IXLink absence, and places where CUDA compatibility fields are suspicious or zeroed.

## LLAISYS implementation implications

For assignment/backend work, I would treat MR-V100/CoreX as a CUDA-compatible backend with a non-NVIDIA execution model, not as a NVIDIA GPU clone.

- Runtime layer: keep using CUDA-style discovery/memcpy/stream APIs where they work, but isolate telemetry and diagnostics behind an `ixsmi`/IX-ML path rather than assuming NVML.
- Kernel layer: do not hardcode warp-size 32 assumptions. Prefer `warpSize`/device attributes and write reductions in a way that can be audited for 64-lane warps.
- Test layer: keep CPU reference paths as the source of truth, then compare device outputs across shapes that stress contiguous, strided, sliced, and broadcast-like access patterns.
- Performance layer: collect coarse bandwidth/GEMM smoke numbers first, then use `ixsys` only after a kernel is correct; otherwise profiler output is just expensive noise.
- Serving layer: if revisiting SGLang-Omni, start with environment compatibility and allocator/NVML assumptions before chasing model-level bugs.

## Commands to extend the study

```bash
# Re-run the compact hardware probe from inside the repo.
export LD_LIBRARY_PATH=/usr/local/corex/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:$PYTHONPATH
python3 tools/probe_mrv100_hardware.py --output /tmp/mrv100_probe_quick.json

# Re-run with PyTorch transfer/GEMM smoke benchmarks.
python3 tools/probe_mrv100_hardware.py --microbench --output /tmp/mrv100_probe_microbench.json

# Confirm script and committed attribute dump are structurally valid.
python3 -m py_compile tools/probe_mrv100_hardware.py
python3 -m json.tool docs/hardware/mrv100_cuda_device_attributes.json >/tmp/attrs.checked.json

# Explore profiler surface before using it on real kernels.
/usr/local/corex/bin/ixsys --help 2>&1 | head -80
strings /usr/local/corex/bin/ixsys | grep -Ei 'cuda|kernel|memcpy|summary|trace' | head -80
```

