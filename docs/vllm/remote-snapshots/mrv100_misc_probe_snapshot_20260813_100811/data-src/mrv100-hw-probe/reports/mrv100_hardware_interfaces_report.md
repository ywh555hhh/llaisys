# Iluvatar MR-V100 / CoreX hardware interface probe

Date: 2026-08-11
Machine: 天数 / `iluvatar-mrv100-node-0004` style container
Workdir: `/data/src/mrv100-hw-probe`
GPU: `Iluvatar MR-V100`, 32 GiB, single card

## Executive summary

This card exposes useful hardware details through four practical layers:

1. `ixsmi`: board identity, clocks, memory, power, ECC, PCIe link, topology, process/utilization monitoring.
2. PCI/sysfs/lspci: PCI device IDs, BAR mappings, NUMA affinity, PCIe negotiated speed/width, Linux driver binding.
3. CUDA-compatible runtime/driver APIs via `cuda-python`: compute capability, SM count, warp size, L2, memory bus width, memory clock, shared memory/register limits, GPUDirect flags, unified addressing, ECC.
4. PyTorch/CoreX: high-level device properties and enough runtime behavior to run bandwidth/GEMM probes.

Most detailed architectural fields came from CUDA driver/runtime attributes, not from `ixsmi`. `ixsmi` is better for live ops/board telemetry; CUDA APIs are better for programming limits.

## Available interfaces

### `ixsmi`

Binary:

```text
/usr/local/corex/bin/ixsmi
```

Useful commands:

```bash
ixsmi
ixsmi -q
ixsmi -L
ixsmi --query-gpu=<fields> --format=csv
ixsmi dmon
ixsmi pmon
ixsmi topo -m
ixsmi ixlink -s
```

Important supported `--query-gpu` fields include:

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
clocks.current.sm, clocks.current.memory,
clocks.max.sm, clocks.max.memory,
ecc.mode.current, compute_mode, pstate
```

### CUDA/Python API

The machine has CUDA-compatible Python bindings:

```python
from cuda import cudart
from cuda import cuda
```

Useful calls:

```python
cudart.cudaGetDeviceProperties(0)
cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_..., dev)
```

This is the richest programming interface for hardware limits.

### PyTorch/CoreX

CoreX provides PyTorch at:

```text
/usr/local/corex/lib64/python3/dist-packages/torch
```

Useful calls:

```python
import torch
torch.cuda.get_device_properties(0)
torch.cuda.get_device_capability(0)
torch.cuda.mem_get_info(0)
```

### PCI/sysfs

Useful Linux paths/commands:

```bash
lspci -s 15:00.0 -vvv
/sys/bus/pci/devices/0000:15:00.0/current_link_speed
/sys/bus/pci/devices/0000:15:00.0/current_link_width
/sys/bus/pci/devices/0000:15:00.0/resource
/sys/bus/pci/devices/0000:15:00.0/local_cpulist
readlink -f /sys/bus/pci/devices/0000:15:00.0/driver
```

## Board and driver identity

From `ixsmi -q`:

```text
Product Name: Iluvatar MR-V100
Serial Number: 23230261115496
GPU UUID: GPU-75401a56-2224-5257-8e2c-8f6dac7c37bc
VBIOS Version: 2.0.17
Board ID: 0x1400
Board Part Number: MR-V100-00
Driver Version: 4.4.0
IX-ML: 4.4.0
CUDA Version reported by ixsmi: 10.2
Compute Mode: Default
ECC Mode: Enabled
```

Linux PCI identity:

```text
PCI bus id: 00000000:15:00.0
Vendor: 0x1e3e
Device: 0x0002
Class: 0x120000 Processing accelerator
Kernel driver: iluvatar
Device node: /dev/iluvatar3
```

## PCIe and host topology

From `ixsmi`, sysfs, and lspci:

```text
PCIe generation: Gen4 current / Gen4 max
PCIe width: x16 current / x16 max
Current link speed: 16.0 GT/s PCIe
Current link width: 16
NUMA node: 0
CPU affinity: 0-27,56-83
Physical slot: 14
```

PCI BAR/resource map:

```text
BAR0: 0x214000000000-0x2147ffffffff, 32 GiB, 64-bit prefetchable
BAR2: 0xb0100000-0xb013ffff, 256 KiB, non-prefetchable
```

Topology:

```text
ixsmi topo -m: GPU0 <-> NIC0 is SYS
NIC0: mlx5_0
No active ixlink device reported on this single-card instance.
```

Interpretation:

- PCIe theoretical one-direction payload bandwidth for Gen4 x16 is about 31.5 GB/s before software overhead.
- Measured pinned H2D/D2H was about 28 GB/s, which is plausible for this link.

## GPU memory and storage-visible regions

From `ixsmi` and CUDA runtime:

```text
GPU memory total: 32768 MiB / 34359738368 bytes
Used at idle: ~68 MiB
Free at idle: ~32700 MiB
BAR0 aperture: 32 GiB prefetchable PCI memory window
L2 cache: 16777216 bytes = 16 MiB
```

From host/container storage:

```text
Container root filesystem: 30 GiB overlay, about 27 GiB free during probe
/data: 196 GiB ext4 loop device, about 185 GiB free during probe
Host RAM visible in container: 32 GiB
Swap: none
Underlying host also exposes NVMe INTEL SSDPE2KX040T8 3.7T, but container persistence uses /data.
```

## CUDA runtime architectural attributes

From `cuda-python` driver/runtime APIs:

```text
Compute capability: 7.1
SM / multiprocessor count: 16
Warp size: 64
Max clock rate: 1500000 kHz current / 1600 MHz max via ixsmi
Memory clock rate: 1600000 kHz
Global memory bus width: 2048 bits
L2 cache size: 16 MiB
Total global memory: 32 GiB
ECC enabled: yes
Unified addressing: yes
Managed memory: no
Concurrent kernels: yes
Async engine count: 1
GPUDirect RDMA supported: yes
Memory pools supported: yes
Virtual address management supported: yes
```

Execution limits:

```text
Max threads per block: 4096
Max threads per SM: 8192
Max block dim: 4096 x 4096 x 256
Max grid dim: 2147483647 x 65535 x 65535
Shared memory per block: 131072 bytes = 128 KiB
Shared memory per SM: 131072 bytes = 128 KiB
Registers per block: 262144
Registers per SM: 262144
Total constant memory: 16384 bytes
```

Notable unsupported/zero fields observed:

```text
Managed memory: 0
Cooperative launch: 0
Cooperative multi-device launch: 0
Stream priorities: 0
Texture/surface limit fields mostly report 0 in this compatibility layer
```

## Theoretical bandwidth estimates from runtime fields

Runtime reports:

```text
memoryClockRate = 1600000 kHz
memoryBusWidth = 2048 bits = 256 bytes
```

Using the common CUDA estimate:

```text
Theoretical memory bandwidth ~= 2 * memoryClockRate_kHz * (memoryBusWidth_bits / 8) / 1e6
                              ~= 2 * 1,600,000 * 256 / 1e6
                              ~= 819.2 GB/s decimal
```

Caveat: this is inferred from exposed CUDA-style fields. It is useful as an upper-bound sanity check, not a vendor datasheet claim.

## Measured transfer and bandwidth probes

Small PyTorch/CoreX microbenchmarks were run with `torch 2.7.1+corex.4.4.0`. Treat results as smoke measurements, not final benchmark numbers.

### PCIe host/device transfer

Best observed for 1 GiB transfers:

```text
H2D pageable: ~12.2 GB/s
D2H pageable: ~12.8 GB/s
H2D pinned:   ~28.0 GB/s
D2H pinned:   ~28.5 GB/s
```

Pinned transfer is close to PCIe Gen4 x16 practical bandwidth. Pageable transfer is roughly half that.

### Device memory bandwidth

Best observed:

```text
D2D copy, 2 GiB payload counted one-way: ~292 GB/s
Vector add, read two arrays + write one array: ~585 GB/s effective traffic
```

If D2D copy is counted as read + write traffic, its effective memory traffic is also about 584 GB/s. This is below the 819 GB/s theoretical estimate, but directionally consistent for a simple PyTorch-level benchmark.

### GEMM rough compute throughput

Best observed `torch.matmul` throughput:

```text
FP32 n=8192:   ~24.36 TFLOPS
FP16 n=8192:   ~96.10 TFLOPS
BF16 n=8192:   ~96.07 TFLOPS
```

These are empirical CoreX/PyTorch GEMM numbers from this container, not peak spec-sheet values. They are still useful as a quick sanity check for compute capability.

## Practical takeaways for AI Infra work

1. `ixsmi -q` is enough for ops telemetry: clocks, power, memory, ECC, PCIe state, process list.
2. `ixsmi --query-gpu=... --format=csv` is the easiest scriptable monitoring surface.
3. `cuda-python` is the best way to get programmer-facing architecture limits: SM count, warp size, shared memory, registers, L2, memory bus width, and feature flags.
4. PyTorch gives enough high-level properties and can run meaningful smoke benchmarks, but exposes fewer low-level architectural fields than CUDA driver attributes.
5. PCI/sysfs is necessary for Linux-level placement: NUMA affinity, BAR aperture, PCIe negotiated width/speed, and driver binding.
6. For this specific MR-V100/CoreX machine, the most important performance shape is: PCIe Gen4 x16, 32 GiB VRAM, 16 SM, warp size 64, 16 MiB L2, 2048-bit memory bus, ~819 GB/s theoretical memory bandwidth, ~96 TFLOPS observed FP16/BF16 GEMM in PyTorch.

## Logs

```text
/data/src/mrv100-hw-probe/logs/01_system_interfaces.txt
/data/src/mrv100-hw-probe/logs/02_programming_interfaces.txt
/data/src/mrv100-hw-probe/logs/03_bandwidth_flops_microbench.txt
```
