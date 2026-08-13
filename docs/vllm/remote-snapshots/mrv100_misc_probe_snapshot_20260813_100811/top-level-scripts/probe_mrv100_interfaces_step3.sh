set -u
BASE=/data/src/mrv100-hw-probe
mkdir -p "$BASE/logs" "$BASE/reports"
export LD_LIBRARY_PATH=/usr/local/corex/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/lib/python3.10/site-packages:${PYTHONPATH:-}
{
  echo '=== corrected ixsmi csv query ==='
  /usr/local/corex/bin/ixsmi --query-gpu=index,name,uuid,serial,pci.bus_id,pci.device_id,pci.sub_device_id,driver_version,cuda_version,memory.total,memory.used,memory.free,temperature.gpu,temperature.memory,gpu.power.draw,gpu.current.power.limit,gpu.default.power.limit,clocks.current.sm,clocks.current.memory,clocks.max.sm,clocks.max.memory,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,pcie.tx.throughput,pcie.rx.throughput,utilization.gpu,utilization.memory,compute_mode,pstate --format=csv 2>&1 || true
  echo '=== microbench torch bandwidth/flops ==='
  python3 - <<'PY'
import time, math, json, statistics, traceback, os
try:
    import torch
    torch.set_grad_enabled(False)
    print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
    dev='cuda:0'
    torch.cuda.set_device(0)
    props=torch.cuda.get_device_properties(0)
    print('device', props.name, 'total_memory', props.total_memory, 'sm', props.multi_processor_count, 'cc', props.major, props.minor)
    def sync(): torch.cuda.synchronize()
    def bench_func(label, func, bytes_moved, warmup=3, iters=10):
        times=[]
        for _ in range(warmup):
            func(); sync()
        for _ in range(iters):
            t0=time.perf_counter(); func(); sync(); t1=time.perf_counter(); times.append(t1-t0)
        med=statistics.median(times)
        best=min(times)
        print(f'{label}: best_s={best:.6f} median_s={med:.6f} best_GBps={bytes_moved/best/1e9:.2f} median_GBps={bytes_moved/med/1e9:.2f}')
    # H2D/D2H with pageable and pinned if available
    for mib in [64, 256, 512, 1024]:
        nbytes=mib*1024*1024
        nelem=nbytes//4
        print(f'--- transfer size {mib} MiB ---')
        cpu=torch.empty(nelem, dtype=torch.float32)
        gpu=torch.empty(nelem, dtype=torch.float32, device=dev)
        bench_func(f'H2D pageable {mib}MiB', lambda: gpu.copy_(cpu, non_blocking=False), nbytes, iters=8)
        bench_func(f'D2H pageable {mib}MiB', lambda: cpu.copy_(gpu, non_blocking=False), nbytes, iters=8)
        try:
            pcpu=torch.empty(nelem, dtype=torch.float32, pin_memory=True)
            bench_func(f'H2D pinned {mib}MiB', lambda: gpu.copy_(pcpu, non_blocking=True), nbytes, iters=8)
            bench_func(f'D2H pinned {mib}MiB', lambda: pcpu.copy_(gpu, non_blocking=True), nbytes, iters=8)
        except Exception as e:
            print(f'pinned {mib}MiB unavailable:', repr(e))
        del cpu,gpu
        torch.cuda.empty_cache()
    # Device memory copy and vector ops
    for mib in [256, 512, 1024, 2048]:
        nbytes=mib*1024*1024
        nelem=nbytes//4
        print(f'--- device memory size {mib} MiB ---')
        a=torch.empty(nelem, dtype=torch.float32, device=dev)
        b=torch.empty(nelem, dtype=torch.float32, device=dev)
        c=torch.empty(nelem, dtype=torch.float32, device=dev)
        a.fill_(1.0); b.fill_(2.0); sync()
        bench_func(f'D2D copy {mib}MiB', lambda: c.copy_(a), nbytes, iters=12)
        bench_func(f'vector add read2write1 {mib}MiB', lambda: torch.add(a,b,out=c), nbytes*3, iters=12)
        del a,b,c
        torch.cuda.empty_cache()
    # GEMM empirical FLOPS
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        print('--- matmul dtype', dtype, '---')
        for n in [1024, 2048, 4096, 6144, 8192]:
            try:
                a=torch.randn((n,n), device=dev, dtype=dtype)
                b=torch.randn((n,n), device=dev, dtype=dtype)
                c=None
                for _ in range(3): c=a@b; sync()
                times=[]
                for _ in range(6):
                    t0=time.perf_counter(); c=a@b; sync(); t1=time.perf_counter(); times.append(t1-t0)
                best=min(times); med=statistics.median(times); flops=2*n*n*n
                print(f'matmul n={n} dtype={str(dtype)} best_s={best:.6f} median_s={med:.6f} best_TFLOPS={flops/best/1e12:.2f} median_TFLOPS={flops/med/1e12:.2f}')
                del a,b,c
                torch.cuda.empty_cache()
            except Exception as e:
                print(f'matmul n={n} dtype={dtype} FAIL {type(e).__name__}: {e}')
                torch.cuda.empty_cache()
except Exception as e:
    print('bench_error', repr(e)); traceback.print_exc()
PY
} 2>&1 | tee "$BASE/logs/03_bandwidth_flops_microbench.txt"
