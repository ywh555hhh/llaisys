set -u
BASE=/data/src/mrv100-hw-probe
mkdir -p "$BASE/logs" "$BASE/reports"
export LD_LIBRARY_PATH=/usr/local/corex/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/lib/python3.10/site-packages:${PYTHONPATH:-}
{
  echo '=== ixsmi query field helps ==='
  /usr/local/corex/bin/ixsmi --help-query-gpu 2>&1 || true
  /usr/local/corex/bin/ixsmi --help-query-compute-apps 2>&1 || true
  echo '=== ixsmi filtered query ==='
  /usr/local/corex/bin/ixsmi --query-gpu=index,name,uuid,serial,pci.bus_id,pci.device_id,pci.sub_device_id,driver_version,memory.total,memory.used,memory.free,temperature.gpu,temperature.memory,power.draw,power.limit,clocks.current.sm,clocks.current.memory,clocks.max.sm,clocks.max.memory,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,utilization.gpu,utilization.memory,compute_mode --format=csv 2>&1 || true
  echo '=== ixsmi topology ==='
  /usr/local/corex/bin/ixsmi topo -h 2>&1 || true
  /usr/local/corex/bin/ixsmi topo -m 2>&1 || true
  /usr/local/corex/bin/ixsmi topo --matrix 2>&1 || true
  echo '=== ixsmi ixlink ==='
  /usr/local/corex/bin/ixsmi ixlink -h 2>&1 || true
  /usr/local/corex/bin/ixsmi ixlink -s 2>&1 || true
  echo '=== ixsmi dmon one sample ==='
  timeout 2 /usr/local/corex/bin/ixsmi dmon 2>&1 || true
  echo '=== ixsys / ixkn-cli help ==='
  /usr/local/corex/bin/ixsys --help 2>&1 | sed -n '1,160p' || true
  /usr/local/corex/bin/ixkn-cli --help 2>&1 | sed -n '1,160p' || true
  echo '=== torch and cuda runtime props ==='
  python3 - <<'PY'
import os, sys, json, time, traceback
print('python', sys.version)
try:
    import torch
    print('torch.version', torch.__version__)
    print('torch.version.cuda', torch.version.cuda)
    print('cuda_available', torch.cuda.is_available())
    print('device_count', torch.cuda.device_count())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print('torch_device_properties_repr', p)
        fields = []
        for name in dir(p):
            if name.startswith('_'):
                continue
            try:
                v=getattr(p,name)
            except Exception:
                continue
            if isinstance(v,(int,float,str,bool,tuple,list)) or v is None:
                fields.append((name,v))
        for k,v in sorted(fields): print('torch_prop', k, repr(v))
        print('torch_capability', torch.cuda.get_device_capability(0))
        print('torch_name', torch.cuda.get_device_name(0))
        print('torch_mem_get_info', torch.cuda.mem_get_info(0))
except Exception as e:
    print('torch_error', repr(e)); traceback.print_exc()
print('--- cuda-python cudart ---')
try:
    from cuda import cudart
    err, count = cudart.cudaGetDeviceCount()
    print('cudaGetDeviceCount', err, count)
    for i in range(count):
        r = cudart.cudaGetDeviceProperties(i)
        print('raw_cudaGetDeviceProperties_type', type(r), r)
        if isinstance(r, tuple):
            err, prop = r
        else:
            err, prop = None, r
        print('device', i, 'err', err)
        for k in sorted([x for x in dir(prop) if not x.startswith('_')]):
            try: v=getattr(prop,k)
            except Exception: continue
            if isinstance(v,(int,float,str,bytes,bool,tuple,list)) or v is None:
                if isinstance(v, bytes):
                    try: v=v.decode(errors='ignore').rstrip('\x00')
                    except Exception: pass
                print('cudart_prop', k, repr(v))
except Exception as e:
    print('cudart_error', repr(e)); traceback.print_exc()
print('--- cuda-python driver attributes ---')
try:
    from cuda import cuda
    print('cuInit', cuda.cuInit(0))
    err, dev = cuda.cuDeviceGet(0)
    print('cuDeviceGet', err, dev)
    attrs=[]
    enum = cuda.CUdevice_attribute
    for name in dir(enum):
        if name.startswith('CU_DEVICE_ATTRIBUTE_'):
            try:
                attr=getattr(enum,name)
                ret=cuda.cuDeviceGetAttribute(attr, dev)
                attrs.append((name,ret))
            except Exception as e:
                attrs.append((name,('ERR',repr(e))))
    for name,ret in attrs:
        print('cu_attr', name, ret)
except Exception as e:
    print('cuda_driver_error', repr(e)); traceback.print_exc()
PY
  echo '=== host memory/disk topology ==='
  free -h || true
  df -hT / /data /mnt 2>/dev/null || true
  lsblk -o NAME,MODEL,SIZE,TYPE,MOUNTPOINT,FSTYPE,ROTA,TRAN 2>/dev/null || true
  numactl --hardware 2>/dev/null || true
  lscpu | sed -n '1,120p'
} 2>&1 | tee "$BASE/logs/02_programming_interfaces.txt"
