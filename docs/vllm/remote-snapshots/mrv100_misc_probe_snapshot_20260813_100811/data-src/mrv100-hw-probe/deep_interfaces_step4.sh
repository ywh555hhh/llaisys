set -u
BASE=/data/src/mrv100-hw-probe
mkdir -p "$BASE/logs" "$BASE/artifacts"
export LD_LIBRARY_PATH=/usr/local/corex/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/lib/python3.10/site-packages:${PYTHONPATH:-}
{
  echo '=== corex tree high level ==='
  find /usr/local/corex -maxdepth 2 -type d | sort
  echo '=== corex include candidates ==='
  find /usr/local/corex -maxdepth 4 -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cuh' \) | sort | sed -n '1,400p'
  echo '=== corex library candidates ==='
  find /usr/local/corex -maxdepth 4 -type f \( -name '*.so' -o -name '*.so.*' -o -name '*.a' \) | sort | sed -n '1,500p'
  echo '=== library names matching ix/cuda/core/rt/ml ==='
  find /usr/local/corex -maxdepth 4 -type f \( -name '*.so' -o -name '*.so.*' \) | grep -Ei 'ix|cuda|core|runtime|ml|blas|dnn|rtc|prof|trace|comm|ucx' | sort || true
  echo '=== python modules under corex dist-packages ==='
  python3 - <<'PY'
import pkgutil, sys, os
paths=['/usr/local/corex/lib64/python3/dist-packages','/usr/local/lib/python3.10/site-packages']
for p in paths:
    print('## path',p)
    if os.path.isdir(p):
        mods=sorted({m.name for m in pkgutil.iter_modules([p])})
        for name in mods[:300]: print(name)
        print('count', len(mods))
PY
  echo '=== loaded modules and modinfo iluvatar ==='
  lsmod | grep -iE 'iluvatar|ix|corex|cuda|nvidia|vfio|mlx' || true
  modinfo iluvatar 2>&1 || true
  echo '=== udev/device node details ==='
  stat /dev/iluvatar3 2>&1 || true
  udevadm info --query=all --name=/dev/iluvatar3 2>&1 || true
  echo '=== ioctl-ish strings in ixsmi and libs ==='
  strings /usr/local/corex/bin/ixsmi | grep -Ei 'ioctl|/dev/|ixml|ml|device|query|temperature|memory|power|pcie|bandwidth|link' | head -200 || true
  echo '=== ixsys help and strings hints ==='
  /usr/local/corex/bin/ixsys --help 2>&1 | sed -n '1,240p' || true
  strings /usr/local/corex/bin/ixsys | grep -Ei 'trace|profile|metric|bandwidth|pcie|memory|kernel|device|json|csv|output|help' | head -240 || true
  echo '=== ucx info relevant ==='
  /usr/local/corex/bin/ucx_info -v 2>&1 | sed -n '1,160p' || true
  /usr/local/corex/bin/ucx_info -d 2>&1 | sed -n '1,260p' || true
  echo '=== dump selected CUDA attrs to json ==='
  python3 - <<'PY'
import json, traceback
out={}
try:
    from cuda import cuda, cudart
    out['cuInit']=str(cuda.cuInit(0))
    err, dev = cuda.cuDeviceGet(0)
    out['cuDeviceGet']={'err':str(err),'dev':str(dev)}
    attrs={}
    for name in dir(cuda.CUdevice_attribute):
        if name.startswith('CU_DEVICE_ATTRIBUTE_'):
            try:
                ret=cuda.cuDeviceGetAttribute(getattr(cuda.CUdevice_attribute,name), dev)
                attrs[name]={'err':str(ret[0]),'value':ret[1] if len(ret)>1 else None}
            except Exception as e:
                attrs[name]={'error':repr(e)}
    out['driver_attributes']=attrs
    err,count=cudart.cudaGetDeviceCount(); out['runtime_count']={'err':str(err),'count':count}
    if count:
        err, prop=cudart.cudaGetDeviceProperties(0)
        props={}
        for k in dir(prop):
            if k.startswith('_'): continue
            try: v=getattr(prop,k)
            except Exception: continue
            if isinstance(v, bytes):
                v=v.decode(errors='ignore').rstrip('\x00')
            if isinstance(v,(int,float,str,bool)) or v is None:
                props[k]=v
        out['runtime_properties']=props
except Exception as e:
    out['error']=repr(e); out['traceback']=traceback.format_exc()
open('/data/src/mrv100-hw-probe/artifacts/cuda_device_attributes.json','w').write(json.dumps(out,indent=2,sort_keys=True))
print(json.dumps(out,indent=2,sort_keys=True)[:8000])
PY
} 2>&1 | tee "$BASE/logs/04_deep_interfaces.txt"
