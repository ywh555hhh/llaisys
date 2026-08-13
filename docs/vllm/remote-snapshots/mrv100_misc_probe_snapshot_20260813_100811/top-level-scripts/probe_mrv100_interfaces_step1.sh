set -u
BASE=/data/src/mrv100-hw-probe
mkdir -p "$BASE/logs" "$BASE/reports"
{
  echo '=== timestamp ==='; date; hostname; uname -a
  echo '=== corex bin ==='; ls -lh /usr/local/corex/bin | sed -n '1,220p'
  echo '=== ixsmi help variants ==='; /usr/local/corex/bin/ixsmi --help 2>&1 || true; /usr/local/corex/bin/ixsmi -h 2>&1 || true
  echo '=== ixsmi normal ==='; /usr/local/corex/bin/ixsmi || true
  echo '=== ixsmi query attempts ==='; for a in -q --query --list-gpus --display=MEMORY --display=CLOCK --display=PERFORMANCE --display=PCI; do echo "--- ixsmi $a"; /usr/local/corex/bin/ixsmi $a 2>&1 || true; done
  echo '=== lspci gpu ==='; command -v lspci || true; lspci -nn | grep -iE 'iluvatar|3d|vga|display|15:00' || true
  echo '=== lspci verbose 15:00 ==='; lspci -s 15:00.0 -vvv 2>&1 || true
  echo '=== sysfs pci 0000:15:00.0 ==='; DEV=/sys/bus/pci/devices/0000:15:00.0; if [ -d "$DEV" ]; then for f in vendor device subsystem_vendor subsystem_device class numa_node local_cpulist current_link_speed current_link_width max_link_speed max_link_width enable irq driver_override resource; do echo "--- $f"; cat "$DEV/$f" 2>/dev/null || true; done; echo '--- driver'; readlink -f "$DEV/driver" 2>/dev/null || true; echo '--- iommu_group'; readlink -f "$DEV/iommu_group" 2>/dev/null || true; echo '--- files'; find "$DEV" -maxdepth 1 -type f -printf '%f\n' | sort; fi
  echo '=== /proc/driver-ish ==='; find /proc/driver /proc/ix* /proc/corex* /proc/iluvatar* -maxdepth 3 -type f 2>/dev/null | sort | xargs -r -I{} sh -c 'echo --- {}; sed -n "1,120p" {} 2>/dev/null'
  echo '=== /dev gpu-ish ==='; ls -lh /dev | grep -iE 'ix|core|cuda|nvidia|dri|gpu|iluvatar' || true; find /dev -maxdepth 2 -iname '*ix*' -o -iname '*core*' -o -iname '*cuda*' -o -iname '*nvidia*' -o -iname '*gpu*' 2>/dev/null | sort
  echo '=== corex env files ==='; find /usr/local/corex -maxdepth 3 -type f \( -iname '*version*' -o -iname '*device*' -o -iname '*smi*' -o -iname '*cuda*' \) | sort | sed -n '1,220p'
} 2>&1 | tee "$BASE/logs/01_system_interfaces.txt"
