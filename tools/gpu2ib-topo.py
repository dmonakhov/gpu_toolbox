#!/usr/bin/env python3
import os
import re
import subprocess
import json

SYSFS_PCI_DEV = "/sys/bus/pci/devices"
SYS_CLASS_IB = "/sys/class/infiniband"
PCI_RE = re.compile(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]")

def run_cmd(cmd):
    return subprocess.check_output(cmd, text=True).strip()

def normalize_bdf(bdf: str) -> str:
    b = bdf.strip().lower()
    parts = b.split(":")
    if len(parts) == 3:
        dom, bus, devf = parts
        dom = f"{int(dom,16):04x}"
        return f"{dom}:{bus}:{devf}"
    return b

def get_gpus():
    """Return list of dicts {index, bdf}."""
    out = run_cmd([
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id",
        "--format=csv,noheader,nounits"
    ])
    gpus = []
    for line in out.splitlines():
        idx, bdf = [x.strip() for x in line.split(",")]
        gpus.append({"index": idx, "bdf": normalize_bdf(bdf)})
    return gpus

def get_ib_devices():
    """Return dict ibdev_name -> normalized BDF."""
    ibs = {}
    if not os.path.isdir(SYS_CLASS_IB):
        return ibs
    for dev in sorted(os.listdir(SYS_CLASS_IB)):
        link = os.path.join(SYS_CLASS_IB, dev, "device")
        if not os.path.exists(link):
            continue
        real = os.path.realpath(link)
        matches = PCI_RE.findall(real)
        if matches:
            ibs[dev] = normalize_bdf(matches[-1])
    return ibs

def pci_chain(bdf):
    path = os.path.join(SYSFS_PCI_DEV, bdf)
    if not os.path.exists(path):
        return []
    real = os.path.realpath(path)
    return [m.lower() for m in PCI_RE.findall(real)]

def longest_common_prefix(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return a[:i]

def pci_node_is_bridge(node_bdf):
    class_path = os.path.join(SYSFS_PCI_DEV, node_bdf, "class")
    try:
        with open(class_path) as f:
            val = int(f.read().strip(), 16)
        return ((val >> 16) & 0xff) == 0x06
    except Exception:
        return False

def map_gpus_to_ib():
    gpus = get_gpus()
    ibs = get_ib_devices()
    ib_chains = {name: pci_chain(bdf) for name, bdf in ibs.items()}

    result = []
    for gpu in gpus:
        gpu_bdf = gpu["bdf"]
        gpu_chain = pci_chain(gpu_bdf)
        colocated = []
        for ib_name, ib_bdf in ibs.items():
            prefix = longest_common_prefix(gpu_chain, ib_chains[ib_name])
            if prefix and pci_node_is_bridge(prefix[-1]):
                colocated.append({"name": ib_name, "bdf": ib_bdf})
        result.append({
            "gpu_index": gpu["index"],
            "gpu_bdf": gpu_bdf,
            "ib_devices": colocated
        })
    return result

if __name__ == "__main__":
    mapping = map_gpus_to_ib()
    print(json.dumps(mapping, indent=2))
