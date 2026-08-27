# nvidia_interrupt_monitor

Tools to detect **lost MSI-X interrupts** on NVIDIA GPUs: the case where
the GPU's internal interrupt aggregator has a pending bit **latched
forever** because the interrupt message never reached the CPU.

## The failure class

MSI-X interrupts are posted PCIe memory writes, not wires. Anything that
briefly pauses or replays interrupt delivery can drop one on the floor:

- suspend/resume
- PCIe AER recovery windows
- host kernel kexec
- hypervisor live update / live migration pauses in cloud environments

NVIDIA GPUs (Turing and later) track "interrupt already sent" in an
on-GPU aggregator (`INTR_LEAF[0..15]` in the BAR0 VF block at
`0xb80000`). If the MSI-X write is lost, the corresponding pending bit
stays latched, the GPU believes the message was delivered and never
re-sends it, the driver ISR never runs, and every completion event
behind that vector is silently gone.

The workload symptom is nasty: variable 100ms+ GPU kernel-dispatch
stalls and heavily degraded jobs with **zero errors anywhere** -- no
Xid, clean dmesg, clean ECC, every standard health check green. These
tools were extracted from a months-long production investigation of
exactly that syndrome on a large H200 training fleet; the final
detector is one 32-bit register read.

## Detection rule

On a healthy GPU the ISR drains a pending bit within microseconds. So:

> A bit in `INTR_LEAF[i]` that stays pending across several reads spread
> over seconds, while enabled in `LEAF_EN_SET[i]` and with a clean MSI-X
> table/PBA, is a lost interrupt.

All tools here implement this rule at different scales. All detection is
strictly **read-only** (mmap of the PCI `resource0` sysfs file plus a
handful of 32-bit MMIO reads) and needs root. Register offsets come from
NVIDIA's open-gpu-kernel-modules (`dev_vm.h`, `intr_cpu_tu102.c`); the
layout exists on Turing+ and is validated on Hopper (H100/H200).

## Recovery (for context)

Toggling the leaf enable mask (`LEAF_EN_CLEAR` then `LEAF_EN_SET`)
re-arms the GPU's interrupt send latch: the GPU re-emits the message and
the stuck bit drains in microseconds, without disturbing running work.
The driver performs this toggle itself during channel setup, which is
why (re)initializing certain GPU libraries can "accidentally" heal an
affected GPU, and a VM reboot always does. A dedicated heal tool is
intentionally not part of this repo yet; the monitor only detects.

## Tools

| Tool                        | Scale           | What it does                                              |
|-----------------------------|-----------------|-----------------------------------------------------------|
| `tools/nv_intr_dump.c`      | one host        | Full dump of aggregator + MSI-X state, stuck-bit verdict  |
| `nv_intr_fleet_scan.sh`     | whole cluster   | kubectl-exec fleet sweep, a few register reads per GPU    |
| `cmd/nv_intr_monitor`       | per-node daemon | Go daemon (k8s DaemonSet), Prometheus metrics + healthz   |
| `tools/nv_bar0_recorder.c`  | one host, deep  | 10-100 kHz per-GPU sampler of leaf/PBA state              |
| `tools/nv_bar0_analyzer.py` | offline         | Occupancy/edge-rate/latched-bit analysis of recorder runs |

Per-tool documentation: [docs/](docs/).

## Quick start

```bash
make                         # builds bin/nv_intr_dump, bin/nv_bar0_recorder,
                             # bin/nv_intr_monitor

# One host: is any GPU holding a latched interrupt?
sudo bin/nv_intr_dump

# Whole k8s fleet (uses existing per-node privileged pods, e.g. DCGM):
./nv_intr_fleet_scan.sh scan -j 32

# Continuous monitoring on every node:
kubectl apply -f deploy/daemonset.yaml
# then alert on: nvidia_intr_gpu_stuck == 1
```

## Safety

- Detection tools only ever `mmap(PROT_READ)` BAR0 and issue aligned
  32-bit reads: the same access the driver performs constantly. They do
  not touch the driver, CUDA, or running workloads.
- The single exception is `nv_bar0_recorder --enable-all-leaves`, which
  temporarily writes the leaf enable mask (documented in
  [docs/nv_bar0_recorder.md](docs/nv_bar0_recorder.md)); it is off by
  default.
- Reading BAR0 of a GPU that is mid-reset can return garbage; every tool
  sanity-checks `PMC_BOOT_0` (chip id, BAR0+0x0) before trusting reads.

## License

MIT.
