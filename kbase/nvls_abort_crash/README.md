# nvls_abort_crash: aborting a job mid-NVLS-collective crashes the NVLink fabric

Demonstrates a crash caused by aborting (SIGTERM) a PyTorch job while
NCCL NVLS (NVLink SHARP in-switch reduction) collectives are in
progress.

A tight `dist.all_reduce()` loop on the default NCCL transport runs
cleanly indefinitely. Send SIGTERM to `torchrun` while collectives are
in flight and within ~1 s the kernel emits a fabric-error burst:

- NVSwitch SXid 12028 "egress non-posted PRIV error" on all NVSwitches
- NVRM Xid 137 "TLC RX interrupt ... PRIV Error" on the GPUs
- NVRM Xid 94 "Contained: SM" attributed to the workload processes
- Ranks die with `cudaErrorContained: Invalid access of peer GPU
  memory over nvlink`

The burst looks like a fabric failure but is not one: subsequent
workloads on the same host run correctly (a later abort-mid-NVLS just
fires the avalanche again). The trigger is the NVLS path
specifically: with `NCCL_NVLS_ENABLE=0` NCCL keeps using NVLink P2P
but the same abort sequence exits cleanly. This is exactly
why nvidia-resiliency-ext requires `NCCL_NVLS_ENABLE=0` for
in-process restart ("SHARP raises an exception" after a hang):
https://github.com/NVIDIA/nvidia-resiliency-ext/blob/main/docs/source/release-notes.md?plain=1#L159

Observed on AWS p5e (8x H200 SXM, 4 NVSwitches), driver 580.105.08,
CUDA 13.0, `nvcr.io/nvidia/pytorch:25.11-py3`.

## Examples

### Crash (NVLS enabled -- the default on NVSwitch systems):

```bash
NCCL_NVLS_ENABLE=1 ./trivial_repro.sh
# ...
# === STATUS ===
# BUG REPRODUCED -- ~230 Xid/SXid lines in dmesg
```

### Clean control (NVLS disabled, NCCL stays on NVLink P2P):

```bash
NCCL_NVLS_ENABLE=0 ./trivial_repro.sh
# ...
# === STATUS ===
# clean -- no Xid/SXid
```

## WARNING

The NVLS-enabled example produces a scary Xid/SXid avalanche (~230
dmesg lines: SXid 12028, Xid 137, Xid 94) but does NOT break the
fabric -- subsequent workloads run correctly. Still, be careful on
managed hosts: fleet health monitors and Xid-based alerting will see
the burst and may flag or drain the node.
