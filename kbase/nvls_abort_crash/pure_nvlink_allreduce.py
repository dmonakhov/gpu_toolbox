#!/usr/bin/env python3
"""pure_nvlink_allreduce: minimal NCCL allreduce loop on NVLink.

Continuously runs dist.all_reduce() in a tight loop on the default
NCCL transport (NVLink P2P + SHM). This is the workload that
reproduces the SXid 12028 + Xid 137/94 fabric burst on AWS p5e.
On a freshly-rebooted host the kernel emits the burst at
~880 s (~2.75M iters) with the default knobs below.

We deliberately do NOT set NCCL_NVLS_ENABLE (nor NCCL_P2P_DISABLE /
NCCL_SHM_DISABLE), so NCCL picks the default NVLink transport with
NVLS (SHARP in-switch reduction) enabled. The workaround is
NCCL_NVLS_ENABLE=0 alone: NVLink P2P stays in use, but the same
loop then runs cleanly indefinitely.

Launch: torchrun --nproc_per_node=8 pure_nvlink_allreduce.py

Knobs (env vars):
  PNA_NCCL_MB    64    allreduce buffer size in MB
  PNA_LOG_EVERY  500   rank-0 progress log cadence
"""
import os
import time
import signal
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist


NCCL_MB   = int(os.environ.get('PNA_NCCL_MB',   64))
LOG_EVERY = int(os.environ.get('PNA_LOG_EVERY', 500))


def main():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    dist.init_process_group(backend='nccl', device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()

    nccl_buf = torch.randn(NCCL_MB * 1024 * 1024 // 4,
                           device=device, dtype=torch.float32)

    # Warmup -- forces NCCL comm ring + NVLink registration
    nccl_buf.mul_(1.001)
    dist.all_reduce(nccl_buf)
    torch.cuda.synchronize(device)
    dist.barrier()

    if rank == 0:
        print(f"pure_nvlink_allreduce: world={world} NCCL_MB={NCCL_MB} "
              f"(expecting NVLink P2P transport)", flush=True)

    stop = False
    def _stop(signo, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    t0 = time.time()
    iters = 0
    while not stop:
        dist.all_reduce(nccl_buf)
        iters += 1
        if iters % LOG_EVERY == 0 and rank == 0:
            elapsed = time.time() - t0
            rate = iters / elapsed
            print(f"pure_nvlink_allreduce: iter={iters} elapsed={elapsed:.1f}s "
                  f"rate={rate:.0f} iter/s", flush=True)

    torch.cuda.synchronize(device)
    if rank == 0:
        elapsed = time.time() - t0
        print(f"pure_nvlink_allreduce: stopped iters={iters} elapsed={elapsed:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
