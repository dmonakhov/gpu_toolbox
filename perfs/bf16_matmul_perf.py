#!/usr/bin/env python3
# Ultra-minimal multi-GPU BF16 matmul burner (SM>=90)

import argparse, threading, torch

def worker(gpu_id, N, runtime_s, warmup_s, results):
    torch.cuda.set_device(gpu_id)
    dev = torch.device(f"cuda:{gpu_id}")
    sm = torch.cuda.get_device_capability(dev)[0] * 10 + torch.cuda.get_device_capability(dev)[1]
    if sm < 90:
        raise RuntimeError(f"GPU{gpu_id}: requires SM>=90, found SM{sm}")

    uuid = torch.cuda.get_device_properties(dev).uuid
    dtype = torch.bfloat16

    a = torch.randn((N, N), device=dev, dtype=dtype)
    b = torch.randn((N, N), device=dev, dtype=dtype)
    c = torch.empty((N, N), device=dev, dtype=dtype)

    # --- Warmup phase (GPU time based) ---
    warmup_start = torch.cuda.Event(enable_timing=True)
    warmup_end = torch.cuda.Event(enable_timing=True)
    warmup_start.record()
    iters = 0
    while True:
        torch.matmul(a, b, out=c)
        iters += 1
        warmup_end.record()
        torch.cuda.synchronize()
        if warmup_start.elapsed_time(warmup_end) / 1e3 >= warmup_s:
            break

    # --- Timed benchmark (GPU time based) ---
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    iters = 0
    ev_start.record()

    while True:
        torch.matmul(a, b, out=c)
        iters += 1
        ev_end.record()
        torch.cuda.synchronize()
        gpu_time_s = ev_start.elapsed_time(ev_end) / 1e3
        if gpu_time_s >= runtime_s:
            break

    # Final timing & TFLOPs
    gpu_s = ev_start.elapsed_time(ev_end) / 1e3
    tflops = (2 * (N ** 3) * iters) / (gpu_s * 1e12)
    results[gpu_id] = (gpu_id, uuid, iters, tflops)

def main():
    p = argparse.ArgumentParser(description="BF16 matmul burner with CUDA event-based timing (tab TFLOPs report)")
    p.add_argument("--size", type=int, default=8192)
    p.add_argument("--runtime", type=float, default=10.0, help="Benchmark duration (seconds)")
    p.add_argument("--warmup", type=float, default=5.0, help="Warmup duration (seconds)")
    args = p.parse_args()

    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("No GPUs found")

    results, threads = {}, []
    for i in range(n):
        t = threading.Thread(target=worker, args=(i, args.size, args.runtime, args.warmup, results))
        t.start(); threads.append(t)
    for t in threads: t.join(timeout=(args.runtime + args.warmup + 60))

    print("gpu_id\tuuid\titers\ttflops")
    total = 0.0
    for gpu_id, uuid, iters, tflops in sorted(results.values()):
        total += tflops
        print(f"{gpu_id}\t{uuid}\t{iters}\t{tflops:.2f}")
    print(f"# total_tflops\t{total:.2f}")

if __name__ == "__main__":
    main()

