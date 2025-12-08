# XID Tracer

A Go program that uses eBPF to trace NVIDIA GPU XID errors via the `nvidia:nvidia_dev_xid` kernel tracepoint.

## Why Kernel Tracepoints Instead of NVML?

The NVML event API (`nvmlDeviceRegisterEvents` / `nvmlEventSetWait`) is unreliable for capturing XID errors during rapid-fire event bursts. In production, we observed that when a GPU failure triggers a cascade of XID errors (e.g., XID 172 -> XID 48 -> XID 13), NVML often misses the initial root-cause event (XID 172) while only reporting subsequent events.

**Root causes of NVML unreliability:**
- Internal event queue in closed-source `libnvidia-ml.so` has unknown fixed size
- Events that arrive before `nvmlDeviceRegisterEvents()` completes are lost forever
- Queue overflow silently drops events with no indication to userspace
- No API exists to query or configure the queue size

**The kernel tracepoint (`nvidia:nvidia_dev_xid`) is authoritative:**
- Events are captured synchronously in kernel space before any userspace queue
- The NVIDIA driver calls `trace_nvidia_dev_xid()` directly in the XID reporting path
- No events can be lost due to userspace buffer overflow
- eBPF perf buffers provide reliable delivery with overflow detection

This makes kernel tracepoint monitoring the only reliable method for production GPU health monitoring where missing the root-cause XID error is unacceptable.
