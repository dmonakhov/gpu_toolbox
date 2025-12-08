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

## Prerequisites

- Linux kernel 5.7+ with BTF support (`CONFIG_DEBUG_INFO_BTF=y`)
- NVIDIA driver loaded (creates the `nvidia:nvidia_dev_xid` tracepoint)
- Go 1.21+
- clang/llvm for eBPF compilation
- Root privileges or `CAP_BPF` capability

## Install Build Dependencies

```bash
sudo apt-get install -y clang llvm libbpf-dev

# Install bpf2go (Go tool to compile eBPF)
go install github.com/cilium/ebpf/cmd/bpf2go@latest
```

## Quick Start

```bash
# Install Go dependencies
make deps

# Generate eBPF bindings and build
make build

# Run (requires root)
sudo ./xid_tracer
```

## Output Example

```
2024/12/08 15:30:00 Waiting for NVIDIA XID events... (press Ctrl+C to exit)
2024/12/08 15:30:00 Tracepoint: nvidia:nvidia_dev_xid
XID event received: code=172 pci=0000:75:00 msg=SM ICACHE, Uncorrectable SRAM error in GPC 7 TPC 6
XID event received: code=48 pci=0000:75:00 msg=pid=3432060, name=python3, Ch 0000000a
```

## Files

- `main.go` - Go program that loads eBPF and reads events
- `xid.bpf.c` - eBPF program attached to the tracepoint
- `vmlinux.h` - Minimal kernel type definitions (regenerate for full BTF support)
- `Makefile` - Build automation

## Regenerating vmlinux.h

For full kernel type support, regenerate vmlinux.h from your running kernel:

```bash
# Requires bpftool and BTF-enabled kernel
make vmlinux
```

Or manually:
```bash
bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h
```

## Verifying Tracepoint Exists

```bash
# Check if NVIDIA tracepoint is available
ls /sys/kernel/debug/tracing/events/nvidia/

# View tracepoint format (shows field layout)
cat /sys/kernel/debug/tracing/events/nvidia/nvidia_dev_xid/format
```

## Troubleshooting

### "NVIDIA tracepoint not found"
- Ensure nvidia driver is loaded: `lsmod | grep nvidia`
- The tracepoint only exists when the driver is loaded

### "Loading eBPF objects: field XxxYyy: invalid argument"
- Kernel BTF may be missing or incompatible
- Regenerate vmlinux.h: `make vmlinux`

### "Operation not permitted"
- Run with root privileges: `sudo ./xid_tracer`
- Or add CAP_BPF capability: `sudo setcap cap_bpf+ep ./xid_tracer`

## How It Works

1. The eBPF program attaches to `nvidia:nvidia_dev_xid` tracepoint
2. When an XID error occurs, the kernel fires the tracepoint
3. Our eBPF program captures the event data (error code, PCI device, message)
4. Data is sent to userspace via perf event buffer
5. Go program reads and prints the events

## Comparison with Other Methods

| Method          | Reliability | Latency       | Complexity |
|-----------------|-------------|---------------|------------|
| **This (eBPF)** | Highest     | ~microseconds | Medium     |
| dmesg scanning  | High        | ~seconds      | Low        |
| NVML events     | Medium      | ~milliseconds | Low        |

eBPF tracepoint monitoring is the most reliable method because:
- Events are captured directly in kernel space
- No userspace queue that can overflow (like NVML)
- No log parsing delays (like dmesg)
