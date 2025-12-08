//go:build ignore

// eBPF program to capture NVIDIA XID events from the nvidia:nvidia_dev_xid tracepoint.
// This program is compiled by bpf2go and loaded by the Go program.

#include "vmlinux.h"
// Note: bpf_helpers and bpf_core_read are included in our minimal vmlinux.h

// Event structure passed to userspace
struct xid_event {
    __u32 error_code;
    char pci_dev[16];
    char msg[256];
};

// Perf event map for sending events to userspace
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, sizeof(__u32));
} events SEC(".maps");

// Tracepoint context structure for nvidia:nvidia_dev_xid
// Based on the tracepoint format from:
// /sys/kernel/debug/tracing/events/nvidia/nvidia_dev_xid/format
//
// Expected format:
//   field:__data_loc char[] dev;
//   field:u32 error_code;
//   field:__data_loc char[] msg;
struct trace_event_nvidia_dev_xid {
    // Common tracepoint fields (from struct trace_entry)
    __u16 common_type;
    __u8 common_flags;
    __u8 common_preempt_count;
    __s32 common_pid;

    // nvidia_dev_xid specific fields
    __u32 __data_loc_dev;   // offset << 16 | length for dev string
    __u32 error_code;
    __u32 __data_loc_msg;   // offset << 16 | length for msg string
};

SEC("tracepoint/nvidia/nvidia_dev_xid")
int handle_nvidia_xid(struct trace_event_nvidia_dev_xid *ctx) {
    struct xid_event event = {};

    // Get error code directly
    event.error_code = ctx->error_code;

    // Extract dev string using __data_loc encoding
    // Upper 16 bits = offset from ctx
    __u32 dev_loc = ctx->__data_loc_dev;
    __u32 dev_offset = dev_loc >> 16;
    // Use fixed size read - verifier doesn't like variable sizes
    bpf_probe_read_str(&event.pci_dev, sizeof(event.pci_dev), (char *)ctx + dev_offset);

    // Extract msg string using __data_loc encoding
    __u32 msg_loc = ctx->__data_loc_msg;
    __u32 msg_offset = msg_loc >> 16;
    // Use fixed size read
    bpf_probe_read_str(&event.msg, sizeof(event.msg), (char *)ctx + msg_offset);

    // Send event to userspace via perf buffer
    bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, &event, sizeof(event));

    return 0;
}

char LICENSE[] SEC("license") = "GPL";
