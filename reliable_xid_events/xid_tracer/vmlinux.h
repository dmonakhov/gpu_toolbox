// Minimal vmlinux.h for XID tracer
// For production use, generate with: bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h

#ifndef __VMLINUX_H__
#define __VMLINUX_H__

typedef unsigned char __u8;
typedef signed char __s8;
typedef unsigned short __u16;
typedef signed short __s16;
typedef unsigned int __u32;
typedef signed int __s32;
typedef unsigned long long __u64;
typedef signed long long __s64;

typedef __u8 u8;
typedef __s8 s8;
typedef __u16 u16;
typedef __s16 s16;
typedef __u32 u32;
typedef __s32 s32;
typedef __u64 u64;
typedef __s64 s64;

typedef __u16 __be16;
typedef __u32 __be32;
typedef __u64 __be64;

typedef __u32 __wsum;

// BPF map types
enum bpf_map_type {
    BPF_MAP_TYPE_UNSPEC = 0,
    BPF_MAP_TYPE_HASH = 1,
    BPF_MAP_TYPE_ARRAY = 2,
    BPF_MAP_TYPE_PROG_ARRAY = 3,
    BPF_MAP_TYPE_PERF_EVENT_ARRAY = 4,
};

#define BPF_F_CURRENT_CPU 0xffffffffULL

#endif /* __VMLINUX_H__ */

// ============================================================================
// BPF Helper Definitions (normally from bpf/bpf_helpers.h)
// ============================================================================

#ifndef __BPF_HELPERS__
#define __BPF_HELPERS__

#define SEC(NAME) __attribute__((section(NAME), used))

#define __uint(name, val) int (*name)[val]
#define __type(name, val) typeof(val) *name
#define __array(name, val) typeof(val) *name[]

// BPF helper function declarations
static long (*bpf_probe_read)(void *dst, __u32 size, const void *unsafe_ptr) = (void *) 4;
static long (*bpf_probe_read_str)(void *dst, __u32 size, const void *unsafe_ptr) = (void *) 45;
static long (*bpf_perf_event_output)(void *ctx, void *map, __u64 flags, void *data, __u64 size) = (void *) 25;
static long (*bpf_trace_printk)(const char *fmt, __u32 fmt_size, ...) = (void *) 6;

// Clang built-ins for BPF
#define bpf_printk(fmt, ...)                                    \
({                                                              \
    char ____fmt[] = fmt;                                       \
    bpf_trace_printk(____fmt, sizeof(____fmt), ##__VA_ARGS__);  \
})

#endif /* __BPF_HELPERS__ */

// ============================================================================
// BPF CO-RE Definitions (normally from bpf/bpf_core_read.h)
// ============================================================================

#ifndef __BPF_CORE_READ__
#define __BPF_CORE_READ__

#define BPF_CORE_READ(src, field) \
    ({ typeof((src)->field) __val; bpf_probe_read(&__val, sizeof(__val), &(src)->field); __val; })

#endif /* __BPF_CORE_READ__ */
