#!/usr/bin/python3
import argparse
import time
import multiprocessing
from bcc import BPF, PerfType, PerfSWConfig

# Arguments
parser = argparse.ArgumentParser(description="Monitor instantaneous CPU core usage of a process.")
parser.add_argument("-p", "--pid", type=int, required=True, help="Target PID (TGID) to monitor")
parser.add_argument("-s", "--sample", type=int, default=10, help="Sample  interval in microseconds")
parser.add_argument("-r", "--report", type=int, default=10, help="Report intervall in seconds")
args = parser.parse_args()

# Detect actual number of CPUs to size the BPF loop correctly
num_cpus = multiprocessing.cpu_count()

# BPF Program
bpf_text = """
#include <linux/sched.h>

// Map: Index = CPU ID, Value = 1 if Target PID is running, 0 otherwise
BPF_ARRAY(cpu_occupancy, u8, %d);

// Histogram to store the distribution
BPF_HISTOGRAM(dist, u32);

// Target TGID
const static u32 TARGET_TGID = %d;

// 1. TRACE SCHEDULER SWITCHES
// We simply mark the current CPU as "Occupied" or "Free" based on who is running next.
RAW_TRACEPOINT_PROBE(sched_switch) {
    // args[0] = prev, args[1] = next
    struct task_struct *next = (struct task_struct *)ctx->args[1];

    u32 cpu = bpf_get_smp_processor_id();
    u8 is_target = 0;

    // If the next task is our target, mark this CPU as occupied (1)
    if (next->tgid == TARGET_TGID) {
        is_target = 1;
    }
    // Otherwise, it remains 0 (clearing the state if it was previously 1)
    
    cpu_occupancy.update(&cpu, &is_target);
    return 0;
}

// 2. HIGH FREQUENCY SAMPLER
// iterate over all CPUs to sum the current usage
int do_sample(struct pt_regs *ctx) {
    u32 sum = 0;
    int key = 0;
    u8 *val;

    // Unroll loop for performance and verifier happiness
    #pragma unroll
    for (int i = 0; i < %d; i++) {
        key = i;
        val = cpu_occupancy.lookup(&key);
        if (val && *val == 1) {
            sum++;
        }
    }

    dist.increment(sum);
    return 0;
}
""" % (num_cpus, args.pid, num_cpus)

# Load BPF
# We must allow a slightly larger instruction limit for the loop over 192 cores
b = BPF(text=bpf_text)

# Attach the sampler to CPU 0 (acting as the ticker)
b.attach_perf_event(
    ev_type=PerfType.SOFTWARE,
    ev_config=PerfSWConfig.CPU_CLOCK,
    fn_name="do_sample",
    sample_period=args.sample * 1000,
    cpu=0 
)

print(f"Tracking PID {args.pid} on {num_cpus} detected cores...")
print(f"Sampling active core count every {args.sample}us. Dumping histogram every {args.report}s.")
print("Ctrl-C to stop.")

try:
    while True:
        time.sleep(args.report)
        print(f"\n[{time.strftime('%H:%M:%S')}] Active Core Concurrency Distribution:")
        b["dist"].print_linear_hist("Active Cores")
        b["dist"].clear()

except KeyboardInterrupt:
    print("Detaching...")
