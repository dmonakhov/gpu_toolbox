# nv_bar0_recorder + nv_bar0_analyzer

Deep-dive pair for a single host: the recorder samples every GPU's
`INTR_LEAF[0..15]` and MSI-X PBA at 10-100 kHz while a workload of your
choice runs; the analyzer turns the trace into occupancy, edge-rate, and
latched-bit reports.

Use it when the one-shot tools are not enough: to see *which* bits an
application fires and how fast, to map leaf bits to GPU engines, to
measure how long pending bits live, or to catch a latch happening
during a suspected delivery-loss event.

## Recorder

```bash
gcc -O2 -Wall -pthread -o nv_bar0_recorder tools/nv_bar0_recorder.c

sudo ./nv_bar0_recorder --outdir run1 --rate 10000 --duration 300
sudo ./nv_bar0_recorder --gpu 0000:53:00.0 --rate 100000 --cpu-base 8 --sched-fifo
sudo ./nv_bar0_recorder --mode aggregate --agg-window-ms 100   # small output
```

- One sampling thread per GPU; `--cpu-base N` pins gpu k to CPU N+k
  (needed to sustain 100 kHz; each MMIO read costs ~1 us).
- `--mode raw` (default) streams 80-byte records per sample (~8 MB/s per
  GPU at 100 kHz). `--mode aggregate` keeps only per-window counters in
  JSONL (~1000x smaller).
- Output dir: `manifest.json` (initial `LEAF_EN_SET` + MSI-X snapshot),
  `raw_trace/gpu_<bdf>.bin` or `occupancy/gpu_<bdf>.jsonl`,
  `run_report.json` (achieved rate, jitter, enable-mask drift check).
- `--enable-all-leaves` (only write mode in the tool): sets
  `LEAF_EN_SET[i]=0xffffffff` at start and restores the snapshotted mask
  on exit, to observe engines whose pending latch is gated on the enable
  bit. Requires write access to BAR0; skip it unless you know you need
  it.

## Analyzer

```bash
python3 tools/nv_bar0_analyzer.py run1        # writes summary.json + summary.md
```

Reports per GPU:

- **Latched-bit verdict** first: any bit pending for ~100% of the run
  with zero edges is latched; if enabled, it is a lost interrupt.
- Per-leaf and per-bit occupancy (fraction of samples pending), masked
  by `LEAF_EN_SET` (disabled bits cannot fire MSI-X).
- Rising-edge counts/rates per bit -- run different single-engine
  workloads and diff these to map bits to engines.
- MSI-X PBA occupancy.

Raw `.bin` traces compress ~50x with zstd; the analyzer reads `.bin.zst`
directly. Timestamps are `CLOCK_MONOTONIC`, so rising edges can be
joined against kernel-side ISR probe timestamps (e.g. a bpftrace probe
on the driver ISR) for interrupt-latency studies.
