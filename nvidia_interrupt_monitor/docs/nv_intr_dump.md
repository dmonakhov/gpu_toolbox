# nv_intr_dump

Single-host diagnostic: dumps the interrupt aggregator (`INTR_LEAF`,
`LEAF_EN_SET`, `TOP`) and the PCIe MSI-X table/PBA of every NVIDIA GPU,
then gives a per-GPU verdict on latched (stuck-pending) bits.

## Build / run

```bash
gcc -O2 -Wall -o nv_intr_dump tools/nv_intr_dump.c
sudo ./nv_intr_dump                    # all GPUs
sudo ./nv_intr_dump 0000:53:00.0       # one GPU
sudo ./nv_intr_dump --quiet            # verdict lines only
sudo ./nv_intr_dump --repeat 5 --interval-ms 2000   # stricter persistence
```

## How the verdict works

Each GPU is read `--repeat` times (default 3), `--interval-ms` apart
(default 1000). A bit pending in **every** read is persistent; if its
`LEAF_EN_SET` enable bit is also set the GPU is declared `STUCK`:

```
VERDICT 0000:53:00.0 STUCK leaf=0 bits=0x00010001 persisted=3/3 span_ms=2000 role="ENGINE_NOTIFICATION (NONSTALL)"
VERDICT 0000:64:00.0 OK
```

Persistent-but-disabled bits are printed as `INFO` lines: they cannot
fire an MSI-X, but are still unusual.

Exit codes: `0` all healthy, `1` stuck bits found, `2` error (also used
for unrecognized GPU architectures; `--force` scans them anyway).

## Reading the dump

- `TOP` nonzero at idle means some subtree has pending work the ISR
  never drained.
- On a lost-interrupt victim the MSI-X table and PBA are typically
  **clean** -- the loss is upstream of PCIe, inside the GPU's send
  latch. A masked vector or set PBA bit points to a different problem
  (host-side masking / ISR starvation).
- Leaf roles (Hopper): leaves 0-1 engine completion notifications
  (NONSTALL) -- the leaf where lost completions hurt workloads; 2-5 UVM;
  6-7 engine stall; 8-11 runlist.
