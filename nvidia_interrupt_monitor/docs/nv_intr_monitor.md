# nv_intr_monitor (Go daemon)

Per-node daemon that periodically reads every NVIDIA GPU's `INTR_LEAF`
registers over BAR0 and reports latched (lost) interrupt vectors.
Intended to run as a privileged Kubernetes DaemonSet
(`deploy/daemonset.yaml`) or a systemd service. Zero dependencies,
static binary.

## Flags

```
-interval 30s      time between scans
-confirm 3         consecutive scans a bit must stay pending to count as latched
-listen :9834      metrics/health listen address
-force             scan unrecognized GPU architectures too
-state PATH        write JSON status after every scan (atomic rename)
-oneshot           run -confirm scans -oneshot-interval apart, print JSON, exit
-oneshot-interval 1s
```

`-oneshot` exit codes: `0` healthy, `1` latched bits found, `2` scan
errors -- convenient for node health-check pipelines:

```bash
sudo ./nv_intr_monitor -oneshot | jq .latched
```

## Endpoints

- `/metrics` -- Prometheus text format:
  - `nvidia_intr_gpu_stuck{bdf}` 0/1 -- **the metric to alert on**
  - `nvidia_intr_stuck_bit{bdf,leaf,bit,role}` 1 per latched bit
  - `nvidia_intr_stuck_bit_age_seconds{bdf,leaf,bit}`
  - `nvidia_intr_leaf_pending_bits{bdf,leaf}` popcount from the last
    scan (normal to be nonzero briefly on a busy GPU)
  - `nvidia_intr_monitor_scans_total`, `..._scan_errors_total`,
    `..._gpus`
- `/healthz` -- 200 `ok` / 503 `stuck:N`. Do NOT use as a liveness
  probe: restarting the monitor does not fix the GPU.
- `/status` -- full JSON snapshot (per-GPU registers + latched list).

## Detection logic

A `(gpu, leaf, bit)` pending in `-confirm` consecutive scans while
enabled in `LEAF_EN_SET` is declared latched and logged:

```
[WARN] latched interrupt: bdf=0000:53:00.0 leaf=0 bit=16 role=ENGINE_NOTIFICATION_NONSTALL pending_since=... (3 consecutive scans)
```

If the bit later reads 0 (ISR ran, e.g. after a heal or driver-side
enable-mask toggle) the state clears and an `[OK]` line is logged. With
the defaults, worst-case time-to-detect is ~90 s; false positives would
require a bit to stay pending for 60+ s across three samples, which does
not happen on a serviced GPU.
