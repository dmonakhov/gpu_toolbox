# nv_intr_fleet_scan.sh

Cluster-wide sweep for GPUs with latched interrupt vectors. Runs through
`kubectl exec` into an existing per-node privileged pod (default label
`k8s-app=dcgm-exporter`; any pod with root and host `/sys` works). The
payload is pure bash + `dd` + `od` -- nothing to ship to the nodes.

A full scan is cheap: 2 x 16 register reads per GPU, executed on N nodes
in parallel. Scanning thousands of GPUs takes about a minute.

## Usage

```bash
# 1. Verify the payload works on one node
./nv_intr_fleet_scan.sh selftest -H <node-name>
./nv_intr_fleet_scan.sh selftest -H <node-name> --debug   # if it doesn't

# 2. Fleet-wide
./nv_intr_fleet_scan.sh scan -j 32
./nv_intr_fleet_scan.sh scan -l app=my-privileged-ds -n monitoring

# 3. Re-summarize old results
./nv_intr_fleet_scan.sh summary LOG/nv-intr-scan-<ts>
```

## Detection

Per GPU the payload reads `INTR_LEAF[0..15]` twice, 1 second apart
(`STUCK_INTERVAL_SEC` env in the payload). A bit set in both reads is
latched -- healthy pending bits live for microseconds:

```
0000:53:00.0 0x18804580 0:00010001 STUCK
0000:64:00.0 0x18804580 - OK
```

## Outputs

```
LOG/nv-intr-scan-<ts>/
  scan.csv          node,bdf,pmc_boot_0,stuck_bits,verdict
  stuck_hosts.txt   affected node names, one per line
  raw/<node>.txt    raw payload output per node
  raw/<node>.err    stderr / exec failures per node
  metadata.json     scan parameters
```

The summary fails loudly (exit 2) when pods were listed but zero GPU
rows came back -- that is a broken scan, not a healthy fleet.

Follow up on any STUCK host with `nv_intr_dump` for enable masks, MSI-X
state, and leaf roles.
