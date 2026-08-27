#!/usr/bin/env python3
"""
nv_bar0_analyzer.py -- offline analysis of a nv_bar0_recorder raw-mode run.

Inputs:
  <run_dir>/manifest.json
  <run_dir>/raw_trace/gpu_<bdf>.bin[.zst]

Outputs (written next to the input):
  summary.json   per-GPU per-leaf-bit numbers
  summary.md     human-readable table

What it computes (per GPU):

Occupancy:
  - leaf_occupancy[i]: fraction of samples where LEAF[i] != 0
  - bit_occupancy[i][b]: fraction of samples where (LEAF[i] >> b) & 1
  - bit_occupancy_masked[i][b]: same, AND-ed with LEAF_EN_SET[i] bit b.
    Bits not enabled at LEAF_EN cannot reach the MSI-X send latch, so
    only masked bits matter for lost-interrupt detection.

Rising edges (per-bit transitions 0->1):
  - rising_edge_count[i][b], rising_edge_rate_hz[i][b]

Engine mapping hint:
  - bits with non-zero rising_edge_count under workload W are the
    engines W activates. Compare across workloads to map bits to
    engines (gpu-burn -> GR; nvenc -> NVENC; etc.).

Latched-bit detection (the lost-interrupt fingerprint):
  - a bit with occupancy ~100% and zero rising edges over the whole run
    is latched: it was pending before the run started and never moved.
    On a healthy GPU the ISR clears pending bits within us..ms, so
    steady 100% occupancy of an ENABLED bit means the MSI-X message for
    it was lost and the ISR never ran.

Sample-format support: v1 (96 B, pba_bits[3]) and v2 (80 B, pba_bits
scalar). Format detected from manifest.sample_format_version or
sample_record_bytes.

Usage:
  python3 nv_bar0_analyzer.py <run_dir>
  python3 nv_bar0_analyzer.py <run_dir> --top-bits 10
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.stderr.write("ERROR: numpy required. pip install numpy\n")
    sys.exit(2)


SAMPLE_DTYPE_V1 = np.dtype([
    ("ts_ns", "<u8"),
    ("intr_leaf", "<u4", 16),
    ("pba_bits", "<u8", 3),
], align=False)
SAMPLE_DTYPE_V2 = np.dtype([
    ("ts_ns", "<u8"),
    ("intr_leaf", "<u4", 16),
    ("pba_bits", "<u8"),
], align=False)

# Fraction-of-samples threshold above which a zero-edge bit counts as
# latched. Not 1.0 exactly: tolerate a handful of torn/glitched reads.
LATCHED_OCC_THRESHOLD = 0.999


def detect_format(manifest):
    """Return (dtype, version, sample_bytes)."""
    v = manifest.get("sample_format_version")
    if v == 2 or manifest.get("sample_record_bytes") == 80:
        return SAMPLE_DTYPE_V2, 2, 80
    if v == 1 or manifest.get("sample_record_bytes") == 96 or v is None:
        return SAMPLE_DTYPE_V1, 1, 96
    raise SystemExit(f"unknown sample format: v={v} bytes={manifest.get('sample_record_bytes')}")


def _load_trace_bytes(bin_path: Path) -> bytes:
    """Return raw bytes from .bin or .bin.zst."""
    if bin_path.suffix == ".zst":
        try:
            import zstandard  # pip install zstandard
        except ImportError:
            import subprocess
            return subprocess.check_output(
                ["zstd", "-d", "-c", "-q", str(bin_path)]
            )
        with open(bin_path, "rb") as f:
            return zstandard.ZstdDecompressor().stream_reader(f).read()
    return bin_path.read_bytes()


def analyze_gpu(bin_path: Path, leaf_en_set, fmt_dtype, sample_bytes):
    """Per-GPU stats. leaf_en_set is list of 16 uint32 (from manifest)."""
    raw = _load_trace_bytes(bin_path)
    file_bytes = len(raw)
    n_drop = file_bytes % sample_bytes
    if n_drop:
        sys.stderr.write(
            f"WARN {bin_path.name}: dropping trailing {n_drop} bytes "
            f"(file size not a multiple of {sample_bytes})\n"
        )
        raw = raw[: file_bytes - n_drop]
    data = np.frombuffer(raw, dtype=fmt_dtype)
    n = len(data)
    if n == 0:
        return None

    ts = data["ts_ns"]
    duration_s = float(ts[-1] - ts[0]) / 1e9 if n > 1 else 0.0
    observed_hz = (n - 1) / duration_s if duration_s > 0 else 0.0

    leaves = data["intr_leaf"]  # (N, 16) uint32

    leaf_any_count = (leaves != 0).sum(axis=0).astype(np.int64)  # (16,)

    bit_occ = np.zeros((16, 32), dtype=np.int64)
    rising = np.zeros((16, 32), dtype=np.int64)
    falling = np.zeros((16, 32), dtype=np.int64)
    prev = leaves[:-1]
    curr = leaves[1:]
    rising_mat = curr & ~prev
    falling_mat = prev & ~curr
    for b in range(32):
        bit_occ[:, b] = ((leaves >> b) & 1).sum(axis=0)
        rising[:, b] = ((rising_mat >> b) & 1).sum(axis=0)
        falling[:, b] = ((falling_mat >> b) & 1).sum(axis=0)

    # Mask occupancy / rising with the enable mask: only bits that
    # could actually have fired through to MSI-X.
    en_set_arr = np.asarray(leaf_en_set, dtype=np.uint32)  # (16,)
    en_mask = np.zeros((16, 32), dtype=bool)
    for b in range(32):
        en_mask[:, b] = ((en_set_arr >> b) & 1).astype(bool)
    bit_occ_masked = np.where(en_mask, bit_occ, 0)
    rising_masked = np.where(en_mask, rising, 0)

    # Latched bits: near-100% occupancy with zero edges the whole run.
    latched = []
    for i in range(16):
        for b in range(32):
            if rising[i][b] == 0 and falling[i][b] == 0 \
                    and bit_occ[i][b] >= n * LATCHED_OCC_THRESHOLD:
                latched.append({
                    "leaf": i, "bit": b,
                    "enabled": bool(en_mask[i][b]),
                    "occupancy": float(bit_occ[i][b] / n),
                })

    # PBA occupancy (first 9 vectors used on Hopper; the rest are
    # reserved-1s in v1).
    if "pba_bits" in fmt_dtype.fields and fmt_dtype["pba_bits"].shape == ():
        pba = data["pba_bits"]  # (N,) uint64
    else:
        pba = data["pba_bits"][:, 0]  # (N,) uint64 -- vec 0..63 from v1
    pba_occ = np.zeros(9, dtype=np.int64)
    for v in range(9):
        pba_occ[v] = ((pba >> v) & 1).sum()

    return {
        "samples": n,
        "duration_s": duration_s,
        "observed_hz": observed_hz,
        "leaf_any_occupancy": (leaf_any_count / n).tolist(),
        "leaf_any_count": leaf_any_count.tolist(),
        "leaf_en_set": [int(x) for x in en_set_arr],
        "bit_occupancy": bit_occ.tolist(),
        "bit_occupancy_masked": bit_occ_masked.tolist(),
        "rising_edge_count": rising.tolist(),
        "rising_edge_count_masked": rising_masked.tolist(),
        "falling_edge_count": falling.tolist(),
        "pba_occupancy_count": pba_occ.tolist(),
        "pba_occupancy_fraction": (pba_occ / n).tolist(),
        "latched_bits": latched,
    }


def fmt_pct(num, den):
    if den == 0:
        return "  -   "
    p = 100.0 * num / den
    if p >= 0.01:
        return f"{p:6.2f}%"
    if num > 0:
        return f"{p:6.3f}%"
    return "  -   "


def make_summary_md(workload_tag, per_gpu, top_bits=8):
    lines = []
    lines.append(f"# nv_bar0_recorder summary -- workload `{workload_tag}`")
    lines.append("")
    lines.append("## Latched-bit verdict")
    lines.append("")
    any_latched = False
    for bdf, g in per_gpu.items():
        for lb in g["latched_bits"]:
            any_latched = True
            sev = "LOST INTERRUPT" if lb["enabled"] else "latched-but-disabled"
            lines.append(
                f"- **{bdf}** leaf {lb['leaf']} bit {lb['bit']}: pending for "
                f"{100*lb['occupancy']:.2f}% of the run, zero edges -- {sev}"
            )
    if not any_latched:
        lines.append("No latched bits: every observed pending bit was drained")
        lines.append("by the ISR during the run. [OK]")
    lines.append("")
    lines.append("## Per-GPU effective rate")
    lines.append("")
    lines.append("| BDF | samples | duration_s | observed_hz |")
    lines.append("|-----|---------|------------|-------------|")
    for bdf, g in per_gpu.items():
        lines.append(f"| {bdf} | {g['samples']} | {g['duration_s']:.3f} | {g['observed_hz']:.0f} |")
    lines.append("")
    lines.append("## Per-leaf occupancy (any-bit-pending, fraction of samples)")
    lines.append("")
    header = "| BDF | " + " | ".join(f"L{i}" for i in range(12)) + " |"
    sep    = "|-----|" + "|".join(["-----" for _ in range(12)]) + "|"
    lines.append(header); lines.append(sep)
    for bdf, g in per_gpu.items():
        cells = [bdf]
        for i in range(12):
            cells.append(fmt_pct(g["leaf_any_count"][i], g["samples"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## NONSTALL subtree (LEAF[0..1] = ENGINE_NOTIFICATION)")
    lines.append("")
    lines.append("Bits in this subtree, masked by LEAF_EN_SET. This is where")
    lines.append("engine-completion notifications live; a stuck bit here silences")
    lines.append("completions while leaving every other health check green.")
    lines.append("")
    lines.append("| BDF | LEAF[0] occupancy any-bit | LEAF[1] occupancy any-bit | LEAF[0] rising edges/s | LEAF[1] rising edges/s |")
    lines.append("|-----|---------------------------|---------------------------|------------------------|------------------------|")
    for bdf, g in per_gpu.items():
        l0_any = sum(g["bit_occupancy_masked"][0])
        l1_any = sum(g["bit_occupancy_masked"][1])
        l0_edges = sum(g["rising_edge_count_masked"][0])
        l1_edges = sum(g["rising_edge_count_masked"][1])
        lines.append(
            f"| {bdf} | "
            f"{fmt_pct(l0_any, g['samples'])} | "
            f"{fmt_pct(l1_any, g['samples'])} | "
            f"{l0_edges / max(g['duration_s'], 1e-9):.1f} | "
            f"{l1_edges / max(g['duration_s'], 1e-9):.1f} |"
        )
    lines.append("")
    lines.append(f"## Top {top_bits} active bits per GPU (masked, by rising-edge count)")
    lines.append("")
    for bdf, g in per_gpu.items():
        lines.append(f"### {bdf}")
        scored = []
        for i in range(16):
            for b in range(32):
                rc = g["rising_edge_count_masked"][i][b]
                if rc == 0:
                    continue
                scored.append((i, b, rc, g["bit_occupancy_masked"][i][b]))
        scored.sort(key=lambda x: -x[2])
        if not scored:
            lines.append("_no masked rising edges in this run_")
            lines.append("")
            continue
        lines.append("| leaf.bit | rising_edges | rising_hz | occupancy |")
        lines.append("|----------|--------------|-----------|-----------|")
        for i, b, rc, occ in scored[:top_bits]:
            lines.append(
                f"| {i}.{b} | {rc} | "
                f"{rc / max(g['duration_s'], 1e-9):.1f} | "
                f"{fmt_pct(occ, g['samples'])} |"
            )
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `bit_occupancy_masked` excludes pending bits not enabled in")
    lines.append("  `LEAF_EN_SET`; disabled bits cannot fire an MSI-X.")
    lines.append("- A latched ENABLED bit is a lost interrupt. See project README")
    lines.append("  for the failure mechanism and recovery options.")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path,
                   help="path to a nv_bar0_recorder output directory")
    p.add_argument("--top-bits", type=int, default=8,
                   help="how many top bits to show per GPU (default 8)")
    p.add_argument("--quiet", action="store_true",
                   help="don't print summary to stdout, just write files")
    args = p.parse_args()

    run = args.run_dir
    if not run.is_dir():
        sys.exit(f"ERROR: not a directory: {run}")
    manifest_path = run / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"ERROR: missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    fmt_dtype, fmt_v, sample_bytes = detect_format(manifest)
    sys.stderr.write(f"# format v{fmt_v} ({sample_bytes} bytes/record)\n")

    workload_tag = "unknown"
    wt_path = run / "workload_tag.txt"
    if wt_path.exists():
        workload_tag = wt_path.read_text().strip() or "unknown"

    raw_dir = run / "raw_trace"
    if not raw_dir.is_dir():
        sys.exit(f"ERROR: no raw_trace/ in {run} (aggregate-mode not yet supported)")

    per_gpu = {}
    gpu_entries = {g["bdf"]: g for g in manifest["gpus"]}
    # Accept both gpu_<bdf>.bin and gpu_<bdf>.bin.zst; prefer .zst.
    raw_files = {}
    for f in raw_dir.glob("gpu_*.bin"):
        raw_files.setdefault(f.stem, f)
    for f in raw_dir.glob("gpu_*.bin.zst"):
        stem = f.name[: -len(".bin.zst")]
        raw_files[stem] = f
    for stem in sorted(raw_files):
        bin_path = raw_files[stem]
        bdf = stem.removeprefix("gpu_")
        if bdf not in gpu_entries:
            sys.stderr.write(f"WARN {bin_path.name}: not in manifest, skipping\n")
            continue
        leaf_en_set = [int(x, 16) for x in gpu_entries[bdf]["leaf_en_set_initial"]]
        sys.stderr.write(f"# analyzing {bdf}...\n")
        stats = analyze_gpu(bin_path, leaf_en_set, fmt_dtype, sample_bytes)
        if stats is None:
            sys.stderr.write("  empty\n")
            continue
        per_gpu[bdf] = stats

    out_json = run / "summary.json"
    out_md = run / "summary.md"
    out_json.write_text(json.dumps({
        "workload_tag": workload_tag,
        "format_version": fmt_v,
        "per_gpu": per_gpu,
    }, indent=2))
    md = make_summary_md(workload_tag, per_gpu, top_bits=args.top_bits)
    out_md.write_text(md)
    sys.stderr.write(f"# wrote {out_json}\n")
    sys.stderr.write(f"# wrote {out_md}\n")
    if not args.quiet:
        print(md)


if __name__ == "__main__":
    main()
