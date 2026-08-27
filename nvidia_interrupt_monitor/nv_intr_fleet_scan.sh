#!/bin/bash
# nv_intr_fleet_scan.sh -- detect NVIDIA GPUs with latched (stuck-pending)
# interrupt vectors across a Kubernetes fleet, via a handful of BAR0
# register reads per GPU.
#
# Failure class (see README.md): an MSI-X message lost in transit (for
# example across a hypervisor live-update/live-migration pause or a host
# kexec) leaves the GPU's internal INTR_LEAF pending bit latched forever.
# The driver never runs the ISR, completions on that vector go silent, and
# every standard health check stays green.
#
# Detection is passive and cheap: read INTR_LEAF[0..15] twice, a second
# apart. On a healthy GPU pending bits are drained by the ISR within
# microseconds, so a bit set in BOTH reads is latched.
#
#   stuck[i] = read1(LEAF[i]) AND read2(LEAF[i]);  any stuck != 0 => STUCK
#
# Register offsets (Turing and later, validated on Hopper H100/H200,
# source: NVIDIA open-gpu-kernel-modules dev_vm.h):
#   LEAF[i] pending = BAR0 + 0xb81000 + i*4
#   PMC_BOOT_0      = BAR0 + 0x0  (sanity check; expect non-zero chip ID)
#
# The scan runs through `kubectl exec` into an existing per-node privileged
# pod (default: the DCGM exporter daemonset; override with -l). The pod
# payload is pure bash + dd + od -- no python, no binaries to ship.
#
# Modes:
#   scan [-j N] [-l LABEL] [-n NS] [-o DIR]   scan fleet, report stuck hosts
#   summary DIR                               re-summarize a previous scan
#   selftest [-H HOST | --pod NS/NAME] [-d]   run payload on one pod / locally
#   help
#
# Follow up on any STUCK host with tools/nv_intr_dump for the full picture
# (enable masks, MSI-X table, per-leaf roles).

set -euo pipefail

LABEL="${LABEL:-k8s-app=dcgm-exporter}"
PARALLEL_JOBS="${PARALLEL_JOBS:-16}"

if [ -t 1 ]; then
    RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; NC='\033[0m'
else
    RED=''; YELLOW=''; GREEN=''; NC=''
fi

usage() {
    cat <<EOF
Usage: $0 MODE [OPTIONS]

Modes:
  scan               Scan the fleet for GPUs with latched interrupt vectors
  summary DIR        Re-summarize a previous scan
  selftest [-H HOST | --pod NS/NAME] [-d|--debug]
                     Run the inline pod payload locally (or via 'kubectl exec'
                     on a specific pod) and print its raw output. With
                     --debug, runs an extended diagnostic payload to figure
                     out why BAR0 reads might fail on a pod.
  help               Show this message

scan options:
  -j, --jobs N         Parallel kubectl exec workers (default: $PARALLEL_JOBS)
  -l, --label SEL      Pod label selector (default: $LABEL)
  -n, --namespace NS   Namespace (default: all)
  -o, --output DIR     Output directory (default: LOG/nv-intr-scan-<ts>)
  -H, --host HOSTNAME  Restrict to one node (substring match on nodeName)
  -q, --quiet          Suppress per-host OK lines (print only stuck hosts)

Detection rule per GPU:
  for i in 0..15: stuck[i] = read32(BAR0+0xb81000+i*4) in two reads 1s apart
  any stuck[i] != 0  =>  STUCK (latched pending vector, lost interrupt)

Examples:
  # Test the payload on one known node first
  $0 selftest -H my-gpu-node-name
  # Fleet-wide
  $0 scan -j 32
  $0 scan -q -o /tmp/intr-scan-today
  $0 summary /tmp/intr-scan-today
EOF
    exit 0
}

# ----------------------------------------------------------------------------
# Inline pod payload: pure-bash BAR0 reader using dd + od.
#
# Why not python: minimal exporter pods often do not have python3.
#
# Why dd + od: dd with bs=4 issues a 32-bit MMIO read via the kernel's
# pci resource read path (sized by bs). od -tx4 prints the 4 bytes as one
# 32-bit value in native byte order, matching little-endian MMIO reads
# on x86/arm64.
#
# Per-line output (one line per GPU on the pod's host):
#   <bdf> <pmc_boot_0_hex|-> <stuck_summary|-> <verdict>
# stuck_summary: comma-separated leaf:hex pairs, e.g. "0:00010001"
# Verdicts: STUCK, OK, ERR_NO_RESOURCE, ERR_PMC_READ, ERR_PMC_BAD,
#           ERR_LEAF_READ
# ----------------------------------------------------------------------------
read -r -d '' POD_PAYLOAD <<'PAYLOAD' || true
set -u

LEAF_BASE=12128256          # 0xb81000
NUM_LEAVES=16
STUCK_INTERVAL_SEC=${STUCK_INTERVAL_SEC:-1}

# Locate GPUs visible to this pod
gpus=()
for d in /proc/driver/nvidia/gpus/*; do
    b=$(basename "$d")
    case "$b" in 0000:*) gpus+=("$b");; esac
done

if [ ${#gpus[@]} -eq 0 ]; then
    echo "ERROR no_gpus_found_in_/proc/driver/nvidia/gpus" >&2
    exit 1
fi

# Read a 32-bit value at byte-offset $2 from MMIO-backed file $1. Returns
# 8-char lowercase hex (no leading 0x) on stdout, or empty on failure.
read_u32_at() {
    local f="$1" off="$2"
    dd if="$f" bs=4 count=1 skip="$off" iflag=skip_bytes status=none 2>/dev/null \
        | od -An -tx4 -N4 | tr -d ' \n'
}

# Read LEAF[0..15] into the global array leaves[]. Returns 1 on any failure.
read_leaves() {
    local resource="$1" i v
    leaves=()
    for ((i = 0; i < NUM_LEAVES; i++)); do
        v=$(read_u32_at "$resource" $((LEAF_BASE + i * 4)))
        [ -n "$v" ] || return 1
        leaves+=("$v")
    done
    return 0
}

for bdf in "${gpus[@]}"; do
    resource="/sys/bus/pci/devices/${bdf}/resource0"
    if [ ! -r "$resource" ]; then
        echo "${bdf} - - ERR_NO_RESOURCE"
        continue
    fi
    pmc=$(read_u32_at "$resource" 0)
    if [ -z "$pmc" ]; then
        echo "${bdf} - - ERR_PMC_READ"
        continue
    fi
    if [ "$pmc" = "00000000" ] || [ "$pmc" = "ffffffff" ]; then
        echo "${bdf} 0x${pmc} - ERR_PMC_BAD"
        continue
    fi
    if ! read_leaves "$resource"; then
        echo "${bdf} 0x${pmc} - ERR_LEAF_READ"
        continue
    fi
    pass1=("${leaves[@]}")
    sleep "$STUCK_INTERVAL_SEC"
    if ! read_leaves "$resource"; then
        echo "${bdf} 0x${pmc} - ERR_LEAF_READ"
        continue
    fi
    stuck_summary=""
    for ((i = 0; i < NUM_LEAVES; i++)); do
        stuck=$(( 0x${pass1[i]} & 0x${leaves[i]} ))
        if [ "$stuck" -ne 0 ]; then
            stuck_summary="${stuck_summary:+${stuck_summary},}$(printf '%d:%08x' "$i" "$stuck")"
        fi
    done
    if [ -n "$stuck_summary" ]; then
        echo "${bdf} 0x${pmc} ${stuck_summary} STUCK"
    else
        echo "${bdf} 0x${pmc} - OK"
    fi
done
PAYLOAD


# ----------------------------------------------------------------------------
# Inline DEBUG payload: emits everything needed to diagnose why BAR0 reads
# fail on a given pod. Use with: selftest --debug -H <host>
# ----------------------------------------------------------------------------
read -r -d '' DEBUG_PAYLOAD <<'DEBUG_PL' || true
echo "=== id / whoami ==="
id 2>&1
echo ""
echo "=== uname -a ==="
uname -a 2>&1
echo ""
echo "=== /proc/self/status caps ==="
grep -E '^(Uid|CapPrm|CapEff|CapBnd):' /proc/self/status 2>&1
echo ""
echo "=== which dd / dd --version ==="
which dd 2>&1
dd --version 2>&1 | head -3
echo ""
echo "=== /proc/driver/nvidia/gpus listing ==="
ls /proc/driver/nvidia/gpus 2>&1
echo ""
first_gpu=$(ls /proc/driver/nvidia/gpus 2>/dev/null | head -1)
echo "=== first GPU: ${first_gpu:-none} ==="
if [ -n "$first_gpu" ]; then
    res="/sys/bus/pci/devices/$first_gpu/resource0"
    echo "=== resource0 stat / readable test ==="
    stat "$res" 2>&1
    test -r "$res" && echo "test -r: yes" || echo "test -r: no"
    echo ""
    echo "=== dd attempt: bs=4 count=1 skip=0 (STDERR visible) ==="
    dd if="$res" bs=4 count=1 skip=0 iflag=skip_bytes 2>&1 | od -An -tx4 -N4
    echo "exit=$?"
fi
echo ""
echo "=== /proc/iomem (nvidia entries) ==="
grep -i nvidia /proc/iomem 2>&1 | head -5
echo ""
echo "=== kernel lockdown ==="
cat /sys/kernel/security/lockdown 2>/dev/null || echo "(no lockdown)"
echo ""
echo "=== /proc/cmdline (look for iomem= setting) ==="
cat /proc/cmdline 2>&1
echo ""
echo "=== Done. ==="
DEBUG_PL


# ----------------------------------------------------------------------------
# SCAN
# ----------------------------------------------------------------------------
do_scan() {
    local NS_FLAG="-A"
    local OUTPUT_DIR=""
    local QUIET=0
    local HOST_FILTER=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--namespace) NS_FLAG="-n $2"; shift 2;;
            -j|--jobs) PARALLEL_JOBS="$2"; shift 2;;
            -o|--output) OUTPUT_DIR="$2"; shift 2;;
            -l|--label) LABEL="$2"; shift 2;;
            -H|--host) HOST_FILTER="$2"; shift 2;;
            -q|--quiet) QUIET=1; shift;;
            -h|--help) usage;;
            *) echo "Unknown option: $1" >&2; usage;;
        esac
    done

    if [ -z "$OUTPUT_DIR" ]; then
        mkdir -p LOG
        OUTPUT_DIR="LOG/nv-intr-scan-$(date +%Y-%m-%d.%s)"
    fi
    mkdir -p "$OUTPUT_DIR/raw"

    {
        echo "Fleet interrupt scan"
        echo "===================="
        echo "Label:    $LABEL"
        echo "Output:   $OUTPUT_DIR"
        echo "Parallel: $PARALLEL_JOBS"
        echo ""
    } >&2

    cat > "$OUTPUT_DIR/metadata.json" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "epoch": $(date +%s),
  "label": "$LABEL",
  "namespace": "${NS_FLAG#-n }",
  "kubectl_context": "$(kubectl config current-context 2>/dev/null || echo unknown)",
  "detection_rule": "leaf[i] bit pending in two reads 1s apart => latched"
}
EOF

    if [ -n "$HOST_FILTER" ]; then
        echo "Host filter: $HOST_FILTER" >&2
    fi

    kubectl get pods $NS_FLAG -l "$LABEL" \
        -o jsonpath='{range .items[*]}{.metadata.namespace}/{.metadata.name}/{.spec.nodeName}{"\n"}{end}' \
        > "$OUTPUT_DIR/pods.all.txt"

    if [ -n "$HOST_FILTER" ]; then
        awk -F/ -v h="$HOST_FILTER" '$3 ~ h' "$OUTPUT_DIR/pods.all.txt" \
            > "$OUTPUT_DIR/pods.txt"
    else
        cp "$OUTPUT_DIR/pods.all.txt" "$OUTPUT_DIR/pods.txt"
    fi

    local POD_COUNT
    POD_COUNT=$(wc -l < "$OUTPUT_DIR/pods.txt")
    echo "Found $POD_COUNT pods" >&2
    if [ "$POD_COUNT" -eq 0 ]; then
        if [ -n "$HOST_FILTER" ]; then
            echo "No pods matched host filter '$HOST_FILTER' under label '$LABEL'" >&2
            echo "All pods (first 5):" >&2
            head -5 "$OUTPUT_DIR/pods.all.txt" | sed 's/^/  /' >&2
        else
            echo "No pods matched label '$LABEL'" >&2
        fi
        exit 1
    fi

    echo "node,bdf,pmc_boot_0,stuck_bits,verdict" > "$OUTPUT_DIR/scan.csv"

    export OUTPUT_DIR POD_PAYLOAD

    cat "$OUTPUT_DIR/pods.txt" \
        | awk -F/ 'NF==3 {print $1, $2, $3}' \
        | xargs -P "$PARALLEL_JOBS" -n 3 -- bash -c '
            set -u
            ns="$1"; pod="$2"; node="$3"
            outdir="$OUTPUT_DIR"
            raw="$outdir/raw/$node.txt"
            err="$outdir/raw/$node.err"
            if ! kubectl exec -n "$ns" "$pod" -- bash -c "$POD_PAYLOAD" \
                    >"$raw" 2>"$err"; then
                echo "EXEC_FAILED $node $pod" >>"$err"
            fi
            # Aggregate into CSV. Payload rows look like:
            #   0000:53:00.0 0x18804580 0:00010001 STUCK
            grep "^0000:" "$raw" 2>/dev/null \
                | awk -v node="$node" "{print node \",\" \$1 \",\" \$2 \",\" \$3 \",\" \$4}" \
                > "$outdir/scan.csv.tmp.$node"
        ' _

    find "$OUTPUT_DIR" -maxdepth 1 -name 'scan.csv.tmp.*' -print0 \
        | sort -z \
        | xargs -0 cat >> "$OUTPUT_DIR/scan.csv" 2>/dev/null || true
    rm -f "$OUTPUT_DIR"/scan.csv.tmp.* 2>/dev/null || true

    summary "$OUTPUT_DIR" "$QUIET"
}


# ----------------------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------------------
summary() {
    local DIR="$1"
    local QUIET="${2:-0}"

    if [ ! -f "$DIR/scan.csv" ]; then
        echo "ERROR: $DIR/scan.csv not found" >&2
        exit 1
    fi

    local TOTAL_GPU TOTAL_HOST STUCK_GPU STUCK_HOST OK_GPU ERR_GPU PODS_LISTED
    TOTAL_GPU=$(awk -F, 'NR>1' "$DIR/scan.csv" | wc -l)
    TOTAL_HOST=$(awk -F, 'NR>1 {print $1}' "$DIR/scan.csv" | sort -u | wc -l)
    STUCK_GPU=$(awk -F, 'NR>1 && $5=="STUCK"' "$DIR/scan.csv" | wc -l)
    STUCK_HOST=$(awk -F, 'NR>1 && $5=="STUCK" {print $1}' "$DIR/scan.csv" | sort -u | wc -l)
    OK_GPU=$(awk -F, 'NR>1 && $5=="OK"' "$DIR/scan.csv" | wc -l)
    ERR_GPU=$(awk -F, 'NR>1 && $5!="STUCK" && $5!="OK"' "$DIR/scan.csv" | wc -l)
    PODS_LISTED=0
    [ -f "$DIR/pods.txt" ] && PODS_LISTED=$(wc -l < "$DIR/pods.txt")

    local EXEC_FAILED_HOSTS PERMISSION_DENIED_HOSTS
    EXEC_FAILED_HOSTS=0
    PERMISSION_DENIED_HOSTS=0
    if [ -d "$DIR/raw" ] && ls "$DIR"/raw/*.err >/dev/null 2>&1; then
        EXEC_FAILED_HOSTS=$({ grep -l "EXEC_FAILED" "$DIR"/raw/*.err || true; } 2>/dev/null | wc -l)
        PERMISSION_DENIED_HOSTS=$({ grep -lE 'Permission denied|EACCES' \
                                          "$DIR"/raw/*.err || true; } 2>/dev/null | wc -l)
    fi

    {
        echo ""
        echo "=== Fleet interrupt scan summary ==="
        echo "  Pods listed:     $PODS_LISTED"
        echo "  Hosts scanned:   $TOTAL_HOST"
        echo "  GPUs scanned:    $TOTAL_GPU"
        printf "  ${GREEN}HEALTHY GPUs:    %d${NC}\n" "$OK_GPU"
        printf "  ${RED}STUCK GPUs:      %d  (on %d hosts)${NC}\n" "$STUCK_GPU" "$STUCK_HOST"
        if [ "$ERR_GPU" -gt 0 ]; then
            printf "  ${YELLOW}ERROR rows:      %d${NC}\n" "$ERR_GPU"
        fi
        if [ "$EXEC_FAILED_HOSTS" -gt 0 ]; then
            printf "  ${YELLOW}EXEC_FAILED hosts: %d${NC}\n" "$EXEC_FAILED_HOSTS"
            [ "$PERMISSION_DENIED_HOSTS" -gt 0 ] && \
                printf "  ${YELLOW}  - permission denied: %d${NC}\n" "$PERMISSION_DENIED_HOSTS"
        fi
        echo ""
    } >&2

    # Loud failure: listed pods but 0 GPU rows means the scan is broken,
    # not "no victims".
    if [ "$PODS_LISTED" -gt 0 ] && [ "$TOTAL_GPU" -eq 0 ]; then
        echo "" >&2
        printf "${RED}*** SCAN FAILED ***${NC}\n" >&2
        echo "Listed $PODS_LISTED pods but parsed 0 GPU rows. Inspect: $DIR/raw/" >&2
        if [ "$EXEC_FAILED_HOSTS" -gt 0 ]; then
            echo "Sample error from $DIR/raw/:" >&2
            local first_err
            first_err=$(ls "$DIR"/raw/*.err 2>/dev/null | head -1)
            [ -n "$first_err" ] && head -5 "$first_err" | sed 's/^/  /' >&2
        fi
        echo "" >&2
        echo "CSV: $DIR/scan.csv (empty data rows)" >&2
        return 2
    fi

    if [ "$TOTAL_GPU" -lt $((PODS_LISTED / 2)) ] && [ "$PODS_LISTED" -gt 8 ]; then
        echo "" >&2
        printf "${YELLOW}*** WARNING: fewer than half of listed pods returned data ***${NC}\n" >&2
        echo "Listed=$PODS_LISTED  scanned=$TOTAL_GPU  exec_failed=$EXEC_FAILED_HOSTS" >&2
    fi

    if [ "$STUCK_HOST" -gt 0 ]; then
        echo "=== Stuck hosts (per-GPU detail) ==="
        awk -F, '
            NR==1 { next }
            $5=="STUCK" { bad[$1]=1 }
            { rows[NR]=$0 }
            END {
                for (i=2; i<=NR; i++) {
                    n=split(rows[i], f, ",")
                    if (n>=1 && (f[1] in bad)) {
                        flag = (f[5]=="STUCK") ? "*** STUCK ***" : ""
                        printf "%-50s %-15s %-12s %-24s %-8s %s\n", f[1], f[2], f[3], f[4], f[5], flag
                    }
                }
            }
        ' "$DIR/scan.csv"
        echo ""
        echo "=== Affected host list (one per line) ==="
        awk -F, 'NR>1 && $5=="STUCK" {print $1}' "$DIR/scan.csv" | sort -u | tee "$DIR/stuck_hosts.txt"
        echo ""
        echo "Next step: run tools/nv_intr_dump on the affected hosts for the"
        echo "full picture (enable masks, MSI-X table, per-leaf roles)."
    elif [ "$TOTAL_GPU" -gt 0 ]; then
        echo "No GPUs with latched interrupt vectors detected."
    fi

    if [ "$ERR_GPU" -gt 0 ]; then
        echo ""
        echo "=== Error rows (first 20) ==="
        awk -F, 'NR>1 && $5!="STUCK" && $5!="OK"' "$DIR/scan.csv" | head -20
    fi

    echo ""
    echo "Full CSV: $DIR/scan.csv"
    if [ "$STUCK_HOST" -gt 0 ]; then
        echo "Affected hosts: $DIR/stuck_hosts.txt"
    fi
    return 0
}


# ----------------------------------------------------------------------------
# SELFTEST
# ----------------------------------------------------------------------------
do_selftest() {
    local POD=""
    local HOST=""
    local DEBUG=0
    while [[ $# -gt 0 ]]; do
        case $1 in
            --pod) POD="$2"; shift 2;;
            -H|--host) HOST="$2"; shift 2;;
            -l|--label) LABEL="$2"; shift 2;;
            -d|--debug) DEBUG=1; shift;;
            -h|--help) usage;;
            *) echo "Unknown option: $1" >&2; exit 1;;
        esac
    done

    # If --host given but no --pod, resolve host -> pod via label.
    if [ -z "$POD" ] && [ -n "$HOST" ]; then
        local pods_list match=""
        pods_list=$(kubectl get pods -A -l "$LABEL" \
            -o jsonpath='{range .items[*]}{.metadata.namespace}/{.metadata.name}/{.spec.nodeName}{"\n"}{end}')
        while IFS= read -r line; do
            case "$line" in
                */*/*"$HOST"*)
                    match="${line%/*}"
                    break
                    ;;
            esac
        done <<< "$pods_list"
        if [ -z "$match" ]; then
            echo "ERROR no pod found for host '$HOST' (label='$LABEL')" >&2
            echo "Sample pods under that label:" >&2
            echo "$pods_list" | head -5 | sed 's/^/  /' >&2
            exit 1
        fi
        POD="$match"
        echo "Resolved host=$HOST -> pod=$POD" >&2
    fi

    if [ -n "$POD" ]; then
        local ns="${POD%%/*}" name="${POD##*/}"
        if [ "$DEBUG" = "1" ]; then
            echo "Running DEBUG payload via kubectl exec on $POD" >&2
            kubectl exec -n "$ns" "$name" -- bash -c "$DEBUG_PAYLOAD"
            local rc=$?
            echo "exit=$rc" >&2
            return $rc
        fi
        echo "Running payload via kubectl exec on $POD" >&2
        kubectl exec -n "$ns" "$name" -- bash -c "$POD_PAYLOAD"
        local rc=$?
        echo "exit=$rc" >&2
        return $rc
    fi

    echo "Running payload locally (needs root and an NVIDIA GPU; on other" >&2
    echo "machines it reports ERR_NO_RESOURCE / no_gpus_found)." >&2
    echo "" >&2
    bash -c "$POD_PAYLOAD"
}

MODE="${1:-help}"
shift || true

case "$MODE" in
    scan)        do_scan "$@" ;;
    summary)     [ -n "${1:-}" ] || { echo "summary needs DIR" >&2; exit 1; }
                 summary "$1" 0 ;;
    selftest)    do_selftest "$@" ;;
    help|-h|--help|"") usage ;;
    *) echo "Unknown mode: $MODE" >&2; usage ;;
esac
