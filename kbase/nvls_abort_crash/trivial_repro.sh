#!/bin/bash
# Trivial reproducer for the NVLink-NCCL teardown crash on H200.
#
# Bug shape: a tight dist.all_reduce() loop on the default NCCL
# transport (NVLink P2P) runs cleanly indefinitely. SIGTERM to
# torchrun while collectives are in flight reliably fires an
# NVSwitch SXid 12028 + GPU Xid 137 + Xid 94 burst within ~1 s
# of the SIGTERM.
#
# The crash time tracks the SIGTERM moment, not absolute wall time
# or iteration count. DURATION_SEC=100 and DURATION_SEC=900 both
# fire the burst <2 s after SIGTERM is delivered to the ranks.
# Steady-state behaviour is clean.
#
# Usage:
#   ./trivial_repro.sh                # default 30 s, sends SIGTERM
#   DURATION_SEC=60 ./trivial_repro.sh
#
# Run on a freshly-rebooted p5e (8x H200). Expect the burst in
# dmesg right after the SIGTERM line below.
set -uo pipefail

DURATION_SEC="${DURATION_SEC:-30}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=== trivial_repro: DURATION_SEC=$DURATION_SEC ==="
date '+wall-clock start: %F %T %Z'
echo "uptime at start: $(awk '{print $1}' /proc/uptime) s"
echo

torchrun --nproc_per_node=8 "$HERE/pure_nvlink_allreduce.py" &
TORCHRUN_PID=$!
echo "torchrun pid=$TORCHRUN_PID -- letting it run ${DURATION_SEC}s"

sleep "$DURATION_SEC"

echo
echo "=== sending SIGTERM to torchrun pid=$TORCHRUN_PID ==="
date '+SIGTERM at: %F %T %Z'
kill -TERM "$TORCHRUN_PID" 2>/dev/null || true
wait "$TORCHRUN_PID" 2>/dev/null
echo "torchrun exited rc=$?"

# Let the burst (if any) land in dmesg.
sleep 5

echo
echo "=== dmesg | grep -E 'Xid|SXid|nvAssertFailed' (last 80 lines) ==="
# nvAssertFailedNoLog lines (KGSP service called when no KGSP interrupt
# pending; pRecord->idx == reqIdx @ journal.c:841) fire interleaved
# with the Xid 94 bursts and are part of the same teardown signature.
dmesg -T 2>/dev/null | grep -E 'Xid|SXid|nvAssertFailed' | tail -80 \
    || echo "(no dmesg access -- run as root or with CAP_SYSLOG)"

echo
echo "=== STATUS ==="
# NB: grep -c prints "0" AND exits 1 on zero matches, so `|| echo 0`
# would append a second line and break the -gt comparisons below.
XID_COUNT=$(dmesg 2>/dev/null | grep -cE 'Xid|SXid' || true)
KGSP_COUNT=$(dmesg 2>/dev/null | grep -c 'KGSP service called when no KGSP' || true)
JOURNAL_COUNT=$(dmesg 2>/dev/null | grep -c 'pRecord->idx == reqIdx' || true)
XID_COUNT=${XID_COUNT:-0}; KGSP_COUNT=${KGSP_COUNT:-0}; JOURNAL_COUNT=${JOURNAL_COUNT:-0}
echo "  Xid/SXid lines:           $XID_COUNT"
echo "  KGSP-service assertions:  $KGSP_COUNT"
echo "  journal.c:841 assertions: $JOURNAL_COUNT"
if [ "$XID_COUNT" -gt 0 ]; then
    echo "BUG REPRODUCED -- $XID_COUNT Xid/SXid lines in dmesg"
    echo "(expect ~230 lines per burst on a fresh host; lower counts mean"
    echo " an earlier burst already drained the visible window)"
else
    echo "clean -- no Xid/SXid"
fi
