#!/bin/bash
# Skyview observation ingest script.
# Fetches live OPERA radar + MSG RSS satellite + MTG-I1 FCI, stores derived
# render frames, and prunes via the observation ring buffer. The MTG source
# de-duplicates before downloading, so frequent ticks only fetch new cycles.
#
# Guards (added after the 2026-07 disk-full incident, where a single ingest run
# hung and held the lock for ~6 days: pruning stopped, the disk filled to 100%,
# and every later tick failed with "No space left on device"):
#   * Disk preflight  — abort loudly when the data filesystem is nearly full
#     instead of letting ingest fill it to 100% and wedge the box.
#   * tmp sweep        — remove ingest tmp leftovers from killed/crashed runs
#     (these accumulated to ~20 GB and filled the disk).
#   * Run timeout      — a hung run can no longer hold the lock indefinitely.

cd "$(dirname "$0")/.."

LOCKFILE="/tmp/skyview-observations.lock"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
    exit 0
fi

DATA_DIR="${SKYVIEW_OBS_DATA_DIR:-data/observations}"
mkdir -p "$DATA_DIR"

# --- tmp sweep --------------------------------------------------------------
# Ingest stages downloads/renders under $DATA_DIR/tmp and moves them into place
# atomically. A run killed mid-write (e.g. by the timeout below or an OOM) leaks
# its staging dir; left unbounded these once grew to ~20 GB. Runs are capped at
# RUN_TIMEOUT, so anything older than an hour is certainly not from a live run.
find "$DATA_DIR/tmp" -mindepth 1 -mmin +60 -exec rm -rf {} + 2>/dev/null || true

# --- Disk-space preflight ---------------------------------------------------
# Refuse to run when free space on the data filesystem is below the floor. A
# full disk makes ingest fail mid-write and, at 100%, wedges pruning and even
# sshd. Abort with a clear, non-zero status so the failure is visible in the
# journal instead of silently corrupting frames.
MIN_FREE_MB="${SKYVIEW_OBS_MIN_FREE_MB:-1024}"
FREE_MB=$(df -Pm "$DATA_DIR" | awk 'NR==2 {print $4}')
MOUNT=$(df -P "$DATA_DIR" | awk 'NR==2 {print $6}')
if [ -n "$FREE_MB" ] && [ "$FREE_MB" -lt "$MIN_FREE_MB" ]; then
    echo "skyview-observations: ABORT - only ${FREE_MB} MB free on ${MOUNT}, need ${MIN_FREE_MB} MB. Skipping ingest to avoid filling the disk." >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-.venv-obs/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="${PYTHON_BIN_FALLBACK:-venv/bin/python}"
fi
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

# --- Run with a hard timeout ------------------------------------------------
# The lock above is non-blocking, so a run that hangs forever silently starves
# every later tick (this is exactly what wedged ingest for ~6 days). Cap the run
# so a stuck fetch/render is TERM'd (then KILL'd) and the lock frees for the
# next tick.
RUN_TIMEOUT="${SKYVIEW_OBS_TIMEOUT:-300}"
PYTHONPATH="${PYTHONPATH:-backend}" timeout --kill-after=30 "$RUN_TIMEOUT" \
    "$PYTHON_BIN" -m backend.observations.ingest_obs \
    --source "${SKYVIEW_OBS_SOURCE:-all}" --log-level "${SKYVIEW_OBS_LOG_LEVEL:-INFO}"
rc=$?
if [ "$rc" -eq 124 ]; then
    echo "skyview-observations: ingest timed out after ${RUN_TIMEOUT}s and was killed." >&2
fi
exit "$rc"
