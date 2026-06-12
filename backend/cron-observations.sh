#!/bin/bash
# Skyview observation ingest script.
# Fetches live OPERA radar + MSG RSS satellite, stores derived render frames,
# and prunes via the observation ring buffer.

cd "$(dirname "$0")/.."

LOCKFILE="/tmp/skyview-observations.lock"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
    exit 0
fi
trap 'rm -f "$LOCKFILE"' EXIT
touch "$LOCKFILE"

PYTHON_BIN="${PYTHON_BIN:-.venv-obs/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="${PYTHON_BIN_FALLBACK:-venv/bin/python}"
fi
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

PYTHONPATH="${PYTHONPATH:-backend}" "$PYTHON_BIN" -m backend.observations.ingest_obs --source both --log-level "${SKYVIEW_OBS_LOG_LEVEL:-INFO}"
