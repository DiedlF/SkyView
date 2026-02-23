#!/bin/bash
# Skyview data ingestion cron script
# Run every 10 minutes to check for new data
# ICON-D2 and ICON-EU are checked independently — they publish at different times
# D2: runs every 3h, ~1h publication delay
# EU: runs every 6h, ~2h publication delay
#
# Config-driven: reads ingest_config.yaml for variable groups.
# D2: full grid (746×1215), EU: cropped to D2 bounds.
# Retention: latest run only (configurable in ingest_config.yaml).

cd "$(dirname "$0")"

LOCKFILE="/tmp/skyview-ingest.lock"
LOCKFILE_AGE_MAX=3600  # 60 minutes (increased for larger download set)
INGEST_PROFILE="${SKYVIEW_INGEST_PROFILE:-skyview_core}"
export SKYVIEW_INGEST_RATE_LIMIT="${SKYVIEW_INGEST_RATE_LIMIT:-10M}"

# Check for stale lock (process died without cleanup)
if [ -f "$LOCKFILE" ]; then
    LOCK_AGE=$(($(date +%s) - $(stat -c %Y "$LOCKFILE" 2>/dev/null || echo 0)))
    if [ $LOCK_AGE -gt $LOCKFILE_AGE_MAX ]; then
        echo "WARNING: Removing stale lock (age: ${LOCK_AGE}s)" >&2
        rm -f "$LOCKFILE"
    else
        # Lock is fresh, another instance is running (silent exit)
        exit 0
    fi
fi

# Acquire lock
touch "$LOCKFILE"
trap "rm -f $LOCKFILE" EXIT

# Prefer project venv python when available
PYTHON_BIN="${PYTHON_BIN:-../venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

# Check and ingest ICON-D2 (2.2km, 48h, full grid)
"$PYTHON_BIN" ingest.py --model icon-d2 --profile "$INGEST_PROFILE" --check-only 2>/dev/null
if [ $? -eq 0 ]; then
    "$PYTHON_BIN" ingest.py --model icon-d2 --profile "$INGEST_PROFILE" --steps all
fi

# ICON-EU ingest disabled for now (meteogram focus on ICON-D2)
# "$PYTHON_BIN" ingest.py --model icon-eu --profile "$INGEST_PROFILE" --check-only 2>/dev/null
# if [ $? -eq 0 ]; then
#     "$PYTHON_BIN" ingest.py --model icon-eu --profile "$INGEST_PROFILE" --steps all
# fi

# Lock automatically removed by trap on exit
