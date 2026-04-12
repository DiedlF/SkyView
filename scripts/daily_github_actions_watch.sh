#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/root/.openclaw/workspace/skyview"
cd "$REPO_DIR"

OUT=$(python3 scripts/check_github_actions_failures.py --mark-seen 2>&1 || true)

if [[ -z "$OUT" || "$OUT" == "NO_FAILURES" ]]; then
  exit 0
fi

openclaw system event --text "$OUT" --mode now
