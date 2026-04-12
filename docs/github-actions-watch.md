# GitHub Actions failure watcher

Repo-local helper to detect new failed GitHub Actions runs and suggest likely fixes.

## Requirements

- `gh` CLI installed and authenticated
- access to `DiedlF/SkyView`

## Usage

From the repo root:

```bash
python3 scripts/check_github_actions_failures.py
```

Show all recent failures instead of only new ones:

```bash
python3 scripts/check_github_actions_failures.py --all
```

Persist reported failures as seen:

```bash
python3 scripts/check_github_actions_failures.py --mark-seen
```

JSON output:

```bash
python3 scripts/check_github_actions_failures.py --json
```

## State file

Seen failed run IDs are stored in:

```text
.state/github-actions-watch.json
```

## Typical cron/heartbeat-friendly invocation

```bash
cd /path/to/SkyView && python3 scripts/check_github_actions_failures.py --mark-seen
```
