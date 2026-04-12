#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_REPO = "DiedlF/SkyView"
DEFAULT_LIMIT = 10
STATE_PATH = Path(".state/github-actions-watch.json")


def run_gh(args: list[str]) -> str:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"gh {' '.join(args)} failed")
    return proc.stdout


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"seen_failed_runs": []}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"seen_failed_runs": []}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def fetch_recent_runs(repo: str, limit: int) -> list[dict[str, Any]]:
    out = run_gh([
        "run",
        "list",
        "--repo",
        repo,
        "--limit",
        str(limit),
        "--json",
        "databaseId,displayTitle,headBranch,headSha,status,conclusion,event,workflowName,createdAt,updatedAt,url",
    ])
    data = json.loads(out)
    return data if isinstance(data, list) else []


def fetch_failed_log(repo: str, run_id: int) -> str:
    return run_gh(["run", "view", str(run_id), "--repo", repo, "--log-failed"])


def first_matching_line(log: str, patterns: list[str]) -> str | None:
    for line in log.splitlines():
        for pattern in patterns:
            if re.search(pattern, line, flags=re.IGNORECASE):
                return line.strip()
    return None


def diagnose_failure(log: str) -> dict[str, str | list[str]]:
    failed_test = first_matching_line(
        log,
        [
            r"FAILED\s+tests?/.*",
            r"AssertionError:",
            r"E\s+AssertionError:",
            r"traceback \(most recent call last\)",
        ],
    )
    qa_msg = first_matching_line(log, [r"Precedence regression:", r"check_.*failed", r"FAIL:"])
    missing_import = first_matching_line(log, [r"ModuleNotFoundError:", r"ImportError:"])
    syntax_error = first_matching_line(log, [r"SyntaxError:", r"IndentationError:"])
    readiness = first_matching_line(log, [r"Server did not become ready", r"Connection refused", r"timed out"])

    likely_cause = "Unknown failure"
    suggestions: list[str] = []

    if qa_msg and "Precedence regression" in qa_msg:
        likely_cause = "Regression expectation no longer matches current symbol/classification logic"
        suggestions = [
            "Update the regression/QA expectation if the new behavior is intended",
            "Otherwise revert the precedence/classification logic change",
        ]
    elif failed_test:
        likely_cause = "Pytest regression or unit test failure"
        suggestions = [
            "Inspect the named failing test and align either implementation or test expectation",
            "Reproduce locally with the same pytest selection used in CI",
        ]
    elif missing_import:
        likely_cause = "Missing dependency or broken import path"
        suggestions = [
            "Check requirements and PYTHONPATH handling in CI",
            "Verify renamed/moved modules are still imported correctly",
        ]
    elif syntax_error:
        likely_cause = "Syntax error in committed Python code"
        suggestions = [
            "Run py_compile or pytest locally before push",
            "Inspect the referenced file/line and fix the syntax issue",
        ]
    elif readiness:
        likely_cause = "Server startup or health-check readiness issue"
        suggestions = [
            "Inspect backend startup logs and health endpoint timing",
            "Increase readiness timeout only if startup is genuinely slower but healthy",
        ]

    headline = qa_msg or failed_test or missing_import or syntax_error or readiness or "No concise failure line found"
    return {
        "headline": headline,
        "likely_cause": likely_cause,
        "suggestions": suggestions,
    }


def format_report(run: dict[str, Any], diagnosis: dict[str, Any]) -> str:
    lines = [
        f"SkyView GitHub Actions failure: {run.get('workflowName', 'unknown workflow')}",
        f"Run ID: {run.get('databaseId')}",
        f"Title: {run.get('displayTitle', '')}",
        f"Branch: {run.get('headBranch', '')}",
        f"Commit: {str(run.get('headSha', ''))[:7]}",
        f"Event: {run.get('event', '')}",
        f"URL: {run.get('url', '')}",
        f"Headline: {diagnosis.get('headline', '')}",
        f"Likely cause: {diagnosis.get('likely_cause', '')}",
    ]
    suggestions = diagnosis.get("suggestions") or []
    if suggestions:
        lines.append("Suggested fixes:")
        lines.extend([f"- {s}" for s in suggestions])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check recent GitHub Actions failures for SkyView and suggest likely fixes.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--all", action="store_true", help="Report all recent failed runs, not only new ones")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text")
    parser.add_argument("--state-path", default=str(STATE_PATH))
    parser.add_argument("--mark-seen", action="store_true", help="Persist reported failed run IDs as seen")
    args = parser.parse_args()

    state_path = Path(args.state_path)
    state = load_state(state_path)
    seen = {int(x) for x in state.get("seen_failed_runs", []) if str(x).isdigit()}

    runs = fetch_recent_runs(args.repo, args.limit)
    failed = [r for r in runs if r.get("conclusion") == "failure"]
    if not args.all:
        failed = [r for r in failed if int(r.get("databaseId", 0)) not in seen]

    reports: list[dict[str, Any]] = []
    for run in failed:
        run_id = int(run["databaseId"])
        try:
            log = fetch_failed_log(args.repo, run_id)
            diagnosis = diagnose_failure(log)
        except Exception as exc:
            diagnosis = {
                "headline": f"Could not fetch failed log: {exc}",
                "likely_cause": "Unable to inspect logs",
                "suggestions": ["Verify gh auth and run visibility", "Retry with: gh run view <id> --log-failed"],
            }
        reports.append({"run": run, "diagnosis": diagnosis})

    if args.mark_seen and reports:
        updated = sorted(seen | {int(r['run']['databaseId']) for r in reports})
        state["seen_failed_runs"] = updated
        save_state(state_path, state)

    if args.json:
        payload = [
            {
                "run": item["run"],
                "diagnosis": item["diagnosis"],
                "report": format_report(item["run"], item["diagnosis"]),
            }
            for item in reports
        ]
        print(json.dumps(payload, indent=2))
    else:
        if not reports:
            print("NO_FAILURES")
        else:
            print("\n\n".join(format_report(item["run"], item["diagnosis"]) for item in reports))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
