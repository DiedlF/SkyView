#!/usr/bin/env python3
"""Authenticate against the EUMETSAT Data Store using EUMDAC.

Reads the API key pair (consumer key + secret) from the environment or from
`backend/.env` (the same file the SkyView server loads), exchanges it for an
OAuth2 access token, and — unless run with --token-only — verifies the token
actually works by listing a few Data Store collections.

The Data Store uses an API key PAIR, NOT your portal username/password. Get the
pair from https://user.eumetsat.int → "API key".

Usage:
  python3 scripts/eumetsat_auth.py                 # load backend/.env, verify
  python3 scripts/eumetsat_auth.py --env-file path/to/.env
  python3 scripts/eumetsat_auth.py --token-only     # just mint/print a token
  python3 scripts/eumetsat_auth.py --write-eumdac   # also persist to ~/.eumdac/credentials
  python3 scripts/eumetsat_auth.py --collection EO:EUM:DAT:MSG:HRSEVIRI-RSS

Exit codes:
  0 success · 2 missing credentials · 3 eumdac not installed · 4 auth/API failure
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Resolve repo paths relative to this file so the script works from any cwd.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_ENV = _REPO_ROOT / "backend" / ".env"

KEY_VAR = "EUMETSAT_CONSUMER_KEY"
SECRET_VAR = "EUMETSAT_CONSUMER_SECRET"


def load_env_file(path: Path) -> int:
    """Parse shell-style `export KEY="VALUE"` / `KEY=VALUE` lines into os.environ.

    Mirrors backend/app.py:_load_env_file — real environment variables already
    set take precedence (the file does not override them). Returns the number of
    keys loaded from the file.
    """
    loaded = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                line = re.sub(r"^export\s+", "", line)
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
                    loaded += 1
    except FileNotFoundError:
        return 0
    except Exception as exc:  # pragma: no cover - defensive
        print(f"warning: could not read {path}: {exc}", file=sys.stderr)
    return loaded


def mask(secret: str) -> str:
    """Show only the first 4 / last 2 chars so logs never leak full secrets."""
    if not secret:
        return "<empty>"
    if len(secret) <= 8:
        return secret[0] + "…"
    return f"{secret[:4]}…{secret[-2:]} (len {len(secret)})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify EUMETSAT Data Store credentials via EUMDAC.")
    parser.add_argument("--env-file", type=Path, default=_DEFAULT_ENV,
                        help=f"Env file to load if vars are unset (default: {_DEFAULT_ENV}).")
    parser.add_argument("--token-only", action="store_true",
                        help="Only mint and print an access token; skip the API connectivity check.")
    parser.add_argument("--write-eumdac", action="store_true",
                        help="Also persist the pair to ~/.eumdac/credentials (eumdac set-credentials equivalent).")
    parser.add_argument("--collection", default=os.environ.get("EUCOMP_SAT_COLLECTION"),
                        help="Optional collection id to describe as a deeper check (e.g. MSG RSS).")
    args = parser.parse_args()

    # 1) Source credentials: real env first, then the env file.
    if KEY_VAR not in os.environ or SECRET_VAR not in os.environ:
        n = load_env_file(args.env_file)
        if n:
            print(f"Loaded {n} var(s) from {args.env_file}")

    key = os.environ.get(KEY_VAR, "").strip()
    secret = os.environ.get(SECRET_VAR, "").strip()
    if not key or not secret:
        print(
            f"ERROR: {KEY_VAR} and/or {SECRET_VAR} not set.\n"
            f"  Set them in {args.env_file} (copy from backend/.env.example) or export them.\n"
            f"  Get the API key pair from https://user.eumetsat.int → 'API key'.",
            file=sys.stderr,
        )
        return 2
    print(f"Credentials found: key={mask(key)}  secret={mask(secret)}")

    # 2) Import eumdac (lazy so the help/usage path works without it installed).
    try:
        import eumdac
    except ImportError:
        print(
            "ERROR: the 'eumdac' package is not installed.\n"
            "  Install it with:  pip install eumdac\n"
            "  (also listed in backend/requirements.txt for the observation layer).",
            file=sys.stderr,
        )
        return 3

    # 3) Mint an access token from the key/secret pair.
    try:
        token = eumdac.AccessToken((key, secret))
        token_str = str(token)
        expires = getattr(token, "expiration", None)
    except Exception as exc:
        print(f"ERROR: failed to obtain access token: {exc}", file=sys.stderr)
        return 4
    print(f"✓ Access token obtained (expires: {expires}).")

    # 4) Optionally persist to ~/.eumdac/credentials for the eumdac CLI/library.
    if args.write_eumdac:
        cred_dir = Path.home() / ".eumdac"
        cred_dir.mkdir(parents=True, exist_ok=True)
        cred_file = cred_dir / "credentials"
        cred_file.write_text(f"{key},{secret}\n", encoding="utf-8")
        try:
            cred_file.chmod(0o600)
        except OSError:
            pass
        print(f"✓ Wrote {cred_file} (chmod 600).")

    if args.token_only:
        return 0

    # 5) Deeper check: connect to the Data Store and confirm the token works.
    try:
        datastore = eumdac.DataStore(token)
        if args.collection:
            collection = datastore.get_collection(args.collection)
            print(f"✓ Collection reachable: {args.collection} — {getattr(collection, 'title', collection)}")
        else:
            sample = []
            for i, collection in enumerate(datastore.collections):
                sample.append(str(collection))
                if i >= 4:
                    break
            print(f"✓ Data Store reachable. Sample collections: {', '.join(sample) or '(none returned)'}")
    except Exception as exc:
        print(f"ERROR: token minted but Data Store query failed: {exc}", file=sys.stderr)
        return 4

    print("\nAll good — EUMETSAT credentials are valid and the Data Store is reachable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
