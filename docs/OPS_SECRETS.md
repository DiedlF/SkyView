# OPS_SECRETS.md — Secret Management & Rotation

## Overview

Skyview uses operator-configurable secrets:

| Variable | Purpose | Required for |
|---|---|---|
| `SKYVIEW_MARKER_AUTH_SECRET` | HMAC-SHA256 signing of marker auth tokens | Marker editing by clients |
| `SKYVIEW_ADMIN_USER` | HTTP Basic username | Admin dashboard and private ops endpoints |
| `SKYVIEW_ADMIN_PASSWORD` | HTTP Basic password | Admin dashboard and private ops endpoints |
| `SKYVIEW_CORS_ORIGINS` | Allowlist of trusted frontend origins | Production deploys |
| `EUMETSAT_CONSUMER_KEY` | EUMETSAT Data Store API key (consumer key) | Satellite observation ingest |
| `EUMETSAT_CONSUMER_SECRET` | EUMETSAT Data Store API secret | Satellite observation ingest |

Copy `backend/.env.example` to `backend/.env` and fill in the values you need.
Secrets are loaded from `backend/.marker_auth_secret.env` (or `.env`) at server startup
**before** any routes are active. The loader also respects real environment variables — if
the variable is already set in the environment, the file is ignored for that key.

---

## SKYVIEW_MARKER_AUTH_SECRET

### Requirements

- Minimum length: **16 characters** (enforced)
- Recommended length: **32+ characters** (use 64 hex chars from `openssl rand`)
- Must not be one of the known-weak defaults (`""`, `"dev-marker-secret-change-me"`)

### Generating a strong secret

```bash
openssl rand -hex 32
# example output: a3f8c2e1d4b5…
```

Write it to the env file:

```bash
echo 'SKYVIEW_MARKER_AUTH_SECRET="<your-secret-here>"' > backend/.marker_auth_secret.env
chmod 600 backend/.marker_auth_secret.env
```

### Startup enforcement

At server startup, Skyview checks the secret and prints a **visible banner to stderr**:

- **Missing** secret → logged at `ERROR` level; marker editing is fully disabled.
- **Weak** secret → logged at `WARNING` level; marker editing is disabled.
- **Strong** secret → no banner; marker editing is enabled.

This makes mis-configuration impossible to miss in logs or systemd journal:

```
journalctl -u skyview --since today | grep -i "skyview security"
```

### Rotation procedure

Rotating the secret **invalidates all existing client tokens** (tokens are HMAC-signed;
a new secret produces different signatures). Clients will need to re-authenticate.

1. Generate a new secret: `openssl rand -hex 32`
2. Update `backend/.marker_auth_secret.env`
3. Restart the server: `systemctl restart skyview` (or your equivalent)
4. Verify startup log is clean (no security banner)
5. Inform clients that their marker auth token has been reset (they will get a new one
   automatically on next `/api/marker_auth` call)

Token TTL is `12 hours` by default (`TOKEN_TTL_SECONDS` in `marker_auth.py`). If you
rotate the secret, all tokens issued under the old secret become invalid immediately.

---

## SKYVIEW_ADMIN_USER / SKYVIEW_ADMIN_PASSWORD

### Requirements

- Both variables must be set for `/admin` and private ops endpoints to work.
- Use a long random password; missing credentials fail closed with HTTP 503.
- Protected endpoints include `/admin`, `/api/admin/*`, `/api/cache_stats`,
  `/api/perf_stats`, `/api/usage_stats`, and feedback list/update routes.

### Example

```bash
SKYVIEW_ADMIN_USER="admin"
SKYVIEW_ADMIN_PASSWORD="<long-random-password>"
```

After changing either value, restart the server and open `/admin`; the browser should
prompt for HTTP Basic credentials.

---

## SKYVIEW_CORS_ORIGINS

### Requirements

- Must be set to the **real public hostname(s)** of the frontend before any public deploy.
- Default (unset) allows only `localhost` origins — safe for local dev, not for production.

### Example

```bash
# In backend/.env or system environment:
SKYVIEW_CORS_ORIGINS="https://skyview.example.com"

# Multiple origins (comma-separated):
SKYVIEW_CORS_ORIGINS="https://skyview.example.com,https://staging.example.com"
```

**Never** use `SKYVIEW_CORS_ORIGINS=*` on a production server with marker auth enabled —
it would allow any website to make credentialed requests.

### Rotation / domain change

If your domain changes:
1. Update `SKYVIEW_CORS_ORIGINS` in the env file
2. Restart the server
3. Verify with: `curl -I -H "Origin: https://newdomain.com" https://skyview.example.com/api/timesteps`
   and check for `Access-Control-Allow-Origin` in the response headers.

---

## EUMETSAT_CONSUMER_KEY / EUMETSAT_CONSUMER_SECRET

Credentials for the EUMETSAT Data Store, used by the satellite half of the
observation layer (MSG RSS / MTG via EUMDAC). See
`docs/OBSERVATION_LAYER_IMPLEMENTATION_PLAN.md`.

### Requirements

- This is an **API key pair** (consumer key + secret), **not** your EUMETSAT
  portal username/password. Get it from <https://user.eumetsat.int> → "API key".
- Both values must be set for satellite ingest to authenticate.
- Long-lived secrets (no automatic expiry); EUMDAC exchanges them for a
  short-lived OAuth2 token (~1 h, auto-refreshed) per session.

### Collection access — NRT licence (required, separate step)

A valid key is **not sufficient** to download. The SEVIRI **L1.5 image**
collection used for the satellite overlay,
**`EO:EUM:DAT:MSG:MSG15-RSS`** (Rapid Scan, 5-min; note the older id
`EO:EUM:DAT:MSG:HRSEVIRI-RSS` is dead and 404s), is licence-gated: downloads
return `403` with body `NRTLicense required to access this collection` until the
**Near-Real-Time licence for that specific collection** is accepted in the Data
Store web UI (<https://data.eumetsat.int> → open the collection → accept its
licence). Caveats learned in live testing (2026-06-12):

- Accepting other licences (e.g. the MSG Cloud Mask) does **not** cover the
  image-data NRT licence — accept it on the `MSG15-RSS` collection specifically.
- Acceptance can take **up to ~1 h** to propagate to the API gateway.
- The **MTG-I1 FCI** comparison layer adds a second licence-gated collection,
  **`EO:EUM:DAT:0665`** (FCI L1c HRFI, the `vis_06` source; `EUCOMP_MTG_COLLECTION`
  overridable). Accept its NRT licence the same way — it is independent of the
  `MSG15-RSS` acceptance. Note FCI L1c products are large (one ZIP per repeat
  cycle); the ingest de-duplicates before downloading, so the licence/download
  path is only exercised on a genuinely new cycle.
- Auth/search succeed before the licence is accepted; only the **download**
  401/403s — so a passing `eumetsat_auth.py` does not by itself prove you can
  pull product files.

### Provisioning

```bash
cp backend/.env.example backend/.env
chmod 600 backend/.env
# edit backend/.env and set EUMETSAT_CONSUMER_KEY / EUMETSAT_CONSUMER_SECRET
```

### Verify

```bash
python3 scripts/eumetsat_auth.py            # loads backend/.env, mints a token, lists collections
python3 scripts/eumetsat_auth.py --token-only   # just confirm the pair mints a token
python3 scripts/eumetsat_auth.py --write-eumdac # also persist to ~/.eumdac/credentials for the eumdac CLI
```

Exit codes: `0` ok · `2` missing creds · `3` `eumdac` not installed · `4` auth/API failure.

### Rotation

Regenerate the pair in the EUMETSAT portal (revokes the old one), update
`backend/.env`, then re-run `scripts/eumetsat_auth.py` to confirm. **Rotate
immediately if a key/secret is ever pasted into chat, logs, or a ticket.**

---

## File permissions checklist

```bash
chmod 600 backend/.marker_auth_secret.env
chmod 600 backend/.env          # if used
```

These files must **not** be committed to git. Verify with:

```bash
git check-ignore -v backend/.marker_auth_secret.env backend/.env
```

Both should be covered by `.gitignore`.
