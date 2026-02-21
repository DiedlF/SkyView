# OPS_SECRETS.md — Secret Management & Rotation

## Overview

Skyview uses two operator-configurable secrets:

| Variable | Purpose | Required for |
|---|---|---|
| `SKYVIEW_MARKER_AUTH_SECRET` | HMAC-SHA256 signing of marker auth tokens | Marker editing by clients |
| `SKYVIEW_CORS_ORIGINS` | Allowlist of trusted frontend origins | Production deploys |

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
