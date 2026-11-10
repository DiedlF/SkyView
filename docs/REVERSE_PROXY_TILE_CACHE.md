# Reverse Proxy / CDN Caching

## Production setup: IPv64 CDN64 (mode 3, "direct to target")

The public site is served by the IPv64 CDN64 edge servers, which connect to the
app on the VPS at plain HTTP `:8501`. The edge terminates TLS (HTTP/2 + HTTP/3),
redirects HTTP to HTTPS and compresses text responses, so the app does neither.

**The edge caches only what the app marks cacheable** (`max-age` / `s-maxage` /
`Expires`; never `no-store`, `no-cache`, `private`, `Set-Cookie` or requests with
`Authorization`; max 8 MB per object). The cache must also be switched on for the
name in the IPv64 panel. A cached answer carries `X-Proxy-Cache: HIT`.

| Response | `Cache-Control` | Why |
|---|---|---|
| HTML (`/`, `*.html`) | `no-cache` | carries the asset versions; every page load reaches the app |
| `app.js?v=N`, `style.css?v=N`, ... | `public, max-age=31536000, immutable` | versioned URL never changes content - **bump `v` in `index.html` whenever the file changes** |
| other static files | `public, max-age=86400` | icons, symbol SVGs |
| overlay tile with current `v=` token | `public, max-age=3600` | URL pins the data, see below |
| overlay tile otherwise | `public, max-age=300` | old client, `time=latest`, no `model`, substep, missing EU fill |
| empty tile (step not ingested) | `no-store` | a cached hole would outlive the missing data |
| observation frame by frame id | `public, max-age=21600, immutable` | frames are written once per id |
| observation frame `latest` / by time | `public, max-age=60` | resolves to a newer frame over time |
| `/api/observations/frames` | `public, max-age=20` | polled every 60 s per tab |
| `/api/ingest_pulse` | `public, max-age=10` | polled every 12 s per tab |
| `/api/timesteps`, everything else | none | must be fresh when the pulse reports a change |

**Tile version token.** `/api/timesteps` returns `tileVersion`, a hash of every
ingested model step. The frontend appends it to tile URLs as `v=`. When a run or
a single step lands the token changes, so the URLs change: browsers and the edge
never show a mix of old and new tiles, and matching tiles can be cached for an
hour instead of five minutes.

**Origin lock-down.** `deploy/setup-firewall.sh` (run on every deploy) lets only
the edge servers in `deploy/cdn64-edge-ips.txt` reach `:8501`; SSH stays open.
The same file is the list of proxies whose `X-Forwarded-*` headers the app
trusts (`backend/trusted_proxies.py`), which is what makes `request.client` the
real visitor for rate limits and visit statistics. If IPv64 changes its servers
(<https://ipv64.net/cdn64-server.php>), edit that one file and deploy.

**Check after a deploy**

```bash
H=https://<public-host>
curl -sI "$H/app.js?v=73" | grep -i -E 'cache-control|x-proxy-cache'   # immutable; HIT on 2nd call
curl -s "$H/api/timesteps" | python3 -c 'import json,sys; print(json.load(sys.stdin)["tileVersion"])'
curl -s -m 5 http://<vps-ip>:8501/api/health || echo "origin closed (expected)"
```

---

# Generic reverse-proxy tile caching (Nginx/Caddy)

Reference for running Skyview behind your own proxy instead of the CDN.

## Goal
Reduce repeated overlay tile load on backend by caching `/api/overlay_tile/*` at proxy layer.

This is the preferred first implementation step for the tile-delivery recommendations in `PERFORMANCE_RECOMMENDATIONS_2026-04-26.md`: a proxy hit avoids FastAPI, Python cache lookups, data loading, colorization, and PNG encoding entirely.

## Recommended cache key inputs
- path includes `z/x/y`
- query includes `layer`, `time`, optional `model`, optional `substep`, the `v` version token, and any future rendering params

Because the full request URI includes `time`, `model`, `substep` and `v`, cache entries naturally separate by timestep and run context.

---

## Nginx example

```nginx
proxy_cache_path /var/cache/nginx/skyview levels=1:2 keys_zone=skyview_tiles:100m max_size=5g inactive=30m use_temp_path=off;

server {
  listen 80;
  server_name skyview.local;

  location /api/overlay_tile/ {
    proxy_pass http://127.0.0.1:8501;
    proxy_http_version 1.1;
    proxy_set_header Host $host;

    proxy_cache skyview_tiles;
    proxy_cache_key "$scheme$request_method$host$request_uri";
    proxy_cache_valid 200 5m;
    proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
    add_header X-Proxy-Cache $upstream_cache_status always;
  }

  location / {
    proxy_pass http://127.0.0.1:8501;
    proxy_set_header Host $host;
  }
}
```

---

## Caddy example

```caddyfile
skyview.local {
  @tiles path /api/overlay_tile/*

  reverse_proxy @tiles 127.0.0.1:8501 {
    header_up Host {host}
  }

  reverse_proxy 127.0.0.1:8501 {
    header_up Host {host}
  }

  # If using caddy-cache plugin, configure tile route with short TTL (5m) and key=request URI.
}
```

---

## Validation checklist
1. Request same tile twice and verify second is proxy HIT (`X-Proxy-Cache` or plugin equivalent).
2. Switch timestep and ensure different query string produces distinct cache entry.
3. Switch `substep` and ensure the cache key separates the variants.
4. Confirm backend tile misses decrease under repeat pan/zoom via `/api/perf_stats` or admin perf telemetry.
5. Confirm visual output is unchanged for desktop, mobile, and substep-capable overlays.
6. Ensure cache max size and inactive TTL are bounded for your disk budget.
