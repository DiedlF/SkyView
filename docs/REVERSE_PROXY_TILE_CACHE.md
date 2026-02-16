# Reverse Proxy Tile Caching (Nginx/Caddy)

## Goal
Reduce repeated overlay tile load on backend by caching `/api/overlay_tile/*` at proxy layer.

## Recommended cache key inputs
- path includes `z/x/y`
- query includes `layer`, `time`, optional `model`, optional rendering params

Because `time`/`model` are part of query, cache entries naturally separate by timestep/run context.

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
3. Confirm backend `/api/status` tile misses decrease under repeat pan/zoom.
4. Ensure cache max size and inactive TTL are bounded for your disk budget.
