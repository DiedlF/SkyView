# Skyview Architecture Review — 2026-02-19

**Scope:** Architecture, performance, and maintainability findings from code review.  
**Status:** Findings documented; quick wins partially applied (see bug fixes below).

---

## 🐛 Bug Fixes Applied

### Spurious "Timestep fallback active" banner at Δ=0h

**Symptom:** The yellow banner "Timestep fallback active: requested Thu 19.02. 16 UTC, using Thu 19.02. 16 UTC (Δ 0.00h)" appears even though requested and used valid times are identical.

**Root cause:** When the strict EU run file is missing (e.g. `2026021906/010.npz` — old run pruned from disk), `_load_eu_data_strict_or_nearby` falls back to the nearest available EU timestep. If a newer run (`2026021912/step=4`) covers the exact same valid time (16:00 UTC), `dist_h = 0.0` but `recovered = True`. The `api_symbols` handler keys only on `recovered`, not on `deltaHours`, so it sets `fallbackTimestep.active = True` and the frontend shows the banner.

**Fix (applied):**
- **Backend (`app.py`):** Only set `fallback_timestep` when `deltaHours > 0.01`. A Δ=0 recovery is a transparent run substitution, not a temporal mismatch.
- **Frontend (`app.js`):** Added `delta <= 0.01` early-exit guard in `updateFallbackBanner` as defense-in-depth.

**Log evidence:**
```
ERROR  Data not found: .../icon-eu/2026021906/010.npz
WARNING ICON-EU strict load failed: run=2026021906 step=10
WARNING ICON-EU recovered via nearby timestep: strict=2026021906/10 -> fallback=2026021912/4 (Δ=0.00h)
```

---

## 🏗️ Architecture

### 1. `app.py` is a 2,868-line monolith — highest maintainability risk

The file currently mixes: endpoint logic, marker auth (HMAC/base64), feedback CRUD, admin storage/logs, EU fallback resolution, grid cell binning, Nominatim proxy, DWD HEAD requests, and PID file management.

**Recommended split using FastAPI `APIRouter`:**

```
backend/
├── routers/
│   ├── symbols.py       # /api/symbols
│   ├── wind.py          # /api/wind
│   ├── overlay.py       # /api/overlay + /api/overlay_tile
│   ├── point.py         # /api/point
│   ├── markers.py       # /api/marker_*
│   ├── feedback.py      # /api/feedback
│   └── admin.py         # /api/admin/*, /api/status, /api/perf_stats
├── services/
│   ├── model_select.py  # EU/D2 fallback resolution (see #2)
│   └── grid.py          # Grid binning shared logic (see #3)
└── app.py               # FastAPI app setup only (~100 lines)
```

### 2. D2/EU fallback logic is heavily duplicated

`api_symbols` and `api_wind` share ~150 lines of essentially identical logic:
- Resolve strict EU time
- Load EU data
- Slice bbox for both models
- Build pre-binned `lat_groups`/`lon_groups` for D2 and EU
- Per-cell source selection (in-domain check, signal check, fallback)

A `GridContext` dataclass would eliminate this entirely:

```python
@dataclass
class GridContext:
    lat: np.ndarray; lon: np.ndarray
    domain: tuple[float, float, float, float]  # lat_min/max, lon_min/max
    lat_groups: list; lon_groups: list          # pre-binned
    eu: Optional['GridContext'] = None          # EU fallback, same structure

def build_grid_context(d, bbox, cell_size, pad, d_eu=None) -> GridContext: ...
```

Both `api_symbols` and `api_wind` would then just call `build_grid_context()` and pass the result to their respective aggregators.

### 3. `_try_load_eu_fallback` runs on every overlay tile request

`api_overlay_tile` calls `_try_load_eu_fallback(time, cfg)` unconditionally for every D2 tile, including tiles entirely within the D2 domain. This triggers `_resolve_eu_time_strict` + potential `load_data` on every request.

**Fix:** Gate the EU fallback call behind a domain check:
```python
eu_fb = None
if model_used == "icon_d2" and _tile_overlaps_outside_d2(lat_min, lat_max, lon_min, lon_max):
    eu_fb = _try_load_eu_fallback(time, cfg)
```

---

## ⚡ Performance

### 4. Blocking calls in async handlers — event loop stall (bug)

Two places block the asyncio event loop entirely:

**`api_location_search`:**
```python
time.sleep(min(wait_s, 1.0))  # blocks all concurrent requests for up to 1s!
r = requests.get("https://nominatim.openstreetmap.org/...")  # blocking HTTP
```

**`_fetch_dwd_run_fully_available_at` (called from `api_status`):**
```python
r = requests.head(url, timeout=6)  # blocks event loop for up to 6s
```

**Fix:** Use `httpx` (async HTTP) and `asyncio.sleep`:
```python
# pip install httpx
import httpx, asyncio

await asyncio.sleep(wait_s)
async with httpx.AsyncClient() as client:
    r = await client.get(url, timeout=8)
```

Or for minimal code change, run via `run_in_executor`:
```python
loop = asyncio.get_event_loop()
r = await loop.run_in_executor(None, lambda: requests.get(url, timeout=8))
```

### 5. `data_cache` LRU size of 8 is too small for concurrent tile bursts

The `load_data` LRU cache is hardcoded at 8 entries. Under a concurrent tile burst (user switches timestep → 20+ tiles fire simultaneously, each for D2 + EU), cache thrashing occurs. Since this is already pattern-matched to the other caches, it should be env-configurable:

```python
DATA_CACHE_MAX_ITEMS = int(os.environ.get('SKYVIEW_DATA_CACHE_MAX_ITEMS', '24'))
```

Also: `load_data` with selective keys has a race — if two concurrent requests both miss the same cache key, both load from .npz. Add a singleflight guard here similar to `computed_field_cache`.

### 6. Per-cell Python loop is the hot path

`api_symbols` and `api_wind` iterate cells in a nested Python `for` loop (up to ~900 cells at zoom 9). The pre-binning strategy is good but the aggregation call per cell still has Python overhead. For the non-ww (stratiform) path, `classify_cloud_type` could be vectorized across all cells at once using NumPy, avoiding per-cell function call overhead.

---

## 🔧 Maintainability

### 7. All state is module-level mutable globals

`fallback_stats`, `fallback_current`, `data_cache`, `_eu_strict_cache`, `feedback_rates`, `last_nominatim_request` — all module-level globals that:
- Make unit testing require module reloading or monkeypatching
- Break silently with multiple workers (currently just a log warning)
- Create invisible coupling between otherwise independent code paths

**Recommended:** Encapsulate into a `AppState` class injected via FastAPI dependency injection, or at minimum group them into clearly named objects per concern.

### 8. Marker auth reinvents JWT

~80 lines of manual HMAC + base64 in `app.py` implement a custom `{payload_b64}.{sig_b64}` scheme that is functionally a simplified JWT. Either:
- Move to `python-jose` / `authlib` for proper JWT (standard, auditable)
- Or at minimum move to `backend/marker_auth.py` so it can be tested in isolation

### 9. `api_point` still loads all variables

```python
d = load_data(run, step, model_used)  # no keys filter — loads all 14 variables
```
This is the only endpoint that hasn't been migrated to selective key loading. The point endpoint only needs the weather variables + wind; the full load wastes memory and evicts the LRU cache faster.

### 10. Latent bug in selective `load_data` merging

If a partial cache entry exists (some keys loaded) and a new request needs additional keys, the code:
1. Eviction check runs before the new data is fully merged
2. Two concurrent requests for the same missing key both fall through to NPZ load

The logic in the `if cache_key in data_cache:` merge block on partial hits is subtle and untested. A singleflight pattern (already used for `computed_field_cache`) would fix both issues.

### 11. QA scripts are not pytest-compatible

`qa_smoke.py`, `qa_contract.py`, `qa_regression.py`, `qa_perf.py` are standalone `if __name__ == "__main__"` scripts. Wrapping them as `pytest` test cases would enable:
- `pytest tests/` in CI with clean pass/fail
- Coverage reports
- Parallel test execution

---

## Quick Wins Summary

| # | Item | Effort | Impact |
|---|------|--------|--------|
| ✅ | Fix Δ=0h spurious fallback banner (backend + frontend) | Done | Bug fixed |
| 4 | Fix `time.sleep()` + `requests` → async in `api_location_search` | 30 min | Prevents event loop stall |
| 4 | Fix `requests.head()` → async in `_fetch_dwd_run_fully_available_at` | 30 min | Same |
| 5 | `data_cache` size → env-configurable (default 24) | 5 min | Better tile burst behavior |
| 3 | Gate `_try_load_eu_fallback` behind bbox-vs-D2 check | 30 min | Saves work on every inner tile |
| 9 | Add key filter to `api_point` `load_data` call | 15 min | Less memory pressure |
| 2/3 | Extract grid binning into shared `GridContext` helper | 2–3h | Removes ~150 lines of duplication |
| 1 | Split `app.py` into FastAPI routers | 1–2 days | Major maintainability improvement |
| 11 | Wrap QA scripts as pytest tests | 1h | Enables CI |
| 8 | Move marker auth to own module | 1h | Testable, auditable |

---

## Notes

- Multi-worker deployment is currently unsafe: all state is process-local. Fix global state (#7) before adding workers.
- CORS still defaults to localhost allowlist; production deployment should set `SKYVIEW_CORS_ORIGINS` to actual hostname.
- The `_eu_strict_cache` (64 entries, process-local) will reset on every server restart. Not a bug, but worth noting for observability.
