# 🔍 Skyview Code-Review

**Datum:** 2026-02-17
**Scope:** Backend (Python/FastAPI) + Frontend (Vanilla JS/HTML/CSS)
**Status note (2026-04-26):** Historical review snapshot. Several high-priority findings have since been addressed, including default CORS hardening, marker auth extraction/startup policy, data-cache LRU/singleflight, symbol/ww alignment, ingest cleanup hardening, Leaflet SRI, and global frontend error banners. Use this file for context only; current open work is tracked in `../../TODO.md` and `../README.md`.

---

## 1. Sicherheitsbedenken

### 🔴 Kritisch: CORS erlaubt alle Origins
**Datei:** `backend/app.py`, Zeile 46–50
```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```
**Problem:** `allow_origins=["*"]` erlaubt jeder beliebigen Domain, die API aufzurufen. In Kombination mit dem Marker-Auth-System können fremde Websites im Namen des Nutzers Marker setzen.
**Verbesserung:** Origins auf die eigene Domain beschränken, z.B. `allow_origins=["https://skyview.example.com"]`.

### 🟡 Mittel: Nominatim-Proxy ohne Rate-Limiting
**Datei:** `backend/app.py`, Zeile ~1310–1330
```python
r = requests.get("https://nominatim.openstreetmap.org/search", ...)
```
**Problem:** Jeder API-Aufruf an `/api/location_search` löst eine Nominatim-Anfrage aus. Nominatim hat strikte Usage-Policies (max 1 req/s). Ein Angreifer kann die App als Open Proxy missbrauchen und den Server IP-sperren lassen.
**Verbesserung:** Serverseitiges Rate-Limiting (z.B. 1 req/s global) und Ergebnis-Caching.

### 🟡 Mittel: Feedback-Endpoint ohne Rate-Limiting oder Auth
**Datei:** `backend/app.py`, Zeile ~550–565
**Problem:** `/api/feedback` ist unauthentifiziert und unbegrenzt aufrufbar. Ein Angreifer kann die `feedback.json` beliebig aufblasen (Disk-Filling-Angriff).
**Verbesserung:** Rate-Limiting pro IP, maximale Dateigröße, oder clientId-basierte Begrenzung.

### 🟡 Mittel: Marker-Auth-Secret im Klartext, schwacher Fallback
**Datei:** `backend/app.py`, Zeile ~535
```python
MARKER_AUTH_SECRET = os.environ.get("SKYVIEW_MARKER_AUTH_SECRET", "")
```
**Problem:** Ohne konfiguriertes Secret (`MARKER_AUTH_CONFIGURED = False`) wird Marker-Bearbeitung vollständig deaktiviert, aber der Default-Marker wird trotzdem jedem Client angezeigt. Das ist korrekt, aber die Prüfung `len(MARKER_AUTH_SECRET) >= 16 and MARKER_AUTH_SECRET != "dev-marker-secret-change-me"` ist eine schwache Validierung.
**Verbesserung:** Secret-Stärke mit `secrets.compare_digest` validieren; Warnung beim Start loggen, wenn Secret fehlt.

### 🟢 Gering: `sys.path.insert(0, ...)` statt relative Imports
**Datei:** `backend/app.py`, Zeile 20
**Problem:** `sys.path.insert(0, os.path.dirname(__file__))` ist fragil und kann zu Shadow-Imports führen.
**Verbesserung:** Projekt als Python-Package strukturieren mit `pyproject.toml` und relativen Imports.

---

## 2. Potenzielle Bugs und Edge Cases

### 🔴 Kritisch: Race Condition bei Marker-Datei-Schreibzugriffen
**Datei:** `backend/app.py`, Zeile ~570–590 (`_write_markers`)
```python
markers = [m for m in _read_markers() if m.get("clientId") != client_id]
markers.append(marker)
_write_markers(markers)
```
**Problem:** Read-Modify-Write ohne Locking. Bei gleichzeitigen Requests können Marker verloren gehen. Dasselbe gilt für `append_feedback`.
**Verbesserung:** File-Locking via `fcntl.flock()` oder atomares Schreiben mit einer leichtgewichtigen DB (SQLite).

### 🟡 Mittel: `data_cache` wächst unbegrenzt
**Datei:** `backend/app.py`, Zeile 107 + `load_data()`
```python
data_cache: Dict[str, Dict[str, Any]] = {}
```
**Problem:** Jeder geladene Run/Step bleibt im Speicher. Bei vielen Modellen/Steps führt das zu Memory-Exhaustion. Es gibt keinen Eviction-Mechanismus.
**Verbesserung:** LRU-Cache mit Maximalzahl Einträge (z.B. `functools.lru_cache` oder manuelles LRU mit `collections.OrderedDict`).

### 🟡 Mittel: `classify_point` in `app.py` dupliziert Logik aus `classify.py`
**Datei:** `backend/app.py`, Zeile 117–140 vs. `backend/classify.py`, Zeile 35–70
**Problem:** Die Klassifizierungslogik existiert zweimal (einmal skalar in `app.py`, einmal vektorisiert in `classify.py`). Änderungen müssen an beiden Stellen gemacht werden — hohes Drift-Risiko.
**Verbesserung:** `classify_point` aus `app.py` entfernen und `classify.py` konsistent nutzen, oder `classify_point` in `classify.py` als kanonische Version definieren und importieren.

### 🟡 Mittel: `_resolve_eu_time_strict` parst beliebige Zeitstrings unsicher
**Datei:** `backend/app.py`, Zeile ~1580
```python
target = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
```
**Problem:** Wenn `time_str` kein ISO-Format ist (z.B. `"latest"`), wirft `fromisoformat` eine Exception. Diese wird zwar gefangen, aber der gesamte EU-Fallback wird dann stillschweigend übersprungen.
**Verbesserung:** Explizite Prüfung auf "latest" vor dem Parsing.

### 🟡 Mittel: Wind-Endpoint nutzt `np.where` statt vorberechnete Gruppen
**Datei:** `backend/app.py`, Zeile ~450–480 (`api_wind`)
**Problem:** Im Gegensatz zum `/api/symbols`-Endpoint, der Pre-Binning nutzt, berechnet `/api/wind` per Aggregationszelle `np.where(lat_mask)` — deutlich langsamer bei großen Viewports.
**Verbesserung:** Gleiche Pre-Binning-Strategie wie bei Symbols verwenden.

### 🟡 Mittel: `classify.py` Convective-Masken-Reihenfolge fehlerhaft
**Datei:** `backend/classify.py`, Zeile 50–58
```python
cloud_type[conv_mask & ((hbas_sc <= 0) | (clcl < 5))] = "blue_thermal"
cloud_type[conv_mask & ((lpi > 7) | ...)] = "cb"
```
**Problem:** `cb`-Zuweisung überschreibt `blue_thermal` für Zellen, die beide Bedingungen erfüllen (hbas_sc ≤ 0 UND lpi > 7). In `app.py:classify_point` wird korrekt priorisiert (blue_thermal hat Vorrang wenn hbas_sc ≤ 0). Die vektorisierte Version in `classify.py` hat also abweichendes Verhalten.
**Verbesserung:** In `classify.py` explizit `& (cloud_type != "blue_thermal")` zur cb-Maske hinzufügen (analog zu `cu_con_mask`).

### 🟢 Gering: `_typeToWw` Mapping in `symbols.js` hat falsche Codes
**Datei:** `frontend/symbols.js`, Zeile ~128–140
```javascript
'drizzle_light': 50, 'drizzle_moderate': 51, 'drizzle_dense': 53,
```
**Problem:** WMO-Code 50 ist "leichter intermittierender Sprühregen", 51 ist "leichter kontinuierlicher Sprühregen", 53 ist "mäßiger kontinuierlicher Sprühregen". Die Mapping-Namen suggerieren eine andere Zuordnung als die tatsächlichen WMO-Codes. Vergleich mit `weather_codes.py` nötig.
**Verbesserung:** Codes gegen die Backend-`ww_to_symbol`-Funktion abgleichen.

### 🟢 Gering: `ingest.py` nutzt `rm -rf` statt sicherere Methode
**Datei:** `backend/ingest.py`, Zeile ~290
```python
subprocess.run(["rm", "-rf", old_path], capture_output=True)
```
**Verbesserung:** `shutil.rmtree()` verwenden — plattformunabhängig und sicherer.

---

## 3. Performance-Optimierungen

### 🟡 Mittel: Per-Pixel-Loop im Overlay-Fallback
**Datei:** `backend/overlay_render.py`, Zeile ~175–185
```python
for yy in range(h):
    for xx in range(w):
        ...
        color = cmap_fn(v[yy, xx])
```
**Problem:** Der Fallback-Pfad in `colorize_layer_vectorized` iteriert pixelweise in Python — extrem langsam bei großen Bildern. Dies wird für alle Layer erreicht, die nicht in den expliziten `if`-Branches behandelt werden.
**Verbesserung:** Sicherstellen, dass alle konfigurierten Layer einen vektorisierten Pfad haben (aktuell fehlt z.B. `lpi`).

### 🟡 Mittel: Overlay-Image-Rendering in `/api/overlay` nutzt Per-Pixel-Loop
**Datei:** `backend/app.py`, Zeile ~1470–1478
```python
for y in range(h_merc):
    for x in range(w_merc):
        val = cropped_merc[y, x]
        color = cmap_fn(val)
```
**Problem:** Der ältere `/api/overlay`-Endpoint rendert immer pixelweise, obwohl `colorize_layer_vectorized` existiert.
**Verbesserung:** `colorize_layer_vectorized` auch hier nutzen und das Ergebnis in ein PIL-Image konvertieren.

### 🟡 Mittel: Boundary-Berechnung ist O(n×m) mit Segmentliste
**Datei:** `backend/app.py`, Zeile ~210–230 und `backend/ingest.py`, Zeile ~310–340
**Problem:** Die Boundary-Berechnung erstellt für jede Gitterzelle bis zu 4 Segmente. Bei 746×1215 Gitterpunkten können das ~3.6M Segmente sein. Dies wird als JSON gespeichert und zum Frontend gesendet.
**Verbesserung:** Marching-Squares-Algorithmus oder ConvexHull/ConcaveHull für kompaktere Randdarstellung.

### 🟢 Gering: Doppelte `resolve_time`-Aufrufe
**Datei:** `backend/app.py`, `/api/symbols` Zeile ~250 + ~280
**Problem:** `resolve_time_with_cache_context` wird aufgerufen, danach `_resolve_eu_time_strict` erneut (das intern auch `resolve_time` aufruft). Bei mehreren EU-Fallback-Prüfungen pro Request summiert sich das.
**Verbesserung:** EU-Auflösung einmal durchführen und das Ergebnis wiederverwenden.

### 🟢 Gering: SVG-Preloading aller 100 WMO-Codes
**Datei:** `frontend/symbols.js`, Zeile 7–12
```javascript
for (let i = 0; i < 100; i++) {
  WW_SYMBOLS[i] = `/geodata/ww-symbols/wmo4677_ww${code}.svg`;
}
```
**Problem:** Es werden URLs für alle 100 Codes generiert, aber nur ~27 tatsächlich preloaded. Die restlichen 73 URLs existieren vermutlich nicht.
**Verbesserung:** `WW_SYMBOLS`-Map nur für tatsächlich verwendete Codes erstellen.

---

## 4. Lesbarkeit und Wartbarkeit

### 🟡 Mittel: `app.py` ist ein 1900-Zeilen-Monolith
**Datei:** `backend/app.py`
**Problem:** Die Hauptdatei enthält Routing, Geschäftslogik, Caching, Auth, Marker-Management, Overlay-Rendering, Boundary-Berechnung und mehr. Extrem schwer zu navigieren und zu testen.
**Verbesserung:** Weitere Aufteilung in Router-Module (z.B. `routes/symbols.py`, `routes/overlay.py`, `routes/markers.py`) mit FastAPI-APIRouter.

### 🟡 Mittel: Magic Numbers durchgehend
**Datei:** Diverse
- `app.py`: `cape_ml > 50`, `cloud_depth > 4000`, `lpi > 7`, `ceiling >= 20000`, `clcl >= 30`
- `soaring.py`: `SINK_RATE = 0.7`, `SAFETY_MARGIN = 200.0`, `GLIDE_RATIO = 40.0`
- `overlay_render.py`: `mmh / 5.0`, `cape / 950.0`, alle Farbwerte

**Problem:** Meteorologische Schwellwerte sind ohne Erklärung hartcodiert. Änderungen erfordern Suchen im gesamten Codebase.
**Verbesserung:** Benannte Konstanten in einem zentralen `constants.py` definieren mit Kommentaren zur meteorologischen Begründung.

### 🟡 Mittel: Duplizierte `cell_sizes`-Dict
**Datei:** `backend/app.py`, Zeile ~245 und ~395 und `frontend/point_data.py`, Zeile 9
```python
cell_sizes = {5: 2.0, 6: 1.0, 7: 0.5, 8: 0.25, ...}
```
**Problem:** Identisches Dict an 3+ Stellen definiert. DRY-Verletzung.
**Verbesserung:** Einmal in `constants.py` definieren und importieren.

### 🟡 Mittel: Massiver Code-Duplizierung bei EU-Fallback
**Datei:** `backend/app.py`, `/api/symbols` (~80 Zeilen) und `/api/wind` (~40 Zeilen)
**Problem:** Die EU-Fallback-Logik (Laden, BBox-Slicing, Gruppen-Berechnung) ist in beiden Endpoints nahezu identisch kopiert.
**Verbesserung:** Gemeinsame Helper-Funktion `_load_blended_source(time, model, bbox, keys)` extrahieren.

### 🟢 Gering: Frontend I18N inline als riesiges Object-Literal
**Datei:** `frontend/app.js`, Zeile 1–200
**Problem:** ~200 Zeilen HTML-Strings (inkl. SVG) sind inline im JS. Das macht die Datei unübersichtlich und die Übersetzungen schwer wartbar.
**Verbesserung:** I18N-Strings in separate JSON-Dateien auslagern oder zumindest in eine eigene `i18n.js`.

### 🟢 Gering: Keine Type-Hints bei einigen Backend-Funktionen
**Datei:** `backend/app.py`, z.B. `classify_point()`, `_freshness_minutes_from_run()`
**Verbesserung:** Konsistent Type-Hints hinzufügen für bessere IDE-Unterstützung und Dokumentation.

---

## 5. Sauberkeit und Best Practices

### 🟡 Mittel: `import time` doppelt / shadowed
**Datei:** `backend/app.py`, Zeile 7 und Zeile 81
```python
import time  # Zeile 7 (top-level)
import time  # Zeile 81 (innerhalb Middleware)
```
**Problem:** `time` wird top-level importiert und dann erneut innerhalb der Middleware-Funktion. Außerdem importiert Zeile 12 `from time import perf_counter`, was Verwirrung stiftet.
**Verbesserung:** Redundanten Import in der Middleware entfernen.

### 🟡 Mittel: `@app.on_event("startup")` ist deprecated
**Datei:** `backend/app.py`, Zeile 89
**Problem:** FastAPI empfiehlt seit v0.93+ die `lifespan`-Handler statt `on_event`.
**Verbesserung:** Migration zu `@asynccontextmanager` Lifespan-Pattern.

### 🟡 Mittel: Globale mutable Dicts als Prozess-Zähler
**Datei:** `backend/app.py`, Zeile 52–62 (`api_error_counters`, `fallback_stats`)
**Problem:** Nicht thread-safe bei Multi-Worker-Deployment (z.B. Gunicorn mit mehreren Workers). Zähler gehen bei Worker-Restart verloren.
**Verbesserung:** Für Produktion: Prometheus-Metriken oder `multiprocessing.Value`. Für Single-Worker: explizit dokumentieren.

### 🟢 Gering: Frontend lädt Leaflet von CDN ohne SRI
**Datei:** `frontend/index.html`, Zeile 9–10
```html
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
```
**Problem:** Ohne Subresource Integrity (SRI) Hash könnte ein kompromittiertes CDN bösartigen Code einschleusen.
**Verbesserung:** `integrity="sha256-..."` und `crossorigin="anonymous"` hinzufügen.

### 🟢 Gering: Kein Error-Boundary im Frontend
**Datei:** `frontend/app.js`
**Problem:** Unbehandelte Promise-Rejections (z.B. bei Netzwerkfehlern in `loadTimesteps`) werden nur via `console.error` geloggt. Der Nutzer sieht ggf. eine leere Karte ohne Hinweis.
**Verbesserung:** `window.addEventListener('unhandledrejection', ...)` mit User-Feedback.

---

## Zusammenfassung

| Schwere | Anzahl |
|---------|--------|
| 🔴 Kritisch | 2 |
| 🟡 Mittel | 16 |
| 🟢 Gering | 8 |

**Top-3-Prioritäten:**
1. **CORS einschränken** und **Race Condition bei Marker-Dateien** fixen (Sicherheit + Datenverlust)
2. **`data_cache` begrenzen** (Memory-Exhaustion in Produktion)
3. **`app.py` aufteilen** und **duplizierte Klassifizierungslogik** konsolidieren (Wartbarkeit, Bug-Prävention)

Das Projekt ist insgesamt solide strukturiert — besonders die Auslagerung in `symbol_logic.py`, `overlay_render.py`, `overlay_data.py`, `soaring.py` und `point_data.py` zeigt gute Modularisierung. Die meteorologische Logik ist korrekt implementiert und die EU-Fallback-Strategie ist durchdacht. Die Hauptrisiken liegen im Betrieb (Speicher, Sicherheit) und in der Drift-Gefahr durch duplizierte Logik.
