"""Central configuration for the SkyView observation layer (radar + satellite).

Ports the standalone ``eucomposites`` scaffold into SkyView. All endpoints and
product identifiers were verified against the EUMETNET ORD API documentation and
the EUMETSAT Data Store as of June 2026. Treat these as defaults overridable via
environment variables.

Derived observation frames are written under the repository
``data/observations/`` tree by default so they sit alongside the ICON stores and
are covered by ``.gitignore``. Native provider files are temporary ingest inputs
unless a caller explicitly downloads them elsewhere for debugging.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Output layout — anchored at <repo>/data/observations by default.
# backend/observations/config.py -> parents[2] == repo root.
# --------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_backend_env(path: Path) -> None:
    """Load backend/.env for observation CLIs without overriding real env vars."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                line = re.sub(r"^export\s+", "", line)
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except FileNotFoundError:
        pass


_load_backend_env(_REPO_ROOT / "backend" / ".env")

_DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "observations"

DATA_ROOT = Path(
    os.environ.get("EUCOMP_DATA_ROOT")
    or os.environ.get("SKYVIEW_OBS_DATA_ROOT")
    or str(_DEFAULT_DATA_ROOT)
).expanduser()

# Derived render products land under these source directories.
RADAR_DIR = DATA_ROOT / "radar"
SAT_DIR = DATA_ROOT / "satellite"
MTG_DIR = DATA_ROOT / "mtg"
LI_DIR = DATA_ROOT / "li"
STATE_DIR = DATA_ROOT / ".state"  # last-seen markers for the poller
TMP_DIR = DATA_ROOT / "tmp"


# --------------------------------------------------------------------------
# Radar: EUMETNET OPERA via the Open Radar Data (ORD) API / S3 cache
# --------------------------------------------------------------------------
# NOTE (verified June 2026): ORD onboarding to MeteoGate finalised 20 May 2026.
# Whitelisting is NO LONGER NEEDED -- the API and S3 cache are openly available.
@dataclass(frozen=True)
class RadarConfig:
    # EDR-compliant API via the MeteoGate gateway (Swagger at <base>/docs).
    edr_base: str = os.environ.get(
        "EUCOMP_RADAR_EDR_BASE", "https://api.meteogate.eu/eu-eumetnet-weather-radar"
    )
    # Public S3 24-hour rolling cache (open; use --no-sign-request / unsigned).
    s3_endpoint: str = os.environ.get(
        "EUCOMP_RADAR_S3_ENDPOINT", "https://s3.waw3-1.cloudferro.com"
    )
    s3_bucket: str = os.environ.get("EUCOMP_RADAR_S3_BUCKET", "openradar-24h")

    # Optional MeteoGate API key. Anonymous access is OPEN but rate-limited
    # (watch the `x-ratelimit-remaining` response header); a key raises the limit.
    # Get one at https://devportal.meteogate.eu/. Sent as a request header.
    api_key: str = os.environ.get("EUCOMP_RADAR_API_KEY", "")
    api_key_header: str = os.environ.get("EUCOMP_RADAR_API_KEY_HEADER", "apikey")

    # MQTT notification service for push-based "new frame" triggers (confirmed
    # June 2026): wss://radar.meteogate.eu:8884/ordmqtt, user/pass everyone.
    # OPERA max-reflectivity topic: ORD/eu.eumetnet/0-20010-0-OPERA/DBZH
    mqtt_host: str = os.environ.get("EUCOMP_RADAR_MQTT_HOST", "radar.meteogate.eu")
    mqtt_path: str = "/ordmqtt"
    mqtt_user: str = "everyone"
    mqtt_password: str = "everyone"
    mqtt_port: int = 8884  # wss; use 1883 for plain mqtt://

    # The 5-minute product: CIRRUS instantaneous max reflectivity composite.
    # Query via EDR with these verified parameters:
    #   standard_name=DBZH, method=comp, location_id=0-20010-0-OPERA, format=ODIM
    standard_name: str = "DBZH"   # RATE (rain rate) or ACRR (accumulation) also valid
    method: str = "comp"          # composite (vs "scan"/"point" for single-site)
    location_id: str = "0-20010-0-OPERA"  # OPERA composite WIGOS id (confirmed)
    odim_format: str = "ODIM"
    cadence_seconds: int = 300

    poll_interval: int = 60  # how often to check for a new frame


# --------------------------------------------------------------------------
# Satellite: EUMETSAT Data Store via EUMDAC
# --------------------------------------------------------------------------
# MSG Rapid Scanning Service (RSS) is the only true 5-minute Europe source today.
# MTG FCI is 10-minute (higher res). Switch collection_id to move between them.
@dataclass(frozen=True)
class SatelliteConfig:
    # EUMDAC reads credentials from ~/.eumdac/credentials or the env vars below.
    # Verify with: python3 scripts/eumetsat_auth.py
    consumer_key: str = os.environ.get("EUMETSAT_CONSUMER_KEY", "")
    consumer_secret: str = os.environ.get("EUMETSAT_CONSUMER_SECRET", "")

    # Collection IDs (confirm via `eumdac describe`):
    #   MSG RSS Level 1.5:   "EO:EUM:DAT:MSG:MSG15-RSS"      (5-min, Europe)
    #   MTG FCI Level 1c:    "EO:EUM:DAT:0662"               (10-min)
    collection_id: str = os.environ.get(
        "EUCOMP_SAT_COLLECTION", "EO:EUM:DAT:MSG:MSG15-RSS"
    )
    cadence_seconds: int = 300
    poll_interval: int = 60

    # Region of interest (lon/lat bbox) for tailoring/cropping to Europe.
    roi_bbox: tuple[float, float, float, float] = (-15.0, 32.0, 45.0, 72.0)


# --------------------------------------------------------------------------
# Satellite (new generation): MTG-I1 (Meteosat-12) FCI via the Data Store
# --------------------------------------------------------------------------
# Separate source from MSG RSS so the two render side-by-side for comparison.
# MTG-I1 sits at the 0° prime position (true nadir over Europe -> far less of
# the inclined-orbit cloud parallax that MSG-4 RSS shows). FCI has no broadband
# HRV; the closest high-res visible analogue to SEVIRI HRV is the ``vis_06``
# channel (0.5 km in HRFI). FD repeat cycle is 10 minutes, so this is a context/
# comparison layer, not a 5-minute loop match for radar.
#
# IMPORTANT (cost): the full FCI L1c stream is ~440 GB/day. Two guards keep this
# VPS-safe: (1) ingest de-duplicates against the manifest BEFORE downloading (see
# ingest_mtg), so a frequent cron tick only fetches genuinely new cycles; and
# (2) only the northern (Europe) chunks are downloaded as individual Data Store
# entries (see ``chunk_fraction`` below), not the whole disk.
@dataclass(frozen=True)
class MtgConfig:
    # Shares EUMDAC credentials with the MSG source.
    consumer_key: str = os.environ.get("EUMETSAT_CONSUMER_KEY", "")
    consumer_secret: str = os.environ.get("EUMETSAT_CONSUMER_SECRET", "")

    # Collection IDs (confirm via `eumdac describe`):
    #   FCI L1c HRFI (0.5 km vis_06):  "EO:EUM:DAT:0665"   (default; high-res)
    #   FCI L1c FDHSI (1 km vis_06):   "EO:EUM:DAT:0662"
    collection_id: str = os.environ.get("EUCOMP_MTG_COLLECTION", "EO:EUM:DAT:0665")
    channel: str = os.environ.get("EUCOMP_MTG_CHANNEL", "vis_06")
    cadence_seconds: int = 600  # FD repeat cycle (10 min)
    poll_interval: int = 120

    # Chunk subsetting (bandwidth/disk control): an FCI L1c full disk is split
    # into many chunks stacked south->north; Europe (40–60°N) is the northern
    # band. We download only the top ``chunk_fraction`` of body chunks (plus the
    # trailer) instead of the whole disk. 0.35 ≈ EUMETSAT's "Q4" northern-quarter
    # coverage with margin on the southern (40°N) edge. ``EUCOMP_MTG_CHUNKS`` pins
    # an explicit range (e.g. "29-40") if you know the product's chunk layout.
    chunk_fraction: float = float(os.environ.get("EUCOMP_MTG_CHUNK_FRACTION", "0.35"))
    chunks: str = os.environ.get("EUCOMP_MTG_CHUNKS", "")

    # Region of interest (lon/lat bbox) for cropping to Europe before resample.
    roi_bbox: tuple[float, float, float, float] = (-15.0, 32.0, 45.0, 72.0)


# --------------------------------------------------------------------------
# Lightning: MTG-I1 (Meteosat-12) Lightning Imager (LI) via the Data Store
# --------------------------------------------------------------------------
# LI is a separate optical lightning instrument on MTG-I1 (not an FCI channel),
# so it renders as its own coloured, mostly-transparent overlay that sits on top
# of whichever satellite/basemap layer is shown — it pairs equally with the MSG
# HRV and the MTG vis_06 images. MSG/SEVIRI has no lightning sensor, so there is
# no equivalent for the HRV layer beyond this same overlay.
#
# We use the gridded "Accumulated Flashes" (AF) product: a sparse 2-D field on
# the FCI 2 km geostationary grid (10-min product, 30-s accumulation internally).
# Cheap: one ~2–3 MB single-chunk product per cycle (~330 MB/day), so unlike FCI
# there is no chunk subsetting — we just download the .nc and render it.
@dataclass(frozen=True)
class LiConfig:
    # Shares EUMDAC credentials with the MSG/MTG sources.
    consumer_key: str = os.environ.get("EUMETSAT_CONSUMER_KEY", "")
    consumer_secret: str = os.environ.get("EUMETSAT_CONSUMER_SECRET", "")

    # Collection IDs (confirm via `eumdac describe`):
    #   LI L2 Accumulated Flashes (AF):       "EO:EUM:DAT:0686"  (default)
    #   LI L2 Accumulated Flash Area (AFA):   "EO:EUM:DAT:0687"
    #   LI L2 Accumulated Flash Radiance:     "EO:EUM:DAT:0688"
    collection_id: str = os.environ.get("EUCOMP_LI_COLLECTION", "EO:EUM:DAT:0686")
    dataset: str = os.environ.get("EUCOMP_LI_DATASET", "flash_accumulation")
    cadence_seconds: int = 600  # AF product cadence (10 min)
    poll_interval: int = 120

    # Region of interest (lon/lat bbox) for cropping to Europe before resample.
    roi_bbox: tuple[float, float, float, float] = (-15.0, 32.0, 45.0, 72.0)


@dataclass(frozen=True)
class Config:
    radar: RadarConfig = field(default_factory=RadarConfig)
    satellite: SatelliteConfig = field(default_factory=SatelliteConfig)
    mtg: MtgConfig = field(default_factory=MtgConfig)
    li: LiConfig = field(default_factory=LiConfig)


def ensure_dirs() -> None:
    for d in (RADAR_DIR, SAT_DIR, MTG_DIR, LI_DIR, STATE_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Target grid for reprojection / serving
# --------------------------------------------------------------------------
# Observations are reprojected onto a regular lat/lon grid (the same convention
# as ICON Zarr output) so the rendered PNG can be georeferenced as a plain
# lat/lon image overlay. The window is intentionally a bit WIDER than the
# ICON-EU/D2 forecast crop (``d2_bounds`` in backend/ingest_config.yaml) so radar
# and satellite frames give surrounding context beyond the forecast domain. The
# resolution matches ICON-D2 native (0.02° ≈ 2 km).
#
# IMPORTANT: this grid is the single source of truth for both the render extent
# (``reproject_odim`` / ``target_area_definition``) and the bbox advertised by
# the serving layer (``routers/observations.py``). Render extent and served bbox
# MUST stay equal or the frontend image overlay will be misregistered.
@dataclass(frozen=True)
class GridSpec:
    lat_min: float = 40.0
    lat_max: float = 60.0
    lon_min: float = -8.0
    lon_max: float = 26.0
    resolution: float = 0.02

    @property
    def n_lat(self) -> int:
        return int(round((self.lat_max - self.lat_min) / self.resolution)) + 1

    @property
    def n_lon(self) -> int:
        return int(round((self.lon_max - self.lon_min) / self.resolution)) + 1

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_lat, self.n_lon)

    def lat_at(self, i: int) -> float:
        return self.lat_min + i * self.resolution

    def lon_at(self, j: int) -> float:
        return self.lon_min + j * self.resolution


TARGET_GRID = GridSpec()

# Ring-buffer retention for observation frames (seconds). Unlike forecast data
# (keep_runs=1) we keep a rolling window the UI can animate. Default 5 h
# (60 frames at 5-minute cadence).
RETENTION_SECONDS = int(os.environ.get("EUCOMP_RETENTION_SECONDS", str(5 * 3600)))
RETENTION_MAX_FRAMES = int(os.environ.get("EUCOMP_RETENTION_MAX_FRAMES", "0")) or None
