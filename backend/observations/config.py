"""Central configuration for the SkyView observation layer (radar + satellite).

Ports the standalone ``eucomposites`` scaffold into SkyView. All endpoints and
product identifiers were verified against the EUMETNET ORD API documentation and
the EUMETSAT Data Store as of June 2026. Treat these as defaults overridable via
environment variables.

Data is written under the repository ``data/observations/`` tree by default so it
sits alongside the ICON Zarr stores and is covered by ``.gitignore``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Output layout — anchored at <repo>/data/observations by default.
# backend/observations/config.py -> parents[2] == repo root.
# --------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "observations"

DATA_ROOT = Path(
    os.environ.get("EUCOMP_DATA_ROOT")
    or os.environ.get("SKYVIEW_OBS_DATA_ROOT")
    or str(_DEFAULT_DATA_ROOT)
).expanduser()

# Native source-of-truth files (ODIM HDF5 / NetCDF) land here in Phase 0/1.
RADAR_DIR = DATA_ROOT / "radar"
SAT_DIR = DATA_ROOT / "satellite"
STATE_DIR = DATA_ROOT / ".state"  # last-seen markers for the poller


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
    #   MSG RSS Level 1.5:   "EO:EUM:DAT:MSG:HRSEVIRI-RSS"   (5-min, Europe)
    #   MTG FCI Level 1c:    "EO:EUM:DAT:0662"               (10-min)
    collection_id: str = os.environ.get(
        "EUCOMP_SAT_COLLECTION", "EO:EUM:DAT:MSG:HRSEVIRI-RSS"
    )
    cadence_seconds: int = 300
    poll_interval: int = 60

    # Region of interest (lon/lat bbox) for tailoring/cropping to Europe.
    roi_bbox: tuple[float, float, float, float] = (-15.0, 32.0, 45.0, 72.0)


@dataclass(frozen=True)
class Config:
    radar: RadarConfig = field(default_factory=RadarConfig)
    satellite: SatelliteConfig = field(default_factory=SatelliteConfig)


def ensure_dirs() -> None:
    for d in (RADAR_DIR, SAT_DIR, STATE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Target grid for reprojection / serving
# --------------------------------------------------------------------------
# Observations are reprojected onto a regular lat/lon grid (the same convention
# as ICON Zarr output) so they reuse SkyView's overlay-tile pipeline. The bounds
# mirror ``d2_bounds`` in backend/ingest_config.yaml and the resolution matches
# ICON-D2 native (0.02° ≈ 2 km), giving a 746×1215 grid over the SkyView region.
@dataclass(frozen=True)
class GridSpec:
    lat_min: float = 43.18
    lat_max: float = 58.08
    lon_min: float = -3.94
    lon_max: float = 20.34
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
# (keep_runs=1) we keep a rolling window the UI can animate. Default 3 h.
RETENTION_SECONDS = int(os.environ.get("EUCOMP_RETENTION_SECONDS", str(3 * 3600)))
RETENTION_MAX_FRAMES = int(os.environ.get("EUCOMP_RETENTION_MAX_FRAMES", "0")) or None
