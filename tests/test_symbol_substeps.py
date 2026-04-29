from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import numpy as np


BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from services.symbol_compute import compute_symbols_payload  # noqa: E402
from symbol_logic import aggregate_symbol_cell  # noqa: E402


def _symbol_arrays(*, substep_minutes: int, include_precomputed: bool, live_fields: bool = False) -> dict:
    lat = np.array([47.8, 48.0, 48.2], dtype=np.float32)
    lon = np.array([10.8, 11.0, 11.2], dtype=np.float32)
    shape = (3, 3)
    valid = datetime(2026, 4, 29, 12, 0)
    if substep_minutes:
        valid += timedelta(minutes=substep_minutes)

    cape = 20.0 if live_fields else 0.0
    out = {
        "lat": lat,
        "lon": lon,
        "validTime": valid.isoformat() + "Z",
        "ww": np.zeros(shape, dtype=np.float32),
        "ceiling": np.zeros(shape, dtype=np.float32),
        "clcl": np.zeros(shape, dtype=np.float32),
        "clcm": np.zeros(shape, dtype=np.float32),
        "clch": np.zeros(shape, dtype=np.float32),
        "cape_ml": np.full(shape, cape, dtype=np.float32),
        "cape_ml_hourly_max": np.zeros(shape, dtype=np.float32),
        "cin_ml": np.zeros(shape, dtype=np.float32),
        "htop_dc": np.full(shape, 1800.0, dtype=np.float32),
        "hbas_sc": np.full(shape, 1000.0, dtype=np.float32),
        "hbas_sc_hourly_max": np.zeros(shape, dtype=np.float32),
        "htop_sc": np.full(shape, 1800.0, dtype=np.float32),
        "htop_sc_hourly_max": np.zeros(shape, dtype=np.float32),
        "lpi": np.zeros(shape, dtype=np.float32),
        "lpi_max": np.zeros(shape, dtype=np.float32),
        "hsurf": np.zeros(shape, dtype=np.float32),
        "mh": np.zeros(shape, dtype=np.float32),
    }
    if include_precomputed:
        out["sym_code"] = np.zeros(shape, dtype=np.int16)
        out["cb_hm"] = np.full(shape, -1, dtype=np.int16)
    return out


def _compute(*, substep_minutes: int, substep_mode: bool = False):
    calls = []

    def resolve_time(time, model):
        return "2026042912", 1, "icon_d2"

    def load_data(run, step, model, keys=None, substep_minutes=0):
        calls.append({"keys": set(keys or []), "substep": substep_minutes})
        return _symbol_arrays(
            substep_minutes=substep_minutes,
            include_precomputed=("sym_code" in set(keys or [])),
            live_fields=substep_mode,
        )

    payload = compute_symbols_payload(
        zoom=12,
        bbox="47.99,10.99,48.01,11.01",
        time="2026-04-29T12:00:00Z",
        model="icon_d2",
        symbol_mode="native",
        resolve_time_with_cache_context=resolve_time,
        load_data=load_data,
        load_eu_data_strict=lambda *args, **kwargs: None,
        freshness_minutes_from_run=lambda run: 0,
        strict_window_hours=0.5,
        load_coverage_damping_cfg=lambda: {"enabled": False},
        substep_minutes=substep_minutes,
        substep_mode=substep_mode,
    )
    return payload, calls


def test_hourly_symbols_keep_precomputed_path():
    payload, calls = _compute(substep_minutes=0)

    assert calls[0]["substep"] == 0
    assert "sym_code" in calls[0]["keys"]
    assert payload["diagnostics"]["substepMinutes"] == 0
    assert payload["diagnostics"]["substepMode"] is False
    assert {s["type"] for s in payload["symbols"]} == {"clear"}


def test_substep_symbols_use_live_substep_fields():
    payload, calls = _compute(substep_minutes=15, substep_mode=True)

    assert calls[0]["substep"] == 15
    assert "sym_code" not in calls[0]["keys"]
    assert "lpi" in calls[0]["keys"]
    assert payload["validTime"] == "2026-04-29T12:15:00Z"
    assert payload["diagnostics"]["substepMinutes"] == 15
    assert payload["diagnostics"]["substepMode"] is True
    assert {s["type"] for s in payload["symbols"]} == {"cu_hum"}


def test_substep_mode_minute_zero_uses_live_symbol_fields():
    payload, calls = _compute(substep_minutes=0, substep_mode=True)

    assert calls[0]["substep"] == 0
    assert "sym_code" not in calls[0]["keys"]
    assert "lpi" in calls[0]["keys"]
    assert payload["validTime"] == "2026-04-29T12:00:00Z"
    assert payload["diagnostics"]["substepMinutes"] == 0
    assert payload["diagnostics"]["substepMode"] is True
    assert {s["type"] for s in payload["symbols"]} == {"cu_hum"}


def test_substep_live_aggregation_uses_cell_local_weather_indices():
    shape = (3, 3)
    cli = np.array([1, 2], dtype=int)
    clo = np.array([1, 2], dtype=int)
    ww = np.zeros(shape, dtype=np.float32)
    ww[np.ix_(cli, clo)] = 2.0
    ceiling = np.zeros(shape, dtype=np.float32)
    ceiling[np.ix_(cli, clo)] = 1500.0
    zeros = np.zeros(shape, dtype=np.float32)

    sym, cb_hm, best_i, best_j = aggregate_symbol_cell(
        cli=cli,
        clo=clo,
        cell_ww=ww[np.ix_(cli, clo)],
        ceil_arr=ceiling,
        c_clcl=zeros,
        c_clcm=zeros,
        c_clch=zeros,
        c_cape_hourly_max=zeros,
        c_htop_dc=zeros,
        c_hbas_sc_hourly_max=zeros,
        c_htop_sc_hourly_max=zeros,
        c_lpi_max=zeros,
        c_hsurf=zeros,
        c_mh=zeros,
        classify_point_fn=lambda **_kwargs: "clear",
        zoom=10,
        pre_has_cape=False,
        pre_has_ceil=True,
    )

    assert sym == "st"
    assert cb_hm == 15
    assert best_i in cli
    assert best_j in clo
