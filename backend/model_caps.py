"""Shared model capability payloads."""

from __future__ import annotations


def get_models_payload() -> dict:
    return {
        "models": [
            {
                "name": "icon-d2",
                "label": "ICON-D2 (2.2km)",
                "maxHours": 48,
                "timesteps": list(range(1, 49)),
                "resolution": 2.2,
                "updateInterval": 3,
            },
            {
                "name": "icon-eu",
                "label": "ICON-EU (6.5km)",
                "maxHours": 120,
                "timesteps": list(range(49, 79)) + list(range(81, 121, 3)),
                "resolution": 6.5,
                "updateInterval": 6,
            },
        ]
    }
