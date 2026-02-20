from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict, deque
from threading import Lock
from typing import Any, Deque, Dict, Optional, Tuple


@dataclass
class AppState:
    # fallback metrics/state
    fallback_stats: Dict[str, int] = field(default_factory=dict)
    fallback_current: Dict[str, Any] = field(default_factory=lambda: {"updatedAt": None, "endpoints": {}})

    # strict EU resolve + missing-data backoff
    eu_strict_cache: "OrderedDict[tuple[str, float], tuple[Optional[tuple[str, int, str]], float]]" = field(default_factory=OrderedDict)
    eu_missing_until_mono: float = 0.0

    # location search state
    last_nominatim_request: float = 0.0
    location_search_lock: Lock = field(default_factory=Lock)
    location_search_rate: Dict[str, Deque[float]] = field(default_factory=dict)
    location_search_cache: "OrderedDict[tuple[str, int], tuple[dict, float]]" = field(default_factory=OrderedDict)

    # feedback anti-spam state
    feedback_rates: Dict[str, Deque[float]] = field(default_factory=dict)
