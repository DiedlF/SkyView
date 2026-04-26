"""Small ICAO-code lookup helpers for marker location search."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

ICAO_QUERY_RE = re.compile(r"^[A-Z]{4}$")
ICAO_IN_TEXT_RE = re.compile(r"\b[A-Z]{4}\b")


CURATED_AIRPORTS: List[Dict[str, Any]] = [
    {"icao": "EDDM", "name": "Munich Airport", "displayName": "Munich Airport EDDM, DE", "lat": 48.3538, "lon": 11.7861, "type": "airport", "country": "DE"},
    {"icao": "EDNY", "name": "Friedrichshafen Airport", "displayName": "Friedrichshafen Airport EDNY, DE", "lat": 47.6713, "lon": 9.5115, "type": "airport", "country": "DE"},
    {"icao": "EDMA", "name": "Augsburg Airport", "displayName": "Augsburg Airport EDMA, DE", "lat": 48.4253, "lon": 10.9317, "type": "airport", "country": "DE"},
    {"icao": "EDMO", "name": "Oberpfaffenhofen Airfield", "displayName": "Oberpfaffenhofen Airfield EDMO, DE", "lat": 48.0814, "lon": 11.2831, "type": "airfield", "country": "DE"},
    {"icao": "EDJA", "name": "Memmingen Airport", "displayName": "Memmingen Airport EDJA, DE", "lat": 47.9888, "lon": 10.2395, "type": "airport", "country": "DE"},
    {"icao": "EDDS", "name": "Stuttgart Airport", "displayName": "Stuttgart Airport EDDS, DE", "lat": 48.6899, "lon": 9.2219, "type": "airport", "country": "DE"},
    {"icao": "EDTD", "name": "Donaueschingen-Villingen Airfield", "displayName": "Donaueschingen-Villingen Airfield EDTD, DE", "lat": 47.9733, "lon": 8.5222, "type": "airfield", "country": "DE"},
    {"icao": "EDQE", "name": "Burg Feuerstein Airfield", "displayName": "Burg Feuerstein Airfield EDQE, DE", "lat": 49.7942, "lon": 11.1336, "type": "airfield", "country": "DE"},
    {"icao": "EDMU", "name": "Unterwoessen Airfield", "displayName": "Unterwoessen Airfield EDMU, DE", "lat": 47.7389, "lon": 12.4564, "type": "airfield", "country": "DE"},
    {"icao": "LOWI", "name": "Innsbruck Airport", "displayName": "Innsbruck Airport LOWI, AT", "lat": 47.2602, "lon": 11.3439, "type": "airport", "country": "AT"},
    {"icao": "LOIJ", "name": "St. Johann in Tirol Airfield", "displayName": "St. Johann in Tirol Airfield LOIJ, AT", "lat": 47.5172, "lon": 12.4497, "type": "airfield", "country": "AT"},
    {"icao": "LOIH", "name": "Hohenems-Dornbirn Airfield", "displayName": "Hohenems-Dornbirn Airfield LOIH, AT", "lat": 47.3850, "lon": 9.7000, "type": "airfield", "country": "AT"},
    {"icao": "LOWZ", "name": "Zell am See Airport", "displayName": "Zell am See Airport LOWZ, AT", "lat": 47.2922, "lon": 12.7875, "type": "airport", "country": "AT"},
    {"icao": "LSZH", "name": "Zurich Airport", "displayName": "Zurich Airport LSZH, CH", "lat": 47.4581, "lon": 8.5481, "type": "airport", "country": "CH"},
    {"icao": "LSZS", "name": "Engadin Airport Samedan", "displayName": "Engadin Airport Samedan LSZS, CH", "lat": 46.5341, "lon": 9.8841, "type": "airport", "country": "CH"},
    {"icao": "LSZL", "name": "Locarno Airport", "displayName": "Locarno Airport LSZL, CH", "lat": 46.1610, "lon": 8.8780, "type": "airport", "country": "CH"},
    {"icao": "LFLP", "name": "Annecy Airport", "displayName": "Annecy Airport LFLP, FR", "lat": 45.9300, "lon": 6.1060, "type": "airport", "country": "FR"},
    {"icao": "LFNA", "name": "Gap-Tallard Airfield", "displayName": "Gap-Tallard Airfield LFNA, FR", "lat": 44.4620, "lon": 6.0370, "type": "airfield", "country": "FR"},
    {"icao": "LIDT", "name": "Trento-Mattarello Airfield", "displayName": "Trento-Mattarello Airfield LIDT, IT", "lat": 46.0210, "lon": 11.1260, "type": "airfield", "country": "IT"},
    {"icao": "LIMW", "name": "Aosta Airport", "displayName": "Aosta Airport LIMW, IT", "lat": 45.7380, "lon": 7.3680, "type": "airport", "country": "IT"},
]


def normalize_icao_query(query: str) -> Optional[str]:
    q = str(query or "").strip().upper()
    return q if ICAO_QUERY_RE.match(q) else None


def extract_icao(*values: Any) -> Optional[str]:
    for value in values:
        text = str(value or "").upper()
        match = ICAO_IN_TEXT_RE.search(text)
        if match:
            return match.group(0)
    return None


def item_icao(item: Dict[str, Any]) -> Optional[str]:
    explicit = str(item.get("icao") or "").strip().upper()
    if ICAO_QUERY_RE.match(explicit):
        return explicit
    return extract_icao(item.get("name"), item.get("displayName"), item.get("display_name"))


def curated_matches(icao: str) -> List[Dict[str, Any]]:
    code = str(icao or "").strip().upper()
    if not ICAO_QUERY_RE.match(code):
        return []
    return [dict(item) for item in CURATED_AIRPORTS if item["icao"] == code]


def seed_contains_icao(item: Dict[str, Any], icao: str) -> bool:
    return item_icao(item) == str(icao or "").strip().upper()


def known_icao_codes(items: Iterable[Dict[str, Any]]) -> set[str]:
    codes = {item["icao"] for item in CURATED_AIRPORTS}
    for item in items:
        code = item_icao(item)
        if code:
            codes.add(code)
    return codes
