"""Pure tests for MTG-I1 FCI L1c product helpers."""

from __future__ import annotations

import datetime as dt
import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from observations.satellite_mtg import (  # noqa: E402
    chunk_number,
    is_trailer,
    parse_chunk_spec,
    product_sensing_time,
    read_fci_valid_time,
    select_europe_chunks,
)


# A realistic WMO-format FCI L1c HRFI chunk filename: the sensing window is the
# pair after `_OPE_`; we key the frame on the sensing start.
_FCI_CHUNK = (
    "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-HRFI-FD--CHK-BODY--DIS-NC4E"
    "_C_EUMT_20260612120100_L2PF_OPE_20260612115917_20260612120000_N_JLS_C_0073_0067.nc"
)


def _chunk_name(num: int, *, kind: str = "BODY") -> str:
    return (
        f"W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-HRFI-FD--CHK-{kind}--DIS-NC4E"
        f"_C_EUMT_20260612120100_L2PF_OPE_20260612115917_20260612120000_N_JLS_C_0073_{num:04d}.nc"
    )


def test_read_fci_valid_time_uses_sensing_start():
    got = read_fci_valid_time(_FCI_CHUNK)
    assert got == dt.datetime(2026, 6, 12, 11, 59, 17, tzinfo=dt.timezone.utc)


def test_read_fci_valid_time_returns_none_for_unknown_name():
    assert read_fci_valid_time("not-an-fci-product.nc") is None


def test_product_sensing_time_prefers_sensing_start_attr():
    class FakeProduct:
        sensing_start = dt.datetime(2026, 6, 12, 12, 0, 0)

        def __str__(self):  # pragma: no cover - not used when attr present
            return "ignored"

    got = product_sensing_time(FakeProduct())
    assert got == dt.datetime(2026, 6, 12, 12, 0, 0, tzinfo=dt.timezone.utc)


def test_product_sensing_time_falls_back_to_identifier():
    class FakeProduct:
        sensing_start = None

        def __str__(self):
            return _FCI_CHUNK

    got = product_sensing_time(FakeProduct())
    assert got == dt.datetime(2026, 6, 12, 11, 59, 17, tzinfo=dt.timezone.utc)


def test_chunk_number_and_trailer():
    assert chunk_number(_chunk_name(40)) == 40
    assert chunk_number("no-chunk-here.nc") is None
    assert is_trailer(_chunk_name(41, kind="TRAIL")) is True
    assert is_trailer(_chunk_name(40)) is False


def test_parse_chunk_spec():
    assert parse_chunk_spec("29-32,40") == {29, 30, 31, 32, 40}
    assert parse_chunk_spec("") is None
    assert parse_chunk_spec(None) is None


def test_select_europe_chunks_keeps_northern_fraction_plus_trailer():
    # 40 body chunks (1..40) + a named trailer. North = high numbers; 0.25 keeps
    # the top 10 body chunks (31..40), and the trailer is always included.
    entries = [_chunk_name(n) for n in range(1, 41)] + [_chunk_name(41, kind="TRAIL")]
    selected = select_europe_chunks(entries, fraction=0.25)
    nums = sorted(chunk_number(e) for e in selected if not is_trailer(e))
    assert nums == list(range(31, 41))
    assert any(is_trailer(e) for e in selected)


def test_select_europe_chunks_explicit_overrides_fraction():
    entries = [_chunk_name(n) for n in range(1, 41)] + [_chunk_name(41, kind="TRAIL")]
    selected = select_europe_chunks(entries, fraction=0.25, explicit={29, 30, 40})
    nums = sorted(chunk_number(e) for e in selected if not is_trailer(e))
    assert nums == [29, 30, 40]


def test_select_europe_chunks_unknown_layout_falls_back_to_all():
    entries = ["weird_product_part_a.nc", "weird_product_part_b.nc"]
    assert select_europe_chunks(entries) == sorted(entries)
