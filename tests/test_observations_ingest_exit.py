"""Unit tests for the observation ingest process exit code.

The cron/systemd unit surfaces this rc, so a run where nothing worked must be
visible as a failure while a quiet run (nothing new upstream, or every source
skipped inside its cadence window) must stay green. Pure: ``run_once`` is stubbed,
so no network or filesystem is touched.
"""

from __future__ import annotations

import os
import sys

import pytest

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from observations import ingest_obs  # noqa: E402


@pytest.fixture
def stub_run_once(monkeypatch):
    """Stub run_once with a canned (results, report) pair."""

    def _install(results: dict, attempted: set, errored: set):
        def fake_run_once(source, cfg=None, *, report=None):
            if report is not None:
                report["attempted"] = attempted
                report["errored"] = errored
            return results

        monkeypatch.setattr(ingest_obs, "run_once", fake_run_once)

    return _install


def test_exit_nonzero_when_every_attempted_source_failed(stub_run_once):
    stub_run_once({"radar": None, "mtg": None}, {"radar", "mtg"}, {"radar", "mtg"})
    assert ingest_obs.main(["--source", "all"]) == 1


def test_exit_zero_on_partial_failure(stub_run_once):
    # One source still produced a frame -> served data is current.
    stub_run_once({"radar": "202606031230", "mtg": None}, {"radar", "mtg"}, {"mtg"})
    assert ingest_obs.main(["--source", "all"]) == 0


def test_exit_zero_when_nothing_new_upstream(stub_run_once):
    # Attempted but benign: no product published yet, no exception raised.
    stub_run_once({"radar": None}, {"radar"}, set())
    assert ingest_obs.main(["--source", "all"]) == 0


def test_exit_zero_when_all_sources_cadence_skipped(stub_run_once):
    # Nothing attempted -> not a failure.
    stub_run_once({"radar": None}, set(), set())
    assert ingest_obs.main(["--source", "all"]) == 0


def test_exit_nonzero_for_single_failed_source(stub_run_once):
    stub_run_once({"satellite": None}, {"satellite"}, {"satellite"})
    assert ingest_obs.main(["--source", "satellite"]) == 1
