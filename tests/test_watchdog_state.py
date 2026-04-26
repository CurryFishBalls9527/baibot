"""Tests for the watchdog state file (offsets, dedupe, atomic write)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradingagents.watchdog import state as state_mod


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("BAIBOT_RESULTS_DIR", str(tmp_path))
    yield


def test_load_state_creates_default_when_missing(tmp_path):
    s = state_mod.load_state(
        default_event_path=tmp_path / "events.jsonl",
        default_log_path=tmp_path / "service.log",
    )
    assert s.events_jsonl.offset == 0
    assert s.alert_dedupe == {}


def test_save_then_load_round_trip(tmp_path):
    s = state_mod.load_state(
        default_event_path=tmp_path / "events.jsonl",
        default_log_path=tmp_path / "service.log",
    )
    s.events_jsonl.offset = 1234
    s.events_jsonl.inode = 99
    state_mod.mark_alerted(s, "drift:mechanical:AAOI:2026-04-24")
    s.last_tick["event_tail"] = "2026-04-24T19:00:00Z"
    state_mod.save_state(s)

    s2 = state_mod.load_state(
        default_event_path=tmp_path / "events.jsonl",
        default_log_path=tmp_path / "service.log",
    )
    assert s2.events_jsonl.offset == 1234
    assert s2.events_jsonl.inode == 99
    assert "drift:mechanical:AAOI:2026-04-24" in s2.alert_dedupe
    assert s2.last_tick["event_tail"] == "2026-04-24T19:00:00Z"


def test_already_alerted_within_ttl(tmp_path):
    s = state_mod.load_state(
        default_event_path=tmp_path / "events.jsonl",
        default_log_path=tmp_path / "service.log",
    )
    state_mod.mark_alerted(s, "x")
    assert state_mod.already_alerted(s, "x", ttl_hours=24) is True
    assert state_mod.already_alerted(s, "y", ttl_hours=24) is False


def test_already_alerted_past_ttl(tmp_path):
    s = state_mod.load_state(
        default_event_path=tmp_path / "events.jsonl",
        default_log_path=tmp_path / "service.log",
    )
    # Inject a 30-day-old timestamp directly.
    old = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    s.alert_dedupe["stale"] = old
    assert state_mod.already_alerted(s, "stale", ttl_hours=24) is False


def test_save_prunes_stale_dedupe_entries(tmp_path):
    s = state_mod.load_state(
        default_event_path=tmp_path / "events.jsonl",
        default_log_path=tmp_path / "service.log",
    )
    old = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    s.alert_dedupe["old"] = old
    state_mod.mark_alerted(s, "fresh")
    state_mod.save_state(s)

    s2 = state_mod.load_state(
        default_event_path=tmp_path / "events.jsonl",
        default_log_path=tmp_path / "service.log",
    )
    assert "fresh" in s2.alert_dedupe
    assert "old" not in s2.alert_dedupe


def test_atomic_write_uses_tmp_rename(tmp_path):
    s = state_mod.load_state(
        default_event_path=tmp_path / "events.jsonl",
        default_log_path=tmp_path / "service.log",
    )
    state_mod.save_state(s)
    state_path = state_mod.state_path()
    assert state_path.exists()
    # No leftover .tmp file
    assert not state_path.with_suffix(state_path.suffix + ".tmp").exists()
