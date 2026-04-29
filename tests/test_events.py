"""Tests for the structured-event emitter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradingagents.automation import events


@pytest.fixture(autouse=True)
def _isolate_results_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BAIBOT_RESULTS_DIR", str(tmp_path))
    yield


def _read_events(tmp_path: Path) -> list[dict]:
    p = tmp_path / "service_logs" / "events.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_emit_event_writes_jsonl_line(tmp_path):
    events.emit_event(
        events.Categories.ORDER_REJECT,
        level="error",
        variant="mechanical",
        symbol="AAOI",
        code="42210000",
        message="stale-quote retry exhausted",
        context={"sl_old": 22.71, "sl_new": 22.45},
    )
    rows = _read_events(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["category"] == "order_reject"
    assert row["level"] == "error"
    assert row["variant"] == "mechanical"
    assert row["symbol"] == "AAOI"
    assert row["code"] == "42210000"
    assert row["context"]["sl_new"] == 22.45
    assert row["fingerprint"] == "order_reject:mechanical:AAOI:42210000"
    assert "ts" in row and row["ts"].endswith("Z")
    assert row["pid"] > 0
    assert row["seq"] >= 1


def test_emit_event_appends_in_order(tmp_path):
    for i in range(3):
        events.emit_event(
            events.Categories.JOB_FAILED,
            message=f"failure {i}",
        )
    rows = _read_events(tmp_path)
    assert [r["message"] for r in rows] == ["failure 0", "failure 1", "failure 2"]
    # seq is monotonic
    seqs = [r["seq"] for r in rows]
    assert seqs == sorted(seqs)


def test_emit_event_invalid_level_falls_back_to_warning(tmp_path):
    events.emit_event(
        events.Categories.NAKED_POSITION,
        level="totally-bogus",
        message="x",
    )
    rows = _read_events(tmp_path)
    assert rows[0]["level"] == "warning"


def test_emit_event_never_raises(tmp_path, monkeypatch):
    # Force the write path to blow up — emit must still return cleanly.
    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("os.write", _boom)
    # No exception expected.
    events.emit_event(
        events.Categories.JOB_FAILED, message="should not raise"
    )


def test_emit_event_default_fingerprint(tmp_path):
    events.emit_event(
        events.Categories.DRIFT_DETECTED,
        variant="chan_v2",
        symbol="NVDA",
        message="ghost row",
    )
    row = _read_events(tmp_path)[0]
    assert row["fingerprint"] == "drift_detected:chan_v2:NVDA"


def test_categories_enum_stable():
    # Wire-protocol contract: these strings must not change without
    # bumping the watchdog. Lock them down.
    assert events.Categories.ORDER_REJECT == "order_reject"
    assert events.Categories.WASH_TRADE_REJECT == "wash_trade_reject"
    assert events.Categories.BRACKET_LEG_MISSING == "bracket_leg_missing"
    assert events.Categories.POSITION_STRANDED == "position_stranded"
    assert events.Categories.NAKED_POSITION == "naked_position"
    assert events.Categories.EXIT_SKIPPED == "exit_skipped"
    assert events.Categories.RS_FILTER_BYPASSED == "rs_filter_bypassed"
    assert events.Categories.ADD_ON_SUPERSESSION == "add_on_supersession"
    assert events.Categories.DRIFT_DETECTED == "drift_detected"
    assert events.Categories.RECONCILER_TICK == "reconciler_tick"
    assert events.Categories.JOB_FAILED == "job_failed"
    assert events.Categories.ACTIVITY_GAP == "activity_gap"
    assert events.Categories.STALE_QUOTE_RETRY == "stale_quote_retry"
