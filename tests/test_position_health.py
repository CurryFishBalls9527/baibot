"""Tests for PR1 round-3 — open-position health snapshot + prompt + markdown."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.automation.position_health import (
    collect_position_snapshot,
    compose_health_markdown,
    render_health_prompt,
)


def _fake_db(state: dict | None):
    db = MagicMock()
    db.get_position_state.return_value = state
    return db


def _fake_position(symbol="AAPL", current_price=150.0, qty=10, side="long"):
    return SimpleNamespace(
        symbol=symbol, current_price=current_price, qty=qty, side=side,
    )


# ───────────────────────────── collect_position_snapshot ─────────────────────────────

class TestCollectSnapshot:
    def test_happy_path_shape(self):
        state = {
            "entry_price": 100.0,
            "entry_date": (date.today() - timedelta(days=3)).isoformat(),
            "highest_close": 120.0,
            "current_stop": 90.0,
            "base_pattern": "VCP",
            "regime_at_entry": "RISK_ON",
            "rs_at_entry": 85,
            "stage_at_entry": 2,
        }
        db = _fake_db(state)
        feats = {"sma_50": 110.0, "rs_percentile": 78.0, "adx_14": 25.0}
        snap = collect_position_snapshot(
            db=db, symbol="AAPL",
            position=_fake_position(current_price=130.0),
            variant="mechanical_v2",
            features_fn=lambda s: feats,
        )
        for key in (
            "symbol", "variant", "entry_price", "entry_date", "current_price",
            "highest_close_since_entry", "current_stop", "base_pattern",
            "regime_at_entry", "rs_at_entry", "stage_at_entry", "hold_days",
            "unrealized_pct", "mfe_so_far_pct", "distance_to_stop_pct",
            "unrealized_pnl", "features", "distance_to_50dma_pct", "rs_delta",
        ):
            assert key in snap, f"missing key: {key}"

    def test_derived_numbers(self):
        entry = (date.today() - timedelta(days=5)).isoformat()
        state = {
            "entry_price": 100.0, "entry_date": entry,
            "highest_close": 120.0, "current_stop": 90.0,
            "rs_at_entry": 80, "base_pattern": "VCP",
        }
        snap = collect_position_snapshot(
            db=_fake_db(state), symbol="AAPL",
            position=_fake_position(current_price=115.0, qty=10),
            variant="mechanical",
            features_fn=lambda s: {"sma_50": 105.0, "rs_percentile": 70},
        )
        assert snap["hold_days"] == 5
        assert snap["unrealized_pct"] == pytest.approx(0.15, rel=1e-3)
        assert snap["mfe_so_far_pct"] == pytest.approx(0.20, rel=1e-3)
        # (115-90)/115
        assert snap["distance_to_stop_pct"] == pytest.approx(0.2174, rel=1e-3)
        # (115-105)/105
        assert snap["distance_to_50dma_pct"] == pytest.approx(0.0952, rel=1e-3)
        assert snap["unrealized_pnl"] == pytest.approx(150.0)
        assert snap["rs_delta"] == pytest.approx(-10.0)

    def test_missing_state_degrades_gracefully(self):
        snap = collect_position_snapshot(
            db=_fake_db({}), symbol="AAPL",
            position=_fake_position(current_price=100.0, qty=1),
            variant="mechanical",
        )
        assert snap["unrealized_pct"] is None
        assert snap["hold_days"] is None
        assert snap["distance_to_stop_pct"] is None
        assert snap["distance_to_50dma_pct"] is None
        assert snap["rs_delta"] is None
        assert snap["features"] == {}

    def test_db_error_does_not_raise(self):
        db = MagicMock()
        db.get_position_state.side_effect = RuntimeError("boom")
        snap = collect_position_snapshot(
            db=db, symbol="AAPL",
            position=_fake_position(current_price=100.0),
            variant="mechanical",
        )
        assert snap["entry_price"] is None
        assert snap["unrealized_pct"] is None

    def test_features_fn_error_does_not_raise(self):
        state = {"entry_price": 100.0, "rs_at_entry": 70}
        def bad(_s):
            raise RuntimeError("features down")
        snap = collect_position_snapshot(
            db=_fake_db(state), symbol="AAPL",
            position=_fake_position(current_price=110.0),
            variant="mechanical", features_fn=bad,
        )
        assert snap["features"] == {}
        assert snap["rs_delta"] is None
        assert snap["distance_to_50dma_pct"] is None


# ───────────────────────────── render_health_prompt ─────────────────────────────

class TestRenderPrompt:
    def test_missing_fields_render_as_na(self):
        snap = {"symbol": "AAPL", "variant": "mechanical"}
        out = render_health_prompt(snap)
        assert "AAPL" in out
        assert "n/a" in out
        assert "### Health:" in out  # the output-format instruction line

    def test_percent_formatting(self):
        snap = {
            "symbol": "AAPL", "variant": "mechanical",
            "unrealized_pct": 0.1234,
            "distance_to_stop_pct": -0.05,
            "mfe_so_far_pct": 0.30,
            "features": {"rs_percentile": 80},
        }
        out = render_health_prompt(snap)
        assert "+12.34%" in out
        assert "-5.00%" in out
        assert "+30.00%" in out


# ───────────────────────────── compose_health_markdown ─────────────────────────────

class TestComposeMarkdown:
    def test_header_and_snapshot_present(self):
        snap = {
            "symbol": "AAPL", "variant": "mechanical",
            "hold_days": 7, "unrealized_pct": 0.10,
            "entry_price": 100.0, "current_price": 110.0,
            "current_stop": 95.0, "distance_to_stop_pct": 0.1364,
            "mfe_so_far_pct": 0.15, "highest_close_since_entry": 115.0,
            "base_pattern": "VCP", "regime_at_entry": "RISK_ON",
        }
        body = "### Health: HEALTHY\n**Read:** intact\n**Watch:** 95\n**Action:** hold"
        md = compose_health_markdown(snap, body)
        assert "# AAPL (mechanical)" in md
        assert "held 7 days" in md
        assert "+10.00%" in md
        assert "$100.00" in md
        assert "$95.00" in md
        assert "VCP" in md
        assert "### Health: HEALTHY" in md
        assert "**Action:** hold" in md

    def test_handles_missing_numbers(self):
        snap = {
            "symbol": "NVDA", "variant": "chan",
            "hold_days": None, "unrealized_pct": None,
            "entry_price": None, "current_price": None,
            "current_stop": None, "distance_to_stop_pct": None,
            "mfe_so_far_pct": None, "highest_close_since_entry": None,
        }
        md = compose_health_markdown(snap, "")
        assert "n/a" in md
        assert "# NVDA (chan)" in md


# ───────────────────────────── run_held_position_review ─────────────────────────────

class TestRunHeldPositionReview:
    def test_kill_switch_short_circuits(self):
        from tradingagents.automation.trade_review import run_held_position_review
        # broker MagicMock asserts not called when disabled.
        broker = MagicMock()
        res = run_held_position_review(
            db=MagicMock(), broker=broker, variant_name="mechanical",
            config={"held_position_review_enabled": False},
        )
        assert res["status"] == "disabled"
        broker.get_positions.assert_not_called()

    def test_no_positions_returns_zero(self, tmp_path, monkeypatch):
        from tradingagents.automation import trade_review
        monkeypatch.chdir(tmp_path)
        broker = MagicMock()
        broker.get_positions.return_value = []
        res = trade_review.run_held_position_review(
            db=MagicMock(), broker=broker, variant_name="mechanical",
            config={"held_position_review_dry_run": True},
        )
        assert res["held"] == 0
        assert res["analyzed"] == 0
        assert res["dry_run"] is True

    def test_dry_run_writes_to_dryrun_dir(self, tmp_path, monkeypatch):
        from tradingagents.automation import trade_review
        monkeypatch.chdir(tmp_path)
        fake_position = _fake_position("AAPL", current_price=110.0, qty=10)
        broker = MagicMock()
        broker.get_positions.return_value = [fake_position]

        state = {
            "entry_price": 100.0,
            "entry_date": (date.today() - timedelta(days=2)).isoformat(),
            "highest_close": 112.0, "current_stop": 95.0,
            "base_pattern": "VCP", "rs_at_entry": 80,
        }
        db = _fake_db(state)

        # Stub the LLM client factory so we don't hit OpenAI.
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = SimpleNamespace(
            content=(
                "### Health: HEALTHY\n"
                "**Read:** trend intact above 50-DMA.\n"
                "**Watch:** break of $105.\n"
                "**Action:** hold, raise stop to $100."
            )
        )
        fake_client = MagicMock()
        fake_client.get_llm.return_value = fake_llm
        review_day = date(2026, 4, 22)
        with patch("tradingagents.llm_clients.factory.create_llm_client",
                   return_value=fake_client):
            res = trade_review.run_held_position_review(
                db=db, broker=broker, variant_name="mechanical",
                config={"held_position_review_dry_run": True},
                review_date=review_day,
                features_fn=lambda s: {"sma_50": 105.0, "rs_percentile": 75},
            )
        assert res["held"] == 1
        assert res["analyzed"] == 1
        out_file = (
            tmp_path / "results" / "daily_reviews_dryrun"
            / review_day.isoformat() / "mechanical_AAPL_HELD.md"
        )
        assert out_file.exists()
        contents = out_file.read_text()
        assert "### Health: HEALTHY" in contents
        assert "AAPL" in contents

    def test_budget_cap_respected(self, tmp_path, monkeypatch):
        from tradingagents.automation import trade_review
        monkeypatch.chdir(tmp_path)
        positions = [
            _fake_position(f"SYM{i}", current_price=100.0, qty=1) for i in range(5)
        ]
        broker = MagicMock()
        broker.get_positions.return_value = positions
        db = _fake_db({"entry_price": 90.0, "rs_at_entry": 80})

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = SimpleNamespace(content="### Health: WATCH")
        fake_client = MagicMock()
        fake_client.get_llm.return_value = fake_llm
        with patch("tradingagents.llm_clients.factory.create_llm_client",
                   return_value=fake_client):
            res = trade_review.run_held_position_review(
                db=db, broker=broker, variant_name="mechanical",
                config={"held_position_review_dry_run": True,
                        "held_review_max_calls": 2},
                review_date=date(2026, 4, 22),
            )
        assert res["held"] == 5
        assert res["analyzed"] == 2
        assert res["skipped_budget"] == 3

    def test_llm_failure_does_not_stop_loop(self, tmp_path, monkeypatch):
        from tradingagents.automation import trade_review
        monkeypatch.chdir(tmp_path)
        positions = [
            _fake_position("AAA", current_price=100.0, qty=1),
            _fake_position("BBB", current_price=100.0, qty=1),
        ]
        broker = MagicMock()
        broker.get_positions.return_value = positions
        db = _fake_db({"entry_price": 90.0})

        fake_llm = MagicMock()
        fake_llm.invoke.side_effect = RuntimeError("rate limit")
        fake_client = MagicMock()
        fake_client.get_llm.return_value = fake_llm
        with patch("tradingagents.llm_clients.factory.create_llm_client",
                   return_value=fake_client):
            res = trade_review.run_held_position_review(
                db=db, broker=broker, variant_name="mechanical",
                config={"held_position_review_dry_run": True},
                review_date=date(2026, 4, 22),
            )
        # Both positions still produce files — LLM failure → inline error note.
        assert res["held"] == 2
        assert res["analyzed"] == 2
        assert res["failed_llm"] == 2
        for sym in ("AAA", "BBB"):
            p = (
                tmp_path / "results" / "daily_reviews_dryrun"
                / "2026-04-22" / f"mechanical_{sym}_HELD.md"
            )
            assert p.exists()
            assert "LLM health read failed" in p.read_text()
