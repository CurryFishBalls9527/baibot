"""Regression test for the intraday entry scan candidate-set gating.

Bug context (2026-04-28):
    `run_intraday_entry_scan` iterated `preflight.screened_symbols`, which is
    the union of every screener row (incl. rows the screener actively
    rejected). The screener still populates buy_point/buy_limit_price on
    `no_base` / `late_stage` rows, so `_trade_rule_based_setup` happily
    fired on UNH (rs=2.08, candidate_status=no_base) once price entered
    the placeholder buy zone. Result: real fills on the mechanical and
    mechanical_v2 paper accounts.

This test pins the fix: only iterate `preflight.approved_symbols`
(rule_watch_candidate=True), which is the same set the daily-analysis
path uses.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from tradingagents.automation.orchestrator import Orchestrator
from tradingagents.automation.prescreener import MinerviniPreflight


def _make_preflight() -> MinerviniPreflight:
    rows = [
        # Rule-entry candidate — should be iterated.
        {
            "symbol": "MU",
            "rule_entry_candidate": True,
            "rule_watch_candidate": True,
            "candidate_status": "leader_continuation_actionable",
            "rs_percentile": 97.92,
            "buy_point": 100.0,
            "buy_limit_price": 107.0,
            "stage_number": 2,
            "base_label": "vcp",
        },
        # Watch-only candidate — should also be iterated (price may
        # tick into buy zone intraday).
        {
            "symbol": "WULF",
            "rule_entry_candidate": False,
            "rule_watch_candidate": True,
            "candidate_status": "leader_continuation_watch",
            "rs_percentile": 95.83,
            "buy_point": 21.6,
            "buy_limit_price": 23.11,
            "stage_number": 2,
            "base_label": "vcp",
        },
        # Rejected by screener — must NOT be iterated even though
        # buy_point/buy_limit_price are populated.
        {
            "symbol": "UNH",
            "rule_entry_candidate": False,
            "rule_watch_candidate": False,
            "candidate_status": "no_base",
            "rs_percentile": 2.08,
            "buy_point": 358.56,
            "buy_limit_price": 383.66,
            "stage_number": 1,
            "base_label": "none",
        },
    ]
    df = pd.DataFrame(rows)
    return MinerviniPreflight(
        trade_date="2026-04-28",
        market_regime="confirmed_uptrend",
        confirmed_uptrend=True,
        approved_symbols=["MU", "WULF"],
        blocked_symbols=["UNH"],
        screened_symbols=["MU", "WULF", "UNH"],
        screen_df=df,
    )


def _make_orchestrator():
    o = Orchestrator.__new__(Orchestrator)
    o.broker = MagicMock()
    o.broker.is_market_open.return_value = True
    o.broker.get_account.return_value = SimpleNamespace(
        equity=100_000.0, cash=50_000.0, portfolio_value=100_000.0,
    )
    o.broker.get_positions.return_value = []
    o.db = MagicMock()
    o.db.was_stopped_today.return_value = False
    o.config = {"variant_name": "mechanical"}
    o.notifier = SimpleNamespace(enabled=False, send=lambda *a, **kw: None)
    o._latest_minervini_preflight = _make_preflight()
    o._stock_positions = lambda positions: []
    o.reconcile_orders = MagicMock(return_value={})
    o._trade_rule_based_setup = MagicMock(
        return_value={"symbol": "?", "action": "SKIP", "traded": False}
    )
    return o


def test_intraday_scan_skips_screener_rejects():
    """UNH (rule_watch=False, no_base) must not reach _trade_rule_based_setup."""
    orch = _make_orchestrator()

    orch.run_intraday_entry_scan()

    called_symbols = {
        call.args[0]["symbol"] for call in orch._trade_rule_based_setup.call_args_list
    }
    assert "UNH" not in called_symbols, (
        f"UNH (rule_watch=False, no_base) leaked into entry scan: {called_symbols}"
    )
    assert called_symbols == {"MU", "WULF"}, (
        f"Expected only approved candidates iterated; got {called_symbols}"
    )


def test_intraday_scan_iterates_watch_candidates():
    """Watch-only candidates (rule_watch=True, rule_entry=False) must still
    be checked — price can tick into the buy zone intraday."""
    orch = _make_orchestrator()

    orch.run_intraday_entry_scan()

    called_symbols = {
        call.args[0]["symbol"] for call in orch._trade_rule_based_setup.call_args_list
    }
    assert "WULF" in called_symbols


def test_rs_defensive_gate_blocks_low_rs_setups():
    """Defensive RS gate inside _trade_rule_based_setup must REFUSE the
    trade (not just emit a warning) when rs_percentile is below the
    configured threshold. Second line of defence if the caller-side
    candidate filter regresses."""
    orch = _make_orchestrator()
    # Reinstate the real method so we exercise the RS guard.
    del orch._trade_rule_based_setup
    orch.config = {
        "variant_name": "mechanical",
        "minervini_min_rs_percentile": 70,
        "minervini_max_stage_number": 3,
        "minervini_use_close_range_filter": False,
    }
    # _execute_structured_signal must NOT be called for a blocked RS entry.
    orch._execute_structured_signal = MagicMock(
        return_value={"action": "BUY", "traded": True}
    )
    orch._is_leader_continuation_setup = lambda setup: False
    orch._entries_allowed_for_setup = lambda setup, regime: True
    orch._current_exposure = lambda account, positions: 0.5
    orch._target_exposure_for_setup = lambda setup, regime: 0.7
    orch._find_position = lambda positions, symbol: None
    orch.broker.get_latest_price = MagicMock(return_value=362.88)
    orch._to_float = lambda v: float(v) if v is not None else None

    setup = {
        "symbol": "UNH",
        "rule_entry_candidate": False,
        "candidate_status": "no_base",
        "rs_percentile": 2.08,
        "buy_point": 358.56,
        "buy_limit_price": 383.66,
        "stage_number": 1,
        "base_label": "none",
        "initial_stop_pct": 0.05,
        "initial_stop_price": 340.0,
        "market_regime": "confirmed_uptrend",
    }
    account = SimpleNamespace(equity=100_000.0)

    result = orch._trade_rule_based_setup(setup, account, [])

    assert result["traded"] is False
    assert result["action"] == "SKIP"
    assert "RS gate" in result["screen_rejected"]
    orch._execute_structured_signal.assert_not_called()
