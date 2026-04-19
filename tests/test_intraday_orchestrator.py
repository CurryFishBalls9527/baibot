"""Tests for IntradayOrchestrator (NR4 + gap-reclaim mechanical)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


class _StubRiskResult:
    def __init__(self, passed=True, reason=""):
        self.passed = passed
        self.reason = reason


def _build_orchestrator(dry_run=True, universe=None, extra_config=None):
    """Build an IntradayOrchestrator with broker/sizer/risk/db stubbed."""
    tmpdir = tempfile.mkdtemp()
    db_path = str(Path(tmpdir) / "intraday_test.db")

    # Patch the heavy imports so __init__ doesn't hit real Alpaca / SQLite
    with patch("tradingagents.automation.intraday_orchestrator.AlpacaBroker") as Broker, \
         patch("tradingagents.automation.intraday_orchestrator.PositionSizer") as Sizer, \
         patch("tradingagents.automation.intraday_orchestrator.RiskEngine") as RiskEng, \
         patch("tradingagents.automation.intraday_orchestrator.PortfolioTracker") as Tracker, \
         patch("tradingagents.automation.intraday_orchestrator.build_notifier") as Notify:

        broker = MagicMock()
        broker.is_market_open.return_value = True
        broker.get_account.return_value = MagicMock(
            equity=100_000, cash=100_000, buying_power=100_000, daily_pl=0,
            daily_pl_pct=0,
        )
        broker.get_positions.return_value = []
        broker.get_open_orders.return_value = []
        broker.get_latest_price.return_value = 100.0
        broker.get_clock.return_value = MagicMock(is_open=True, next_open="", next_close="")
        broker.submit_bracket_order.return_value = MagicMock(
            status="filled", filled_qty=10, filled_avg_price=100.0,
            order_id="oid-1",
        )
        broker.close_position.return_value = MagicMock(
            status="submitted", filled_qty=0, filled_avg_price=None, order_id="cid-1",
        )
        Broker.return_value = broker

        sizer = MagicMock()
        sizer.calculate.return_value = MagicMock(
            symbol="AAPL", side="buy", qty=10, order_type="market",
            time_in_force="gtc",
        )
        Sizer.return_value = sizer

        risk = MagicMock()
        risk.check_order.return_value = _StubRiskResult(passed=True)
        RiskEng.return_value = risk

        Tracker.return_value = MagicMock()
        Notify.return_value = MagicMock()

        from tradingagents.automation.intraday_orchestrator import IntradayOrchestrator

        config = {
            "alpaca_api_key": "k", "alpaca_secret_key": "s",
            "paper_trading": True,
            "db_path": db_path,
            "variant_name": "intraday_mechanical",
            "intraday_dry_run": dry_run,
            "intraday_universe_symbols": universe or ["AAPL", "MSFT", "NVDA"],
            "intraday_backtest_config": {
                "max_positions": 4,
                "max_position_pct": 0.08,
                "stop_loss_pct": 0.03,
                "trail_stop_pct": 0.04,
                "interval_minutes": 15,
                "allow_nr4_breakout": True,
                "nr4_min_volume_ratio": 2.0,
            },
            **(extra_config or {}),
        }
        orch = IntradayOrchestrator(config)
        # Replace stubs created during __init__ with accessible references
        orch.broker = broker
        orch.sizer = sizer
        orch.risk_engine = risk
        return orch, broker, sizer, risk


def _make_nr4_signal_frame(symbol="AAPL") -> pd.DataFrame:
    """A 25-row frame where the last row has a valid nr4_breakout signal.

    We bypass prepare_signals feature math by having the tests patch it to
    return this handcrafted frame directly.
    """
    idx = pd.date_range("2026-04-17 08:30", periods=25, freq="15min")
    rows = []
    for i, ts in enumerate(idx):
        rows.append({
            "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
            "volume": 10000, "entry_signal": False, "setup_family": "none",
            "candidate_score": 0.0, "volume_ratio": 1.0,
            "breakout_distance_pct": 0.0,
            "session_date": ts.date(),
        })
    # Flip the last row into an NR4 signal
    rows[-1].update({
        "entry_signal": True, "setup_family": "nr4_breakout",
        "candidate_score": 5.0, "volume_ratio": 2.3,
        "breakout_distance_pct": 0.012,
    })
    return pd.DataFrame(rows, index=idx)


class IntradayOrchestratorTests(unittest.TestCase):

    def test_init_loads_universe_and_backtest_config(self):
        orch, _, _, _ = _build_orchestrator(
            universe=["AAPL", "MSFT", "NVDA", "TSLA"],
        )
        self.assertEqual(len(orch.universe), 4)
        self.assertEqual(orch.bt_config.max_positions, 4)
        self.assertEqual(orch.bt_config.interval_minutes, 15)
        self.assertTrue(orch.bt_config.allow_nr4_breakout)
        self.assertEqual(orch.bt_config.nr4_min_volume_ratio, 2.0)
        self.assertTrue(orch.dry_run)

    def test_scan_returns_market_closed_when_broker_closed(self):
        orch, broker, _, _ = _build_orchestrator()
        broker.is_market_open.return_value = False
        result = orch.scan()
        self.assertEqual(result["status"], "market_closed")

    def test_scan_no_signals_returns_empty_entries(self):
        orch, broker, _, _ = _build_orchestrator()
        with patch.object(orch, "_fetch_bars", return_value={}):
            result = orch.scan()
        self.assertEqual(result["entries"], [])

    def test_scan_dry_run_logs_but_does_not_submit(self):
        orch, broker, sizer, risk = _build_orchestrator(
            universe=["AAPL"], dry_run=True,
        )
        frame = _make_nr4_signal_frame()
        with patch.object(orch, "_fetch_bars", return_value={"AAPL": frame}), \
             patch.object(orch.backtester, "prepare_signals", return_value=frame):
            result = orch.scan()
        entries = result["entries"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["symbol"], "AAPL")
        self.assertTrue(entries[0]["dry_run"])
        self.assertFalse(entries[0]["traded"])
        broker.submit_bracket_order.assert_not_called()

    def test_scan_live_mode_submits_bracket_order(self):
        orch, broker, sizer, risk = _build_orchestrator(
            universe=["AAPL"], dry_run=False,
        )
        frame = _make_nr4_signal_frame()
        with patch.object(orch, "_fetch_bars", return_value={"AAPL": frame}), \
             patch.object(orch.backtester, "prepare_signals", return_value=frame):
            result = orch.scan()
        self.assertEqual(len(result["entries"]), 1)
        self.assertTrue(result["entries"][0]["traded"])
        broker.submit_bracket_order.assert_called_once()
        call_kwargs = broker.submit_bracket_order.call_args.kwargs
        # Stop at -3% off the $100 price
        self.assertAlmostEqual(call_kwargs["stop_loss_price"], 97.0, places=2)
        # Bracket forces time_in_force=day for intraday
        submitted_order = broker.submit_bracket_order.call_args.args[0]
        self.assertEqual(submitted_order.time_in_force, "day")

    def test_scan_dedups_same_bar_within_session(self):
        orch, broker, sizer, _ = _build_orchestrator(
            universe=["AAPL"], dry_run=True,
        )
        frame = _make_nr4_signal_frame()
        with patch.object(orch, "_fetch_bars", return_value={"AAPL": frame}), \
             patch.object(orch.backtester, "prepare_signals", return_value=frame):
            first = orch.scan()
            second = orch.scan()
        self.assertEqual(len(first["entries"]), 1)
        self.assertEqual(len(second["entries"]), 0)
        self.assertEqual(sizer.calculate.call_count, 1)

    def test_scan_rejects_disallowed_setup_family(self):
        orch, _, sizer, _ = _build_orchestrator(
            universe=["AAPL"], dry_run=True,
        )
        frame = _make_nr4_signal_frame()
        # Mark the latest row as a family we explicitly don't allow live yet
        frame.iloc[-1, frame.columns.get_loc("setup_family")] = "opening_drive_continuation"
        with patch.object(orch, "_fetch_bars", return_value={"AAPL": frame}), \
             patch.object(orch.backtester, "prepare_signals", return_value=frame):
            result = orch.scan()
        self.assertEqual(result["entries"], [])
        sizer.calculate.assert_not_called()

    def test_scan_respects_max_positions_cap(self):
        orch, broker, sizer, _ = _build_orchestrator(
            universe=["AAPL", "MSFT"], dry_run=True,
        )
        existing = [
            MagicMock(symbol="XYZ", qty=10, avg_entry_price=50, current_price=51,
                      unrealized_pl=10, unrealized_plpc=0.02, market_value=500),
        ] * 4
        broker.get_positions.return_value = existing
        result = orch.scan()
        self.assertTrue(result.get("maxed"))
        self.assertEqual(result["entries"], [])

    def test_flatten_all_dry_run(self):
        orch, broker, _, _ = _build_orchestrator(dry_run=True)
        broker.get_positions.return_value = [
            MagicMock(symbol="AAPL", qty=10),
            MagicMock(symbol="MSFT", qty=20),
        ]
        orch._entered_today["AAPL:2026-04-17"] = pd.Timestamp("2026-04-17 14:30")
        result = orch.flatten_all()
        self.assertEqual(result["positions_closed"], 2)
        self.assertTrue(all(c["dry_run"] for c in result["closed"]))
        broker.close_position.assert_not_called()
        self.assertEqual(orch._entered_today, {})

    def test_flatten_all_live_cancels_orders_then_closes(self):
        orch, broker, _, _ = _build_orchestrator(dry_run=False)
        broker.get_positions.return_value = [
            MagicMock(symbol="AAPL", qty=10),
        ]
        broker.get_open_orders.return_value = [
            MagicMock(symbol="AAPL", order_id="stop-1"),
            MagicMock(symbol="OTHER", order_id="nope-1"),
        ]
        result = orch.flatten_all()
        self.assertEqual(result["positions_closed"], 1)
        # Only the AAPL open order should be canceled
        broker.cancel_order.assert_called_once_with("stop-1")
        broker.close_position.assert_called_once_with("AAPL")

    def test_aliases_and_stubs(self):
        orch, broker, _, _ = _build_orchestrator()
        # run_daily_analysis, run_intraday_entry_scan both route to scan
        with patch.object(orch, "scan", return_value={"entries": []}) as scan:
            orch.run_daily_analysis()
            orch.run_intraday_entry_scan()
        self.assertEqual(scan.call_count, 2)
        # Noop stubs
        self.assertEqual(orch.run_daily_reflection(), {})
        self.assertEqual(orch.take_market_snapshot(), {})

    def test_get_status_contains_strategy_tag(self):
        orch, broker, _, _ = _build_orchestrator(
            universe=["AAPL", "MSFT", "NVDA"], dry_run=True,
        )
        status = orch.get_status()
        self.assertEqual(status["strategy"], "intraday_mechanical")
        self.assertEqual(status["universe_size"], 3)
        self.assertTrue(status["dry_run"])


if __name__ == "__main__":
    unittest.main()
