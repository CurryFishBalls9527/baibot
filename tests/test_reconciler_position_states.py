"""Regression tests for the position_states + broker-stop reconciliation passes.

Covers the drift failure modes enumerated in
`memory/project_bracket_leg_id_bug.md`:
  - ghost rows (DB has row, Alpaca doesn't hold)
  - ID drift (DB leg IDs point to cancelled orders)
  - orphan positions (Alpaca holds, DB has no row)
  - ratchet drift (DB current_stop above broker SL, not synced)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from tradingagents.automation.reconciler import OrderReconciler
from tradingagents.broker.models import OrderResult, Position
from tradingagents.storage.database import TradingDatabase


def _mkdb(tmp_path: Path, variant: str = "mechanical") -> TradingDatabase:
    return TradingDatabase(str(tmp_path / "test.db"), variant=variant)


def _pos(symbol: str, qty: float = 100, avg_entry_price: float = 100.0,
         current_price: float = 110.0, qty_available: float | None = None) -> Position:
    return Position(
        symbol=symbol,
        qty=qty,
        side="long",
        avg_entry_price=avg_entry_price,
        current_price=current_price,
        market_value=qty * current_price,
        unrealized_pl=(current_price - avg_entry_price) * qty,
        unrealized_plpc=(current_price / avg_entry_price) - 1,
        qty_available=qty_available,
    )


def _order(order_id: str, order_type: str = "stop", stop_price=None,
           limit_price=None) -> OrderResult:
    return OrderResult(
        order_id=order_id,
        symbol="AAPL",
        side="sell",
        qty=100,
        notional=None,
        order_type=order_type,
        status="held" if order_type == "stop" else "new",
        stop_price=stop_price,
        limit_price=limit_price,
    )


# ── ghost pass ──────────────────────────────────────────────────────────


class TestGhostRowDeletion:
    def test_deletes_row_when_broker_not_holding(self, tmp_path):
        db = _mkdb(tmp_path)
        db.upsert_position_state("AAPL", {
            "entry_price": 100.0, "entry_date": "2026-04-01",
            "highest_close": 110.0, "current_stop": 95.0,
            "partial_taken": False, "stop_order_id": "old-stop",
        })
        broker = MagicMock()
        broker.get_positions.return_value = []  # nothing held
        broker.get_live_orders.return_value = []

        rec = OrderReconciler(broker=broker, db=db, variant="mechanical")
        result = rec._reconcile_position_states()

        assert result["ghosts"] == 1
        assert db.get_position_state("AAPL") is None

    def test_dry_run_does_not_delete(self, tmp_path):
        db = _mkdb(tmp_path)
        db.upsert_position_state("AAPL", {
            "entry_price": 100.0, "entry_date": "2026-04-01",
            "highest_close": 110.0, "current_stop": 95.0,
            "partial_taken": False,
        })
        broker = MagicMock()
        broker.get_positions.return_value = []
        broker.get_live_orders.return_value = []
        rec = OrderReconciler(broker=broker, db=db, variant="mechanical")
        result = rec._reconcile_position_states(dry_run=True)

        assert result["ghosts"] == 1
        assert db.get_position_state("AAPL") is not None  # untouched


# ── ID fix pass ─────────────────────────────────────────────────────────


class TestLegIDDriftCorrection:
    def test_updates_db_when_broker_ids_differ(self, tmp_path):
        db = _mkdb(tmp_path)
        db.upsert_position_state("AAPL", {
            "entry_price": 100.0, "entry_date": "2026-04-01",
            "highest_close": 110.0, "current_stop": 95.0,
            "partial_taken": False,
            "stop_order_id": "stale-stop",
            "tp_order_id": "stale-tp",
        })
        broker = MagicMock()
        broker.get_positions.return_value = [_pos("AAPL")]
        broker.get_live_orders.return_value = [
            _order("new-stop", "stop", stop_price=95.0),
            _order("new-tp", "limit", limit_price=120.0),
        ]

        rec = OrderReconciler(broker=broker, db=db, variant="mechanical")
        result = rec._reconcile_position_states()

        assert result["id_fixes"] == 1
        state = db.get_position_state("AAPL")
        assert state["stop_order_id"] == "new-stop"
        assert state["tp_order_id"] == "new-tp"

    def test_no_update_when_ids_match(self, tmp_path):
        db = _mkdb(tmp_path)
        db.upsert_position_state("AAPL", {
            "entry_price": 100.0, "entry_date": "2026-04-01",
            "highest_close": 110.0, "current_stop": 95.0,
            "partial_taken": False,
            "stop_order_id": "stop-abc",
            "tp_order_id": "tp-xyz",
        })
        broker = MagicMock()
        broker.get_positions.return_value = [_pos("AAPL")]
        broker.get_live_orders.return_value = [
            _order("stop-abc", "stop", stop_price=95.0),
            _order("tp-xyz", "limit", limit_price=120.0),
        ]
        rec = OrderReconciler(broker=broker, db=db, variant="mechanical")
        result = rec._reconcile_position_states()
        assert result["id_fixes"] == 0

    def test_picks_most_recent_when_multiple_live_stops(self, tmp_path):
        """Manual OCO resubmission leaves cancelled + live orders; most recent wins."""
        from datetime import datetime
        db = _mkdb(tmp_path)
        db.upsert_position_state("AAPL", {
            "entry_price": 100.0, "entry_date": "2026-04-01",
            "highest_close": 110.0, "current_stop": 95.0,
            "partial_taken": False,
            "stop_order_id": "old-stop",
        })
        broker = MagicMock()
        broker.get_positions.return_value = [_pos("AAPL")]
        older = _order("older-stop", "stop", stop_price=95.0)
        older.submitted_at = datetime(2026, 4, 1, 10, 0)
        newer = _order("newer-stop", "stop", stop_price=100.0)
        newer.submitted_at = datetime(2026, 4, 5, 10, 0)
        tp = _order("tp-only", "limit", limit_price=120.0)
        tp.submitted_at = datetime(2026, 4, 1, 10, 0)
        broker.get_live_orders.return_value = [older, newer, tp]

        rec = OrderReconciler(broker=broker, db=db, variant="mechanical")
        rec._reconcile_position_states()
        state = db.get_position_state("AAPL")
        assert state["stop_order_id"] == "newer-stop"


# ── orphan import ───────────────────────────────────────────────────────


class TestOrphanImport:
    def test_imports_position_with_broker_sl_price(self, tmp_path):
        db = _mkdb(tmp_path)
        broker = MagicMock()
        broker.get_positions.return_value = [_pos("AAPL", qty=63,
                                                  avg_entry_price=192.08,
                                                  current_price=200.0)]
        broker.get_live_orders.return_value = [
            _order("sl-1", "stop", stop_price=174.80),
            _order("tp-1", "limit", limit_price=238.18),
        ]

        rec = OrderReconciler(broker=broker, db=db, variant="llm")
        result = rec._reconcile_position_states()

        assert result["orphans"] == 1
        state = db.get_position_state("AAPL")
        assert state["entry_price"] == 192.08
        assert state["current_stop"] == 174.80  # from live SL, NOT 8% fallback
        assert state["stop_order_id"] == "sl-1"
        assert state["tp_order_id"] == "tp-1"
        assert state["stop_type"] == "imported"

    def test_imports_with_fallback_when_no_live_sl(self, tmp_path):
        """Orphan without any live SL — seed at 8% below entry."""
        db = _mkdb(tmp_path)
        broker = MagicMock()
        broker.get_positions.return_value = [_pos("AAPL", qty=100,
                                                  avg_entry_price=100.0)]
        broker.get_live_orders.return_value = []  # no SL/TP exists

        rec = OrderReconciler(broker=broker, db=db, variant="chan")
        result = rec._reconcile_position_states()

        assert result["orphans"] == 1
        state = db.get_position_state("AAPL")
        assert abs(state["current_stop"] - 92.0) < 0.01
        assert state["stop_order_id"] is None

    def test_dry_run_does_not_insert(self, tmp_path):
        db = _mkdb(tmp_path)
        broker = MagicMock()
        broker.get_positions.return_value = [_pos("AAPL")]
        broker.get_live_orders.return_value = []
        rec = OrderReconciler(broker=broker, db=db, variant="chan")
        result = rec._reconcile_position_states(dry_run=True)
        assert result["orphans"] == 1
        assert db.get_position_state("AAPL") is None


# ── broker-stop sync ────────────────────────────────────────────────────


class TestBrokerStopSync:
    def _setup(self, tmp_path, db_stop: float, broker_sl_price: float,
              current_price: float, replace_success: bool = True):
        db = _mkdb(tmp_path)
        db.upsert_position_state("AAPL", {
            "entry_price": 100.0, "entry_date": "2026-04-01",
            "highest_close": 120.0, "current_stop": db_stop,
            "partial_taken": False, "stop_order_id": "old-sl",
        })
        broker = MagicMock()
        broker.get_positions.return_value = [
            _pos("AAPL", qty=100, avg_entry_price=100.0, current_price=current_price)
        ]
        broker.get_live_orders.return_value = [
            _order("old-sl", "stop", stop_price=broker_sl_price),
            _order("tp-1", "limit", limit_price=130.0),
        ]
        if replace_success:
            broker.replace_order.return_value = OrderResult(
                order_id="new-sl", symbol="AAPL", side="sell", qty=100,
                notional=None, order_type="stop", status="accepted",
                stop_price=db_stop,
            )
        else:
            broker.replace_order.side_effect = RuntimeError("broker rejected")
        return db, broker

    def test_pushes_stop_when_db_above_broker(self, tmp_path):
        db, broker = self._setup(
            tmp_path, db_stop=115.0, broker_sl_price=110.0, current_price=118.0
        )
        rec = OrderReconciler(broker=broker, db=db, variant="mech_v2")
        result = rec._sync_broker_stops()

        assert result["synced"] == 1
        broker.replace_order.assert_called_once_with("old-sl", stop_price=115.0)
        # DB stop_order_id updated to the new one returned by replace_order
        state = db.get_position_state("AAPL")
        assert state["stop_order_id"] == "new-sl"

    def test_skips_when_broker_already_matches(self, tmp_path):
        db, broker = self._setup(
            tmp_path, db_stop=110.0, broker_sl_price=110.0, current_price=120.0
        )
        rec = OrderReconciler(broker=broker, db=db, variant="mech_v2")
        result = rec._sync_broker_stops()
        assert result["synced"] == 0
        broker.replace_order.assert_not_called()

    def test_skips_when_db_stop_above_current_price(self, tmp_path):
        """Guard: a stop above mkt would trigger immediately. Never push."""
        db, broker = self._setup(
            tmp_path, db_stop=120.0, broker_sl_price=110.0, current_price=118.0
        )
        rec = OrderReconciler(broker=broker, db=db, variant="mech_v2")
        result = rec._sync_broker_stops()
        assert result["synced"] == 0
        assert result["skipped"] == 1
        broker.replace_order.assert_not_called()

    def test_skips_absurd_tighten_delta(self, tmp_path):
        """Safety: DB 50% above broker SL smells like corruption — don't push."""
        db, broker = self._setup(
            tmp_path, db_stop=165.0, broker_sl_price=100.0, current_price=170.0
        )
        rec = OrderReconciler(broker=broker, db=db, variant="mech_v2")
        result = rec._sync_broker_stops(max_tighten_pct=0.30)
        assert result["synced"] == 0
        assert result["skipped"] == 1
        broker.replace_order.assert_not_called()

    def test_handles_replace_failure_without_db_change(self, tmp_path):
        db, broker = self._setup(
            tmp_path, db_stop=115.0, broker_sl_price=110.0, current_price=118.0,
            replace_success=False,
        )
        rec = OrderReconciler(broker=broker, db=db, variant="mech_v2")
        result = rec._sync_broker_stops()
        assert result["synced"] == 0
        assert result["errors"] == 1
        # DB keeps the old stop_order_id since replace failed
        state = db.get_position_state("AAPL")
        assert state["stop_order_id"] == "old-sl"

    def test_dry_run_does_not_call_replace(self, tmp_path):
        db, broker = self._setup(
            tmp_path, db_stop=115.0, broker_sl_price=110.0, current_price=118.0
        )
        rec = OrderReconciler(broker=broker, db=db, variant="mech_v2")
        result = rec._sync_broker_stops(dry_run=True)
        assert result["synced"] == 1
        broker.replace_order.assert_not_called()


# ── stale stop_order_id auto-recovery (DELL 2026-04-24 case) ────────────


class TestStaleStopAutoRecovery:
    """Position naked because the recorded stop is terminal at the broker.

    Ensures the reconciler clears the stale id and reattaches a fresh stop
    so exit_manager stops retrying replace_order on a dead order_id.
    """

    def _setup(self, tmp_path, db_stop=95.0, current_price=110.0,
               qty=100, attach_succeeds=True):
        db = _mkdb(tmp_path, variant="mech_v2")
        db.upsert_position_state("AAPL", {
            "entry_price": 100.0, "entry_date": "2026-04-01",
            "highest_close": 120.0, "current_stop": db_stop,
            "partial_taken": False, "stop_order_id": "stale-cancelled",
        })
        broker = MagicMock()
        broker.get_positions.return_value = [
            _pos("AAPL", qty=qty, avg_entry_price=100.0,
                 current_price=current_price)
        ]
        broker.get_live_orders.return_value = []  # no live SL/TP at broker
        if attach_succeeds:
            broker.submit_order.return_value = OrderResult(
                order_id="fresh-sl", symbol="AAPL", side="sell", qty=qty,
                notional=None, order_type="stop", status="held",
                stop_price=db_stop,
            )
        else:
            broker.submit_order.side_effect = RuntimeError(
                '{"code":40310000,"message":"wash trade"}'
            )
        return db, broker

    def test_clears_stale_id_and_attaches_fresh_stop(self, tmp_path):
        db, broker = self._setup(tmp_path, db_stop=95.0, current_price=110.0)
        rec = OrderReconciler(broker=broker, db=db, variant="mech_v2")
        result = rec._reconcile_position_states()

        assert result["id_fixes"] == 1
        assert result["errors"] == 0
        broker.submit_order.assert_called_once()
        call = broker.submit_order.call_args.args[0]
        assert call.symbol == "AAPL"
        assert call.side == "sell"
        assert call.order_type == "stop"
        assert call.stop_price == 95.0
        assert call.qty == 100
        state = db.get_position_state("AAPL")
        assert state["stop_order_id"] == "fresh-sl"

    def test_attach_failure_leaves_id_null(self, tmp_path):
        """Wash-trade reject etc. — clear ID, no reattach, next tick retries."""
        db, broker = self._setup(tmp_path, attach_succeeds=False)
        rec = OrderReconciler(broker=broker, db=db, variant="mech_v2")
        result = rec._reconcile_position_states()

        assert result["id_fixes"] == 1  # cleared
        assert result["errors"] == 1    # submit failed
        state = db.get_position_state("AAPL")
        assert state["stop_order_id"] is None

    def test_skips_attach_when_db_stop_above_market(self, tmp_path):
        """Safety: stop above mkt would trigger immediately. Clear only."""
        db, broker = self._setup(
            tmp_path, db_stop=120.0, current_price=110.0
        )
        rec = OrderReconciler(broker=broker, db=db, variant="mech_v2")
        result = rec._reconcile_position_states()

        assert result["id_fixes"] == 1  # ID cleared
        broker.submit_order.assert_not_called()
        state = db.get_position_state("AAPL")
        assert state["stop_order_id"] is None

    def test_dry_run_does_not_clear_or_submit(self, tmp_path):
        db, broker = self._setup(tmp_path)
        rec = OrderReconciler(broker=broker, db=db, variant="mech_v2")
        result = rec._reconcile_position_states(dry_run=True)

        assert result["id_fixes"] == 1
        broker.submit_order.assert_not_called()
        state = db.get_position_state("AAPL")
        assert state["stop_order_id"] == "stale-cancelled"  # untouched

    def test_skips_reattach_when_qty_available_zero(self, tmp_path):
        """EOD-flatten race: bracket child is mid-cancel so qty_available=0.
        Submitting another sell would trip Alpaca's wash-trade guard. Clear
        the stale ID but don't try to reattach."""
        db = _mkdb(tmp_path, variant="mech_v2")
        db.upsert_position_state("AAPL", {
            "entry_price": 100.0, "entry_date": "2026-04-01",
            "highest_close": 120.0, "current_stop": 95.0,
            "partial_taken": False, "stop_order_id": "stale-cancelled",
        })
        broker = MagicMock()
        broker.get_positions.return_value = [
            _pos("AAPL", qty=100, qty_available=0),  # held_for_orders == qty
        ]
        broker.get_live_orders.return_value = []
        rec = OrderReconciler(broker=broker, db=db, variant="mech_v2")
        result = rec._reconcile_position_states()

        assert result["id_fixes"] == 1
        assert result["errors"] == 0
        broker.submit_order.assert_not_called()  # would have hit 40310000
        state = db.get_position_state("AAPL")
        assert state["stop_order_id"] is None  # stale cleared either way

    def test_no_recovery_when_live_sl_exists(self, tmp_path):
        """Healthy position with live SL — no clear, no submit, no error."""
        db, broker = self._setup(tmp_path)
        broker.get_live_orders.return_value = [
            _order("stale-cancelled", "stop", stop_price=95.0),
        ]
        rec = OrderReconciler(broker=broker, db=db, variant="mech_v2")
        result = rec._reconcile_position_states()

        assert result["id_fixes"] == 0
        broker.submit_order.assert_not_called()
        state = db.get_position_state("AAPL")
        assert state["stop_order_id"] == "stale-cancelled"

    def test_id_drift_takes_precedence_over_stale_recovery(self, tmp_path):
        """If a NEW live SL exists with a different id, id-drift fixes it
        rather than clearing — recovery only runs when no live SL at all."""
        db, broker = self._setup(tmp_path)
        broker.get_live_orders.return_value = [
            _order("real-live-sl", "stop", stop_price=95.0),
        ]
        rec = OrderReconciler(broker=broker, db=db, variant="mech_v2")
        result = rec._reconcile_position_states()

        assert result["id_fixes"] == 1
        broker.submit_order.assert_not_called()
        state = db.get_position_state("AAPL")
        assert state["stop_order_id"] == "real-live-sl"


# ── stale tp_order_id auto-clear (DELL post-add-on case) ────────────────


class TestStaleTpAutoClear:
    """DB has tp_order_id but broker has no live TP. Clear without reattach
    — post-add-on positions are stop-only by design."""

    def _setup(self, tmp_path, with_live_sl: bool = True):
        db = _mkdb(tmp_path, variant="mech_v2")
        db.upsert_position_state("AAPL", {
            "entry_price": 100.0, "entry_date": "2026-04-01",
            "highest_close": 120.0, "current_stop": 95.0,
            "partial_taken": False,
            "stop_order_id": "live-stop",
            "tp_order_id": "stale-tp",
        })
        broker = MagicMock()
        broker.get_positions.return_value = [_pos("AAPL")]
        live_orders = []
        if with_live_sl:
            live_orders.append(_order("live-stop", "stop", stop_price=95.0))
        broker.get_live_orders.return_value = live_orders
        return db, broker

    def test_clears_stale_tp_when_no_live_tp(self, tmp_path):
        db, broker = self._setup(tmp_path)
        rec = OrderReconciler(broker=broker, db=db, variant="mech_v2")
        result = rec._reconcile_position_states()

        assert result["id_fixes"] == 1
        broker.submit_order.assert_not_called()  # never reattaches a TP
        state = db.get_position_state("AAPL")
        assert state["tp_order_id"] is None
        assert state["stop_order_id"] == "live-stop"  # untouched

    def test_no_action_when_live_tp_exists(self, tmp_path):
        db, broker = self._setup(tmp_path)
        # Add a matching live TP — id-drift path covers id mismatch separately.
        broker.get_live_orders.return_value = [
            _order("live-stop", "stop", stop_price=95.0),
            _order("stale-tp", "limit", limit_price=120.0),
        ]
        rec = OrderReconciler(broker=broker, db=db, variant="mech_v2")
        result = rec._reconcile_position_states()

        assert result["id_fixes"] == 0
        state = db.get_position_state("AAPL")
        assert state["tp_order_id"] == "stale-tp"  # untouched

    def test_dry_run_does_not_clear(self, tmp_path):
        db, broker = self._setup(tmp_path)
        rec = OrderReconciler(broker=broker, db=db, variant="mech_v2")
        result = rec._reconcile_position_states(dry_run=True)
        assert result["id_fixes"] == 1
        state = db.get_position_state("AAPL")
        assert state["tp_order_id"] == "stale-tp"  # untouched in dry-run
