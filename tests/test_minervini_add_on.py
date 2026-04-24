"""Tests for Minervini pyramid add-on live wire-up.

Covers `Orchestrator._evaluate_minervini_add_on` and
`Orchestrator._calculate_add_on_qty` in isolation (bypassing __init__ to
avoid broker/DB setup).
"""

from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from tradingagents.automation.orchestrator import Orchestrator


@dataclass
class MockPosition:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float


@dataclass
class MockAccount:
    equity: float
    cash: float
    buying_power: float = 0.0
    portfolio_value: float = 0.0


@dataclass
class MockOrder:
    order_id: str
    symbol: str
    side: str
    status: str


@dataclass
class MockOrderResult:
    order_id: str
    status: str = "accepted"
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None


def _make_orch(
    *,
    add_on_enabled: bool = True,
    trigger_1: float = 0.025,
    trigger_2: float = 0.05,
    fraction_1: float = 0.30,
    fraction_2: float = 0.20,
    max_position_pct: float = 0.12,
    risk_per_trade: float = 0.012,
):
    """Build a minimal Orchestrator stub with only what the add-on path needs."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = {
        "minervini_add_on_enabled": add_on_enabled,
        "minervini_add_on_trigger_pct_1": trigger_1,
        "minervini_add_on_trigger_pct_2": trigger_2,
        "minervini_add_on_fraction_1": fraction_1,
        "minervini_add_on_fraction_2": fraction_2,
        "max_position_pct": max_position_pct,
        "risk_per_trade": risk_per_trade,
    }
    orch.broker = MagicMock()
    orch.broker.get_live_orders = MagicMock(return_value=[])
    orch.broker.submit_order = MagicMock(
        return_value=MockOrderResult(order_id="buy-1", status="accepted")
    )
    orch.broker.cancel_order = MagicMock()

    orch.db = MagicMock()
    orch.db.log_signal = MagicMock(return_value=42)
    orch.db.log_trade = MagicMock()
    orch.db.mark_signal_executed = MagicMock()
    orch.db.mark_signal_rejected = MagicMock()

    orch.risk_engine = MagicMock()
    risk_ok = MagicMock()
    risk_ok.passed = True
    risk_ok.reason = ""
    orch.risk_engine.check_order = MagicMock(return_value=risk_ok)
    orch.risk_engine.record_trade = MagicMock()

    notifier = MagicMock()
    notifier.enabled = False
    orch.notifier = notifier

    return orch


def _state(entry: float = 100.0, **overrides):
    base = {
        "entry_price": entry,
        "entry_date": "2026-04-10",
        "highest_close": entry,
        "current_stop": entry * 0.95,
        "partial_taken": False,
        "add_on_1_done": False,
        "add_on_2_done": False,
    }
    base.update(overrides)
    return base


ACCOUNT = MockAccount(equity=100_000.0, cash=50_000.0)


class TestCalculateAddOnQty:
    def test_caps_by_remaining_position_notional(self):
        orch = _make_orch()
        # max_position_pct=0.12 * 100k = $12k cap. Position already has
        # $10k at price 100 (100 shares). Add-on can only buy up to
        # $2k / $102 = 19 shares. But risk budget: 100k * 0.012 * 0.30 = $360,
        # divided by risk_per_share = 102-95 = $7 → floor(51.4) = 51.
        # The notional cap is tighter.
        pos = MockPosition("AAPL", qty=100, avg_entry_price=95.0, current_price=102.0)
        qty = orch._calculate_add_on_qty(
            account=ACCOUNT,
            position=pos,
            price=102.0,
            stop_price=95.0,
            add_fraction=0.30,
        )
        # Expected cap = floor(((100000*0.12) - (100*102)) / 102) = floor(1800/102) = 17
        assert qty == 17

    def test_returns_zero_when_stop_at_or_above_price(self):
        orch = _make_orch()
        pos = MockPosition("AAPL", qty=100, avg_entry_price=95.0, current_price=102.0)
        assert (
            orch._calculate_add_on_qty(
                account=ACCOUNT, position=pos, price=102.0,
                stop_price=102.0, add_fraction=0.30,
            )
            == 0
        )

    def test_returns_zero_when_position_already_at_cap(self):
        orch = _make_orch()
        # Position value already at cap.
        pos = MockPosition("AAPL", qty=120, avg_entry_price=100.0, current_price=100.0)
        qty = orch._calculate_add_on_qty(
            account=ACCOUNT, position=pos, price=100.0,
            stop_price=92.0, add_fraction=0.30,
        )
        assert qty == 0


class TestEvaluateAddOn:
    def test_no_op_when_flag_off(self):
        orch = _make_orch(add_on_enabled=False)
        pos = MockPosition("AAPL", qty=100, avg_entry_price=100.0, current_price=103.0)
        result = orch._evaluate_minervini_add_on(
            position=pos, pos_state=_state(100.0),
            regime_label="confirmed_uptrend",
            account=ACCOUNT, positions=[pos],
        )
        assert result is None
        orch.broker.submit_order.assert_not_called()

    def test_no_op_in_market_correction(self):
        orch = _make_orch()
        pos = MockPosition("AAPL", qty=100, avg_entry_price=100.0, current_price=103.0)
        result = orch._evaluate_minervini_add_on(
            position=pos, pos_state=_state(100.0),
            regime_label="market_correction",
            account=ACCOUNT, positions=[pos],
        )
        assert result is None
        orch.broker.submit_order.assert_not_called()

    def test_no_op_below_first_trigger(self):
        orch = _make_orch()
        # +2.0% is below 2.5% threshold
        pos = MockPosition("AAPL", qty=100, avg_entry_price=100.0, current_price=102.0)
        result = orch._evaluate_minervini_add_on(
            position=pos, pos_state=_state(100.0),
            regime_label="confirmed_uptrend",
            account=ACCOUNT, positions=[pos],
        )
        assert result is None

    def test_fires_at_first_trigger(self):
        orch = _make_orch()
        pos = MockPosition("AAPL", qty=100, avg_entry_price=100.0, current_price=103.0)
        result = orch._evaluate_minervini_add_on(
            position=pos, pos_state=_state(100.0),
            regime_label="confirmed_uptrend",
            account=ACCOUNT, positions=[pos],
        )
        assert result is not None
        assert result["traded"] is True
        assert result["add_on_level"] == 1
        orch.broker.submit_order.assert_called()

    def test_fires_at_second_trigger_skipping_first(self):
        # Position gapped past +5% in one session and add_on_1_done was False.
        # Helper should pick level 2 (the higher) not level 1.
        orch = _make_orch()
        pos = MockPosition("AAPL", qty=100, avg_entry_price=100.0, current_price=106.0)
        result = orch._evaluate_minervini_add_on(
            position=pos, pos_state=_state(100.0),
            regime_label="confirmed_uptrend",
            account=ACCOUNT, positions=[pos],
        )
        assert result is not None
        assert result["add_on_level"] == 2

    def test_skipped_when_add_on_1_already_done_but_not_at_2(self):
        orch = _make_orch()
        pos = MockPosition("AAPL", qty=130, avg_entry_price=100.0, current_price=104.0)
        result = orch._evaluate_minervini_add_on(
            position=pos,
            pos_state=_state(100.0, add_on_1_done=True),
            regime_label="confirmed_uptrend",
            account=ACCOUNT, positions=[pos],
        )
        assert result is None
        orch.broker.submit_order.assert_not_called()

    def test_blocked_by_existing_open_buy(self):
        orch = _make_orch()
        orch.broker.get_live_orders = MagicMock(
            return_value=[
                MockOrder(order_id="pending", symbol="AAPL", side="buy", status="accepted"),
            ]
        )
        pos = MockPosition("AAPL", qty=100, avg_entry_price=100.0, current_price=103.0)
        result = orch._evaluate_minervini_add_on(
            position=pos, pos_state=_state(100.0),
            regime_label="confirmed_uptrend",
            account=ACCOUNT, positions=[pos],
        )
        assert result is None
        # Idempotent: should not submit a second buy
        orch.broker.submit_order.assert_not_called()

    def test_blocked_by_risk_engine(self):
        orch = _make_orch()
        reject = MagicMock()
        reject.passed = False
        reject.reason = "Single-position limit exceeded"
        orch.risk_engine.check_order = MagicMock(return_value=reject)
        pos = MockPosition("AAPL", qty=100, avg_entry_price=100.0, current_price=103.0)
        result = orch._evaluate_minervini_add_on(
            position=pos, pos_state=_state(100.0),
            regime_label="confirmed_uptrend",
            account=ACCOUNT, positions=[pos],
        )
        # Implementation returns None (no-op) when risk engine rejects — we
        # prefer "silent skip, retry next tick" over a noisy rejected signal.
        assert result is None
        orch.broker.submit_order.assert_not_called()

    def test_stop_resync_cancels_existing_sells_and_submits_new(self):
        orch = _make_orch()
        orch.broker.get_live_orders = MagicMock(
            return_value=[
                MockOrder(order_id="sl-old", symbol="AAPL", side="sell", status="held"),
                MockOrder(order_id="tp-old", symbol="AAPL", side="sell", status="held"),
            ]
        )
        # Sequence of submit_order calls: first the add-on BUY, then the
        # stop-resync SELL. Use side_effect to give distinct OrderResults.
        orch.broker.submit_order = MagicMock(
            side_effect=[
                MockOrderResult(order_id="buy-new", status="accepted"),
                MockOrderResult(order_id="stop-new", status="accepted"),
            ]
        )
        pos = MockPosition("AAPL", qty=100, avg_entry_price=100.0, current_price=103.0)
        state = _state(100.0)
        result = orch._evaluate_minervini_add_on(
            position=pos, pos_state=state,
            regime_label="confirmed_uptrend",
            account=ACCOUNT, positions=[pos],
        )
        assert result is not None and result["traded"] is True
        # Both legacy sell legs should have been canceled
        cancel_calls = [c.args[0] for c in orch.broker.cancel_order.call_args_list]
        assert set(cancel_calls) == {"sl-old", "tp-old"}
        # submit_order invoked twice (buy + stop resync)
        assert orch.broker.submit_order.call_count == 2
        # Second submit is a stop order covering projected qty (100 + add_on_qty)
        stop_req = orch.broker.submit_order.call_args_list[1].args[0]
        assert stop_req.order_type == "stop"
        assert stop_req.side == "sell"
        assert stop_req.qty >= 100  # at minimum covers pre-add-on qty; add-on qty is >0
        # pos_state updated to the new stop leg ID; tp_order_id cleared
        assert state["stop_order_id"] == "stop-new"
        assert state["tp_order_id"] is None

    def test_stop_resync_waits_for_cancel_ack_before_new_submit(self):
        """Regression: Alpaca holds cancelled orders in pending_cancel briefly.
        Submitting a new opposite-side stop before the cancel clears triggers
        a 40310000 wash-trade reject. See 2026-04-24 DELL incident. The
        resync must re-poll get_live_orders until no sell leg remains (or
        timeout) before submitting the replacement stop.
        """
        orch = _make_orch()
        # First call: old SL still live. Second call: still pending_cancel.
        # Third call: cleared. The resync submit must not fire until call 3.
        orch.broker.get_live_orders = MagicMock(
            side_effect=[
                [MockOrder(order_id="sl-old", symbol="AAPL", side="sell", status="held")],
                [MockOrder(order_id="sl-old", symbol="AAPL", side="sell", status="pending_cancel")],
                [],  # cleared
            ]
        )
        submit_calls = []
        def _record_submit(req):
            submit_calls.append((req, orch.broker.get_live_orders.call_count))
            return MockOrderResult(order_id=f"o-{len(submit_calls)}", status="accepted")
        orch.broker.submit_order = MagicMock(side_effect=_record_submit)
        pos = MockPosition("AAPL", qty=100, avg_entry_price=100.0, current_price=103.0)
        state = _state(100.0)
        result = orch._evaluate_minervini_add_on(
            position=pos, pos_state=state,
            regime_label="confirmed_uptrend",
            account=ACCOUNT, positions=[pos],
        )
        assert result is not None and result["traded"] is True
        # The stop-resync submit (2nd submit) must have been issued only after
        # get_live_orders returned empty (call #3). If the wait was skipped,
        # the stop submit would see call_count == 2 or less.
        stop_submit_idx = 1  # zero-indexed; submit order: [buy, stop]
        _, live_orders_calls_at_stop_submit = submit_calls[stop_submit_idx]
        assert live_orders_calls_at_stop_submit >= 3, (
            f"Stop submit fired after only {live_orders_calls_at_stop_submit} "
            f"get_live_orders calls — cancel-wait skipped, race still present"
        )
