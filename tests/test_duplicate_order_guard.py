"""Tests for the duplicate-open-order guard."""

from types import SimpleNamespace

import pytest

from tradingagents.automation.orchestrator import Orchestrator
from tradingagents.automation.chan_orchestrator import ChanOrchestrator
from tradingagents.broker.models import Account


def make_order(order_id="o1", symbol="NVDA", side="buy", status="new"):
    return SimpleNamespace(
        order_id=order_id, symbol=symbol, side=side, status=status
    )


class FakeBroker:
    def __init__(self, orders=None, raise_exc=None):
        self._orders = orders or []
        self._raise_exc = raise_exc
        self.calls = []

    def get_open_orders(self, symbol=None):
        if self._raise_exc is not None:
            raise self._raise_exc
        self.calls.append(symbol)
        if symbol:
            return [o for o in self._orders if o.symbol == symbol]
        return self._orders


class BrokerWithoutMethod:
    pass


class BrokerNoSymbolKwarg:
    """Broker whose get_open_orders signature rejects the symbol kwarg."""

    def __init__(self, orders):
        self._orders = orders
        self.call_count = 0

    def get_open_orders(self):
        self.call_count += 1
        return self._orders


def _make_orch(broker):
    o = Orchestrator.__new__(Orchestrator)
    o.broker = broker
    return o


def _make_chan(broker):
    o = ChanOrchestrator.__new__(ChanOrchestrator)
    o.broker = broker
    return o


class TestFindExistingOpenOrder:
    def test_broker_without_method_returns_none(self):
        orch = _make_orch(BrokerWithoutMethod())
        assert orch._find_existing_open_order("NVDA", "buy") is None

    def test_no_orders_returns_none(self):
        orch = _make_orch(FakeBroker(orders=[]))
        assert orch._find_existing_open_order("NVDA", "buy") is None

    def test_matching_open_order_returned(self):
        match = make_order(symbol="NVDA", side="buy", status="new")
        orch = _make_orch(FakeBroker(orders=[match]))
        assert orch._find_existing_open_order("NVDA", "buy") is match

    def test_symbol_mismatch_returns_none(self):
        other = make_order(symbol="AAPL", side="buy", status="new")
        # FakeBroker filters by symbol when symbol kwarg is given,
        # so this exercises the per-order symbol check explicitly.
        orch = _make_orch(FakeBroker(orders=[other]))
        # Bypass FakeBroker's pre-filter by querying with the wrong symbol
        # via direct list — recreate broker that returns regardless.
        class AllBroker:
            def get_open_orders(self, symbol=None):
                return [other]
        orch = _make_orch(AllBroker())
        assert orch._find_existing_open_order("NVDA", "buy") is None

    def test_side_mismatch_returns_none(self):
        sell = make_order(symbol="NVDA", side="sell", status="new")
        orch = _make_orch(FakeBroker(orders=[sell]))
        assert orch._find_existing_open_order("NVDA", "buy") is None

    @pytest.mark.parametrize(
        "status", ["filled", "canceled", "cancelled", "expired", "rejected"]
    )
    def test_terminal_status_returns_none(self, status):
        terminal = make_order(symbol="NVDA", side="buy", status=status)
        orch = _make_orch(FakeBroker(orders=[terminal]))
        assert orch._find_existing_open_order("NVDA", "buy") is None

    def test_case_insensitive_symbol_and_side(self):
        # Broker that ignores the symbol filter so the helper's own
        # case-insensitive comparison is what's under test.
        match = make_order(symbol="nvda", side="BUY", status="NEW")

        class AllBroker:
            def get_open_orders(self, symbol=None):
                return [match]

        orch = _make_orch(AllBroker())
        assert orch._find_existing_open_order("NVDA", "buy") is match

    def test_no_side_filter_matches_any_side(self):
        sell = make_order(symbol="NVDA", side="sell", status="new")
        orch = _make_orch(FakeBroker(orders=[sell]))
        assert orch._find_existing_open_order("NVDA") is sell

    def test_broker_exception_returns_none(self):
        orch = _make_orch(FakeBroker(raise_exc=RuntimeError("boom")))
        assert orch._find_existing_open_order("NVDA", "buy") is None

    def test_broker_without_symbol_kwarg_falls_back(self):
        match = make_order(symbol="NVDA", side="buy", status="new")
        broker = BrokerNoSymbolKwarg([match])
        orch = _make_orch(broker)
        assert orch._find_existing_open_order("NVDA", "buy") is match
        assert broker.call_count == 1

    def test_chan_orchestrator_uses_same_logic(self):
        match = make_order(symbol="AAPL", side="buy", status="accepted")
        orch = _make_chan(FakeBroker(orders=[match]))
        assert orch._find_existing_open_order("AAPL", "buy") is match

    def test_chan_orchestrator_skips_terminal(self):
        filled = make_order(symbol="AAPL", side="buy", status="filled")
        orch = _make_chan(FakeBroker(orders=[filled]))
        assert orch._find_existing_open_order("AAPL", "buy") is None


class TestExecuteOverlayGuard:
    """End-to-end: when a duplicate exists, _execute_overlay_order short-circuits."""

    def _make_account(self):
        return Account(
            account_id="t",
            equity=100000,
            cash=100000,
            buying_power=100000,
            portfolio_value=100000,
            status="ACTIVE",
        )

    def test_guard_blocks_overlay_submit(self):
        existing = make_order(
            order_id="abc", symbol="SPY", side="buy", status="accepted"
        )
        broker = FakeBroker(orders=[existing])

        rejected = []
        submitted = []

        class FakeDB:
            def log_signal(self, **kwargs):
                return 42

            def mark_signal_rejected(self, sid, reason):
                rejected.append((sid, reason))

        class WrappedBroker(FakeBroker):
            def submit_order(self, order_request):
                submitted.append(order_request)
                raise AssertionError("submit_order must not be called")

        wrapped = WrappedBroker(orders=[existing])
        orch = _make_orch(wrapped)
        orch.db = FakeDB()

        result = orch._execute_overlay_order(
            symbol="SPY",
            side="buy",
            qty=10,
            account=self._make_account(),
            reasoning="overlay top-up",
        )

        assert result is not None
        assert result["traded"] is False
        assert result["overlay_managed"] is True
        assert "Existing open buy order abc" in result["risk_rejected"]
        assert rejected == [(42, result["risk_rejected"])]
        assert submitted == []

    def test_no_duplicate_allows_submit_attempt(self):
        """Sanity: guard does not block when broker has no matching open order."""
        broker_orders = []  # no existing orders
        submitted = []

        class FakeDB:
            def log_signal(self, **kwargs):
                return 99

            def mark_signal_rejected(self, sid, reason):
                raise AssertionError(
                    f"Should not reject when no duplicate (got {reason!r})"
                )

            def mark_signal_executed(self, sid):
                pass

            def log_trade(self, **kwargs):
                pass

        class FakeNotifier:
            enabled = False

            def send(self, *args, **kwargs):
                pass

        class StubBroker(FakeBroker):
            def submit_order(self, order_request):
                submitted.append(order_request)
                # Minimal OrderResult-shaped response
                return SimpleNamespace(
                    order_id="new-1",
                    status="accepted",
                    filled_qty=0.0,
                    filled_avg_price=None,
                )

        broker = StubBroker(orders=broker_orders)
        orch = _make_orch(broker)
        orch.db = FakeDB()
        orch.notifier = FakeNotifier()
        orch.config = {"strategy_tag": "test"}

        result = orch._execute_overlay_order(
            symbol="SPY",
            side="buy",
            qty=5,
            account=self._make_account(),
            reasoning="overlay add",
        )

        assert result is not None
        assert result["traded"] is True
        assert len(submitted) == 1
        assert submitted[0].symbol == "SPY"
        assert submitted[0].side == "buy"
