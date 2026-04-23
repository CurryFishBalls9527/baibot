"""Regression test: bracket-order submission must capture SL/TP leg IDs.

Pre-fix bug: Alpaca's initial `submit_order` response for a bracket returns
`legs=[]` because children materialise after the parent is accepted. The
broker captured the empty legs and returned `stop_order_id=None`, so the
orchestrator never persisted them. This silently disabled
`ExitManagerV2._maybe_update_broker_stop` (broker-side stop ratcheting).
21 live positions were affected across mechanical / llm / mechanical_v2
when this was found on 2026-04-22.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class _StubLeg:
    def __init__(self, id_, order_type, stop_price=None, limit_price=None):
        self.id = id_
        self.type = order_type
        self.order_type = order_type
        self.stop_price = stop_price
        self.limit_price = limit_price
        self.side = "sell"
        self.qty = 10
        self.submitted_at = None
        self.filled_at = None
        self.status = "held" if order_type == "stop" else "new"
        self.notional = None
        self.filled_qty = 0
        self.filled_avg_price = None
        self.symbol = "AAPL"
        self.legs = []


class _StubParent:
    def __init__(self, legs=None, status="filled", filled_avg_price=100.0):
        self.status = status
        self.filled_qty = 10
        self.filled_avg_price = filled_avg_price
        self.id = "parent-oid"
        self.symbol = "AAPL"
        self.side = "buy"
        self.qty = 10
        self.notional = None
        self.type = "market"
        self.submitted_at = None
        self.filled_at = None
        self.legs = legs or []


def _build_broker(submit_response, get_order_responses):
    """Build a broker where submit_order returns submit_response and
    successive get_order_by_id calls return items from get_order_responses.
    """
    with patch("tradingagents.broker.alpaca_broker.TradingClient") as TC:
        client = MagicMock()
        client.submit_order.return_value = submit_response
        client.get_order_by_id.side_effect = list(get_order_responses)
        TC.return_value = client

        from tradingagents.broker.alpaca_broker import AlpacaBroker
        from tradingagents.broker.models import OrderRequest

        broker = AlpacaBroker("k", "s", paper=True)
        return broker, OrderRequest, client


class BracketLegIDCaptureTests(unittest.TestCase):
    def test_refetch_populates_leg_ids_when_initial_response_empty(self):
        """Submit returns legs=[], refetch returns both children → IDs captured."""
        populated = _StubParent(
            legs=[
                _StubLeg("tp-oid", "limit", limit_price=115.0),
                _StubLeg("sl-oid", "stop", stop_price=95.0),
            ]
        )
        # Patch time.sleep so the test doesn't actually wait.
        with patch("tradingagents.broker.alpaca_broker.time.sleep"):
            broker, OrderRequest, client = _build_broker(
                submit_response=_StubParent(legs=[]),
                get_order_responses=[populated],
            )
            order = OrderRequest(
                symbol="AAPL", side="buy", qty=10,
                order_type="market", time_in_force="gtc",
            )
            result = broker.submit_bracket_order(
                order, stop_loss_price=95.0, take_profit_price=115.0
            )
        self.assertEqual(result.stop_order_id, "sl-oid")
        self.assertEqual(result.tp_order_id, "tp-oid")
        client.get_order_by_id.assert_called_with("parent-oid")

    def test_refetch_retries_until_legs_appear(self):
        """First refetch empty, second refetch populated → IDs captured on retry."""
        still_empty = _StubParent(legs=[])
        populated = _StubParent(
            legs=[
                _StubLeg("tp-oid-2", "limit", limit_price=120.0),
                _StubLeg("sl-oid-2", "stop", stop_price=90.0),
            ]
        )
        with patch("tradingagents.broker.alpaca_broker.time.sleep"):
            broker, OrderRequest, client = _build_broker(
                submit_response=_StubParent(legs=[]),
                get_order_responses=[still_empty, still_empty, populated],
            )
            order = OrderRequest(
                symbol="AAPL", side="buy", qty=10,
                order_type="market", time_in_force="gtc",
            )
            result = broker.submit_bracket_order(
                order, stop_loss_price=90.0, take_profit_price=120.0
            )
        self.assertEqual(result.stop_order_id, "sl-oid-2")
        self.assertEqual(result.tp_order_id, "tp-oid-2")
        # Three refetches expected: [empty, empty, populated]
        self.assertEqual(client.get_order_by_id.call_count, 3)

    def test_refetch_gives_up_gracefully_if_legs_never_appear(self):
        """Legs never materialise → warning logged, IDs remain None, no crash."""
        never_populated = _StubParent(legs=[])
        # submit_bracket_order has max_attempts=5 by default — simulate all empty
        responses = [never_populated] * 5
        with patch("tradingagents.broker.alpaca_broker.time.sleep"):
            broker, OrderRequest, client = _build_broker(
                submit_response=_StubParent(legs=[]),
                get_order_responses=responses,
            )
            order = OrderRequest(
                symbol="AAPL", side="buy", qty=10,
                order_type="market", time_in_force="gtc",
            )
            result = broker.submit_bracket_order(
                order, stop_loss_price=95.0, take_profit_price=115.0
            )
        self.assertIsNone(result.stop_order_id)
        self.assertIsNone(result.tp_order_id)
        self.assertEqual(client.get_order_by_id.call_count, 5)

    def test_refetch_skipped_if_initial_response_has_legs(self):
        """If Alpaca populates legs on submit (rare), don't refetch."""
        populated_initial = _StubParent(
            legs=[
                _StubLeg("tp-oid-fast", "limit", limit_price=115.0),
                _StubLeg("sl-oid-fast", "stop", stop_price=95.0),
            ]
        )
        broker, OrderRequest, client = _build_broker(
            submit_response=populated_initial,
            get_order_responses=[],
        )
        order = OrderRequest(
            symbol="AAPL", side="buy", qty=10,
            order_type="market", time_in_force="gtc",
        )
        result = broker.submit_bracket_order(
            order, stop_loss_price=95.0, take_profit_price=115.0
        )
        self.assertEqual(result.stop_order_id, "sl-oid-fast")
        self.assertEqual(result.tp_order_id, "tp-oid-fast")
        client.get_order_by_id.assert_not_called()


if __name__ == "__main__":
    unittest.main()
