"""Regression test for bracket TIF respecting OrderRequest.time_in_force.

Pre-fix bug: alpaca_broker.submit_bracket_order hardcoded TimeInForce.GTC,
silently overriding the intraday orchestrator's "day" TIF and risking
overnight positions on parents that filled after EOD flatten.
"""
import unittest
from unittest.mock import MagicMock, patch

from alpaca.trading.enums import TimeInForce


class _StubResult:
    def __init__(self, symbol="AAPL", qty=10):
        self.status = "filled"
        self.filled_qty = qty
        self.filled_avg_price = 100.0
        self.id = "oid-1"
        self.symbol = symbol
        self.side = "buy"
        self.qty = qty
        self.notional = None
        self.type = "market"
        self.submitted_at = None
        self.filled_at = None
        self.legs = []


def _build_broker_with_request_capture():
    captured = {}

    with patch("tradingagents.broker.alpaca_broker.TradingClient") as TC:
        client = MagicMock()
        def _capture(req):
            captured["req"] = req
            return _StubResult(symbol=req.symbol, qty=req.qty)
        client.submit_order.side_effect = _capture
        TC.return_value = client

        from tradingagents.broker.alpaca_broker import AlpacaBroker
        from tradingagents.broker.models import OrderRequest

        broker = AlpacaBroker("k", "s", paper=True)
        return broker, OrderRequest, captured


class BracketTIFTests(unittest.TestCase):

    def test_intraday_day_tif_propagates_to_alpaca(self):
        broker, OrderRequest, captured = _build_broker_with_request_capture()
        order = OrderRequest(
            symbol="AAPL", side="buy", qty=10,
            order_type="market", time_in_force="day",
        )
        broker.submit_bracket_order(order, stop_loss_price=97.0, take_profit_price=200.0)
        self.assertEqual(captured["req"].time_in_force, TimeInForce.DAY)

    def test_daily_gtc_tif_propagates_to_alpaca(self):
        broker, OrderRequest, captured = _build_broker_with_request_capture()
        order = OrderRequest(
            symbol="NVDA", side="buy", qty=5,
            order_type="market", time_in_force="gtc",
        )
        broker.submit_bracket_order(order, stop_loss_price=400.0, take_profit_price=500.0)
        self.assertEqual(captured["req"].time_in_force, TimeInForce.GTC)


if __name__ == "__main__":
    unittest.main()
