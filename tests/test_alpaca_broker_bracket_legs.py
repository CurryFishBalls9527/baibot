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
        # _refetch_bracket_legs default max_attempts=30 — supply enough to
        # exhaust the budget; patched sleep keeps the test instant.
        responses = [never_populated] * 30
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
        self.assertEqual(client.get_order_by_id.call_count, 30)

    def test_refetch_waits_for_fill_before_returning(self):
        """Regression: leg IDs appear before fill → keep polling until fill.

        Pre-fix, refetch returned as soon as leg IDs were populated, even if
        filled_avg_price was still None. Callers that relied on fill price
        (re-anchor) silently skipped, so on fast-movers the broker SL stayed
        anchored to the stale pre-order quote. Now we keep polling until
        legs AND fill (or terminal status) are both available. This test also
        exercises the end-to-end chain: wait for fill → detect drift from
        anchor → replace SL/TP legs.
        """
        legs = [
            _StubLeg("tp-oid", "limit", limit_price=115.0),
            _StubLeg("sl-oid", "stop", stop_price=95.0),
        ]
        # 1st refetch: legs but not filled. 2nd: legs + fill. Should return on 2nd.
        pending = _StubParent(legs=legs, status="pending_new",
                              filled_avg_price=None)
        pending.filled_qty = 0
        filled = _StubParent(legs=legs, status="filled", filled_avg_price=97.5)
        # submit_response has filled_avg_price=None too — the fill should come
        # via refetch, exercising the "copy fill price from refetch" branch.
        submit_resp = _StubParent(legs=[], status="pending_new",
                                  filled_avg_price=None)
        submit_resp.filled_qty = 0
        with patch("tradingagents.broker.alpaca_broker.time.sleep"):
            broker, OrderRequest, client = _build_broker(
                submit_response=submit_resp,
                get_order_responses=[pending, filled],
            )
            # replace_order_by_id mock: return an object whose .id is a plain str.
            replaced_sl = MagicMock()
            replaced_sl.id = "sl-oid-reanchored"
            replaced_tp = MagicMock()
            replaced_tp.id = "tp-oid-reanchored"
            client.replace_order_by_id.side_effect = [replaced_sl, replaced_tp]
            order = OrderRequest(
                symbol="AAPL", side="buy", qty=10,
                order_type="market", time_in_force="gtc",
            )
            # anchor=100, fill=97.5 → drift=250bps, triggers re-anchor.
            # Intended SL buffer = (100-95)/100 = 5% → new SL = 97.5*0.95 = 92.63
            result = broker.submit_bracket_order(
                order, stop_loss_price=95.0, take_profit_price=115.0,
                anchor_price=100.0,
            )
        # filled_avg_price must propagate from refetch.
        self.assertEqual(result.filled_avg_price, 97.5)
        # Refetch polled twice — first pending, second filled.
        self.assertEqual(client.get_order_by_id.call_count, 2)
        # Re-anchor fired because drift > 10bps: SL/TP legs replaced.
        self.assertEqual(result.stop_order_id, "sl-oid-reanchored")
        self.assertEqual(result.tp_order_id, "tp-oid-reanchored")
        self.assertAlmostEqual(result.effective_stop_price, 92.62, places=2)
        # Verify the actual replace request — SL rebased to 5% below fill,
        # TP to 15% above fill.
        sl_call = client.replace_order_by_id.call_args_list[0]
        self.assertEqual(sl_call.args[0], "sl-oid")
        self.assertAlmostEqual(sl_call.args[1].stop_price, 92.62, places=2)

    def test_refetch_returns_on_terminal_status_even_without_fill(self):
        """Canceled/rejected parent → don't keep polling, return promptly."""
        legs = [
            _StubLeg("tp-oid", "limit", limit_price=115.0),
            _StubLeg("sl-oid", "stop", stop_price=95.0),
        ]
        canceled = _StubParent(legs=legs, status="canceled", filled_avg_price=None)
        canceled.filled_qty = 0
        with patch("tradingagents.broker.alpaca_broker.time.sleep"):
            broker, OrderRequest, client = _build_broker(
                submit_response=_StubParent(legs=[]),
                get_order_responses=[canceled],
            )
            order = OrderRequest(
                symbol="AAPL", side="buy", qty=10,
                order_type="market", time_in_force="gtc",
            )
            result = broker.submit_bracket_order(
                order, stop_loss_price=95.0, take_profit_price=115.0,
            )
        self.assertEqual(result.stop_order_id, "sl-oid")
        self.assertEqual(client.get_order_by_id.call_count, 1)

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


class StaleQuoteRetryTests(unittest.TestCase):
    """Regression: stale IEX mid on thin names can push SL above Alpaca's
    base_price, triggering 42210000 "stop_loss.stop_price must be <=
    base_price - 0.01". See 2026-04-24 AAOI incident and 2026-04-20 VSCO.
    """

    def test_retries_with_fresh_price_on_sl_above_base_reject(self):
        from alpaca.common.exceptions import APIError
        first_err = APIError(
            '{"code":42210000,"base_price":"163.32",'
            '"message":"stop_loss.stop_price must be <= base_price - 0.01"}'
        )
        filled = _StubParent(
            legs=[
                _StubLeg("tp-oid", "limit", limit_price=200.0),
                _StubLeg("sl-oid", "stop", stop_price=155.15),
            ],
            status="filled",
            filled_avg_price=163.0,
        )
        with patch("tradingagents.broker.alpaca_broker.TradingClient") as TC, \
             patch("tradingagents.broker.alpaca_broker.time.sleep"):
            client = MagicMock()
            # submit_order: raise on first call, return filled on second.
            client.submit_order.side_effect = [first_err, filled]
            client.get_order_by_id.return_value = filled
            TC.return_value = client

            from tradingagents.broker.alpaca_broker import AlpacaBroker
            from tradingagents.broker.models import OrderRequest

            broker = AlpacaBroker("k", "s", paper=True)
            # Stub fresh-quote fetch: SIP price is $163.28 vs anchor $172.56.
            broker.get_latest_price = MagicMock(return_value=163.28)

            order = OrderRequest(
                symbol="AAOI", side="buy", qty=70,
                order_type="market", time_in_force="gtc",
            )
            # Anchor $172.56, original SL $163.93 (5% below anchor). Retry
            # should clamp SL against fresh price $163.28 → $155.12.
            result = broker.submit_bracket_order(
                order, stop_loss_price=163.93, take_profit_price=198.44,
                anchor_price=172.56,
            )

        # Two submit attempts: first raised, second succeeded.
        self.assertEqual(client.submit_order.call_count, 2)
        # Second submit request should carry the clamped SL (≈$155.11),
        # not the original stale $163.93.
        second_req = client.submit_order.call_args_list[1].args[0]
        new_sl_price = float(second_req.stop_loss.stop_price)
        self.assertLess(new_sl_price, 163.28)
        self.assertAlmostEqual(new_sl_price, 155.11, places=1)
        # Retry produced a non-None result (downstream re-anchor may
        # further mutate stop_order_id; we only care that retry succeeded).
        self.assertIsNotNone(result)

    def test_does_not_retry_on_unrelated_api_error(self):
        """Only the specific SL-vs-base_price reject triggers retry. Other
        42210000 errors (e.g. insufficient buying power) must bubble up."""
        from alpaca.common.exceptions import APIError
        unrelated = APIError('{"code":40310000,"message":"some other error"}')
        with patch("tradingagents.broker.alpaca_broker.TradingClient") as TC, \
             patch("tradingagents.broker.alpaca_broker.time.sleep"):
            client = MagicMock()
            client.submit_order.side_effect = [unrelated]
            TC.return_value = client

            from tradingagents.broker.alpaca_broker import AlpacaBroker
            from tradingagents.broker.models import OrderRequest

            broker = AlpacaBroker("k", "s", paper=True)
            broker.get_latest_price = MagicMock(return_value=100.0)

            order = OrderRequest(
                symbol="AAPL", side="buy", qty=10,
                order_type="market", time_in_force="gtc",
            )
            with self.assertRaises(APIError):
                broker.submit_bracket_order(
                    order, stop_loss_price=95.0, take_profit_price=115.0,
                    anchor_price=100.0,
                )
        # Only one submit attempt — no retry for unrelated errors.
        self.assertEqual(client.submit_order.call_count, 1)

    def test_abandons_retry_when_fresh_quote_unavailable(self):
        """If fresh-quote fetch returns 0/negative, the error bubbles up —
        can't compute a valid clamped SL without a price."""
        from alpaca.common.exceptions import APIError
        reject = APIError(
            '{"code":42210000,"base_price":"90.00",'
            '"message":"stop_loss.stop_price must be <= base_price - 0.01"}'
        )
        with patch("tradingagents.broker.alpaca_broker.TradingClient") as TC, \
             patch("tradingagents.broker.alpaca_broker.time.sleep"):
            client = MagicMock()
            client.submit_order.side_effect = [reject]
            TC.return_value = client

            from tradingagents.broker.alpaca_broker import AlpacaBroker
            from tradingagents.broker.models import OrderRequest

            broker = AlpacaBroker("k", "s", paper=True)
            # Quote fetch returns 0 — broker can't recover.
            broker.get_latest_price = MagicMock(return_value=0.0)

            order = OrderRequest(
                symbol="AAPL", side="buy", qty=10,
                order_type="market", time_in_force="gtc",
            )
            with self.assertRaises(APIError):
                broker.submit_bracket_order(
                    order, stop_loss_price=95.0, take_profit_price=115.0,
                    anchor_price=100.0,
                )
        self.assertEqual(client.submit_order.call_count, 1)


if __name__ == "__main__":
    unittest.main()
