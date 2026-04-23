"""Parity tests for partial-exit + runner take-profit.

Locks in the contract between the Minervini backtester's progressive path
(`research/backtester.py::_backtest_symbol_progressive`, partial block at
lines 604-622) and the live `ExitManagerV2` (`portfolio/exit_manager_v2.py`,
partial branch at lines 177-201).

Scope: single-lot (no add-ons) case — matches current paper_launch_v2.yaml
where `progressive_entries` / add-ons are disabled. Add-on parity is a known
gap (backtester averages cost across lots; live uses the original
entry_price) documented in the Phase-B plan risks.

The two implementations differ in ONE important way:

- Backtester evaluates partial BEFORE the stop check each bar (lines 604-625,
  then stop check at 630).
- ExitManagerV2 evaluates stop BEFORE partial (lines 150-151, then partial
  at 177).

So when a bar satisfies both "price >= partial_trigger" AND "price <= stop",
the backtester books the partial and possibly the stop (on the remainder),
while live books a full SELL at the stop. These tests pin the live ordering
as the ground truth for what the sweep is allowed to shape around.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from tradingagents.portfolio.exit_manager_v2 import ExitManagerV2


@dataclass
class MockPosition:
    symbol: str
    current_price: float
    qty: float
    avg_entry_price: float
    market_value: float = 0.0


def _config(**overrides):
    base = {
        "trail_stop_pct": 0.10,
        "breakeven_trigger_pct": 0.08,
        "breakeven_lock_offset_pct": 0.01,
        "partial_profit_trigger_pct": 0.12,
        "partial_profit_fraction": 0.33,
        "max_hold_days": 60,
        "use_50dma_exit": False,
        "dead_money_enabled": False,
        "regime_aware_trail": False,
    }
    base.update(overrides)
    return base


def _state(entry_price=100.0, days_ago=5, **overrides):
    s = {
        "entry_price": entry_price,
        "entry_date": (date.today() - timedelta(days=days_ago)).isoformat(),
        "highest_close": entry_price,
        "current_stop": entry_price * 0.92,
        "partial_taken": False,
        "stop_order_id": None,
    }
    s.update(overrides)
    return s


class TestPartialFires:
    def test_fires_at_trigger(self):
        em = ExitManagerV2(_config())
        pos = MockPosition("AAPL", 112.01, 100, 100.0)
        decision = em.check_position(pos, {}, _state(100.0))
        assert decision.action == "PARTIAL_SELL"
        assert decision.reason == "partial_profit"
        assert decision.qty == 33
        assert decision.updated_state["partial_taken"] is True

    def test_does_not_fire_below_trigger(self):
        em = ExitManagerV2(_config())
        pos = MockPosition("AAPL", 111.99, 100, 100.0)
        decision = em.check_position(pos, {}, _state(100.0))
        assert decision.action == "HOLD"

    def test_does_not_double_fire(self):
        em = ExitManagerV2(_config())
        pos = MockPosition("AAPL", 120.0, 67, 100.0)
        decision = em.check_position(pos, {}, _state(100.0, partial_taken=True))
        assert decision.action == "HOLD"

    def test_disabled_when_fraction_zero(self):
        em = ExitManagerV2(_config(partial_profit_fraction=0.0))
        pos = MockPosition("AAPL", 150.0, 100, 100.0)
        decision = em.check_position(pos, {}, _state(100.0))
        assert decision.action == "HOLD"


class TestPartialStateUpdate:
    def test_stop_ratchets_to_breakeven_lock(self):
        em = ExitManagerV2(_config())
        pos = MockPosition("AAPL", 113.0, 100, 100.0)
        decision = em.check_position(pos, {}, _state(100.0))
        assert decision.action == "PARTIAL_SELL"
        # lock_price = entry * (1 + breakeven_lock_offset_pct) = 101
        assert decision.updated_state["current_stop"] >= 101.0

    def test_partial_taken_persisted(self):
        em = ExitManagerV2(_config())
        pos = MockPosition("AAPL", 113.0, 100, 100.0)
        d1 = em.check_position(pos, {}, _state(100.0))
        assert d1.updated_state["partial_taken"] is True

        # Simulate re-tick with the updated state and reduced qty (67 after 33 sold).
        pos2 = MockPosition("AAPL", 118.0, 67, 100.0)
        d2 = em.check_position(pos2, {}, d1.updated_state)
        assert d2.action == "HOLD"

    def test_stop_order_id_preserved(self):
        em = ExitManagerV2(_config())
        pos = MockPosition("AAPL", 113.0, 100, 100.0)
        state = _state(100.0, stop_order_id="order-abc")
        decision = em.check_position(pos, {}, state)
        assert decision.updated_state["stop_order_id"] == "order-abc"


class TestStopPreemptsPartial:
    """Live ordering: trailing_stop wins over partial_profit on the same bar.

    Pins the live semantics. The sweep's MFE/PnL reporting must use the same
    ordering to remain faithful to what will actually execute.
    """

    def test_stop_wins_when_both_would_trigger(self):
        em = ExitManagerV2(_config(trail_stop_pct=0.04))
        # highest_close=117, trail_stop = 117 * 0.96 = 112.32
        # price=112.0 sits below trail_stop; price/entry = 1.12 exactly meets partial trigger.
        # Live order: stop check (150) runs first → SELL full qty.
        pos = MockPosition("AAPL", 112.0, 100, 100.0)
        state = _state(100.0, highest_close=117.0, current_stop=112.32)
        decision = em.check_position(pos, {}, state)
        assert decision.action == "SELL"
        assert decision.reason == "trailing_stop"
        assert decision.qty == 100


class TestFractionRounding:
    def test_rounds_down(self):
        em = ExitManagerV2(_config(partial_profit_fraction=0.33))
        pos = MockPosition("AAPL", 113.0, 100, 100.0)
        decision = em.check_position(pos, {}, _state(100.0))
        assert decision.qty == 33

    def test_minimum_one_share(self):
        # 1 * 0.33 = 0.33 → floor = 0, but manager enforces min 1.
        em = ExitManagerV2(_config(partial_profit_fraction=0.33))
        pos = MockPosition("AAPL", 113.0, 1, 100.0)
        decision = em.check_position(pos, {}, _state(100.0))
        assert decision.action == "PARTIAL_SELL"
        assert decision.qty == 1

    def test_half_fraction(self):
        em = ExitManagerV2(_config(partial_profit_fraction=0.50))
        pos = MockPosition("AAPL", 113.0, 100, 100.0)
        decision = em.check_position(pos, {}, _state(100.0))
        assert decision.qty == 50


class TestWalkForwardParity:
    """Walk a synthetic price series and verify the live evaluator matches
    a reference implementation of the backtester's single-lot partial path.

    The reference uses the backtester's FORMULA
    (`price >= entry * (1 + trigger)`, `partial_qty = max(1, int(qty*frac))`,
    stop ratcheted to entry after partial) but applies the LIVE ordering
    (stop preempts partial) so divergence from the backtester's own ordering
    is captured outside this test, not inside.
    """

    def _reference_walk(self, prices, entry, qty, cfg):
        """Reference: single-lot progressive path with live ordering."""
        state = {
            "qty": qty,
            "highest_close": entry,
            "stop": entry * (1.0 - cfg["stop_loss_pct"]),
            "partial_taken": False,
            "exits": [],
        }
        for p in prices:
            state["highest_close"] = max(state["highest_close"], p)
            # Trail ratchet
            trail = state["highest_close"] * (1.0 - cfg["trail_stop_pct"])
            state["stop"] = max(state["stop"], trail)
            # Breakeven
            if p >= entry * (1.0 + cfg["breakeven_trigger_pct"]):
                state["stop"] = max(
                    state["stop"], entry * (1.0 + cfg["breakeven_lock_offset_pct"])
                )
            # Live ordering: stop first
            if p <= state["stop"]:
                state["exits"].append(("SELL", state["qty"], p))
                state["qty"] = 0
                break
            # Partial
            if (
                not state["partial_taken"]
                and cfg["partial_profit_fraction"] > 0.0
                and p >= entry * (1.0 + cfg["partial_profit_trigger_pct"])
            ):
                partial = max(1, int(state["qty"] * cfg["partial_profit_fraction"]))
                state["exits"].append(("PARTIAL_SELL", partial, p))
                state["qty"] -= partial
                state["partial_taken"] = True
                state["stop"] = max(
                    state["stop"],
                    entry * (1.0 + cfg["breakeven_lock_offset_pct"]),
                )
        return state["exits"]

    def _live_walk(self, prices, entry, qty, cfg):
        em = ExitManagerV2(cfg)
        state = _state(entry, current_stop=entry * (1.0 - cfg["stop_loss_pct"]))
        live_qty = qty
        exits = []
        for p in prices:
            pos = MockPosition("AAPL", p, live_qty, entry)
            d = em.check_position(pos, {}, state)
            if d.action == "SELL":
                exits.append(("SELL", live_qty, p))
                break
            if d.action == "PARTIAL_SELL":
                exits.append(("PARTIAL_SELL", d.qty, p))
                live_qty -= d.qty
                state = d.updated_state
                continue
            state = d.updated_state
        return exits

    def test_monotone_up_then_pullback(self):
        cfg = _config(stop_loss_pct=0.08, trail_stop_pct=0.10)
        prices = [100, 105, 110, 113, 118, 120, 115, 108, 105]
        ref = self._reference_walk(prices, 100, 100, cfg)
        live = self._live_walk(prices, 100, 100, cfg)
        assert ref == live
        # Should see partial at 113, then final stop when trail catches up.
        assert ref[0] == ("PARTIAL_SELL", 33, 113)
        # After partial: qty=67, stop ratchets with HC=120 → trail at 108.
        # Price 108 exactly touches stop → SELL.
        assert ref[-1][0] == "SELL"

    def test_straight_up_no_stop(self):
        cfg = _config(stop_loss_pct=0.08, trail_stop_pct=0.10)
        prices = [100, 105, 110, 113, 116, 119, 122]
        ref = self._reference_walk(prices, 100, 100, cfg)
        live = self._live_walk(prices, 100, 100, cfg)
        assert ref == live
        assert len(ref) == 1
        assert ref[0] == ("PARTIAL_SELL", 33, 113)

    def test_gap_down_through_stop_before_partial(self):
        cfg = _config(stop_loss_pct=0.08, trail_stop_pct=0.10)
        # Straight to 113 (partial trigger), then gap to 91 (below 92 stop).
        prices = [100, 113, 91]
        ref = self._reference_walk(prices, 100, 100, cfg)
        live = self._live_walk(prices, 100, 100, cfg)
        assert ref == live
        # Bar 1: partial at 113. Bar 2: stop fires on remaining 67 at 91.
        assert ref == [("PARTIAL_SELL", 33, 113), ("SELL", 67, 91)]

    def test_disabled_partial_matches_baseline(self):
        """With fraction=0, both paths behave like the no-partial baseline."""
        cfg = _config(partial_profit_fraction=0.0, stop_loss_pct=0.08, trail_stop_pct=0.10)
        prices = [100, 105, 110, 113, 118, 120, 115, 108, 105]
        ref = self._reference_walk(prices, 100, 100, cfg)
        live = self._live_walk(prices, 100, 100, cfg)
        assert ref == live
        assert all(e[0] != "PARTIAL_SELL" for e in ref)
