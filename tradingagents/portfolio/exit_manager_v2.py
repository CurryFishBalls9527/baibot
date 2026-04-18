"""Active position management v2 — ports B1→B4L backtester behaviors to live.

Adds on top of v1 (`exit_manager.py`):
- Dead-money stop (B1): exit if bars_held >= max_days AND return_pct < min_gain
- Partial-profit guard (B2): no-op when partial_profit_fraction <= 0
- Breakeven lock offset (B3L): ratchet stop to entry*(1+offset), not just entry
- Regime-dependent trail (B4L): tighter trail in market_correction
- Broker stop ratcheting: on HOLD, if current_stop rose, call
  broker.replace_order(stop_order_id, stop_price=current_stop) so the Alpaca
  OCO stop-leg matches local state.

Wired behind `exit_manager_version: "v2"` config key in the orchestrator. v1
remains the default; existing variants (mechanical/llm/chan) are untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional, Dict

from dateutil.parser import parse

logger = logging.getLogger(__name__)


@dataclass
class ExitDecisionV2:
    """Result of evaluating an open position (v2)."""

    action: str  # "SELL", "PARTIAL_SELL", "HOLD"
    qty: float = 0
    reason: str = ""
    updated_state: Optional[Dict] = None


class ExitManagerV2:
    """Live exit manager with parity to B4L backtester semantics."""

    def __init__(self, config: dict, broker=None):
        # v1 parameters (unchanged defaults)
        self.trail_stop_pct = float(config.get("trail_stop_pct", 0.10))
        self.breakeven_trigger_pct = float(config.get("breakeven_trigger_pct", 0.08))
        self.breakeven_lock_offset_pct = float(config.get("breakeven_lock_offset_pct", 0.01))
        self.partial_profit_trigger_pct = float(config.get("partial_profit_trigger_pct", 0.12))
        self.partial_profit_fraction = float(config.get("partial_profit_fraction", 0.0))
        self.max_hold_days = int(config.get("max_hold_days", 60))
        self.use_50dma_exit = bool(config.get("use_50dma_exit", True))

        # B1 — dead-money stop
        self.dead_money_enabled = bool(config.get("dead_money_enabled", True))
        self.dead_money_min_gain_pct = float(config.get("dead_money_min_gain_pct", 0.03))
        self.dead_money_max_days = int(config.get("dead_money_max_days", 10))

        # B4L — regime-dependent trail
        self.regime_aware_trail = bool(config.get("regime_aware_trail", True))
        self.trail_stop_pct_uptrend = float(config.get("trail_stop_pct_uptrend", 0.10))
        self.trail_stop_pct_pressure = float(config.get("trail_stop_pct_pressure", 0.10))
        self.trail_stop_pct_correction = float(config.get("trail_stop_pct_correction", 0.07))

        self.broker = broker

    # ── regime-aware trail width ─────────────────────────────────────

    def _trail_pct_for_regime(self, regime_label: Optional[str]) -> float:
        if not self.regime_aware_trail or regime_label is None:
            return self.trail_stop_pct
        if regime_label == "confirmed_uptrend":
            return self.trail_stop_pct_uptrend
        if regime_label == "market_correction":
            return self.trail_stop_pct_correction
        return self.trail_stop_pct_pressure

    # ── broker-side stop ratchet ─────────────────────────────────────

    def _maybe_update_broker_stop(
        self,
        symbol: str,
        stop_order_id: Optional[str],
        new_stop_price: float,
        prev_stop_price: float,
    ) -> None:
        """Tell the broker about a ratcheted stop. No-op if no broker / no leg ID.

        Known gap (2026-04-17 smoke test): Alpaca rejects replace_order on
        ACCEPTED orders (code 42210000). Can happen if this is called off-hours
        while the stop leg is queued. Current mitigation: caller's current_stop
        advances anyway, so next RTH tick with further price gain will retry.
        If the broker-side drift ever surfaces in practice, fix by tracking
        desired_stop vs broker_stop separately in position_states and retrying
        whenever desired > broker.
        """
        if self.broker is None or not stop_order_id:
            return
        # Only ratchet UP — never loosen the broker stop.
        if new_stop_price <= prev_stop_price + 1e-6:
            return
        try:
            self.broker.replace_order(stop_order_id, stop_price=round(new_stop_price, 2))
            logger.info(
                f"{symbol}: broker stop ratcheted ${prev_stop_price:.2f} -> ${new_stop_price:.2f} "
                f"(order {stop_order_id})"
            )
        except Exception as e:
            logger.warning(
                f"{symbol}: broker stop update failed ({stop_order_id}): {e}. "
                f"Local stop={new_stop_price:.2f}, broker stop remains stale."
            )

    # ── main evaluation ──────────────────────────────────────────────

    def check_position(
        self,
        position,
        features: Dict,
        position_state: Dict,
        regime_label: Optional[str] = None,
    ) -> ExitDecisionV2:
        """Evaluate an open position and return an exit decision (v2)."""
        price = float(position.current_price)
        entry_price = float(position_state["entry_price"])
        highest_close = max(float(position_state["highest_close"]), price)
        prev_stop = float(position_state["current_stop"])
        current_stop = prev_stop
        partial_taken = bool(position_state.get("partial_taken", False))
        stop_order_id = position_state.get("stop_order_id")
        symbol = getattr(position, "symbol", "?")

        # Regime-aware trailing stop
        trail_pct = self._trail_pct_for_regime(regime_label)
        trail_stop = highest_close * (1.0 - trail_pct)
        current_stop = max(current_stop, trail_stop)

        # Breakeven + lock offset (B3L)
        if price >= entry_price * (1.0 + self.breakeven_trigger_pct):
            lock_price = entry_price * (1.0 + self.breakeven_lock_offset_pct)
            current_stop = max(current_stop, lock_price)

        # Hold duration
        try:
            entry_date = parse(position_state["entry_date"]).date()
        except (ValueError, TypeError):
            entry_date = date.today()
        hold_days = (date.today() - entry_date).days
        return_pct = (price - entry_price) / entry_price if entry_price > 0 else 0.0

        # Exit priority:
        # 1. Hard stop (trailing or breakeven-locked)
        if price <= current_stop:
            return ExitDecisionV2("SELL", float(position.qty), "trailing_stop")

        # 2. Dead-money stop (B1)
        if (
            self.dead_money_enabled
            and hold_days >= self.dead_money_max_days
            and return_pct < self.dead_money_min_gain_pct
        ):
            return ExitDecisionV2(
                "SELL",
                float(position.qty),
                f"dead_money_{self.dead_money_max_days}d_{int(self.dead_money_min_gain_pct*100)}pct",
            )

        # 3. 50-DMA exit
        if (
            self.use_50dma_exit
            and features.get("sma_50")
            and price < float(features["sma_50"])
        ):
            return ExitDecisionV2("SELL", float(position.qty), "lost_50dma")

        # 4. Max hold days
        if hold_days >= self.max_hold_days:
            return ExitDecisionV2("SELL", float(position.qty), "max_hold_days")

        # 5. Partial profit (skipped entirely when fraction <= 0 — B2 fix)
        if (
            not partial_taken
            and self.partial_profit_fraction > 0.0
            and price >= entry_price * (1.0 + self.partial_profit_trigger_pct)
        ):
            partial_qty = max(1, int(float(position.qty) * self.partial_profit_fraction))
            lock_price = entry_price * (1.0 + self.breakeven_lock_offset_pct)
            post_stop = max(current_stop, lock_price)
            # Broker stop ratchet on partial — stop tightened to breakeven+offset
            self._maybe_update_broker_stop(symbol, stop_order_id, post_stop, prev_stop)
            return ExitDecisionV2(
                "PARTIAL_SELL",
                partial_qty,
                "partial_profit",
                updated_state={
                    "entry_price": entry_price,
                    "entry_date": position_state["entry_date"],
                    "highest_close": highest_close,
                    "current_stop": post_stop,
                    "partial_taken": True,
                    "stop_type": "breakeven",
                    "stop_order_id": stop_order_id,
                },
            )

        # 6. Hold — ratchet broker stop if local stop rose
        self._maybe_update_broker_stop(symbol, stop_order_id, current_stop, prev_stop)

        return ExitDecisionV2(
            "HOLD",
            0,
            "",
            updated_state={
                "entry_price": entry_price,
                "entry_date": position_state["entry_date"],
                "highest_close": highest_close,
                "current_stop": current_stop,
                "partial_taken": partial_taken,
                "stop_type": position_state.get("stop_type", "trailing"),
                "stop_order_id": stop_order_id,
            },
        )
