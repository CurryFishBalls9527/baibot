"""Active position management — runs daily for each open position."""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Dict

from dateutil.parser import parse

logger = logging.getLogger(__name__)


@dataclass
class ExitDecision:
    """Result of evaluating an open position."""

    action: str  # "SELL", "PARTIAL_SELL", "HOLD"
    qty: float = 0
    reason: str = ""
    updated_state: Optional[Dict] = None


class ExitManager:
    """Active position management — runs daily for each open position.

    Ports the backtester's exit logic (trailing stop, breakeven, partial profit,
    50 DMA exit, max hold days) into live trading.
    """

    def __init__(self, config: dict):
        self.trail_stop_pct = config.get("trail_stop_pct", 0.10)
        self.breakeven_trigger_pct = config.get("breakeven_trigger_pct", 0.05)
        self.partial_profit_trigger_pct = config.get("partial_profit_trigger_pct", 0.12)
        self.partial_profit_fraction = config.get("partial_profit_fraction", 0.33)
        self.max_hold_days = config.get("max_hold_days", 60)
        self.use_50dma_exit = config.get("use_50dma_exit", True)

    def check_position(self, position, features: Dict, position_state: Dict) -> ExitDecision:
        """Evaluate an open position and return an exit decision.

        Args:
            position: Alpaca Position object (needs .current_price, .qty)
            features: Dict with optional keys: sma_50, rs_percentile, adx_14, etc.
            position_state: Dict with entry_price, entry_date, highest_close,
                           current_stop, partial_taken

        Returns:
            ExitDecision with action, qty, reason, and updated_state
        """
        price = float(position.current_price)
        entry_price = float(position_state["entry_price"])
        highest_close = max(float(position_state["highest_close"]), price)
        current_stop = float(position_state["current_stop"])
        partial_taken = bool(position_state.get("partial_taken", False))

        # Trailing stop: ratchet up as price moves higher
        trail_stop = highest_close * (1.0 - self.trail_stop_pct)
        current_stop = max(current_stop, trail_stop)

        # Breakeven stop: once position is up by trigger %, move stop to entry
        if price >= entry_price * (1.0 + self.breakeven_trigger_pct):
            current_stop = max(current_stop, entry_price)

        # Calculate hold duration
        try:
            entry_date = parse(position_state["entry_date"]).date()
        except (ValueError, TypeError):
            entry_date = date.today()
        hold_days = (date.today() - entry_date).days

        # Check exit conditions in priority order
        if price <= current_stop:
            return ExitDecision("SELL", float(position.qty), "trailing_stop")

        if self.use_50dma_exit and features.get("sma_50") and price < float(features["sma_50"]):
            return ExitDecision("SELL", float(position.qty), "lost_50dma")

        if hold_days >= self.max_hold_days:
            return ExitDecision("SELL", float(position.qty), "max_hold_days")

        # Partial profit taking
        if not partial_taken and price >= entry_price * (1.0 + self.partial_profit_trigger_pct):
            partial_qty = max(1, int(float(position.qty) * self.partial_profit_fraction))
            return ExitDecision(
                "PARTIAL_SELL",
                partial_qty,
                "partial_profit",
                updated_state={
                    "entry_price": entry_price,
                    "entry_date": position_state["entry_date"],
                    "highest_close": highest_close,
                    "current_stop": max(current_stop, entry_price),
                    "partial_taken": True,
                    "stop_type": "breakeven",
                },
            )

        # Hold — update tracking state
        return ExitDecision(
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
            },
        )
