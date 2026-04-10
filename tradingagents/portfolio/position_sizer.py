"""Convert AI signals into concrete position sizes."""

import logging
import math
from typing import Optional

from tradingagents.broker.models import OrderRequest, Account, Position

logger = logging.getLogger(__name__)


class PositionSizer:
    """Translates BUY/SELL/HOLD signals into specific share quantities."""

    def __init__(self, config: dict):
        self.max_position_pct = config.get("max_position_pct", 0.10)
        self.max_total_exposure = config.get("max_total_exposure", 0.80)
        self.risk_per_trade = config.get("risk_per_trade", 0.02)

    def calculate(
        self,
        signal: dict,
        account: Account,
        current_price: float,
        current_position: Optional[Position] = None,
        total_position_value: float = 0.0,
        atr: float = None,
    ) -> Optional[OrderRequest]:
        """
        Calculate order based on signal and account state.

        Args:
            signal: dict with 'action', optional 'confidence', 'stop_loss'
            account: current account state
            current_price: latest price of the symbol
            current_position: existing position if any
            total_position_value: sum of all current position market values

        Returns:
            OrderRequest or None (for HOLD or if already positioned correctly)
        """
        action = signal.get("action", "HOLD").upper()
        symbol = signal.get("symbol", "")
        confidence = signal.get("confidence", 0.5)

        if action == "HOLD":
            return None

        if action == "BUY":
            return self._size_buy(
                symbol, confidence, account, current_price,
                current_position, total_position_value, signal, atr
            )

        if action == "SELL":
            return self._size_sell(symbol, current_position)

        logger.warning(f"Unknown action '{action}' for {symbol}")
        return None

    def _size_buy(
        self, symbol: str, confidence: float, account: Account,
        current_price: float, current_position: Optional[Position],
        total_position_value: float, signal: dict, atr: float = None,
    ) -> Optional[OrderRequest]:

        if current_position and current_position.qty > 0:
            logger.info(f"Already long {current_position.qty} shares of {symbol}, skipping BUY")
            return None

        equity = account.equity
        max_for_this_stock = equity * self.max_position_pct
        remaining_exposure = (equity * self.max_total_exposure) - total_position_value
        available = min(max_for_this_stock, remaining_exposure, float(account.buying_power))

        if available <= 0:
            logger.info(f"No available capital for {symbol}")
            return None

        # Determine risk per share from stop-loss, ATR, or fallback to confidence
        stop_loss = signal.get("stop_loss")
        if stop_loss and stop_loss > 0 and stop_loss < current_price:
            risk_per_share = current_price - stop_loss
        elif atr and atr > 0:
            risk_per_share = atr * 2.0  # 2x ATR as default risk
        else:
            risk_per_share = None

        if risk_per_share and risk_per_share > 0:
            max_risk_amount = equity * self.risk_per_trade
            qty_by_risk = math.floor(max_risk_amount / risk_per_share)
            qty_by_capital = math.floor(available / current_price)
            qty = min(qty_by_risk, qty_by_capital)
        else:
            scale = 0.5 + (confidence * 0.5)
            qty = math.floor((available * scale) / current_price)

        if qty <= 0:
            logger.info(f"Calculated qty=0 for {symbol} at ${current_price:.2f}")
            return None

        logger.info(f"Sized BUY: {qty} shares of {symbol} @ ~${current_price:.2f} (${qty * current_price:,.0f})")
        return OrderRequest(symbol=symbol, side="buy", qty=float(qty))

    def _size_sell(self, symbol: str, current_position: Optional[Position]) -> Optional[OrderRequest]:
        if not current_position or current_position.qty <= 0:
            logger.info(f"No position to sell for {symbol}")
            return None

        qty = current_position.qty
        logger.info(f"Sized SELL: {qty} shares of {symbol}")
        return OrderRequest(symbol=symbol, side="sell", qty=float(qty))
