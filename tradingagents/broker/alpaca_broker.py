"""Alpaca broker implementation for paper and live trading."""

import logging
from typing import List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    StopLimitOrderRequest,
    TrailingStopOrderRequest,
    ClosePositionRequest,
    GetOrdersRequest,
    ReplaceOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderClass, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

from .base_broker import BaseBroker
from .models import OrderRequest, OrderResult, Position, Account, MarketClock

logger = logging.getLogger(__name__)

TIF_MAP = {
    "day": TimeInForce.DAY,
    "gtc": TimeInForce.GTC,
    "ioc": TimeInForce.IOC,
}


class AlpacaBroker(BaseBroker):
    """Alpaca Markets broker — works for both paper and live accounts."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.paper = paper
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        mode = "PAPER" if paper else "LIVE"
        logger.info(f"AlpacaBroker initialized in {mode} mode")

    # ── Account ──────────────────────────────────────────────────────

    def get_account(self) -> Account:
        acct = self.trading_client.get_account()
        return Account(
            account_id=str(acct.id),
            equity=float(acct.equity),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
            portfolio_value=float(acct.portfolio_value or acct.equity),
            status=str(acct.status),
            day_trade_count=int(acct.daytrade_count or 0),
            last_equity=float(acct.last_equity or 0),
        )

    # ── Positions ────────────────────────────────────────────────────

    def get_positions(self) -> List[Position]:
        raw = self.trading_client.get_all_positions()
        return [self._to_position(p) for p in raw]

    def get_position(self, symbol: str) -> Optional[Position]:
        try:
            raw = self.trading_client.get_open_position(symbol)
            return self._to_position(raw)
        except Exception:
            return None

    @staticmethod
    def _to_position(p) -> Position:
        return Position(
            symbol=p.symbol,
            qty=abs(float(p.qty)),
            side=str(p.side),
            avg_entry_price=float(p.avg_entry_price),
            current_price=float(p.current_price),
            market_value=abs(float(p.market_value)),
            unrealized_pl=float(p.unrealized_pl),
            unrealized_plpc=float(p.unrealized_plpc),
        )

    # ── Orders ───────────────────────────────────────────────────────

    def submit_order(self, order: OrderRequest) -> OrderResult:
        tif = TIF_MAP.get(order.time_in_force, TimeInForce.DAY)
        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL

        if order.order_type == "market":
            req = MarketOrderRequest(
                symbol=order.symbol,
                side=side,
                qty=order.qty,
                notional=order.notional,
                time_in_force=tif,
            )
        elif order.order_type == "limit":
            req = LimitOrderRequest(
                symbol=order.symbol,
                side=side,
                qty=order.qty,
                notional=order.notional,
                time_in_force=tif,
                limit_price=order.limit_price,
            )
        elif order.order_type == "stop":
            req = StopOrderRequest(
                symbol=order.symbol,
                side=side,
                qty=order.qty,
                time_in_force=tif,
                stop_price=order.stop_price,
            )
        elif order.order_type == "stop_limit":
            req = StopLimitOrderRequest(
                symbol=order.symbol,
                side=side,
                qty=order.qty,
                time_in_force=tif,
                stop_price=order.stop_price,
                limit_price=order.limit_price,
            )
        elif order.order_type == "trailing_stop":
            req = TrailingStopOrderRequest(
                symbol=order.symbol,
                side=side,
                qty=order.qty,
                time_in_force=tif,
                trail_percent=order.trail_percent,
            )
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")

        result = self.trading_client.submit_order(req)
        logger.info(f"Order submitted: {order.side} {order.qty or order.notional} {order.symbol} -> {result.status}")
        return self._to_order_result(result)

    def submit_bracket_order(
        self,
        order: OrderRequest,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> OrderResult:
        """Submit a bracket order: main entry + automatic stop-loss + take-profit.

        When the main order fills, Alpaca automatically creates:
        - A stop order at stop_loss_price (sells if price drops)
        - A limit order at take_profit_price (sells if price rises)
        Whichever triggers first cancels the other (OCO).
        """
        if order.side != "buy":
            return self.submit_order(order)

        tif = TIF_MAP.get(order.time_in_force, TimeInForce.DAY)

        req = MarketOrderRequest(
            symbol=order.symbol,
            side=OrderSide.BUY,
            qty=order.qty,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            stop_loss={"stop_price": round(stop_loss_price, 2)},
            take_profit={"limit_price": round(take_profit_price, 2)},
        )

        result = self.trading_client.submit_order(req)
        logger.info(
            f"Bracket order: BUY {order.qty} {order.symbol} "
            f"SL=${stop_loss_price:.2f} TP=${take_profit_price:.2f} -> {result.status}"
        )
        return self._to_order_result(result)

    def cancel_order(self, order_id: str) -> None:
        self.trading_client.cancel_order_by_id(order_id)
        logger.info(f"Order cancelled: {order_id}")

    def replace_order(
        self,
        order_id: str,
        stop_price: Optional[float] = None,
        limit_price: Optional[float] = None,
        qty: Optional[float] = None,
    ) -> Optional[OrderResult]:
        """Replace an open order's stop_price / limit_price / qty.

        Used by ExitManagerV2 to ratchet the stop leg of a bracket OCO as the
        trailing stop moves up. Returns the new OrderResult or None on failure.
        """
        kwargs = {}
        if stop_price is not None:
            kwargs["stop_price"] = round(float(stop_price), 2)
        if limit_price is not None:
            kwargs["limit_price"] = round(float(limit_price), 2)
        if qty is not None:
            kwargs["qty"] = int(qty)
        if not kwargs:
            return None
        req = ReplaceOrderRequest(**kwargs)
        try:
            raw = self.trading_client.replace_order_by_id(order_id, req)
        except Exception as e:
            logger.warning(f"replace_order({order_id}, {kwargs}) failed: {e}")
            return None
        logger.info(f"Order replaced: {order_id} {kwargs} -> {raw.id}")
        return self._to_order_result(raw)

    def get_order(self, order_id: str) -> OrderResult:
        raw = self.trading_client.get_order_by_id(order_id)
        return self._to_order_result(raw)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        req = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol] if symbol else None,
        )
        raw = self.trading_client.get_orders(filter=req)
        return [self._to_order_result(order) for order in raw]

    @staticmethod
    def _to_order_result(o) -> OrderResult:
        out = OrderResult(
            order_id=str(o.id),
            symbol=o.symbol,
            side=str(o.side),
            qty=float(o.qty) if o.qty else None,
            notional=float(o.notional) if o.notional else None,
            order_type=str(o.type),
            status=str(o.status),
            filled_qty=float(o.filled_qty or 0),
            filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
            submitted_at=o.submitted_at,
            filled_at=o.filled_at,
        )
        for leg in getattr(o, "legs", None) or []:
            leg_stop = getattr(leg, "stop_price", None)
            leg_limit = getattr(leg, "limit_price", None)
            if leg_stop and out.stop_order_id is None:
                out.stop_order_id = str(leg.id)
            elif leg_limit and out.tp_order_id is None:
                out.tp_order_id = str(leg.id)
        return out

    # ── Close Positions ──────────────────────────────────────────────

    def close_position(self, symbol: str) -> OrderResult:
        result = self.trading_client.close_position(symbol)
        logger.info(f"Position closed: {symbol}")
        return self._to_order_result(result)

    def close_all_positions(self) -> List[OrderResult]:
        results = self.trading_client.close_all_positions(cancel_orders=True)
        logger.info(f"All positions closed ({len(results)} orders)")
        return [self._to_order_result(r) for r in results if hasattr(r, "id")]

    # ── Market Data ──────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        return self.trading_client.get_clock().is_open

    def get_clock(self) -> MarketClock:
        c = self.trading_client.get_clock()
        return MarketClock(
            is_open=c.is_open,
            next_open=c.next_open,
            next_close=c.next_close,
            timestamp=c.timestamp,
        )

    @staticmethod
    def _mid_price(q) -> float:
        ask = float(q.ask_price)
        bid = float(q.bid_price)
        if ask > 0 and bid > 0:
            return (ask + bid) / 2
        return ask or bid

    def get_latest_price(self, symbol: str) -> float:
        quotes = self.data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        )
        return self._mid_price(quotes[symbol])

    def get_latest_prices(self, symbols: List[str]) -> dict:
        quotes = self.data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbols)
        )
        return {sym: self._mid_price(q) for sym, q in quotes.items()}
