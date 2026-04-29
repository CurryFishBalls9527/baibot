"""Alpaca broker implementation for paper and live trading."""

import logging
import time
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
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

from .base_broker import BaseBroker
from .models import OrderRequest, OrderResult, Position, Account, MarketClock
from tradingagents.automation.events import Categories, emit_event

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
        qty_avail = getattr(p, "qty_available", None)
        return Position(
            symbol=p.symbol,
            qty=abs(float(p.qty)),
            side=str(p.side),
            avg_entry_price=float(p.avg_entry_price),
            current_price=float(p.current_price),
            market_value=abs(float(p.market_value)),
            unrealized_pl=float(p.unrealized_pl),
            unrealized_plpc=float(p.unrealized_plpc),
            qty_available=abs(float(qty_avail)) if qty_avail is not None else None,
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

        try:
            result = self.trading_client.submit_order(req)
        except APIError as e:
            code = getattr(e, "code", None)
            category = (
                Categories.WASH_TRADE_REJECT
                if code == 40310000
                else Categories.ORDER_REJECT
            )
            emit_event(
                category,
                level="error",
                symbol=order.symbol,
                code=str(code or ""),
                message=f"submit_order rejected: {e}",
                context={
                    "side": order.side,
                    "qty": order.qty,
                    "order_type": order.order_type,
                    "stop_price": order.stop_price,
                    "limit_price": order.limit_price,
                },
            )
            raise
        logger.info(f"Order submitted: {order.side} {order.qty or order.notional} {order.symbol} -> {result.status}")
        return self._to_order_result(result)

    def submit_bracket_order(
        self,
        order: OrderRequest,
        stop_loss_price: float,
        take_profit_price: float,
        anchor_price: Optional[float] = None,
    ) -> OrderResult:
        """Submit a bracket order: main entry + automatic stop-loss + take-profit.

        When the main order fills, Alpaca automatically creates:
        - A stop order at stop_loss_price (sells if price drops)
        - A limit order at take_profit_price (sells if price rises)
        Whichever triggers first cancels the other (OCO).

        ``anchor_price``: the quote the caller used to derive ``stop_loss_price``
        and ``take_profit_price``. When provided AND the parent fills at a price
        that drifts >10 bps from this anchor, the SL/TP child legs are replaced
        in place to preserve the caller's intended distance (e.g. SL=−3%) from
        the actual fill rather than from a stale quote. This eliminates silent
        bracket miscalibration in fast markets and on stale IEX quotes.
        """
        if order.side != "buy":
            return self.submit_order(order)

        tif = TIF_MAP.get(order.time_in_force, TimeInForce.DAY)

        sl_submit = round(stop_loss_price, 2)
        tp_submit = round(take_profit_price, 2)

        req = MarketOrderRequest(
            symbol=order.symbol,
            side=OrderSide.BUY,
            qty=order.qty,
            # Respect caller's TIF — intraday wants DAY (auto-cancel unfilled
            # parent at EOD); daily/chan want GTC (children persist past today).
            time_in_force=tif,
            order_class=OrderClass.BRACKET,
            stop_loss={"stop_price": sl_submit},
            take_profit={"limit_price": tp_submit},
        )

        try:
            result = self.trading_client.submit_order(req)
        except APIError as e:
            # Stale-quote retry: when our IEX mid lags SIP on a fast-moving
            # thin name (e.g. small-caps like AAOI/VSCO), the SL computed
            # from the stale anchor can land above Alpaca's base_price and
            # hit `stop_loss.stop_price must be <= base_price - 0.01`.
            # Re-fetch quote, clamp SL to preserve the caller's intended
            # SL-distance against the fresh price, retry once.
            msg = str(e)
            if e.code != 42210000 or "stop_loss" not in msg or "base_price" not in msg:
                emit_event(
                    Categories.ORDER_REJECT,
                    level="error",
                    symbol=order.symbol,
                    code=str(getattr(e, "code", "") or ""),
                    message=f"bracket submit rejected: {e}",
                    context={
                        "anchor_price": anchor_price,
                        "sl_submit": sl_submit,
                        "tp_submit": tp_submit,
                        "tif": str(tif),
                    },
                )
                raise
            fresh_price = self.get_latest_price(order.symbol)
            if fresh_price <= 0:
                # Quote fetch failed — can't recover; bubble up.
                emit_event(
                    Categories.ORDER_REJECT,
                    level="error",
                    symbol=order.symbol,
                    code="42210000",
                    message="stale-quote retry aborted — fresh quote unavailable",
                    context={"anchor_price": anchor_price, "sl_submit": sl_submit},
                )
                raise
            sl_pct = (anchor_price - stop_loss_price) / anchor_price if anchor_price else 0.03
            new_sl = round(min(fresh_price * (1 - sl_pct), fresh_price - 0.01), 2)
            logger.warning(
                f"{order.symbol}: SL stale-quote retry "
                f"(anchor=${anchor_price:.2f} fresh=${fresh_price:.2f} "
                f"SL=${sl_submit:.2f}->${new_sl:.2f})"
            )
            req = MarketOrderRequest(
                symbol=order.symbol,
                side=OrderSide.BUY,
                qty=order.qty,
                time_in_force=tif,
                order_class=OrderClass.BRACKET,
                stop_loss={"stop_price": new_sl},
                take_profit={"limit_price": tp_submit},
            )
            sl_submit = new_sl
            try:
                result = self.trading_client.submit_order(req)
            except APIError as retry_e:
                emit_event(
                    Categories.ORDER_REJECT,
                    level="error",
                    symbol=order.symbol,
                    code=str(getattr(retry_e, "code", "") or "42210000"),
                    message=f"stale-quote retry exhausted: {retry_e}",
                    context={
                        "anchor_price": anchor_price,
                        "fresh_price": fresh_price,
                        "sl_old": stop_loss_price,
                        "sl_new": new_sl,
                    },
                )
                raise
            emit_event(
                Categories.STALE_QUOTE_RETRY,
                level="info",
                symbol=order.symbol,
                code="42210000",
                message="stale-quote retry recovered",
                context={
                    "anchor_price": anchor_price,
                    "fresh_price": fresh_price,
                    "sl_old": stop_loss_price,
                    "sl_new": new_sl,
                },
            )
        logger.info(
            f"Bracket order: BUY {order.qty} {order.symbol} "
            f"SL=${sl_submit:.2f} TP=${tp_submit:.2f} -> {result.status}"
        )
        out = self._to_order_result(result)
        # Alpaca's initial submit response for a bracket has `legs=[]` — the SL/TP
        # children only materialize after the parent is accepted/filled. Without
        # capturing their IDs here, position_states.stop_order_id stays None and
        # ExitManagerV2._maybe_update_broker_stop silently no-ops (trail ratchet
        # never propagates to the broker). Refetch the parent by ID to pick the
        # child IDs up; short retry covers the brief window between parent fill
        # and child creation on Alpaca's side.
        if not out.stop_order_id or not out.tp_order_id:
            refetched = self._refetch_bracket_legs(str(result.id))
            if refetched is not None:
                out.stop_order_id = out.stop_order_id or refetched.stop_order_id
                out.tp_order_id = out.tp_order_id or refetched.tp_order_id
                # Refetch also has the authoritative fill price once filled.
                if refetched.filled_avg_price and not out.filled_avg_price:
                    out.filled_avg_price = refetched.filled_avg_price
                    out.filled_qty = refetched.filled_qty
                    out.status = refetched.status
        if not out.stop_order_id or not out.tp_order_id:
            logger.warning(
                f"{order.symbol}: bracket submitted but leg IDs still missing after "
                f"refetch (stop={out.stop_order_id} tp={out.tp_order_id}). Broker-side "
                f"stop ratcheting will be disabled for this position until backfill."
            )
            emit_event(
                Categories.BRACKET_LEG_MISSING,
                level="error",
                symbol=order.symbol,
                message="bracket submitted but leg IDs missing after refetch",
                context={
                    "parent_id": str(result.id),
                    "stop_order_id": out.stop_order_id,
                    "tp_order_id": out.tp_order_id,
                    "filled_avg_price": out.filled_avg_price,
                },
            )
        out.effective_stop_price = sl_submit
        out.effective_take_profit_price = tp_submit

        if anchor_price and anchor_price > 0 and out.filled_avg_price:
            fill = float(out.filled_avg_price)
            drift_bps = abs(fill - anchor_price) / anchor_price * 1e4
            if drift_bps > 10.0:
                sl_pct = (anchor_price - stop_loss_price) / anchor_price
                tp_pct = (take_profit_price - anchor_price) / anchor_price
                new_sl = round(fill * (1 - sl_pct), 2)
                new_tp = round(fill * (1 + tp_pct), 2)
                logger.info(
                    f"{order.symbol}: re-anchoring bracket legs (anchor=${anchor_price:.2f} "
                    f"fill=${fill:.2f} drift={drift_bps:.1f}bps) "
                    f"SL=${sl_submit:.2f}->${new_sl:.2f} TP=${tp_submit:.2f}->${new_tp:.2f}"
                )
                if out.stop_order_id:
                    rep = self.replace_order(out.stop_order_id, stop_price=new_sl)
                    if rep is not None:
                        out.stop_order_id = rep.order_id
                        out.effective_stop_price = new_sl
                if out.tp_order_id:
                    rep = self.replace_order(out.tp_order_id, limit_price=new_tp)
                    if rep is not None:
                        out.tp_order_id = rep.order_id
                        out.effective_take_profit_price = new_tp
        return out

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

        Failures emit `WASH_TRADE_REJECT` (code 40310000) or `ORDER_REJECT`
        events at error-level so the watchdog's `check_event_tail` escalates
        them — silent ratchet rejects were a known watchdog blind spot.
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
        except APIError as e:
            code = getattr(e, "code", None)
            msg = str(e).lower()
            # Benign 42210000 sub-cases that signal the system already
            # converged to the desired state. Don't escalate to ERROR or
            # emit ORDER_REJECT (the watchdog tail-watch picks those up
            # and pages). Both observed live on NOK 2026-04-29:
            #   "order already replaced" — exit_manager_v2 + reconciler
            #     race on the same 5-min tick; one wins, the other hits
            #     the now-terminal old order_id.
            #   "order parameters are not changed" — float-precision diff
            #     in stop_price > 1e-6 but rounds to the same 2-decimal
            #     broker price.
            # Other 42210000 sub-cases (e.g. "stop_loss.stop_price must
            # be <= base_price - 0.01") remain ERROR — those are real
            # entry-side rejections.
            if code == 42210000 and (
                "already replaced" in msg
                or "parameters are not changed" in msg
            ):
                logger.info(
                    "replace_order(%s, %s) benign-skip: %s",
                    order_id, kwargs, e,
                )
                return None
            category = (
                Categories.WASH_TRADE_REJECT
                if code == 40310000
                else Categories.ORDER_REJECT
            )
            emit_event(
                category,
                level="error",
                code=str(code or ""),
                message=f"replace_order rejected: {e}",
                context={"order_id": order_id, **kwargs},
            )
            logger.error(f"replace_order({order_id}, {kwargs}) failed: {e}")
            return None
        except Exception as e:
            emit_event(
                Categories.ORDER_REJECT,
                level="error",
                message=f"replace_order failed (non-API): {e}",
                context={"order_id": order_id, **kwargs},
            )
            logger.error(f"replace_order({order_id}, {kwargs}) failed: {e}")
            return None
        logger.info(f"Order replaced: {order_id} {kwargs} -> {raw.id}")
        return self._to_order_result(raw)

    def get_order(self, order_id: str) -> OrderResult:
        raw = self.trading_client.get_order_by_id(order_id)
        return self._to_order_result(raw)

    def _refetch_bracket_legs(
        self,
        parent_order_id: str,
        max_attempts: int = 30,
        delay_seconds: float = 0.5,
    ) -> Optional[OrderResult]:
        """Re-query a bracket parent by ID to populate leg IDs AND capture fill.

        Alpaca's initial submit response for a bracket has `legs=[]` and
        usually status=`pending_new`. Sequence of state changes:

          1. Parent accepted → legs materialize (leg IDs become available)
          2. Parent fills → filled_avg_price populates

        Previously this returned as soon as the leg IDs were populated, which
        meant callers racing a fast fill often saw `filled_avg_price=None` and
        the caller's post-submit re-anchor (SL/TP shift to actual fill price)
        silently skipped. On fast movers where fill drifted 2%+ from the
        pre-order quote, the broker's SL ended up anchored to the stale quote
        — a compressed effective buffer.

        Now the loop keeps polling until BOTH legs exist AND the parent has
        either filled (filled_avg_price populated) or reached a terminal
        status, or the budget is exhausted. Budget defaults bumped from
        5×0.3s=1.5s to 30×0.5s=15s to cover p99 fills while capping worst-case
        submit latency.
        """
        terminal = {"filled", "canceled", "cancelled", "expired", "rejected"}
        last: Optional[OrderResult] = None
        for attempt in range(max_attempts):
            try:
                raw = self.trading_client.get_order_by_id(parent_order_id)
            except Exception as e:
                logger.warning(
                    f"_refetch_bracket_legs({parent_order_id}) attempt "
                    f"{attempt + 1}/{max_attempts} failed: {e}"
                )
                time.sleep(delay_seconds)
                continue
            last = self._to_order_result(raw)
            ids_ready = bool(last.stop_order_id and last.tp_order_id)
            status_l = str(last.status or "").lower().split(".")[-1]
            settled = bool(last.filled_avg_price) or status_l in terminal
            if ids_ready and settled:
                return last
            time.sleep(delay_seconds)
        if last is not None and (not last.stop_order_id or not last.filled_avg_price):
            logger.warning(
                f"_refetch_bracket_legs({parent_order_id}) timed out: "
                f"legs_ready={bool(last.stop_order_id and last.tp_order_id)} "
                f"filled={bool(last.filled_avg_price)} status={last.status}. "
                f"SL re-anchor will be skipped."
            )
        return last

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        req = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol] if symbol else None,
        )
        raw = self.trading_client.get_orders(filter=req)
        return [self._to_order_result(order) for order in raw]

    # Status values that indicate the order is still alive at the broker.
    # Alpaca's QueryOrderStatus.OPEN does NOT include HELD — bracket / OCO
    # stop-loss legs sit in HELD while they wait for their trigger. Reconciler
    # needs to see them to match DB position_states against live broker state.
    #
    # NOTE: "replaced" is NOT included — it's a terminal status indicating the
    # order was superseded by a new one. The replacement order has its own ID
    # and will appear with one of the live statuses below. Including
    # "replaced" caused stale-ID drift (DELL 2026-04-24): _find_live_sl_tp
    # surfaced a dead order as if it were live, so the reconciler kept
    # writing a cancelled order_id back into position_states.
    _LIVE_ORDER_STATUSES = {
        "new", "accepted", "pending_new", "accepted_for_bidding",
        "held", "pending_replace",
    }

    def get_live_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """Return orders still alive at the broker, including HELD stop legs.

        `get_open_orders` uses Alpaca's `OPEN` status filter which excludes
        HELD — so OCO/bracket stop legs waiting for their trigger don't show
        up there. Reconciliation needs the wider view.
        """
        req = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            symbols=[symbol] if symbol else None,
            limit=500,
        )
        raw = self.trading_client.get_orders(filter=req)
        out: List[OrderResult] = []
        for order in raw:
            status_l = str(order.status).lower()
            # Normalize "OrderStatus.HELD" → "held"
            if "." in status_l:
                status_l = status_l.split(".", 1)[1]
            if status_l in self._LIVE_ORDER_STATUSES:
                out.append(self._to_order_result(order))
        return out

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
            stop_price=float(o.stop_price) if getattr(o, "stop_price", None) else None,
            limit_price=float(o.limit_price) if getattr(o, "limit_price", None) else None,
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
        """Cancel all open orders and submit market-close orders for every
        position on this account. Atomic at Alpaca — no race with bracket
        children that would leave `held_for_orders == qty` and reject the
        close.

        Alpaca returns ``ClosePositionResponse`` objects (symbol, status,
        order_id, body=Order|FailedClosePositionDetails). Success entries
        have an Order in ``body`` which we flatten to OrderResult; failures
        are warned and skipped.
        """
        raw = self.trading_client.close_all_positions(cancel_orders=True)
        out: List[OrderResult] = []
        for r in raw:
            status_code = getattr(r, "status", None)
            body = getattr(r, "body", None)
            symbol = getattr(r, "symbol", None)
            if status_code and 200 <= int(status_code) < 300 and hasattr(body, "id"):
                out.append(self._to_order_result(body))
            else:
                err = getattr(body, "message", None) or str(body) if body else f"http {status_code}"
                logger.warning(
                    "close_all_positions: %s failed (status=%s): %s",
                    symbol, status_code, err,
                )
        logger.info(
            "close_all_positions: %s close order(s) submitted (raw responses=%s)",
            len(out), len(raw),
        )
        return out

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
