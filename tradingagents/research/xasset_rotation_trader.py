"""Cross-asset rotation paper trader.

Daily-frequency strategy: rank a small ETF universe (SPY, QQQ, TLT, IEF, GLD)
by trailing 20-day return on prior session close, hold equal-weight long top-N
through the month, rebalance on the first trading day of each new month.

Why this is structurally separate from `intraday_xsection_paper_trader.py`:
  * Daily / monthly cadence — no intraday loop or bar-grace logic
  * 5-ETF universe — no rolling-DV liquidity filter, no shorts
  * Long-only, persistent positions — no EOD flatten
  * Different data path: Alpaca daily bars (or DuckDB fallback)

Runs once per call (cron-style). The CLI script schedules it via launchd or
a simple sleep loop.

Backtest results (lookahead-clean per `formation_lag_bars` probe):
  * 2018:    -4.24% / DD 12.83% / Sharpe -0.24
  * 2020:   +37.90% / DD 11.09% / Sharpe +1.71
  * 2023_25: +94.75% / DD  9.63% / Sharpe +1.79
2 of 3 periods strongly positive. Mechanism: cross-asset momentum (capture
trends across stocks/bonds/gold via prior-month winners).

Safety defaults: dry_run=True, max_gross_exposure=0.5 cap.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from tradingagents.broker.alpaca_broker import AlpacaBroker

logger = logging.getLogger(__name__)


@dataclass
class XAssetRotationConfig:
    universe: tuple[str, ...] = ("SPY", "QQQ", "TLT", "IEF", "GLD")
    lookback_days: int = 20
    top_n: int = 2
    daily_db_path: str = "research_data/market_data.duckdb"
    log_dir: Path = field(default_factory=lambda: Path("results/xasset_rotation/paper"))
    dry_run: bool = True
    max_gross_exposure: float = 0.5


@dataclass
class XAssetFillRecord:
    rebalance_date: str  # ISO date
    submit_ts: str       # ISO timestamp
    fill_ts: Optional[str]
    symbol: str
    action: str          # "open_long" / "rebalance_close" / "rebalance_add"
    side: str            # "buy" / "sell"
    qty: int
    intent_price: float
    fill_price: Optional[float]
    fill_status: str
    slippage_bps: Optional[float]
    formation_return: Optional[float]
    target_dollars: float
    dry_run: bool
    order_id: Optional[str]
    error: Optional[str] = None


class XAssetRotationTrader:
    def __init__(
        self,
        config: XAssetRotationConfig,
        broker: "AlpacaBroker",
    ):
        self.config = config
        self.broker = broker
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self._fills_path = self.config.log_dir / "fills.jsonl"
        self._signal_path = self.config.log_dir / "rebalances.jsonl"

    # ── Calendar ─────────────────────────────────────────────────────

    def is_rebalance_day(self, today: datetime) -> bool:
        """First trading day of a new calendar month using Alpaca's calendar."""
        try:
            # Alpaca's get_clock + get_calendar gives us the trading-day list.
            from alpaca.trading.requests import GetCalendarRequest

            today_date = today.date()
            cal_req = GetCalendarRequest(
                start=today_date - timedelta(days=10),
                end=today_date,
            )
            cal = self.broker.trading_client.get_calendar(cal_req)
            trading_days = sorted({c.date for c in cal})
        except Exception as exc:
            logger.warning("Alpaca calendar lookup failed (%s) — falling back to weekday rule", exc)
            return today.weekday() < 5 and today.day <= 5

        if today_date not in trading_days:
            return False
        # Today's month vs prior trading day's month — mismatch = rebalance.
        prior = [d for d in trading_days if d < today_date]
        if not prior:
            return True
        return today_date.month != prior[-1].month

    # ── Signal ───────────────────────────────────────────────────────

    def compute_signal(self, asof: datetime) -> dict[str, float]:
        """Trailing N-day return ending YESTERDAY.

        Source: Alpaca daily bars (live source). asof is "now" — we use bars
        with timestamps strictly before today's date so the signal is fully
        prior-day-aware.
        """
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        feed = (
            DataFeed.SIP
            if os.getenv("ALPACA_XSECTION_FEED", "iex").lower() == "sip"
            else DataFeed.IEX
        )
        end = asof.replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=self.config.lookback_days * 4)  # padding
        symbols = list(self.config.universe)
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=start,
                end=end,
                feed=feed,
            )
            bars = self.broker.data_client.get_stock_bars(req)
        except Exception as exc:
            logger.warning("Alpaca daily bars fetch failed: %s — falling back to DuckDB", exc)
            return self._compute_signal_from_duckdb(asof)
        df = bars.df
        if df is None or df.empty:
            logger.warning("Alpaca returned no bars — falling back to DuckDB")
            return self._compute_signal_from_duckdb(asof)
        df = df.reset_index()
        df["date"] = (
            pd.to_datetime(df["timestamp"])
            .dt.tz_convert("America/Chicago")
            .dt.tz_localize(None)
            .dt.normalize()
        )
        # Use only bars strictly before today's date (asof.date()).
        cutoff = pd.Timestamp(asof.date())
        df = df[df["date"] < cutoff]
        out: dict[str, float] = {}
        for sym in symbols:
            sub = df[df["symbol"] == sym].sort_values("date")
            if len(sub) < self.config.lookback_days + 1:
                continue
            close_today = float(sub["close"].iloc[-1])
            close_back = float(sub["close"].iloc[-1 - self.config.lookback_days])
            if close_back <= 0:
                continue
            out[sym] = close_today / close_back - 1.0
        return out

    def _compute_signal_from_duckdb(self, asof: datetime) -> dict[str, float]:
        from .warehouse import MarketDataWarehouse

        cutoff = pd.Timestamp(asof.date())
        start = (cutoff - pd.Timedelta(days=self.config.lookback_days * 4)).strftime("%Y-%m-%d")
        end = (cutoff - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        w = MarketDataWarehouse(self.config.daily_db_path, read_only=True)
        out: dict[str, float] = {}
        try:
            for sym in self.config.universe:
                df = w.get_daily_bars(sym, start, end)
                if df is None or len(df) < self.config.lookback_days + 1:
                    continue
                close = df["close"].sort_index()
                close_today = float(close.iloc[-1])
                close_back = float(close.iloc[-1 - self.config.lookback_days])
                if close_back <= 0:
                    continue
                out[sym] = close_today / close_back - 1.0
        finally:
            w.close()
        return out

    # ── Sizing ───────────────────────────────────────────────────────

    def select_basket(self, signal: dict[str, float]) -> list[str]:
        if len(signal) < self.config.top_n:
            return []
        ranked = sorted(signal.items(), key=lambda kv: kv[1], reverse=True)
        return [sym for sym, _ in ranked[: self.config.top_n]]

    def compute_target_dollars(self) -> float:
        """Per-name dollar allocation. Long-only equal-weight."""
        try:
            account = self.broker.get_account()
            equity = float(account.equity)
        except Exception as exc:
            logger.warning("get_account failed: %s — assuming $100k", exc)
            equity = 100_000.0
        gross = min(self.config.max_gross_exposure, 1.0)
        return equity * gross / max(self.config.top_n, 1)

    # ── Rebalance ────────────────────────────────────────────────────

    def rebalance_to(self, target_symbols: list[str], target_dollars: float,
                    signal: dict[str, float], rebalance_date: str) -> list[XAssetFillRecord]:
        """Move account to target_symbols at equal-weight. Closes positions
        no longer in the basket, opens missing ones, leaves matching positions
        unchanged (no churn).
        """
        records: list[XAssetFillRecord] = []
        try:
            current_positions = self.broker.get_positions()
        except Exception as exc:
            logger.warning("get_positions failed: %s", exc)
            return records
        target_set = set(target_symbols)
        current_universe_positions = {
            p.symbol: p for p in current_positions if p.symbol in self.config.universe
        }
        # Anything in the universe we own but don't want anymore → close.
        to_close = [
            sym for sym in current_universe_positions
            if sym not in target_set
        ]
        # Anything we want but don't own → open.
        to_open = [
            sym for sym in target_symbols
            if sym not in current_universe_positions
        ]
        # We deliberately don't touch positions we already own AND want.

        intents = self._capture_intents(list(target_set | set(to_close)))
        submit_ts = datetime.now(timezone.utc).isoformat()

        # 1. Close stale positions
        for sym in to_close:
            pos = current_universe_positions[sym]
            qty = int(pos.qty)
            records.append(self._close_one(
                sym, qty, intents.get(sym), rebalance_date, submit_ts,
                signal.get(sym),
            ))

        # 2. Open new positions
        for sym in to_open:
            intent_px = intents.get(sym)
            if intent_px is None or intent_px <= 0:
                logger.info("skipping %s — no intent price", sym)
                continue
            qty = int(target_dollars / intent_px)
            if qty <= 0:
                continue
            records.append(self._submit_buy(
                sym, qty, intent_px, target_dollars, rebalance_date, submit_ts,
                signal.get(sym),
            ))
        return records

    def _capture_intents(self, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}
        try:
            quotes = self.broker.get_latest_prices(symbols)
        except Exception as exc:
            logger.warning("get_latest_prices failed: %s", exc)
            return {}
        return {s: float(p) for s, p in quotes.items() if p and p > 0}

    def _close_one(self, sym, qty, intent_price, rebalance_date, submit_ts,
                  formation_return) -> XAssetFillRecord:
        if self.config.dry_run:
            return XAssetFillRecord(
                rebalance_date=rebalance_date, submit_ts=submit_ts, fill_ts=None,
                symbol=sym, action="rebalance_close", side="sell", qty=qty,
                intent_price=intent_price or 0.0, fill_price=None,
                fill_status="dry_run", slippage_bps=None,
                formation_return=formation_return, target_dollars=0.0,
                dry_run=True, order_id=None,
            )
        try:
            close_result = self.broker.close_position(sym)
        except Exception as exc:
            logger.error("close_position(%s) failed: %s", sym, exc)
            return XAssetFillRecord(
                rebalance_date=rebalance_date, submit_ts=submit_ts, fill_ts=None,
                symbol=sym, action="rebalance_close", side="sell", qty=qty,
                intent_price=intent_price or 0.0, fill_price=None,
                fill_status="close_failed", slippage_bps=None,
                formation_return=formation_return, target_dollars=0.0,
                dry_run=False, order_id=None, error=str(exc),
            )
        final = self._poll(close_result.order_id)
        slip = self._slippage_bps(intent_price, final.filled_avg_price, "sell")
        return XAssetFillRecord(
            rebalance_date=rebalance_date, submit_ts=submit_ts,
            fill_ts=final.filled_at.isoformat() if final.filled_at else None,
            symbol=sym, action="rebalance_close", side="sell",
            qty=int(final.filled_qty or 0),
            intent_price=intent_price or 0.0, fill_price=final.filled_avg_price,
            fill_status=final.status, slippage_bps=slip,
            formation_return=formation_return, target_dollars=0.0,
            dry_run=False, order_id=close_result.order_id,
        )

    def _submit_buy(self, sym, qty, intent_price, target_dollars,
                   rebalance_date, submit_ts, formation_return) -> XAssetFillRecord:
        if self.config.dry_run:
            return XAssetFillRecord(
                rebalance_date=rebalance_date, submit_ts=submit_ts, fill_ts=None,
                symbol=sym, action="open_long", side="buy", qty=qty,
                intent_price=intent_price, fill_price=None,
                fill_status="dry_run", slippage_bps=None,
                formation_return=formation_return, target_dollars=target_dollars,
                dry_run=True, order_id=None,
            )
        from tradingagents.broker.models import OrderRequest as _OrderRequest
        order = _OrderRequest(symbol=sym, side="buy", qty=qty,
                              order_type="market", time_in_force="day")
        try:
            result = self.broker.submit_order(order)
        except Exception as exc:
            logger.error("submit_order(%s) failed: %s", sym, exc)
            return XAssetFillRecord(
                rebalance_date=rebalance_date, submit_ts=submit_ts, fill_ts=None,
                symbol=sym, action="open_long", side="buy", qty=qty,
                intent_price=intent_price, fill_price=None,
                fill_status="submit_failed", slippage_bps=None,
                formation_return=formation_return, target_dollars=target_dollars,
                dry_run=False, order_id=None, error=str(exc),
            )
        final = self._poll(result.order_id)
        slip = self._slippage_bps(intent_price, final.filled_avg_price, "buy")
        return XAssetFillRecord(
            rebalance_date=rebalance_date, submit_ts=submit_ts,
            fill_ts=final.filled_at.isoformat() if final.filled_at else None,
            symbol=sym, action="open_long", side="buy",
            qty=int(final.filled_qty or 0),
            intent_price=intent_price, fill_price=final.filled_avg_price,
            fill_status=final.status, slippage_bps=slip,
            formation_return=formation_return, target_dollars=target_dollars,
            dry_run=False, order_id=result.order_id,
        )

    def _poll(self, order_id, attempts=12, delay=1.0):
        import time as _time

        terminal = {"filled", "canceled", "cancelled", "expired", "rejected"}
        last = None
        for _ in range(attempts):
            try:
                last = self.broker.get_order(order_id)
            except Exception as exc:
                logger.warning("get_order(%s) failed: %s", order_id, exc)
                _time.sleep(delay)
                continue
            status_l = str(last.status).lower().split(".")[-1]
            if status_l in terminal or last.filled_avg_price:
                return last
            _time.sleep(delay)
        if last is None:
            from tradingagents.broker.models import OrderResult as _OrderResult
            return _OrderResult(
                order_id=order_id, symbol="?", side="?", qty=None,
                notional=None, order_type="market", status="poll_timeout",
                filled_qty=0.0, filled_avg_price=None,
            )
        return last

    @staticmethod
    def _slippage_bps(intent, fill, side):
        if intent is None or fill is None or intent <= 0:
            return None
        raw = (fill - intent) / intent * 1e4
        return round(raw if side == "buy" else -raw, 2)

    # ── Logging ──────────────────────────────────────────────────────

    def append_records(self, records):
        if not records:
            return
        with self._fills_path.open("a") as fh:
            for r in records:
                fh.write(json.dumps(r.__dict__, default=str) + "\n")

    def append_signal_summary(self, summary):
        with self._signal_path.open("a") as fh:
            fh.write(json.dumps(summary, default=str) + "\n")

    # ── Main entry ───────────────────────────────────────────────────

    def run_once(self, asof: Optional[datetime] = None) -> dict:
        """Single rebalance check. If today is a rebalance day, picks top-N
        and rebalances. Otherwise no-op."""
        asof = asof or datetime.now()
        rebalance_date = asof.date().isoformat()

        if not self.is_rebalance_day(asof):
            summary = {
                "action": "skip",
                "reason": "not_rebalance_day",
                "rebalance_date": rebalance_date,
            }
            self.append_signal_summary(summary)
            return summary

        signal = self.compute_signal(asof)
        if len(signal) < self.config.top_n:
            summary = {
                "action": "skip",
                "reason": "insufficient_signal_data",
                "rebalance_date": rebalance_date,
                "signal_size": len(signal),
            }
            self.append_signal_summary(summary)
            return summary

        target_symbols = self.select_basket(signal)
        target_dollars = self.compute_target_dollars()
        summary = {
            "action": "rebalance",
            "rebalance_date": rebalance_date,
            "target_symbols": target_symbols,
            "target_dollars_per_name": round(target_dollars, 2),
            "signal": {s: round(v, 6) for s, v in signal.items()},
            "dry_run": self.config.dry_run,
        }
        self.append_signal_summary(summary)

        records = self.rebalance_to(
            target_symbols, target_dollars, signal, rebalance_date,
        )
        self.append_records(records)
        summary["close_count"] = sum(1 for r in records if r.action == "rebalance_close")
        summary["open_count"] = sum(1 for r in records if r.action == "open_long")
        return summary
