"""Standalone live runner for the cross-sectional intraday reversion strategy.

This is research-grade infrastructure for collecting real-world fill / slippage
data on the dollar-neutral 30-min-hold variant. It is **deliberately separate**
from `tradingagents/automation/`: no scheduler hook, no risk-engine wiring, no
shared DB, no reconciler. The point is to get fill data without coupling to
production live infra.

Architecture:
  * Mirrors `XSectionReversionBacktester.backtest()`'s selection logic exactly,
    but executes a single rebalance step at wall-clock time.
  * Fetches recent 15-min bars from Alpaca (same pattern as
    `IntradayOrchestrator._fetch_bars`).
  * Captures mid-quote at decision time as the "intent" price, submits market
    orders for the basket, then polls each order to collect the fill price.
  * Logs every intended trade to a JSONL file with both intent and fill, so
    realized slippage can be computed post-hoc from real Alpaca data.

Safety defaults:
  * `dry_run=True` — produces full JSONL of intended actions without
    submitting any orders.
  * `max_gross_exposure=0.5` — caps live gross at 50% of equity (vs the
    backtester's 200%), so blast radius is small for the first paper run.
  * `XSECTION_PAPER_KILL=1` env var aborts the loop on the next tick.

The selection logic must stay in sync with `intraday_xsection_backtester.py`.
A unit test (`test_paper_trader_select_basket_matches_backtester`) is the
guard against drift.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time as time_mod
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

# tradingagents.broker.__init__ has a known circular-import that bites when
# this module is loaded by some test paths. AlpacaBroker is only used as a
# type hint here, so defer it. OrderRequest/OrderResult are constructed at
# runtime; deferred import inside the method that needs them.
if TYPE_CHECKING:  # pragma: no cover
    from tradingagents.broker.alpaca_broker import AlpacaBroker
    from tradingagents.broker.models import OrderRequest, OrderResult

from .intraday_xsection_backtester import (
    XSectionReversionConfig,
    _compute_dollar_volume,
    _filter_session,
)
from .market_context import build_market_context
from .warehouse import MarketDataWarehouse

logger = logging.getLogger(__name__)


def _naive_local_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    """Match the backtester's naive America/Chicago convention."""
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert("America/Chicago").tz_localize(None)


def _to_alpaca_symbol(symbol: str) -> str:
    return symbol.replace("-", ".")


@dataclass
class PaperFillRecord:
    """One row per intended order. Realized slippage = (fill - intent) signed."""
    rebalance_ts: str               # ISO; bar that triggered the decision
    submit_ts: str                  # ISO; when we sent the order to Alpaca
    fill_ts: Optional[str]          # ISO; last poll timestamp the fill became known
    symbol: str
    action: str                     # "open_long" / "open_short" / "flatten"
    side: str                       # "buy" / "sell" — broker side
    qty: int
    intent_price: float             # mid-quote captured before submit
    fill_price: Optional[float]     # filled_avg_price from Alpaca
    fill_status: str                # final order status string
    slippage_bps: Optional[float]   # signed: positive = paid more than intent (bad)
    formation_return: Optional[float]
    regime_label: Optional[str]
    market_score: Optional[int]
    dry_run: bool
    order_id: Optional[str]
    error: Optional[str] = None


@dataclass
class XSectionPaperTraderConfig:
    """Live-runner-specific knobs layered on top of the backtester config."""
    log_dir: Path = field(default_factory=lambda: Path("results/intraday_xsection/paper"))
    dry_run: bool = True
    max_gross_exposure: float = 0.5         # safety cap (backtester default = 1.0)
    bar_grace_seconds: float = 8.0          # wait after bar boundary for Alpaca to publish
    poll_fill_attempts: int = 12            # 12 × 1.0s = 12s budget per order
    poll_fill_delay_seconds: float = 1.0
    bar_lookback_days: int = 5              # enough for the formation window + buffer
    universe_path: str = "research_data/intraday_top250_universe.json"
    daily_db_path: str = "research_data/market_data.duckdb"
    alpaca_data_feed: str = "iex"           # paper accounts default to IEX
    kill_env_var: str = "XSECTION_PAPER_KILL"


class XSectionPaperTrader:
    """One process, one paper account. Walks the rebalance schedule live."""

    def __init__(
        self,
        strategy: XSectionReversionConfig,
        live: XSectionPaperTraderConfig,
        broker: "AlpacaBroker",
        symbols: list[str],
    ):
        if strategy.signal_direction != "reversion":
            # Momentum is allowed by the backtester but the live runner has only
            # been validated against reversion. Fail loudly rather than silently.
            raise ValueError(
                "paper trader currently supports signal_direction='reversion' only"
            )
        self.strategy = strategy
        self.live = live
        self.broker = broker
        self.symbols = list(symbols)
        self.live.log_dir.mkdir(parents=True, exist_ok=True)
        self._stopped = False
        self._jsonl_path = self.live.log_dir / "fills.jsonl"
        self._signal_log_path = self.live.log_dir / "rebalances.jsonl"

    # ── Lifecycle ────────────────────────────────────────────────────

    def install_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            logger.warning("Received signal %s — flagging stop", signum)
            self._stopped = True

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def should_stop(self) -> bool:
        if self._stopped:
            return True
        return os.environ.get(self.live.kill_env_var, "0") == "1"

    # ── Bar fetching ─────────────────────────────────────────────────

    def fetch_bars(self) -> dict[str, pd.DataFrame]:
        """Fetch ``bar_lookback_days`` of ``interval_minutes`` bars for the universe."""
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        feed = (
            DataFeed.SIP
            if self.live.alpaca_data_feed.lower() == "sip"
            else DataFeed.IEX
        )
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.live.bar_lookback_days)
        alpaca_to_universe: dict[str, str] = {
            _to_alpaca_symbol(s): s for s in self.symbols
        }
        alpaca_symbols = list(alpaca_to_universe.keys())
        frames: dict[str, pd.DataFrame] = {}
        batch_size = 100
        for i in range(0, len(alpaca_symbols), batch_size):
            batch = alpaca_symbols[i : i + batch_size]
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame(
                        self.strategy.interval_minutes, TimeFrameUnit.Minute
                    ),
                    start=start,
                    end=end,
                    feed=feed,
                )
                bars = self.broker.data_client.get_stock_bars(request)
            except Exception as exc:
                logger.warning(
                    "bar fetch failed for batch %d-%d: %s",
                    i, i + len(batch), exc,
                )
                continue
            df = bars.df
            if df is None or df.empty:
                continue
            df = df.reset_index()
            df["ts"] = df["timestamp"].apply(_naive_local_timestamp)
            for alpaca_sym, group in df.groupby("symbol"):
                universe_sym = alpaca_to_universe.get(alpaca_sym, alpaca_sym)
                sym_df = (
                    group[["ts", "open", "high", "low", "close", "volume"]]
                    .set_index("ts")
                    .sort_index()
                )
                sym_df = _filter_session(sym_df)
                if not sym_df.empty:
                    frames[universe_sym] = sym_df
        return frames

    # ── Selection (mirrors backtester) ────────────────────────────────

    def select_basket(
        self,
        frames: dict[str, pd.DataFrame],
        rebalance_ts: pd.Timestamp,
    ) -> dict:
        """Run one rebalance step. Returns the same shape as the backtester's
        scheduled rebalance dict, plus diagnostics for logging.

        Exact mirror of `XSectionReversionBacktester.backtest()` lines 786-844.
        """
        cfg = self.strategy
        bars_per_form = max(1, cfg.formation_minutes // cfg.interval_minutes)

        master_index = sorted(set().union(*[df.index for df in frames.values()]))
        master_index = pd.DatetimeIndex(master_index)
        if rebalance_ts not in master_index:
            return {"action": "no_bar", "rebalance_ts": rebalance_ts}
        i = master_index.get_loc(rebalance_ts)
        if i < bars_per_form:
            return {"action": "warmup", "rebalance_ts": rebalance_ts, "bars": i}

        close_df = pd.DataFrame(
            {sym: df["close"] for sym, df in frames.items()}, index=master_index
        )
        bars_per_day = (
            int((dtime(15, 0).hour * 60 + dtime(15, 0).minute
                 - dtime(8, 30).hour * 60 - dtime(8, 30).minute)
                / cfg.interval_minutes) + 1
        )
        liquidity_lookback_bars = bars_per_day * 20
        dv_df = pd.DataFrame(
            {
                sym: _compute_dollar_volume(df, liquidity_lookback_bars)
                for sym, df in frames.items()
            },
            index=master_index,
        )

        regime_label, regime_score = self._regime_lookup(rebalance_ts)
        if cfg.allowed_market_regimes and regime_label not in set(
            cfg.allowed_market_regimes
        ):
            return {
                "action": "regime_blocked",
                "rebalance_ts": rebalance_ts,
                "regime_label": regime_label,
                "regime_score": regime_score,
            }
        if (
            cfg.market_context_min_score is not None
            and (regime_score is None or regime_score < int(cfg.market_context_min_score))
        ):
            return {
                "action": "regime_blocked",
                "rebalance_ts": rebalance_ts,
                "regime_label": regime_label,
                "regime_score": regime_score,
            }
        if (
            cfg.market_context_max_score is not None
            and (regime_score is None or regime_score > int(cfg.market_context_max_score))
        ):
            return {
                "action": "regime_blocked",
                "rebalance_ts": rebalance_ts,
                "regime_label": regime_label,
                "regime_score": regime_score,
            }

        form_idx = i - bars_per_form - cfg.formation_lag_bars
        if form_idx < 0:
            return {"action": "warmup", "rebalance_ts": rebalance_ts}
        ret = (close_df.iloc[i] / close_df.iloc[form_idx]) - 1.0
        # Live runs typically fetch only ~5 days of bars while
        # `_compute_dollar_volume`'s rolling window expects 20 days. With a
        # short fetch, dv_df.iloc[i] is all NaN and `NaN >= threshold` is
        # always False — pandas filters out the entire universe even when the
        # threshold is 0. Treat threshold <= 0 as "skip the filter" so a
        # pre-screened universe (broad250) doesn't need a 20-day warm-up.
        if cfg.min_dollar_volume_avg > 0:
            liq_ok = dv_df.iloc[i] >= cfg.min_dollar_volume_avg
            ret = ret.where(liq_ok).dropna()
        else:
            ret = ret.dropna()
        if ret.empty:
            return {"action": "no_signal", "rebalance_ts": rebalance_ts}

        rank_score = ret.sort_values()
        longs = list(rank_score.head(cfg.n_long).index)
        shorts = list(rank_score.tail(cfg.n_short).index)
        return {
            "action": "rebalance",
            "rebalance_ts": rebalance_ts,
            "longs": longs,
            "shorts": shorts,
            "ret": {sym: float(ret[sym]) for sym in ret.index},
            "regime_label": regime_label,
            "regime_score": regime_score,
        }

    def _regime_lookup(
        self, rebalance_ts: pd.Timestamp
    ) -> tuple[Optional[str], Optional[int]]:
        """Pull prior-session regime label/score. Empty config => (None, None)."""
        cfg = self.strategy
        if (
            cfg.market_context_min_score is None
            and cfg.market_context_max_score is None
            and not cfg.allowed_market_regimes
        ):
            return None, None
        try:
            warehouse = MarketDataWarehouse(self.live.daily_db_path, read_only=True)
        except Exception as exc:
            logger.warning("market_context disabled — warehouse open failed: %s", exc)
            return None, None
        try:
            buffer_begin = (rebalance_ts - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
            end_str = rebalance_ts.strftime("%Y-%m-%d")
            frames = {}
            for symbol in ("SPY", "QQQ", "IWM", "SMH", "^VIX"):
                df = warehouse.get_daily_bars(symbol, buffer_begin, end_str)
                if df is not None and not df.empty:
                    frames[symbol] = df.copy().sort_index()
        finally:
            warehouse.close()
        if not frames:
            return None, None
        context = build_market_context(frames)
        if context.empty:
            return None, None
        context = context.copy()
        context.index = pd.to_datetime(context.index).normalize()
        session_date = pd.Timestamp(rebalance_ts.normalize())
        prior = context.loc[context.index < session_date]
        if prior.empty:
            return None, None
        latest = prior.iloc[-1]
        regime = (
            str(latest["market_regime"])
            if "market_regime" in latest and pd.notna(latest["market_regime"])
            else None
        )
        score = (
            int(latest["market_score"])
            if "market_score" in latest and pd.notna(latest["market_score"])
            else None
        )
        return regime, score

    # ── Order submission ─────────────────────────────────────────────

    def capture_intent_prices(self, symbols: list[str]) -> dict[str, float]:
        """Mid-quote at the moment of decision. Returned in universe symbol form."""
        if not symbols:
            return {}
        alpaca_symbols = [_to_alpaca_symbol(s) for s in symbols]
        try:
            quotes = self.broker.get_latest_prices(alpaca_symbols)
        except Exception as exc:
            logger.warning("get_latest_prices failed: %s — skipping intent capture", exc)
            return {}
        out: dict[str, float] = {}
        # Map Alpaca form back to universe form
        for universe_sym, alpaca_sym in zip(symbols, alpaca_symbols):
            px = quotes.get(alpaca_sym)
            if px is not None and px > 0:
                out[universe_sym] = float(px)
        return out

    def _compute_target_dollars(self) -> tuple[float, float]:
        """Per-name dollar allocation given current account equity and safety cap."""
        try:
            account = self.broker.get_account()
            equity = float(account.equity)
        except Exception as exc:
            logger.warning("get_account failed: %s — using initial_cash fallback", exc)
            equity = self.strategy.initial_cash

        cfg = self.strategy
        # Apply both the strategy's gross-exposure knob AND the live safety cap.
        effective_gross = min(cfg.target_gross_exposure, self.live.max_gross_exposure)
        if cfg.dollar_neutral:
            long_per_name = (
                equity * effective_gross / cfg.n_long if cfg.n_long > 0 else 0.0
            )
            short_per_name = (
                equity * effective_gross / cfg.n_short if cfg.n_short > 0 else 0.0
            )
        else:
            total_names = cfg.n_long + cfg.n_short
            shared = equity * effective_gross / total_names if total_names > 0 else 0.0
            long_per_name = short_per_name = shared
        return long_per_name, short_per_name

    def flatten_all(self, rebalance_ts: pd.Timestamp) -> list[PaperFillRecord]:
        """Close every open position. In dry-run, log only."""
        records: list[PaperFillRecord] = []
        try:
            positions = self.broker.get_positions()
        except Exception as exc:
            logger.warning("get_positions failed: %s", exc)
            return records
        if not positions:
            return records
        intents = self.capture_intent_prices([p.symbol for p in positions])
        submit_ts = datetime.now(timezone.utc).isoformat()
        if self.live.dry_run:
            for pos in positions:
                # AlpacaBroker stores side as `str(p.side)` which yields
                # "PositionSide.LONG" / "PositionSide.SHORT" (enum repr), not
                # "long"/"short". Normalize before comparing or every long is
                # mislabeled as a buy-to-close.
                pos_side = str(pos.side).split(".")[-1].lower()
                records.append(
                    PaperFillRecord(
                        rebalance_ts=rebalance_ts.isoformat(),
                        submit_ts=submit_ts,
                        fill_ts=None,
                        symbol=pos.symbol,
                        action="flatten",
                        side="sell" if pos_side == "long" else "buy",
                        qty=int(pos.qty),
                        intent_price=intents.get(pos.symbol, pos.current_price),
                        fill_price=None,
                        fill_status="dry_run",
                        slippage_bps=None,
                        formation_return=None,
                        regime_label=None,
                        market_score=None,
                        dry_run=True,
                        order_id=None,
                    )
                )
            return records
        # Live: atomic close-all
        try:
            results = self.broker.close_all_positions()
        except Exception as exc:
            logger.error("close_all_positions failed: %s", exc)
            return records
        # Poll each result for filled price.
        for r in results:
            final = self._poll_until_terminal(r.order_id)
            intent_px = intents.get(r.symbol)
            slip = self._slippage_bps(intent_px, final.filled_avg_price, r.side)
            records.append(
                PaperFillRecord(
                    rebalance_ts=rebalance_ts.isoformat(),
                    submit_ts=submit_ts,
                    fill_ts=final.filled_at.isoformat() if final.filled_at else None,
                    symbol=r.symbol,
                    action="flatten",
                    side=r.side,
                    qty=int(final.filled_qty or 0),
                    intent_price=intent_px or 0.0,
                    fill_price=final.filled_avg_price,
                    fill_status=final.status,
                    slippage_bps=slip,
                    formation_return=None,
                    regime_label=None,
                    market_score=None,
                    dry_run=False,
                    order_id=r.order_id,
                )
            )
        return records

    def open_basket(
        self,
        decision: dict,
        long_per_name: float,
        short_per_name: float,
    ) -> list[PaperFillRecord]:
        """Submit market orders for the longs/shorts of one rebalance decision."""
        records: list[PaperFillRecord] = []
        all_targets = decision["longs"] + decision["shorts"]
        intents = self.capture_intent_prices(all_targets)
        submit_ts = datetime.now(timezone.utc).isoformat()
        rebalance_iso = decision["rebalance_ts"].isoformat()

        for sym in decision["longs"]:
            intent_px = intents.get(sym)
            if intent_px is None or intent_px <= 0:
                logger.info("skipping %s long — no intent price", sym)
                continue
            qty = int(long_per_name / intent_px)
            if qty <= 0:
                continue
            records.append(
                self._submit_one(
                    sym=sym,
                    side="buy",
                    qty=qty,
                    intent_price=intent_px,
                    rebalance_iso=rebalance_iso,
                    submit_ts=submit_ts,
                    action="open_long",
                    formation_return=decision["ret"].get(sym),
                    regime_label=decision.get("regime_label"),
                    market_score=decision.get("regime_score"),
                )
            )
        for sym in decision["shorts"]:
            intent_px = intents.get(sym)
            if intent_px is None or intent_px <= 0:
                logger.info("skipping %s short — no intent price", sym)
                continue
            qty = int(short_per_name / intent_px)
            if qty <= 0:
                continue
            records.append(
                self._submit_one(
                    sym=sym,
                    side="sell",
                    qty=qty,
                    intent_price=intent_px,
                    rebalance_iso=rebalance_iso,
                    submit_ts=submit_ts,
                    action="open_short",
                    formation_return=decision["ret"].get(sym),
                    regime_label=decision.get("regime_label"),
                    market_score=decision.get("regime_score"),
                )
            )
        return records

    def _submit_one(
        self,
        sym: str,
        side: str,
        qty: int,
        intent_price: float,
        rebalance_iso: str,
        submit_ts: str,
        action: str,
        formation_return: Optional[float],
        regime_label: Optional[str],
        market_score: Optional[int],
    ) -> PaperFillRecord:
        if self.live.dry_run:
            return PaperFillRecord(
                rebalance_ts=rebalance_iso,
                submit_ts=submit_ts,
                fill_ts=None,
                symbol=sym,
                action=action,
                side=side,
                qty=qty,
                intent_price=intent_price,
                fill_price=None,
                fill_status="dry_run",
                slippage_bps=None,
                formation_return=formation_return,
                regime_label=regime_label,
                market_score=market_score,
                dry_run=True,
                order_id=None,
            )
        from tradingagents.broker.models import OrderRequest as _OrderRequest
        order = _OrderRequest(
            symbol=_to_alpaca_symbol(sym),
            side=side,  # type: ignore[arg-type]
            qty=qty,
            order_type="market",
            time_in_force="day",
        )
        try:
            result = self.broker.submit_order(order)
        except Exception as exc:
            logger.error("submit_order(%s) failed: %s", sym, exc)
            return PaperFillRecord(
                rebalance_ts=rebalance_iso,
                submit_ts=submit_ts,
                fill_ts=None,
                symbol=sym,
                action=action,
                side=side,
                qty=qty,
                intent_price=intent_price,
                fill_price=None,
                fill_status="submit_failed",
                slippage_bps=None,
                formation_return=formation_return,
                regime_label=regime_label,
                market_score=market_score,
                dry_run=False,
                order_id=None,
                error=str(exc),
            )
        final = self._poll_until_terminal(result.order_id)
        slip = self._slippage_bps(intent_price, final.filled_avg_price, side)
        return PaperFillRecord(
            rebalance_ts=rebalance_iso,
            submit_ts=submit_ts,
            fill_ts=final.filled_at.isoformat() if final.filled_at else None,
            symbol=sym,
            action=action,
            side=side,
            qty=int(final.filled_qty or 0),
            intent_price=intent_price,
            fill_price=final.filled_avg_price,
            fill_status=final.status,
            slippage_bps=slip,
            formation_return=formation_return,
            regime_label=regime_label,
            market_score=market_score,
            dry_run=False,
            order_id=result.order_id,
        )

    def _poll_until_terminal(self, order_id: str) -> "OrderResult":
        terminal = {"filled", "canceled", "cancelled", "expired", "rejected"}
        last: Optional[OrderResult] = None
        for _ in range(self.live.poll_fill_attempts):
            try:
                last = self.broker.get_order(order_id)
            except Exception as exc:
                logger.warning("get_order(%s) failed: %s", order_id, exc)
                time_mod.sleep(self.live.poll_fill_delay_seconds)
                continue
            status_l = str(last.status).lower().split(".")[-1]
            if status_l in terminal or last.filled_avg_price:
                return last
            time_mod.sleep(self.live.poll_fill_delay_seconds)
        if last is None:
            # Synthesize a stub so downstream logging doesn't crash.
            from tradingagents.broker.models import OrderResult as _OrderResult
            return _OrderResult(
                order_id=order_id, symbol="?", side="?", qty=None, notional=None,
                order_type="market", status="poll_timeout",
            )
        return last

    @staticmethod
    def _slippage_bps(
        intent: Optional[float],
        fill: Optional[float],
        side: str,
    ) -> Optional[float]:
        """Signed slippage. Positive = unfavorable for us:
            buy:  fill > intent  →  positive bps
            sell: fill < intent  →  positive bps
        """
        if intent is None or fill is None or intent <= 0:
            return None
        raw = (fill - intent) / intent * 1e4
        return round(raw if side == "buy" else -raw, 2)

    # ── Logging ──────────────────────────────────────────────────────

    def append_records(self, records: list[PaperFillRecord]) -> None:
        if not records:
            return
        with self._jsonl_path.open("a") as fh:
            for r in records:
                fh.write(json.dumps(r.__dict__, default=str) + "\n")

    def append_rebalance_summary(self, decision: dict) -> None:
        with self._signal_log_path.open("a") as fh:
            fh.write(json.dumps(decision, default=str) + "\n")

    # ── Top-level rebalance step ─────────────────────────────────────

    def run_one_rebalance(self, rebalance_ts: pd.Timestamp) -> dict:
        """End-to-end: fetch bars, select, capture intent, submit, log."""
        frames = self.fetch_bars()
        if not frames:
            logger.warning("no bars returned — skipping %s", rebalance_ts)
            return {"action": "no_bars", "rebalance_ts": rebalance_ts}
        decision = self.select_basket(frames, rebalance_ts)
        self.append_rebalance_summary(decision)
        if decision.get("action") != "rebalance":
            return decision

        flatten_records = self.flatten_all(rebalance_ts)
        long_per_name, short_per_name = self._compute_target_dollars()
        open_records = self.open_basket(decision, long_per_name, short_per_name)
        self.append_records(flatten_records + open_records)
        return {
            **decision,
            "flatten_count": len(flatten_records),
            "open_count": len(open_records),
        }

    def run_eod_flatten(self, now: pd.Timestamp) -> dict:
        records = self.flatten_all(now)
        self.append_records(records)
        return {"action": "eod_flatten", "count": len(records)}
