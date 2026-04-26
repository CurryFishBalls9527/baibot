"""Intraday mechanical orchestrator — NR4 + gap-reclaim breakout on 15m bars.

Designed to run alongside the existing Minervini and Chan orchestrators on a
separate Alpaca account. Delegates signal generation to
IntradayBreakoutBacktester.prepare_signals so live logic stays bit-for-bit
aligned with the research baseline `gap12 + NR4 vol=2.0` (validated OOS on
2020/2018).

Flow per scheduler tick (every 15 min during market hours):
  1. Fetch last ~7 sessions of 15m bars for each universe symbol.
  2. Run prepare_signals to annotate entry_signal / setup_family per bar.
  3. On the latest completed bar, if entry_signal=True and the setup family
     is allowed, size + risk-check + submit bracket order (stop @ -3%).
  4. End-of-day flatten runs on a separate 15:55 ET scheduler job — NOT from
     the per-tick scan — because we want the flatten to fire even when no
     scan did on the final bar.

Dry-run mode (`intraday_dry_run: true`) skips broker submission and logs
the would-be orders; this is the launch default.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from tradingagents.automation.events import Categories, emit_event
from tradingagents.automation.notifier import build_notifier
from tradingagents.broker.alpaca_broker import AlpacaBroker
from tradingagents.broker.models import Account, OrderRequest, Position
from tradingagents.portfolio.portfolio_tracker import PortfolioTracker
from tradingagents.portfolio.position_sizer import PositionSizer
from tradingagents.research.intraday_backtester import (
    IntradayBacktestConfig,
    IntradayBreakoutBacktester,
)
from tradingagents.risk.risk_engine import RiskEngine
from tradingagents.storage.database import TradingDatabase

logger = logging.getLogger(__name__)

# Setup families whose live deployment we have OOS evidence for.
# gap_reclaim_long and nr4_breakout are the validated families from the
# intraday research line (see memory/project_nr4_finding.md).
_DEFAULT_ALLOWED_FAMILIES: Set[str] = {"gap_reclaim_long", "nr4_breakout"}


def _naive_local_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert a tz-aware (UTC or otherwise) timestamp to naive America/Chicago.

    The backtester assumes naive bar timestamps in the workstation-local
    America/Chicago zone (08:30–15:00 local = US equities regular session).
    Alpaca returns UTC-aware timestamps, so we convert then drop the tz.
    """
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert("America/Chicago").tz_localize(None)


def _to_alpaca_symbol(symbol: str) -> str:
    """Alpaca uses `.` for share class separators where universe files use `-`
    (e.g., `BRK-B` → `BRK.B`). A single bad symbol rejects the whole batch, so
    normalize before the API call."""
    return symbol.replace("-", ".")


class IntradayOrchestrator:
    """Mechanical intraday breakout orchestrator (NR4 + gap-reclaim)."""

    def __init__(self, config: dict):
        self.config = config

        self.broker = AlpacaBroker(
            api_key=config["alpaca_api_key"],
            secret_key=config["alpaca_secret_key"],
            paper=config.get("paper_trading", True),
        )

        self.db = TradingDatabase(
            config.get("db_path", "trading_intraday.db"),
            variant=config.get("variant_name"),
        )
        config.setdefault("strategy_tag", "intraday_mechanical")
        self.notifier = build_notifier(config)

        starting_equity = self.db.get_starting_equity()
        risk_config = {**config, "starting_equity": starting_equity}
        self.risk_engine = RiskEngine(risk_config)
        self.sizer = PositionSizer(config)
        self.tracker = PortfolioTracker(self.broker, self.db)

        self.dry_run = bool(config.get("intraday_dry_run", True))

        bt_config_kwargs = config.get("intraday_backtest_config", {}) or {}
        self.bt_config = IntradayBacktestConfig(**bt_config_kwargs)
        self.backtester = IntradayBreakoutBacktester(self.bt_config)

        self.max_positions = int(
            config.get("max_positions", self.bt_config.max_positions)
        )
        self.interval_minutes = int(self.bt_config.interval_minutes)
        self.lookback_days = int(config.get("intraday_lookback_days", 7))
        self.allowed_families: Set[str] = set(
            config.get("intraday_allowed_families", _DEFAULT_ALLOWED_FAMILIES)
        )

        self.universe = self._load_universe()

        # Per-day dedup: (symbol, date) -> bar timestamp of the entry we took.
        # Cleared each session by flatten_all.
        self._entered_today: Dict[str, pd.Timestamp] = {}

        # Daily-cached Minervini regime label. Computed once per session date
        # from SPY (close > SMA50 AND > SMA200 AND SMA200 rising). Used as a
        # pre-scan gate when bt_config.minervini_regime_filter is set. See
        # memory/project_intraday_minervini_gate.md for backtest evidence.
        self._minervini_regime_cache: Optional[Tuple[date, str]] = None

    # ------------------------------------------------------------------ universe

    def _load_universe(self) -> List[str]:
        path = self.config.get("intraday_universe_path")
        explicit = self.config.get("intraday_universe_symbols")
        if explicit:
            return list(explicit)
        if not path:
            logger.warning(
                "IntradayOrchestrator: no intraday_universe_path or "
                "intraday_universe_symbols set — universe is empty"
            )
            return []
        try:
            payload = json.loads(Path(path).read_text())
        except Exception as e:
            logger.warning("Failed to load intraday universe from %s: %s", path, e)
            return []
        if isinstance(payload, dict):
            return list(payload.get("symbols") or [])
        return list(payload)

    # ------------------------------------------------------------------ data

    def _fetch_bars(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch last `lookback_days` of interval_minutes bars from Alpaca.

        Returns a dict symbol -> OHLCV DataFrame with a naive local-time
        DatetimeIndex matching the backtester's expected format.
        """
        if not symbols:
            return {}
        try:
            from alpaca.data.enums import DataFeed
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        except ImportError:
            logger.warning("alpaca-py not installed — cannot fetch intraday bars")
            return {}

        # Free Alpaca paper subscriptions cannot query recent SIP data; fall
        # back to IEX unless the variant config explicitly requests SIP.
        feed_name = str(self.config.get("alpaca_data_feed", "iex")).lower()
        feed = DataFeed.SIP if feed_name == "sip" else DataFeed.IEX

        client = StockHistoricalDataClient(
            self.config["alpaca_api_key"], self.config["alpaca_secret_key"]
        )
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.lookback_days)

        frames: Dict[str, pd.DataFrame] = {}
        # Alpaca batch API accepts a list; a single request per 100 symbols
        # is plenty for a 52-symbol universe. Normalize BRK-B → BRK.B etc so
        # a single bad symbol doesn't reject the whole batch.
        alpaca_to_universe: Dict[str, str] = {
            _to_alpaca_symbol(s): s for s in symbols
        }
        alpaca_symbols = list(alpaca_to_universe.keys())
        batch_size = 100
        for i in range(0, len(alpaca_symbols), batch_size):
            batch = alpaca_symbols[i : i + batch_size]
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame(self.interval_minutes, TimeFrameUnit.Minute),
                    start=start,
                    end=end,
                    feed=feed,
                )
                bars = client.get_stock_bars(request)
            except Exception as e:
                logger.warning(
                    "Alpaca bar fetch failed for batch %d-%d: %s",
                    i, i + len(batch), e,
                )
                continue
            df = bars.df
            if df is None or df.empty:
                continue
            df = df.reset_index()
            df["ts"] = df["timestamp"].apply(_naive_local_timestamp)
            for alpaca_sym, group in df.groupby("symbol"):
                # Map back to the universe-file form so downstream dedup /
                # position-tracking keys line up with what the rest of the
                # codebase uses.
                universe_sym = alpaca_to_universe.get(alpaca_sym, alpaca_sym)
                sym_df = (
                    group[["ts", "open", "high", "low", "close", "volume"]]
                    .set_index("ts")
                    .sort_index()
                )
                sym_df = self.backtester._filter_regular_session(sym_df)
                if not sym_df.empty:
                    frames[universe_sym] = sym_df
        return frames

    # ------------------------------------------------------------------ regime gate

    def _minervini_regime_today(self) -> Optional[str]:
        """Minervini market regime label for the prior trading session.

        Cached per session date so the daily warehouse is queried once per
        scheduler tick day, not per scan tick. Returns None when the gate
        is disabled, the warehouse is missing, or SPY history is too short.
        """
        if self.bt_config.minervini_regime_filter is None:
            return None
        today = date.today()
        if (
            self._minervini_regime_cache is not None
            and self._minervini_regime_cache[0] == today
        ):
            return self._minervini_regime_cache[1]
        try:
            # Fetch a year of SPY daily bars and reuse the same classifier the
            # backtester uses (`<` boundary — prior session's regime, never
            # today's close).
            series = self.backtester._load_minervini_regime(
                begin=(today - timedelta(days=400)).isoformat(),
                end=today.isoformat(),
            )
            if series.empty:
                self._minervini_regime_cache = (today, "unknown")
                return "unknown"
            today_ts = pd.Timestamp(today)
            valid = series.loc[series.index < today_ts]
            label = str(valid.iloc[-1]) if not valid.empty else "unknown"
        except Exception as e:
            logger.warning("Minervini regime lookup failed: %s", e)
            label = "unknown"
        self._minervini_regime_cache = (today, label)
        return label

    def _minervini_regime_allows_entry(self) -> bool:
        """True iff today's prior-session Minervini regime passes the gate.

        Mirrors the backtester logic: with `confirmed_uptrend_only`, only
        confirmed_uptrend passes; with `any_uptrend`, both confirmed_uptrend
        and uptrend_under_pressure pass. None / unknown filter → permissive.
        """
        filter_mode = self.bt_config.minervini_regime_filter
        if filter_mode is None:
            return True
        regime = self._minervini_regime_today()
        if regime is None:
            return True
        if filter_mode == "confirmed_uptrend_only":
            return regime == "confirmed_uptrend"
        if filter_mode == "any_uptrend":
            return regime in ("confirmed_uptrend", "uptrend_under_pressure")
        return True

    # ------------------------------------------------------------------ scan

    def scan(self) -> Dict[str, Any]:
        """Scheduler entry — look for entries on the latest 15m bar."""
        logger.info("=== Intraday Orchestrator: Scan (dry_run=%s) ===", self.dry_run)
        if not self.broker.is_market_open():
            logger.info("Market closed — skipping intraday scan")
            return {"status": "market_closed", "entries": []}

        if not self._minervini_regime_allows_entry():
            regime = self._minervini_regime_today()
            logger.info(
                "Minervini regime gate: skipping scan (regime=%s, filter=%s)",
                regime, self.bt_config.minervini_regime_filter,
            )
            return {
                "status": "regime_blocked",
                "regime": regime,
                "filter": self.bt_config.minervini_regime_filter,
                "entries": [],
            }

        try:
            account = self.broker.get_account()
            positions = self.broker.get_positions()
        except Exception as e:
            logger.error("Intraday: failed to fetch account state: %s", e)
            return {"error": str(e), "entries": []}

        held_symbols = {p.symbol for p in positions}
        if len(positions) >= self.max_positions:
            logger.info(
                "At max positions (%d/%d), skipping entry scan",
                len(positions), self.max_positions,
            )
            return {"entries": [], "positions": len(positions), "maxed": True}

        candidates = [s for s in self.universe if s not in held_symbols]
        frames = self._fetch_bars(candidates)
        if not frames:
            logger.info("No intraday bars returned for universe — nothing to scan")
            return {"entries": []}

        signals: List[Dict[str, Any]] = []
        for sym, frame in frames.items():
            sig = self._check_symbol_signal(sym, frame)
            if sig is not None:
                signals.append(sig)

        # Rank by candidate_score desc, setup_priority as tiebreaker
        signals.sort(
            key=lambda s: (s["candidate_score"], s["setup_priority"]),
            reverse=True,
        )

        entries: List[Dict[str, Any]] = []
        slots = self.max_positions - len(positions)
        for sig in signals[:slots]:
            try:
                result = self._execute_entry(sig, account, positions)
                entries.append(result)
                if result.get("traded"):
                    positions = self.broker.get_positions()
                    account = self.broker.get_account()
                    if len(positions) >= self.max_positions:
                        break
            except Exception as e:
                logger.warning("Entry execution failed for %s: %s", sig["symbol"], e)
                entries.append(
                    {"symbol": sig["symbol"], "traded": False, "error": str(e)}
                )

        logger.info(
            "Intraday scan: %d candidates, %d signals, %d entries",
            len(candidates), len(signals), len([e for e in entries if e.get("traded")]),
        )
        self._notify_summary(entries)
        return {
            "entries": entries,
            "signals_evaluated": len(signals),
            "candidates_scanned": len(candidates),
            "dry_run": self.dry_run,
        }

    def _check_symbol_signal(
        self, symbol: str, frame: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Return a signal dict if the latest completed bar fires, else None."""
        if len(frame) < 21:  # backtester needs ~20 bars for volume_ratio
            return None
        try:
            features = self.backtester.prepare_signals(frame)
        except Exception as e:
            logger.debug("prepare_signals failed for %s: %s", symbol, e)
            return None

        if features.empty:
            return None
        last = features.iloc[-1]
        if not bool(last.get("entry_signal", False)):
            return None
        family = str(last.get("setup_family", "none"))
        if family not in self.allowed_families:
            return None

        bar_ts = features.index[-1]
        session = str(last.get("session_date"))
        dedup_key = f"{symbol}:{session}"
        if dedup_key in self._entered_today:
            return None

        # Snapshot the structured setup levels that fired this signal so the
        # daily trade-review can later annotate a chart with them. Kept as a
        # plain dict here; serialized to JSON at log_signal time. Each value
        # is sanitised through `_safe_float` at consumption time so NaN / inf
        # can't break JSON encoding.
        def _num(key):
            try:
                v = last.get(key)
                return float(v) if v is not None and v == v else None  # NaN→None
            except Exception:
                return None

        metadata = {
            "setup_family": family,
            "bar_ts": str(bar_ts),
            "session_date": session,
            "entry_close": float(last["close"]),
            "volume_ratio": _num("volume_ratio"),
            "breakout_distance_pct": _num("breakout_distance_pct"),
            "candidate_score": _num("candidate_score"),
            "vwap": _num("vwap"),
            "distance_from_vwap_pct": _num("distance_from_vwap_pct"),
            "opening_range_high": _num("opening_range_high"),
            "opening_range_low": _num("opening_range_low"),
            "prior_session_high": _num("prior_session_high"),
            "prior_session_close": _num("prior_session_close"),
            "prior_session_is_nr": bool(last.get("prior_session_is_nr")),
        }

        return {
            "symbol": symbol,
            "setup_family": family,
            "bar_ts": bar_ts,
            "session": session,
            "dedup_key": dedup_key,
            "close": float(last["close"]),
            "volume_ratio": float(last.get("volume_ratio") or 0.0),
            "breakout_distance_pct": float(last.get("breakout_distance_pct") or 0.0),
            "candidate_score": float(last.get("candidate_score") or 0.0),
            "setup_priority": int(
                self.backtester._setup_priority(family)
            ),
            "metadata": metadata,
        }

    # ------------------------------------------------------------------ entry

    def _execute_entry(
        self,
        signal: Dict[str, Any],
        account: Account,
        positions: List[Position],
    ) -> Dict[str, Any]:
        symbol = signal["symbol"]
        stop_pct = float(self.bt_config.stop_loss_pct)
        # Far-away take-profit so the bracket doesn't short-circuit the EOD
        # flatten exit. NR4 research showed no TP in the winning config; we
        # keep TP as a placeholder to satisfy Alpaca's bracket requirement.
        # 2026-04-23: default tightened from 1.0 (100% = $1,192 limit on $596
        # STX, alarming on the dashboard) to 0.20. Still unreachable intraday,
        # and if a position ever gets stranded overnight the TP is merely
        # "above a plausible ceiling" rather than in fantasy territory.
        tp_pct = float(self.config.get("intraday_bracket_tp_pct", 0.20))

        try:
            current_price = self.broker.get_latest_price(symbol)
        except Exception as e:
            logger.warning("get_latest_price(%s) failed: %s", symbol, e)
            return {"symbol": symbol, "traded": False, "error": f"price: {e}"}

        order_signal = {
            "symbol": symbol,
            "action": "BUY",
            "confidence": 0.7,
            "reasoning": (
                f"{signal['setup_family']} @ bar {signal['bar_ts']} "
                f"(vol_ratio={signal['volume_ratio']:.2f})"
            ),
            "stop_loss_pct": stop_pct,
            "take_profit_pct": tp_pct,
            "stop_loss": current_price * (1 - stop_pct),
            "timeframe": "intraday",
            "source": "intraday_mechanical",
        }

        # Serialize structured setup metadata (ORB levels, NR prior-range,
        # VWAP, etc) for the daily trade review. Wrapped in try/except so a
        # JSON edge case (NaN, inf, exotic types) can never block entry. The
        # metadata column is NULL-tolerant on the read side.
        try:
            import json, math
            def _clean(v):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    return None
                return v
            cleaned = {k: _clean(v) for k, v in signal.get("metadata", {}).items()}
            metadata_json = json.dumps(cleaned, default=str)
        except Exception as e:
            logger.warning("intraday signal_metadata serialize failed: %s", e)
            metadata_json = None

        signal_id = self.db.log_signal(
            symbol=symbol,
            action="BUY",
            confidence=0.7,
            reasoning=order_signal["reasoning"],
            stop_loss=stop_pct,
            take_profit=tp_pct,
            timeframe="intraday",
            signal_metadata=metadata_json,
        )

        total_pos_value = sum(p.market_value for p in positions)
        order_request: Optional[OrderRequest] = self.sizer.calculate(
            signal=order_signal,
            account=account,
            current_price=current_price,
            current_position=None,
            total_position_value=total_pos_value,
        )
        if order_request is None:
            self.db.mark_signal_rejected(signal_id, "sizer returned None")
            return {"symbol": symbol, "traded": False, "reason": "sizer_none"}

        # Intraday must close same-day → force day TIF
        order_request.time_in_force = "day"

        risk_result = self.risk_engine.check_order(
            order_request, account, positions, current_price
        )
        if not risk_result.passed:
            logger.info("%s: risk rejected: %s", symbol, risk_result.reason)
            self.db.mark_signal_rejected(signal_id, risk_result.reason)
            return {"symbol": symbol, "traded": False, "risk_rejected": risk_result.reason}

        sl_price = round(current_price * (1 - stop_pct), 2)
        tp_price = round(current_price * (1 + tp_pct), 2)

        if self.dry_run:
            logger.info(
                "DRY_RUN: would BUY %d %s @ ~$%.2f (SL=$%.2f, TP=$%.2f) via %s",
                order_request.qty, symbol, current_price,
                sl_price, tp_price, signal["setup_family"],
            )
            self._entered_today[signal["dedup_key"]] = signal["bar_ts"]
            self.db.mark_signal_executed(signal_id)
            return {
                "symbol": symbol, "traded": False, "dry_run": True,
                "qty": order_request.qty, "price": current_price,
                "setup_family": signal["setup_family"],
            }

        order_result = self.broker.submit_bracket_order(
            order_request,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            anchor_price=current_price,
        )
        sl_price = order_result.effective_stop_price or sl_price
        tp_price = order_result.effective_take_profit_price or tp_price
        self.risk_engine.record_trade()
        self.db.log_trade(
            symbol=symbol,
            side="buy",
            qty=order_request.qty,
            order_type="bracket",
            status=order_result.status,
            filled_qty=order_result.filled_qty,
            filled_price=order_result.filled_avg_price,
            order_id=order_result.order_id,
            signal_id=signal_id,
            reasoning=order_signal["reasoning"],
        )
        self.db.mark_signal_executed(signal_id)
        self._entered_today[signal["dedup_key"]] = signal["bar_ts"]

        if "fill" in str(order_result.status).lower():
            self.db.upsert_position_state(symbol, {
                "entry_price": order_result.filled_avg_price or current_price,
                "entry_date": date.today().isoformat(),
                "highest_close": order_result.filled_avg_price or current_price,
                "current_stop": sl_price,
                "partial_taken": False,
                "variant": self.config.get("variant_name", "intraday_mechanical"),
                # Setup family (nr4_breakout / orb_breakout / gap_reclaim_long /
                # ...) used as base_pattern so the daily/weekly review can cut
                # stats by setup. Regime left NULL here: intraday doesn't join
                # the daily market-regime score, and the signal_metadata JSON
                # already captures session-level context for charting.
                "base_pattern": signal.get("setup_family"),
            })

        return {
            "symbol": symbol, "traded": True,
            "qty": order_request.qty, "status": order_result.status,
            "setup_family": signal["setup_family"],
        }

    # ------------------------------------------------------------------ flatten

    def flatten_all(self) -> Dict[str, Any]:
        """End-of-day exit: atomically cancel all open orders and market-close
        every position on this intraday account.

        Uses ``broker.close_all_positions(cancel_orders=True)`` so Alpaca
        handles the cancel-then-close atomically. The prior per-symbol
        cancel loop was racing the bracket children — ``get_live_orders``
        sometimes missed the OCO stop leg, and even when it did cancel,
        ``close_position`` could fire before Alpaca released
        ``held_for_orders`` and get rejected with 40310000 (2026-04-23:
        KDP 420sh and STX 20sh stranded overnight).
        """
        logger.info(
            "=== Intraday Orchestrator: EOD Flatten (dry_run=%s) ===",
            self.dry_run,
        )
        variant_name = self.config.get("variant_name", "intraday_mechanical")
        try:
            positions = self.broker.get_positions()
        except Exception as e:
            logger.error("flatten_all: get_positions failed: %s", e)
            emit_event(
                Categories.JOB_FAILED,
                level="error",
                variant=variant_name,
                message="flatten_all: get_positions failed",
                context={"error": str(e)},
            )
            return {"error": str(e), "closed": []}

        if not positions:
            logger.info("flatten_all: no positions to close")
            self._entered_today.clear()
            return {"closed": [], "positions_closed": 0, "dry_run": self.dry_run}

        # Capture entry context BEFORE the close so the trade_outcome hook
        # can use it even if delete_position_state runs first.
        pos_by_symbol = {p.symbol: p for p in positions}
        state_by_symbol = {
            sym: self.db.get_position_state(sym) for sym in pos_by_symbol
        }

        closed: List[Dict[str, Any]] = []
        if self.dry_run:
            for symbol, pos in pos_by_symbol.items():
                logger.info("DRY_RUN: would CLOSE %s %s shares", pos.qty, symbol)
                closed.append({"symbol": symbol, "closed": False, "dry_run": True})
            self._entered_today.clear()
            self._notify_flatten(closed)
            return {
                "closed": closed,
                "positions_closed": len(closed),
                "dry_run": self.dry_run,
            }

        try:
            results = self.broker.close_all_positions()
        except Exception as e:
            logger.error("flatten_all: close_all_positions failed: %s", e, exc_info=True)
            for symbol in pos_by_symbol:
                emit_event(
                    Categories.POSITION_STRANDED,
                    level="error",
                    variant=variant_name,
                    symbol=symbol,
                    message="flatten_all close_all_positions raised",
                    context={"error": str(e), "qty": pos_by_symbol[symbol].qty},
                )
            self._notify_flatten([])
            return {"error": str(e), "closed": [], "positions_closed": 0}

        results_by_symbol = {r.symbol.upper(): r for r in results if getattr(r, "symbol", None)}

        for symbol, pos in pos_by_symbol.items():
            result = results_by_symbol.get(symbol.upper())
            if result is None:
                logger.warning(
                    "flatten_all: no close response for %s — Alpaca may have "
                    "rejected this one. Position likely still held.",
                    symbol,
                )
                emit_event(
                    Categories.POSITION_STRANDED,
                    level="error",
                    variant=variant_name,
                    symbol=symbol,
                    message="flatten_all: no close response from Alpaca",
                    context={"qty": pos.qty, "avg_entry_price": pos.avg_entry_price},
                )
                closed.append({
                    "symbol": symbol, "closed": False,
                    "error": "no close response from Alpaca",
                })
                continue

            fill_price = getattr(result, "filled_avg_price", None)
            closed.append({
                "symbol": symbol, "closed": True,
                "status": getattr(result, "status", None),
            })
            self.db.log_trade(
                symbol=symbol,
                side="sell",
                qty=int(float(pos.qty)),
                order_type="market",
                status=getattr(result, "status", "submitted"),
                filled_qty=getattr(result, "filled_qty", None),
                filled_price=fill_price,
                order_id=getattr(result, "order_id", None),
                reasoning="intraday EOD flatten",
            )
            pos_state_before = state_by_symbol.get(symbol)
            if (
                self.config.get("trade_outcome_live_hook_intraday_enabled", True)
                and pos_state_before
            ):
                try:
                    from tradingagents.automation.trade_outcome import log_closed_trade
                    exit_px = float(fill_price) if fill_price else float(pos.current_price)
                    log_closed_trade(
                        db=self.db,
                        symbol=symbol,
                        pos_state=pos_state_before,
                        exit_price=exit_px,
                        exit_reason="eod_flatten",
                        broker=self.broker,
                        excursion_enabled=self.config.get(
                            "trade_outcome_excursion_enabled", False
                        ),
                    )
                except Exception as e:
                    logger.warning(
                        "intraday flatten trade_outcome hook failed %s: %s",
                        symbol, e,
                    )
            self.db.delete_position_state(symbol)

        self._entered_today.clear()
        self._notify_flatten(closed)
        return {
            "closed": closed,
            "positions_closed": sum(1 for c in closed if c.get("closed")),
            "dry_run": self.dry_run,
        }

    # ------------------------------------------------------------------ aliases / stubs

    def run_daily_analysis(self) -> Dict:
        """Alias for scan — lets ABRunner call a uniform method."""
        return self.scan()

    def run_intraday_entry_scan(self) -> Dict:
        """Alias for scan — matches Minervini Orchestrator's intraday hook name."""
        return self.scan()

    def run_daily_reflection(self) -> Dict:
        """Intraday mechanical has no LLM reflection."""
        return {}

    def run_daily_trade_review(self) -> Dict:
        """Per-trade post-mortems for intraday trades closed today.

        Shares the same implementation as `Orchestrator.run_daily_trade_review`.
        Kill-switched via `daily_trade_review_enabled` config flag.
        """
        from tradingagents.automation.trade_review import run_daily_review

        return run_daily_review(
            db=self.db,
            broker=self.broker,
            variant_name=self.config.get("variant_name", "intraday_mechanical"),
            config=self.config,
        )

    def take_market_snapshot(self) -> Dict:
        """Intraday mechanical needs no pre-market snapshot."""
        return {}

    def reconcile_orders(self) -> Dict:
        """Sync local `trades` rows against broker state (Track P-SYNC)."""
        from .reconciler import OrderReconciler
        reconciler = OrderReconciler(
            broker=self.broker,
            db=self.db,
            variant=self.config.get("variant_name"),
            notifier=self.notifier,
            config=self.config,
        )
        return reconciler.reconcile_once()

    # ------------------------------------------------------------------ status

    def get_status(self) -> Dict:
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        clock = self.broker.get_clock()
        pos_list = [{
            "symbol": p.symbol,
            "qty": float(p.qty),
            "entry": float(p.avg_entry_price),
            "current": float(p.current_price),
            "pl": float(p.unrealized_pl),
            "pl_pct": f"{float(p.unrealized_plpc):.2%}" if p.unrealized_plpc else "0.00%",
        } for p in positions]
        return {
            "account": {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
            },
            "market": {
                "is_open": clock.is_open if clock else False,
            },
            "positions": pos_list,
            "universe_size": len(self.universe),
            "paper_mode": self.config.get("paper_trading", True),
            "strategy": "intraday_mechanical",
            "dry_run": self.dry_run,
        }

    def generate_daily_report(self, save: bool = True) -> Dict:
        status = self.get_status()
        trades = self.db.get_recent_trades(limit=50)
        today_str = date.today().isoformat()
        today_trades = [
            t for t in trades if t.get("timestamp", "").startswith(today_str)
        ]
        report = {
            "date": today_str,
            "strategy": "intraday_mechanical",
            "paper_mode": self.config.get("paper_trading", True),
            "dry_run": self.dry_run,
            "account": status["account"],
            "trade_summary": {
                "total_orders": len(today_trades),
                "filled_orders": sum(1 for t in today_trades if t.get("status") == "filled"),
                "buy_orders": sum(1 for t in today_trades if t.get("side") == "buy"),
                "sell_orders": sum(1 for t in today_trades if t.get("side") == "sell"),
                "symbols": list({t["symbol"] for t in today_trades}),
            },
            "position_summary": status["positions"],
        }
        if save:
            report_dir = Path("results/intraday_reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / f"report_{today_str}.json").write_text(
                json.dumps(report, indent=2, default=str)
            )
        return report

    # ------------------------------------------------------------------ notify

    def _notify_summary(self, entries: List[Dict[str, Any]]):
        if not entries:
            return
        traded = [e for e in entries if e.get("traded")]
        dry = [e for e in entries if e.get("dry_run")]
        if not traded and not dry:
            return
        lines = ["Intraday Scan:"]
        for e in traded:
            lines.append(
                f"  BUY {e['symbol']} x{e.get('qty', '?')} "
                f"({e.get('setup_family', '?')})"
            )
        for e in dry:
            lines.append(
                f"  [DRY] BUY {e['symbol']} x{e.get('qty', '?')} "
                f"({e.get('setup_family', '?')})"
            )
        try:
            self.notifier.send("Intraday Mechanical", "\n".join(lines))
        except Exception:
            pass

    def _notify_flatten(self, closed: List[Dict[str, Any]]):
        if not closed:
            return
        real = [c for c in closed if c.get("closed")]
        dry = [c for c in closed if c.get("dry_run")]
        if not real and not dry:
            return
        lines = ["Intraday EOD Flatten:"]
        for c in real:
            lines.append(f"  CLOSE {c['symbol']} → {c.get('status')}")
        for c in dry:
            lines.append(f"  [DRY] CLOSE {c['symbol']}")
        try:
            self.notifier.send("Intraday Mechanical", "\n".join(lines))
        except Exception:
            pass
