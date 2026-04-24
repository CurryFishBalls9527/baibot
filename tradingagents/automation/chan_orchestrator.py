"""Chan theory orchestrator for automated paper/live trading.

Uses RS-filtered Chan 缠论 structural signals for entries and exits.
Designed to run alongside the existing Minervini orchestrator on a
separate Alpaca account.

Signal flow:
  1. Compute RS ranking from daily data → top N% qualify
  2. Run Chan analysis on 30m bars for qualifying symbols
  3. New buy BSPs → entry signals (with MACD + divergence filters)
  4. Existing positions: sell BSPs or structural stop → exit signals
  5. Position sizer + risk engine → Alpaca bracket orders
"""

import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import duckdb
import numpy as np
import pandas as pd

CHAN_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "chan.py"
if str(CHAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CHAN_ROOT))

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, BI_DIR, KL_TYPE

from tradingagents.broker.alpaca_broker import AlpacaBroker
from tradingagents.broker.models import Account, OrderRequest, Position
from tradingagents.portfolio.portfolio_tracker import PortfolioTracker
from tradingagents.portfolio.position_sizer import PositionSizer
from tradingagents.risk.risk_engine import RiskEngine
from tradingagents.storage.database import TradingDatabase
from tradingagents.automation.notifier import build_notifier
from tradingagents.research.chan_adapter import DuckDBIntradayAPI
from tradingagents.research import MarketDataWarehouse, build_market_context

logger = logging.getLogger(__name__)

_ALPACA_UNSUPPORTED_SYMBOL_PREFIXES = ("^",)


DEFAULT_CHAN_CONFIG = {
    "trigger_step": True,
    "bi_strict": True,
    "skip_step": 0,
    "divergence_rate": 0.8,
    "bsp2_follow_1": False,
    "bsp3_follow_1": False,
    "min_zs_cnt": 1,
    "bs1_peak": False,
    "macd_algo": "area",
    "bs_type": "1,1p,2,2s",
    "max_bs2_rate": 0.618,
    "print_warning": False,
    "zs_algo": "normal",
}


class ChanOrchestrator:
    """Chan 缠论 + RS ranking automated trading orchestrator."""

    def __init__(self, config: dict):
        self.config = config

        self.broker = AlpacaBroker(
            api_key=config["alpaca_api_key"],
            secret_key=config["alpaca_secret_key"],
            paper=config.get("paper_trading", True),
        )

        self.db = TradingDatabase(
            config.get("db_path", "trading_chan.db"),
            variant=config.get("variant_name"),
        )
        config.setdefault("strategy_tag", "chan")
        self.notifier = build_notifier(config)

        starting_equity = self.db.get_starting_equity()
        risk_config = {**config, "starting_equity": starting_equity}
        self.risk_engine = RiskEngine(risk_config)
        self.sizer = PositionSizer(config)
        # Parity with the base Orchestrator: without this, the scheduler's
        # initial snapshot hook and `take_market_snapshot` both no-op'd,
        # leaving `daily_snapshots` empty for chan/chan_v2 variants — the
        # dashboard then shows no equity curve for those variants.
        self.tracker = PortfolioTracker(self.broker, self.db)

        self.intraday_db = config.get("chan_intraday_db", "research_data/intraday_30m_broad.duckdb")
        self.daily_db = config.get("chan_daily_db", "research_data/market_data.duckdb")

        self.rs_top_pct = config.get("chan_rs_top_pct", 0.15)
        self.rs_lookback_days = config.get("chan_rs_lookback_days", 126)
        self.buy_types = set(config.get("chan_buy_types", ["T1", "T2", "T2S"]))
        self.filter_macd_zero = config.get("chan_filter_macd_zero", True)
        self.filter_divergence_max = config.get("chan_filter_divergence_max", 0.6)
        self.max_positions = config.get("max_positions", 8)
        self.default_stop_loss_pct = config.get("default_stop_loss_pct", 0.05)
        self.default_take_profit_pct = config.get("default_take_profit_pct", 0.15)

        self._seen_buy_keys: Set[str] = set()
        self._seen_sell_keys: Dict[str, Set[str]] = {}

        self.dead_money_bars = config.get("chan_dead_money_bars", 0)
        self.dead_money_min_gain = config.get("chan_dead_money_min_gain", 0.05)
        self.ma_trend_filter = config.get("chan_ma_trend_filter", False)

        # C3 regime gate. Off by default to preserve existing chan variant
        # behavior. Set chan_regime_gate_enabled: true + chan_regime_min_score
        # to activate.
        self.regime_gate_enabled = config.get("chan_regime_gate_enabled", False)
        self.regime_min_score = config.get("chan_regime_min_score", 4)
        self._cached_regime: Optional[Dict] = None
        self._regime_cache_date: Optional[str] = None


    def run_daily_analysis(self) -> Dict:
        """Main entry point — called by scheduler every 30m during market hours.

        Scans for signals on completed bars and executes immediately.
        """
        logger.info("=== Chan Orchestrator: Intraday Scan ===")
        t0 = time.perf_counter()

        try:
            account = self.broker.get_account()
            positions = self.broker.get_positions()
            logger.info(
                "Account: equity=$%.2f, cash=$%.2f, %d positions",
                account.equity, account.cash, len(positions),
            )
        except Exception as e:
            logger.error("Failed to get account info: %s", e)
            return {"error": str(e)}

        rs_qualifying = self._compute_rs_ranking()
        logger.info("RS filter: %d symbols qualify (top %.0f%%)",
                     len(rs_qualifying), self.rs_top_pct * 100)

        held_symbols = {p.symbol for p in positions}
        symbols_to_update = rs_qualifying | held_symbols
        self._refresh_intraday_data(symbols_to_update)

        results = {"entries": [], "exits": [], "holds": []}

        exit_results = self._check_exits(positions, account)
        results["exits"] = exit_results

        if exit_results:
            positions = self.broker.get_positions()
            account = self.broker.get_account()

        entry_results = self._check_entries(rs_qualifying, positions, account)
        results["entries"] = entry_results

        elapsed = time.perf_counter() - t0
        logger.info("Chan scan complete in %.1fs: %d entries, %d exits",
                     elapsed, len(entry_results), len(exit_results))

        self._notify_summary(results)
        return results

    def _check_exits(self, positions: List[Position], account: Account) -> List[Dict]:
        """Check existing positions for Chan sell signals and execute immediately."""
        exit_results = []
        if not positions:
            return exit_results

        DuckDBIntradayAPI.DB_PATH = self.intraday_db
        chan_cfg = CChanConfig(DEFAULT_CHAN_CONFIG.copy())

        for pos in positions:
            symbol = pos.symbol
            try:
                sell_signal = self._analyze_chan_exit(symbol, chan_cfg)
                if sell_signal:
                    result = self._execute_signal(
                        symbol=symbol, action="SELL",
                        confidence=sell_signal["confidence"],
                        reasoning=sell_signal["reason"],
                        account=account, positions=positions,
                    )
                    exit_results.append(result)
                elif self.dead_money_bars > 0:
                    dm_signal = self._check_dead_money(pos)
                    if dm_signal:
                        result = self._execute_signal(
                            symbol=symbol, action="SELL",
                            confidence=0.65,
                            reasoning=dm_signal,
                            account=account, positions=positions,
                        )
                        exit_results.append(result)
                    else:
                        logger.info("%s: HOLD — no sell signal", symbol)
                else:
                    logger.info("%s: HOLD — no sell signal", symbol)
            except Exception as e:
                logger.error("Exit check failed for %s: %s", symbol, e)

        return exit_results

    def _load_market_regime(self) -> Optional[Dict]:
        """Return {'market_score', 'market_regime', 'confirmed_uptrend'} or None.

        Cached once per day. Mirrors the approach used by Orchestrator for the
        mechanical overlay regime source. Safe to call repeatedly.
        """
        today = date.today().isoformat()
        if self._cached_regime is not None and self._regime_cache_date == today:
            return self._cached_regime

        try:
            lookback = max(int(self.config.get("minervini_lookback_days", 730)), 400)
            start = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")
            end = today
            symbols = ["SPY", "QQQ", "IWM", "SMH", "^VIX"]
            warehouse = MarketDataWarehouse(
                self.config.get("minervini_db_path", self.daily_db)
            )
            try:
                frames = {s: warehouse.get_daily_bars(s, start, end) for s in symbols}
            finally:
                warehouse.close()
            df = build_market_context(frames)
            if df is None or df.empty:
                return None
            latest = df.iloc[-1]
            score = latest.get("market_score")
            self._cached_regime = {
                "market_score": int(score) if score is not None else None,
                "market_regime": str(latest["market_regime"]),
                "confirmed_uptrend": bool(latest["market_confirmed_uptrend"]),
            }
            self._regime_cache_date = today
            return self._cached_regime
        except Exception as e:
            logger.warning("Chan regime gate: failed to load regime: %s", e)
            return None

    def _check_entries(
        self, rs_qualifying: Set[str], positions: List[Position], account: Account,
    ) -> List[Dict]:
        """Scan RS-qualifying symbols for Chan buy signals and execute immediately."""
        entry_results = []
        held_symbols = {p.symbol for p in positions}

        if len(positions) >= self.max_positions:
            logger.info("At max positions (%d), skipping entry scan", self.max_positions)
            return entry_results

        # Fail-closed freshness: sync broker → DB so the same-day-stop guard below
        # reads the latest bracket leg fills. Entry scans run every 10 min but the
        # reconciler cron fires every 5 min; a stop that fills between ticks would
        # otherwise be invisible here.
        try:
            self.reconcile_orders()
        except Exception as e:
            logger.warning(
                "Chan entry scan aborted: pre-scan reconcile failed (%s). "
                "Skipping this tick rather than entering against a stale DB.",
                e,
            )
            return entry_results

        # C3: regime gate — suppress new entries in weak markets when enabled.
        if self.regime_gate_enabled:
            regime = self._load_market_regime()
            if regime is not None and regime.get("market_score") is not None:
                if regime["market_score"] <= self.regime_min_score:
                    logger.info(
                        "Chan regime gate: score=%s regime=%s <= min=%s — "
                        "suppressing new entries",
                        regime["market_score"],
                        regime["market_regime"],
                        self.regime_min_score,
                    )
                    return entry_results

        DuckDBIntradayAPI.DB_PATH = self.intraday_db
        cfg_dict = DEFAULT_CHAN_CONFIG.copy()
        if self.ma_trend_filter:
            cfg_dict["mean_metrics"] = [20, 60]
        chan_cfg = CChanConfig(cfg_dict)
        candidates = sorted(rs_qualifying - held_symbols)
        blocked_whipsaw = [s for s in candidates if self.db.was_stopped_today(s)]
        if blocked_whipsaw:
            logger.info(
                "Chan whipsaw guard: skipping %d symbols stopped today: %s",
                len(blocked_whipsaw), ",".join(blocked_whipsaw),
            )
            candidates = [s for s in candidates if s not in set(blocked_whipsaw)]
        logger.info("Scanning %d candidates for buy signals...", len(candidates))

        for symbol in candidates:
            if len(positions) + len(entry_results) >= self.max_positions:
                break
            try:
                buy_signal = self._analyze_chan_entry(symbol, chan_cfg)
                if buy_signal:
                    # Load regime for journaling even when the gate is
                    # disabled — the daily review wants regime_at_entry
                    # populated regardless of whether we're using it as a
                    # filter. Cheap (cached per-day in _load_market_regime).
                    regime_info = self._load_market_regime()
                    result = self._execute_signal(
                        symbol=symbol, action="BUY",
                        confidence=buy_signal["confidence"],
                        reasoning=buy_signal["reason"],
                        stop_loss_pct=buy_signal.get("stop_loss_pct", self.default_stop_loss_pct),
                        take_profit_pct=buy_signal.get("take_profit_pct", self.default_take_profit_pct),
                        account=account, positions=positions,
                        entry_context={
                            "base_pattern": buy_signal.get("types"),
                            "regime_at_entry": (
                                regime_info.get("market_regime")
                                if regime_info else None
                            ),
                        },
                        # Extra structured context stashed on the SIGNAL row
                        # via signal_metadata — feeds the daily review.
                        signal_metadata={
                            "t_types": buy_signal.get("types"),
                            "confidence": buy_signal.get("confidence"),
                            "bi_low": buy_signal.get("bi_low"),
                            "bsp_reason": buy_signal.get("reason"),
                            "regime_at_entry": (
                                regime_info.get("market_regime")
                                if regime_info else None
                            ),
                            "market_score": (
                                regime_info.get("market_score")
                                if regime_info else None
                            ),
                        },
                    )
                    entry_results.append(result)
            except Exception as e:
                logger.warning("Entry scan failed for %s: %s", symbol, e)

        return entry_results

    def _refresh_intraday_data(self, symbols: Set[str]):
        """Update 30m bars for the given symbols from Alpaca."""
        api_key = self.config.get("alpaca_api_key", "")
        secret_key = self.config.get("alpaca_secret_key", "")
        if not api_key or not secret_key:
            logger.warning("No Alpaca keys for data refresh — using existing 30m data")
            return

        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        except ImportError:
            logger.warning("alpaca-py not installed — skipping data refresh")
            return

        client = StockHistoricalDataClient(api_key, secret_key)
        conn = duckdb.connect(self.intraday_db)

        end = datetime.now()
        start = end - timedelta(days=7)

        updated = 0
        try:
            for sym in sorted(symbols):
                try:
                    request = StockBarsRequest(
                        symbol_or_symbols=sym,
                        timeframe=TimeFrame(30, TimeFrameUnit.Minute),
                        start=start,
                        end=end,
                    )
                    bars = client.get_stock_bars(request)
                    df = bars.df
                    if df is None or df.empty:
                        continue
                    if isinstance(df.index, pd.MultiIndex):
                        df = df.reset_index()
                    else:
                        df = df.reset_index()
                        df.insert(0, "symbol", sym)

                    now = datetime.utcnow()
                    for _, row in df.iterrows():
                        ts = row.get("timestamp", row.get("time"))
                        conn.execute(
                            "INSERT OR REPLACE INTO bars_30m "
                            "(symbol, ts, open, high, low, close, volume, trade_count, vwap, source, updated_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            [sym, pd.Timestamp(ts).to_pydatetime(),
                             float(row.get("open", 0) or 0), float(row.get("high", 0) or 0),
                             float(row.get("low", 0) or 0), float(row.get("close", 0) or 0),
                             float(row.get("volume", 0) or 0), int(row.get("trade_count", 0) or 0),
                             float(row.get("vwap", 0) or 0), "alpaca", now],
                        )
                    updated += 1
                except Exception as e:
                    logger.debug("30m refresh failed for %s: %s", sym, e)
        finally:
            conn.close()
        logger.info("Refreshed 30m data for %d/%d symbols", updated, len(symbols))

    def bulk_refresh_30m_data(self) -> Dict:
        """Download recent 30m bars for all symbols with fresh daily data.

        Designed to run once each morning before the first intraday scan so
        that the RS ranking and Chan analysis have a broad universe to work
        with.  Uses Alpaca's multi-symbol batch API for speed.
        """
        api_key = self.config.get("alpaca_api_key", "")
        secret_key = self.config.get("alpaca_secret_key", "")
        if not api_key or not secret_key:
            logger.warning("No Alpaca keys — skipping bulk 30m refresh")
            return {"status": "no_keys"}

        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        except ImportError:
            logger.warning("alpaca-py not installed — skipping bulk 30m refresh")
            return {"status": "no_alpaca"}

        # Gather symbols with fresh daily data
        if not Path(self.daily_db).exists():
            logger.warning("Daily DB not found for bulk refresh: %s", self.daily_db)
            return {"status": "no_daily_db"}

        # NOTE: connect RW (not read_only) to match MarketDataWarehouse,
        # which other variants (mechanical/llm/...) open on this same file
        # in the same process. DuckDB rejects mixed RO/RW configs → chan_v2
        # scans fail with "different configuration than existing connections".
        daily_conn = duckdb.connect(self.daily_db)
        try:
            freshness_cutoff = (date.today() - timedelta(days=10)).isoformat()
            fresh_symbols = [r[0] for r in daily_conn.execute(
                "SELECT symbol FROM ("
                "  SELECT symbol, MAX(trade_date) AS latest"
                "  FROM daily_bars GROUP BY symbol"
                ") WHERE latest >= ? AND symbol != 'SPY'",
                [freshness_cutoff],
            ).fetchall()]
        finally:
            daily_conn.close()

        fresh_symbols = [
            sym
            for sym in fresh_symbols
            if sym and not sym.startswith(_ALPACA_UNSUPPORTED_SYMBOL_PREFIXES)
        ]

        if not fresh_symbols:
            logger.warning("No fresh daily symbols found for bulk 30m refresh")
            return {"status": "no_symbols"}

        logger.info("Bulk 30m refresh: %d symbols with fresh daily data", len(fresh_symbols))

        client = StockHistoricalDataClient(api_key, secret_key)
        conn = duckdb.connect(self.intraday_db)

        try:
            # Ensure table exists
            conn.execute(
                "CREATE TABLE IF NOT EXISTS bars_30m ("
                "  symbol VARCHAR, ts TIMESTAMP, open DOUBLE, high DOUBLE,"
                "  low DOUBLE, close DOUBLE, volume DOUBLE, trade_count BIGINT,"
                "  vwap DOUBLE, source VARCHAR, updated_at TIMESTAMP,"
                "  PRIMARY KEY (symbol, ts))"
            )

            end = datetime.now()
            # Incremental: only fetch from the latest bar we already have
            latest_row = conn.execute(
                "SELECT MAX(ts) FROM bars_30m"
            ).fetchone()
            latest_ts = latest_row[0] if latest_row and latest_row[0] else None
            if latest_ts:
                # Small overlap to catch any bars we might have missed
                start = latest_ts - timedelta(hours=1)
                logger.info("Incremental 30m refresh from %s", start)
            else:
                start = end - timedelta(days=90)
                logger.info("Full 30m backfill (no existing data)")
            batch_size = 100
            total_updated = 0
            total_bars = 0
            errors = 0

            for i in range(0, len(fresh_symbols), batch_size):
                batch = fresh_symbols[i : i + batch_size]
                try:
                    request = StockBarsRequest(
                        symbol_or_symbols=batch,
                        timeframe=TimeFrame(30, TimeFrameUnit.Minute),
                        start=start,
                        end=end,
                    )
                    bars = client.get_stock_bars(request)
                    df = bars.df
                    if df is None or df.empty:
                        continue

                    df = df.reset_index()
                    now = datetime.utcnow()
                    rows = []
                    for _, row in df.iterrows():
                        sym = row.get("symbol", "")
                        ts = row.get("timestamp", row.get("time"))
                        rows.append((
                            sym,
                            pd.Timestamp(ts).to_pydatetime(),
                            float(row.get("open", 0) or 0),
                            float(row.get("high", 0) or 0),
                            float(row.get("low", 0) or 0),
                            float(row.get("close", 0) or 0),
                            float(row.get("volume", 0) or 0),
                            int(row.get("trade_count", 0) or 0),
                            float(row.get("vwap", 0) or 0),
                            "alpaca",
                            now,
                        ))

                    if rows:
                        conn.executemany(
                            "INSERT OR REPLACE INTO bars_30m "
                            "(symbol, ts, open, high, low, close, volume, "
                            "trade_count, vwap, source, updated_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            rows,
                        )
                        symbols_in_batch = {r[0] for r in rows}
                        total_updated += len(symbols_in_batch)
                        total_bars += len(rows)

                    logger.info(
                        "Bulk 30m refresh batch %d-%d: %d bars",
                        i, min(i + batch_size, len(fresh_symbols)), len(rows),
                    )
                except Exception as e:
                    errors += 1
                    logger.warning(
                        "Bulk 30m refresh batch %d-%d failed: %s",
                        i, min(i + batch_size, len(fresh_symbols)), e,
                    )
        finally:
            conn.close()
        logger.info(
            "Bulk 30m refresh complete: %d symbols, %d bars, %d batch errors",
            total_updated, total_bars, errors,
        )
        return {
            "status": "ok",
            "symbols_updated": total_updated,
            "bars_inserted": total_bars,
            "errors": errors,
        }

    def _compute_rs_ranking(self) -> Set[str]:
        """Compute RS ranking from daily data, return qualifying symbols."""
        if not Path(self.daily_db).exists():
            logger.warning("Daily DB not found: %s", self.daily_db)
            return set()

        # RW to match MarketDataWarehouse — see note at bulk_refresh_30m_data.
        conn = duckdb.connect(self.daily_db)
        today = date.today()
        lookback_start = today - timedelta(days=int(self.rs_lookback_days * 1.6) + 30)

        try:
            spy_df = conn.execute(
                "SELECT trade_date, close FROM daily_bars "
                "WHERE symbol = 'SPY' AND trade_date >= ? ORDER BY trade_date",
                [lookback_start.isoformat()],
            ).fetchdf()
            if spy_df.empty or len(spy_df) < self.rs_lookback_days:
                logger.warning("Insufficient SPY daily data for RS ranking")
                return set()

            spy_df["trade_date"] = pd.to_datetime(spy_df["trade_date"])
            spy_df = spy_df.set_index("trade_date").sort_index()
            spy_latest = spy_df["close"].iloc[-1]
            spy_past = spy_df["close"].iloc[-self.rs_lookback_days] if len(spy_df) >= self.rs_lookback_days else spy_df["close"].iloc[0]
            spy_ret = (spy_latest - spy_past) / spy_past

            # Only consider symbols with recent daily data (within 5
            # trading days of SPY's latest bar) to avoid stale prices
            # polluting the RS percentile cutoff.
            spy_latest_date = spy_df.index[-1].strftime("%Y-%m-%d")
            freshness_cutoff = (
                spy_df.index[-1] - timedelta(days=10)
            ).strftime("%Y-%m-%d")

            all_symbols = [r[0] for r in conn.execute(
                "SELECT symbol FROM ("
                "  SELECT symbol, MAX(trade_date) AS latest"
                "  FROM daily_bars WHERE trade_date >= ?"
                "  GROUP BY symbol"
                ") WHERE latest >= ?",
                [lookback_start.isoformat(), freshness_cutoff],
            ).fetchall()]
            logger.info(
                "RS ranking: %d symbols with fresh daily data (cutoff %s)",
                len(all_symbols), freshness_cutoff,
            )

            rs_scores: list[tuple[str, float]] = []
            for sym in all_symbols:
                if sym == "SPY":
                    continue
                try:
                    rows = conn.execute(
                        "SELECT close FROM daily_bars WHERE symbol = ? "
                        "AND trade_date >= ? ORDER BY trade_date",
                        [sym, lookback_start.isoformat()],
                    ).fetchdf()
                    if len(rows) < self.rs_lookback_days:
                        continue
                    sym_latest = float(rows["close"].iloc[-1])
                    sym_past = float(rows["close"].iloc[-self.rs_lookback_days])
                    sym_ret = (sym_latest - sym_past) / sym_past
                    rs = sym_ret / spy_ret if spy_ret > 0 else sym_ret - spy_ret
                    rs_scores.append((sym, rs))
                except Exception:
                    continue

            conn.close()

            if not rs_scores:
                return set()

            rs_scores.sort(key=lambda x: x[1], reverse=True)
            cutoff = max(1, int(len(rs_scores) * self.rs_top_pct))
            qualifying = {sym for sym, _ in rs_scores[:cutoff]}
            return qualifying

        except Exception as e:
            logger.error("RS ranking failed: %s", e)
            conn.close()
            return set()



    def _analyze_chan_entry(self, symbol: str, chan_cfg: CChanConfig) -> Optional[Dict]:
        """Run Chan analysis and look for recent buy BSPs."""
        end_date = date.today().isoformat()
        begin_date = (date.today() - timedelta(days=90)).isoformat()

        try:
            DuckDBIntradayAPI.DB_PATH = self.intraday_db
            chan = CChan(
                code=symbol,
                begin_time=begin_date,
                end_time=end_date,
                data_src="custom:DuckDBAPI.DuckDB30mAPI",
                lv_list=[KL_TYPE.K_30M],
                config=chan_cfg,
                autype=AUTYPE.QFQ,
            )
            for snapshot in chan.step_load():
                pass
        except Exception as e:
            logger.debug("Chan init failed for %s: %s: %s", symbol, type(e).__name__, e)
            return None

        lvl = chan[0]
        bsp_list = list(lvl.bs_point_lst.bsp_store_flat_dict.values())
        if not bsp_list:
            return None

        latest_bsp = bsp_list[-1]
        if latest_bsp.is_buy is False:
            return None

        types_set = {t.name.split("_")[-1] for t in latest_bsp.type}
        if not types_set & self.buy_types:
            return None

        bsp_key = f"{symbol}_{latest_bsp.klu.time}"
        if bsp_key in self._seen_buy_keys:
            return None
        self._seen_buy_keys.add(bsp_key)

        bars = list(lvl.klu_iter())
        if len(bars) < 3:
            return None
        recent_bars = bars[-5:]
        bsp_bar_time = latest_bsp.klu.time
        is_recent = any(
            abs(b.time.ts - bsp_bar_time.ts) < 86400 * 3
            for b in recent_bars
            if hasattr(b.time, 'ts') and hasattr(bsp_bar_time, 'ts')
        )
        if not is_recent:
            bsp_idx = None
            for i, b in enumerate(bars):
                if b.time == bsp_bar_time:
                    bsp_idx = i
                    break
            if bsp_idx is not None and (len(bars) - 1 - bsp_idx) > 10:
                return None

        if self.filter_macd_zero and ("T1" in types_set or "T1P" in types_set):
            try:
                dif = float(latest_bsp.klu.macd.DIF)
                dea = float(latest_bsp.klu.macd.DEA)
                if not (dif < 0 and dea < 0):
                    return None
            except (AttributeError, TypeError):
                pass

        if self.filter_divergence_max > 0 and ("T1" in types_set or "T1P" in types_set):
            try:
                div_rate = latest_bsp.features["divergence_rate"]
                if div_rate > self.filter_divergence_max:
                    return None
            except (KeyError, TypeError):
                pass

        if self.ma_trend_filter:
            try:
                from Common.CEnum import TREND_TYPE
                last_klu = bars[-1]
                trend = getattr(last_klu, "trend", {})
                mean_dict = trend.get(TREND_TYPE.MEAN, {})
                ma20 = mean_dict.get(20)
                ma60 = mean_dict.get(60)
                if ma20 is not None and ma60 is not None:
                    last_close_val = float(last_klu.close)
                    if not (last_close_val > ma20 > ma60):
                        return None
            except Exception:
                pass

        bi_low = float(latest_bsp.bi.get_begin_val())
        last_close = float(bars[-1].close)
        stop_dist = (last_close - bi_low) / last_close if last_close > bi_low else 0.05
        stop_loss_pct = max(0.03, min(0.08, stop_dist))

        confidence = 0.7
        if "T1" in types_set:
            confidence = 0.8
        if len(types_set) > 1:
            confidence += 0.05

        return {
            "reason": f"Chan buy signal: {'+'.join(sorted(types_set))} at {bsp_bar_time}",
            "confidence": min(confidence, 0.95),
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": self.default_take_profit_pct,
            "bi_low": bi_low,
            # Structured T-type (T1 / T2 / T2S, or combos) so the outcome
            # hook can persist it as base_pattern for per-setup review cuts.
            "types": "+".join(sorted(types_set)) if types_set else None,
        }

    def _analyze_chan_exit(self, symbol: str, chan_cfg: CChanConfig) -> Optional[Dict]:
        """Run Chan analysis and check for sell BSPs on held position."""
        end_date = date.today().isoformat()
        begin_date = (date.today() - timedelta(days=90)).isoformat()

        try:
            DuckDBIntradayAPI.DB_PATH = self.intraday_db
            chan = CChan(
                code=symbol,
                begin_time=begin_date,
                end_time=end_date,
                data_src="custom:DuckDBAPI.DuckDB30mAPI",
                lv_list=[KL_TYPE.K_30M],
                config=chan_cfg,
                autype=AUTYPE.QFQ,
            )
            for snapshot in chan.step_load():
                pass
        except Exception as e:
            logger.debug("Chan init failed for %s: %s: %s", symbol, type(e).__name__, e)
            return None

        lvl = chan[0]
        bsp_list = list(lvl.bs_point_lst.bsp_store_flat_dict.values())

        for bsp in reversed(bsp_list):
            if bsp.is_buy:
                break
            sell_key = f"{symbol}_{bsp.klu.time}"
            if sell_key in self._seen_sell_keys.get(symbol, set()):
                continue

            types_set = {t.name.split("_")[-1] for t in bsp.type}
            bars = list(lvl.klu_iter())
            if not bars:
                continue

            bsp_idx = None
            for i, b in enumerate(bars):
                if b.time == bsp.klu.time:
                    bsp_idx = i
                    break
            if bsp_idx is not None and (len(bars) - 1 - bsp_idx) > 10:
                continue

            self._seen_sell_keys.setdefault(symbol, set()).add(sell_key)
            return {
                "reason": f"Chan sell signal: {'+'.join(sorted(types_set))} at {bsp.klu.time}",
                "confidence": 0.75,
            }

        zs_list = list(lvl.zs_list)
        if zs_list:
            bars = list(lvl.klu_iter())
            if bars:
                last_close = float(bars[-1].close)
                nearest_zs = None
                for zs in reversed(zs_list):
                    if float(zs.low) < last_close:
                        nearest_zs = zs
                        break
                if nearest_zs and last_close < float(nearest_zs.low):
                    return {
                        "reason": f"ZS structural break: price {last_close:.2f} < ZS low {float(nearest_zs.low):.2f}",
                        "confidence": 0.8,
                    }

        return None

    def _execute_signal(
        self,
        symbol: str,
        action: str,
        confidence: float,
        reasoning: str,
        account: Account,
        positions: List[Position],
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.15,
        entry_context: Optional[Dict] = None,
        signal_metadata: Optional[Dict] = None,
    ) -> Dict:
        """Size, risk-check, and execute a signal via Alpaca."""
        signal = {
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "timeframe": "swing",
            "source": "chan_structural",
        }

        # Serialize signal_metadata (BSP time, T-types, BI low, MACD state,
        # regime) for the daily review. Wrapped + defensive — never blocks
        # entry on a JSON edge case (NaN, Timestamp, etc).
        metadata_json = None
        if signal_metadata:
            try:
                import json as _json, math
                def _clean(v):
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        return None
                    return v
                metadata_json = _json.dumps(
                    {k: _clean(v) for k, v in signal_metadata.items()},
                    default=str,
                )
            except Exception as e:
                logger.warning("chan signal_metadata serialize failed: %s", e)
                metadata_json = None

        signal_id = self.db.log_signal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            stop_loss=stop_loss_pct,
            take_profit=take_profit_pct,
            timeframe="swing",
            signal_metadata=metadata_json,
        )

        current_price = self.broker.get_latest_price(symbol)
        current_position = next((p for p in positions if p.symbol == symbol), None)
        total_pos_value = sum(p.market_value for p in positions)

        if action == "BUY":
            signal["stop_loss"] = current_price * (1 - stop_loss_pct)

        order_request = self.sizer.calculate(
            signal=signal,
            account=account,
            current_price=current_price,
            current_position=current_position,
            total_position_value=total_pos_value,
        )

        if order_request is None:
            logger.info("%s: No trade needed (sizer returned None)", symbol)
            return {"symbol": symbol, "action": action, "traded": False}

        # BUY-only idempotency guard. For SELL exits, the branch at line 945
        # cancels bracket legs (which this guard would otherwise match) before
        # calling close_position — that path is idempotent on its own. Having
        # the guard fire on SELLs caused every legitimate dead-money exit to
        # be silently denied by a bracket TP leg. See 2026-04-24 chan_v2
        # stall on COHR/INSW/OUT/PBR.
        if order_request.side == "buy":
            existing_open_order = self._find_existing_open_order(symbol, order_request.side)
            if existing_open_order is not None:
                reason = (
                    f"Existing open {order_request.side} order "
                    f"{existing_open_order.order_id} [{existing_open_order.status}]"
                )
                logger.info("%s: %s", symbol, reason)
                self.db.mark_signal_rejected(signal_id, reason)
                return {"symbol": symbol, "action": action, "traded": False,
                        "screen_rejected": reason}

        risk_result = self.risk_engine.check_order(
            order_request, account, positions, current_price
        )
        if not risk_result.passed:
            logger.warning("%s: Risk check FAILED: %s", symbol, risk_result.reason)
            self.db.mark_signal_rejected(signal_id, risk_result.reason)
            return {"symbol": symbol, "action": action, "traded": False,
                    "risk_rejected": risk_result.reason}

        logger.info("%s: Submitting %s %d shares @ ~$%.2f",
                     symbol, order_request.side, order_request.qty, current_price)

        if order_request.side == "buy":
            sl_price = round(current_price * (1 - stop_loss_pct), 2)
            tp_price = round(current_price * (1 + take_profit_pct), 2)
            # Chan holds positions overnight — bracket children must persist
            # past today's close (parent + OCO inherit the same TIF).
            order_request.time_in_force = "gtc"
            order_result = self.broker.submit_bracket_order(
                order_request,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                anchor_price=current_price,
            )
            sl_price = order_result.effective_stop_price or sl_price
            tp_price = order_result.effective_take_profit_price or tp_price
            logger.info("%s: Bracket order SL=$%.2f TP=$%.2f", symbol, sl_price, tp_price)
        else:
            # Cancel any live bracket legs pinning the shares, then liquidate.
            # Plain submit_order on a bracketed position fails with Alpaca 40310000
            # because the SL/TP OCO legs leave held_for_orders == position qty.
            # get_live_orders is required: OCO legs sit in HELD (not OPEN) until
            # their trigger fires, so get_open_orders misses them.
            try:
                open_orders = self.broker.get_live_orders(symbol=symbol)
            except TypeError:
                open_orders = self.broker.get_live_orders()
            for o in open_orders or []:
                if str(getattr(o, "symbol", "")).upper() != symbol.upper():
                    continue
                try:
                    self.broker.cancel_order(o.order_id)
                except Exception as exc:
                    logger.debug("cancel_order(%s) failed: %s", o.order_id, exc)
            try:
                order_result = self.broker.close_position(symbol)
            except Exception as exc:
                logger.error("close_position(%s) failed: %s", symbol, exc)
                self.db.mark_signal_rejected(signal_id, f"close_position failed: {exc}")
                return {"symbol": symbol, "action": action, "traded": False,
                        "error": str(exc)}

        self.risk_engine.record_trade()

        self.db.log_trade(
            symbol=symbol,
            side=order_request.side,
            qty=order_request.qty,
            order_type="bracket" if order_request.side == "buy" else order_request.order_type,
            status=order_result.status,
            filled_qty=order_result.filled_qty,
            filled_price=order_result.filled_avg_price,
            order_id=order_result.order_id,
            signal_id=signal_id,
            reasoning=reasoning,
        )
        self.db.mark_signal_executed(signal_id)

        if order_request.side == "buy" and "fill" in str(order_result.status).lower():
            state_payload = {
                "entry_price": order_result.filled_avg_price or current_price,
                "entry_date": date.today().isoformat(),
                "highest_close": order_result.filled_avg_price or current_price,
                "current_stop": sl_price,
                "partial_taken": False,
                "variant": self.config.get("variant_name", "chan"),
            }
            # Entry-time context (T-type + regime) for the daily/weekly reviews.
            # Always persisted even when None — keeps the read path uniform.
            if entry_context:
                if entry_context.get("base_pattern") is not None:
                    state_payload["base_pattern"] = entry_context["base_pattern"]
                if entry_context.get("regime_at_entry") is not None:
                    state_payload["regime_at_entry"] = entry_context["regime_at_entry"]
            self.db.upsert_position_state(symbol, state_payload)

        # On SELL fill, log the closed-trade outcome (daily/weekly review
        # inputs) and clean up position_states. Uses the state captured at
        # entry time — T-type, regime, entry price — which was read BEFORE
        # this function body started (see `pos_state_before` snapshot).
        if order_request.side == "sell" and "fill" in str(order_result.status).lower():
            try:
                if self.config.get("trade_outcome_live_hook_chan_enabled", True):
                    from tradingagents.automation.trade_outcome import log_closed_trade
                    pos_state = self.db.get_position_state(symbol)
                    if pos_state:
                        log_closed_trade(
                            db=self.db,
                            symbol=symbol,
                            pos_state=pos_state,
                            exit_price=float(
                                order_result.filled_avg_price or current_price
                            ),
                            exit_reason=reasoning or action,
                            broker=self.broker,
                            excursion_enabled=self.config.get(
                                "trade_outcome_excursion_enabled", False
                            ),
                        )
                        self.db.delete_position_state(symbol)
            except Exception as e:
                logger.warning(
                    "chan trade_outcome hook failed for %s: %s", symbol, e
                )

        return {
            "symbol": symbol, "action": action, "traded": True,
            "qty": order_request.qty, "status": order_result.status,
        }

    def _check_dead_money(self, pos: Position) -> Optional[str]:
        """Return exit reason if position is stale (held too long with weak gain)."""
        trades = self.db.get_trades_for_symbol(pos.symbol)
        buy_trades = [t for t in trades if t.get("side") == "buy"]
        if not buy_trades:
            return None
        entry_time_str = buy_trades[0].get("timestamp")
        if not entry_time_str:
            return None
        try:
            entry_dt = datetime.fromisoformat(str(entry_time_str))
        except (ValueError, TypeError):
            return None
        now = datetime.now()
        bars_approx = int((now - entry_dt).total_seconds() / 1800)
        if bars_approx < self.dead_money_bars:
            return None
        gain = (pos.current_price - pos.avg_entry_price) / pos.avg_entry_price
        if gain < self.dead_money_min_gain:
            return (
                f"Dead money: held ~{bars_approx} bars "
                f"({gain:.1%} gain < {self.dead_money_min_gain:.0%} threshold)"
            )
        return None

    def _find_existing_open_order(self, symbol: str, side: Optional[str] = None):
        # Use get_live_orders: Alpaca's OPEN filter excludes HELD, and bracket
        # SL/TP legs sit in HELD waiting for their trigger. Missing them makes
        # the guard green-light a duplicate sell while the OCO leg already
        # holds the full qty — Alpaca then rejects with 40310000.
        getter = getattr(self.broker, "get_live_orders", None)
        if not callable(getter):
            getter = getattr(self.broker, "get_open_orders", None)
        if not callable(getter):
            return None
        try:
            orders = getter(symbol=symbol)
        except TypeError:
            orders = getter()
        except Exception as exc:
            logger.warning("Could not fetch live orders for %s: %s", symbol, exc)
            return None

        target_symbol = symbol.upper()
        target_side = side.lower() if side else None
        terminal_statuses = {"filled", "canceled", "cancelled", "expired", "rejected"}
        # AlpacaBroker._to_order_result stores str(enum), e.g. "OrderSide.SELL"
        # and "OrderStatus.HELD". Strip the enum-class prefix so comparisons
        # against plain "sell" / "held" work.
        for order in orders or []:
            order_symbol = str(getattr(order, "symbol", "") or "").upper()
            order_side = str(getattr(order, "side", "") or "").lower().split(".")[-1]
            order_status = str(getattr(order, "status", "") or "").lower().split(".")[-1]
            if order_symbol != target_symbol:
                continue
            if target_side and order_side != target_side:
                continue
            if order_status in terminal_statuses:
                continue
            return order
        return None

    def _notify_summary(self, results: Dict):
        """Send notification summary of today's actions."""
        entries = [r for r in results.get("entries", []) if r.get("traded")]
        exits = [r for r in results.get("exits", []) if r.get("traded")]

        if not entries and not exits:
            return

        lines = ["Chan Strategy Daily Summary:"]
        for r in entries:
            lines.append(f"  BUY {r['symbol']} x{r.get('qty', '?')}")
        for r in exits:
            lines.append(f"  SELL {r['symbol']} x{r.get('qty', '?')}")
        msg = "\n".join(lines)

        try:
            self.notifier.send("Chan Strategy", msg)
        except Exception:
            pass

    def get_status(self) -> Dict:
        """Return current account and position status."""
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        clock = self.broker.get_clock()

        pos_list = []
        for p in positions:
            pos_list.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "entry": float(p.avg_entry_price),
                "current": float(p.current_price),
                "pl": float(p.unrealized_pl),
                "pl_pct": f"{float(p.unrealized_plpc):.2%}" if p.unrealized_plpc else "0.00%",
            })

        trades_today = self.db.get_recent_trades(limit=50)
        today_str = date.today().isoformat()
        today_trades = [t for t in trades_today if t.get("timestamp", "").startswith(today_str)]

        return {
            "account": {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "daily_pl": float(account.daily_pl) if account.daily_pl else 0,
                "daily_pl_pct": f"{float(account.daily_pl_pct):.2%}" if account.daily_pl_pct else "0.00%",
            },
            "market": {
                "is_open": clock.is_open if clock else False,
                "next_open": str(clock.next_open) if clock else "",
                "next_close": str(clock.next_close) if clock else "",
            },
            "positions": pos_list,
            "performance": {},
            "today": {
                "trade_summary": {
                    "total_orders": len(today_trades),
                    "filled_orders": sum(1 for t in today_trades if t.get("status") == "filled"),
                    "symbols": list({t["symbol"] for t in today_trades}),
                },
                "unrealized_pl": sum(float(p.unrealized_pl) for p in positions),
            },
            "watchlist": f"RS top {self.rs_top_pct:.0%} dynamic",
            "paper_mode": self.config.get("paper_trading", True),
            "strategy": "chan",
        }

    def generate_daily_report(self, save: bool = True) -> Dict:
        """Generate a daily performance report."""
        status = self.get_status()
        trades = self.db.get_recent_trades(limit=50)
        today_str = date.today().isoformat()
        today_trades = [t for t in trades if t.get("timestamp", "").startswith(today_str)]

        report = {
            "date": today_str,
            "strategy": "chan",
            "paper_mode": self.config.get("paper_trading", True),
            "account": status["account"],
            "trade_summary": {
                "total_orders": len(today_trades),
                "filled_orders": sum(1 for t in today_trades if t.get("status") == "filled"),
                "buy_orders": sum(1 for t in today_trades if t.get("side") == "buy"),
                "sell_orders": sum(1 for t in today_trades if t.get("side") == "sell"),
                "gross_filled_notional": sum(
                    (t.get("filled_qty") or 0) * (t.get("filled_price") or 0)
                    for t in today_trades
                ),
                "symbols": list({t["symbol"] for t in today_trades}),
            },
            "position_summary": status["positions"],
            "performance": status["performance"],
        }

        if save:
            report_dir = Path("results/chan_reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"report_{today_str}.json"
            import json
            report_path.write_text(json.dumps(report, indent=2, default=str))
            logger.info("Report saved to %s", report_path)

        return report

    def run_daily_reflection(self) -> Dict:
        """Placeholder — Chan strategy doesn't use AI reflection."""
        return {}

    def take_market_snapshot(self) -> Dict:
        """Capture account + portfolio state into `daily_snapshots`.

        Previously a `return {}` placeholder, which meant the dashboard's
        equity curve was empty for chan and chan_v2 variants. Now mirrors
        the base Orchestrator: writes one row per cron tick.
        """
        logger.info("Chan: capturing scheduled market snapshot...")
        return self.tracker.take_daily_snapshot()

    def run_daily_trade_review(self) -> Dict:
        """Per-trade post-mortems for trades that closed today.

        Shares the same implementation as `Orchestrator.run_daily_trade_review`.
        Kill-switched via `daily_trade_review_enabled` config flag.
        """
        from tradingagents.automation.trade_review import run_daily_review

        return run_daily_review(
            db=self.db,
            broker=self.broker,
            variant_name=self.config.get("variant_name", "chan"),
            config=self.config,
        )

    def run_held_position_review(self) -> Dict:
        """Health-check per held Chan position (daily).

        Chan doesn't have a Minervini-style preflight cache, so
        features_fn stays None. The prompt degrades gracefully — no
        SMA-50/RS-now — but the core stats (entry, current, MFE,
        distance-to-stop, hold days, T-type, regime at entry) all work.
        """
        from tradingagents.automation.trade_review import run_held_position_review

        return run_held_position_review(
            db=self.db,
            broker=self.broker,
            variant_name=self.config.get("variant_name", "chan"),
            config=self.config,
        )

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
