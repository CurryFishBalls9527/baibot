"""Chan-daily orchestrator for daily-bar Donchian + segment-level Chan strategy.

Mirrors ChanOrchestrator's interface so the ABRunner / scheduler can drive it
identically to other variants. Key differences vs intraday `ChanOrchestrator`:

- Once-per-day cadence (not 30-min intraday scan)
- Fixed 16-ETF universe (not RS-filtered single stocks)
- Donchian-30 close breakout OR seg-level Chan BSP entries
- ATR-parity sizing with momentum-priority queue
- Exits: Chan T1 sell signal + structural stop (broker-side bracket) + 100-bar time stop

Strategy parameters locked from research (see SUMMARY_OPTIMIZATION_FINAL.md).
Backtest: 9.5% annual / -11% max DD / Calmar 1.18 across 4 OOS periods 2014-2025.
"""

import logging
import math
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import pandas as pd

CHAN_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "chan.py"
if str(CHAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CHAN_ROOT))

from tradingagents.broker.alpaca_broker import AlpacaBroker
from tradingagents.broker.models import Account, OrderRequest, Position
from tradingagents.portfolio.portfolio_tracker import PortfolioTracker
from tradingagents.portfolio.position_sizer import PositionSizer
from tradingagents.research.chan_daily_backtester import (
    ChanDailyBacktestConfig,
    PortfolioChanDailyBacktester,
)
from tradingagents.research.warehouse import MarketDataWarehouse
from tradingagents.risk.risk_engine import RiskEngine
from tradingagents.storage.database import TradingDatabase
from tradingagents.automation.notifier import build_notifier

logger = logging.getLogger(__name__)


DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "USO",
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU",
]


class ChanDailyOrchestrator:
    """Daily-bar Donchian + seg-BSP Chan strategy. One scan per trading day."""

    def __init__(self, config: dict):
        self.config = config

        self.broker = AlpacaBroker(
            api_key=config["alpaca_api_key"],
            secret_key=config["alpaca_secret_key"],
            paper=config.get("paper_trading", True),
        )

        self.db = TradingDatabase(
            config.get("db_path", "trading_chan_daily.db"),
            variant=config.get("variant_name"),
        )
        config.setdefault("strategy_tag", "chan_daily")
        self.notifier = build_notifier(config)

        starting_equity = self.db.get_starting_equity()
        risk_config = {**config, "starting_equity": starting_equity}
        self.risk_engine = RiskEngine(risk_config)
        self.sizer = PositionSizer(config)
        self.tracker = PortfolioTracker(self.broker, self.db)

        self.daily_db = config.get("daily_db_path", "research_data/market_data.duckdb")
        self.universe = config.get("chan_daily_universe", DEFAULT_UNIVERSE)

        # Build the locked-down NEW NEW OPTIMAL config from research.
        # Do NOT change params without re-running scripts/sweep_chan_optimization.py.
        self.bt_cfg = ChanDailyBacktestConfig(
            chan_bs_type="1,1p,2,2s,3a,3b",
            buy_types=("T1", "T1P", "T2", "T2S", "T3A", "T3B"),
            sell_types=("T1",),
            chan_min_zs_cnt=0,
            chan_bi_strict=False,
            require_sure=False,
            sizing_mode="atr_parity",
            risk_per_trade=config.get("risk_per_trade", 0.020),
            position_pct=config.get("position_pct", 0.25),
            max_positions=config.get("max_positions", 6),
            enable_shorts=False,
            entry_mode="donchian_or_seg",
            donchian_period=30,
            chan_macd_algo="slope",
            chan_divergence_rate=0.7,
            chan_bsp2_follow_1=config.get("chan_bsp2_follow_1", True),
            chan_bsp3_follow_1=config.get("chan_bsp3_follow_1", True),
            chan_bsp3_peak=config.get("chan_bsp3_peak", False),
            chan_strict_bsp3=config.get("chan_strict_bsp3", False),
            chan_bsp3a_max_zs_cnt=config.get("chan_bsp3a_max_zs_cnt", 1),
            chan_zs_algo=config.get("chan_zs_algo", "normal"),
            momentum_rank_lookback=63,
            momentum_rank_top_k=10,
            time_stop_bars=100,
            entry_priority_mode="momentum",
            trend_type_filter_mode="trend_only",  # 走势类型: block during active ZS
            daily_db_path=self.daily_db,
        )

        self._last_run_date: Optional[str] = None

    # ------------------------------------------------------------------
    # Top-level scheduler entry points
    # ------------------------------------------------------------------

    def run_daily_analysis(self) -> Dict:
        """Once-per-trading-day pre-market scan + execute. Idempotent within day.

        Called by ABRunner during the scheduler's run_daily_analysis cron tick.
        """
        today_iso = date.today().isoformat()
        if self._last_run_date == today_iso:
            logger.info("ChanDaily: already ran today, skipping")
            return {"status": "skipped_already_ran"}

        logger.info("=== ChanDaily: Daily Scan ===")
        t0 = time.perf_counter()

        try:
            account = self.broker.get_account()
            positions = self.broker.get_positions()
            logger.info("Account: equity=$%.2f cash=$%.2f n_positions=%d",
                         account.equity, account.cash, len(positions))
        except Exception as e:
            logger.error("Failed to get account info: %s", e)
            return {"error": str(e)}

        # Refresh daily data (with retry for DuckDB writer-lock contention)
        self._refresh_daily_data()

        # Compute today's signals
        try:
            sigs = self._compute_signals()
        except Exception as e:
            logger.error("Signal computation failed: %s", e)
            return {"error": str(e)}

        results = {"entries": [], "exits": [], "holds": []}

        # Exits first (free up cash + slot)
        exit_results = self._check_exits(positions, account, sigs)
        results["exits"] = exit_results
        if exit_results:
            try:
                positions = self.broker.get_positions()
                account = self.broker.get_account()
            except Exception as e:
                logger.warning("Post-exit refresh failed: %s", e)

        # Entries
        entry_results = self._check_entries(account, positions, sigs)
        results["entries"] = entry_results

        elapsed = time.perf_counter() - t0
        logger.info("ChanDaily scan complete in %.1fs: %d entries, %d exits",
                     elapsed, len(entry_results), len(exit_results))

        self._notify_summary(results)
        self._last_run_date = today_iso
        return results

    # ------------------------------------------------------------------
    # Data refresh + signal extraction
    # ------------------------------------------------------------------

    def _refresh_daily_data(self):
        """Pull last 14 days of daily bars for the universe via yfinance.

        Retries up to 3× with backoff if another process holds the DuckDB
        writer lock. Failures are logged but do not abort the run — we'll
        proceed with whatever's already in the DB (subject to staleness guard).
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        logger.info("ChanDaily: refreshing daily data %s -> %s for %d symbols",
                     start_date, end_date, len(self.universe))

        wh = None
        for attempt in range(3):
            try:
                wh = MarketDataWarehouse(db_path=self.daily_db)
                counts = wh.fetch_and_store_daily_bars(self.universe, start_date, end_date)
                n_ok = sum(1 for c in counts.values() if c > 0)
                logger.info("ChanDaily refresh OK: %d/%d symbols updated",
                             n_ok, len(self.universe))
                break
            except Exception as e:
                logger.warning("ChanDaily refresh attempt %d failed: %s", attempt + 1, e)
                if wh is not None:
                    try:
                        wh.close()
                    except Exception:
                        pass
                    wh = None
                time.sleep(5 * (attempt + 1))
        if wh is not None:
            try:
                wh.close()
            except Exception as e:
                logger.warning("ChanDaily wh.close warning: %s", e)

    def _compute_signals(self) -> Dict:
        """Return {today, bars, signals_dict, donchian_high/low, momentum_ranks}.

        Reuses the backtester's signal extraction so live decisions match
        backtest decisions bit-for-bit. SAFETY: refuses to use today's calendar
        date as the decision bar (intraday partial close looks like final close
        in yfinance during market hours). Truncates to prior trading day.
        """
        today_str = datetime.now().strftime("%Y-%m-%d")
        begin_str = (datetime.now() - pd.Timedelta(days=600)).strftime("%Y-%m-%d")

        bt = PortfolioChanDailyBacktester(self.bt_cfg)
        bars = bt._load_bars(self.universe, begin_str, today_str)
        if not bars:
            raise RuntimeError("ChanDaily: no bars loaded — check daily_db_path")

        last_dates = [df.index[-1] for df in bars.values()]
        today = max(last_dates)
        logger.info("ChanDaily: most recent universe bar = %s", today.date())

        # Staleness guard
        days_stale = (datetime.now().date() - today.date()).days
        if days_stale > 5:
            raise RuntimeError(
                f"ChanDaily: latest bar {today.date()} is {days_stale} days stale"
            )

        # Truncate today's calendar-date bar (intraday partial close)
        today_cal = datetime.now().date()
        if today.date() >= today_cal:
            cutoff = today - pd.Timedelta(days=1)
            tries = 0
            while not any(cutoff in df.index for df in bars.values()):
                cutoff -= pd.Timedelta(days=1)
                tries += 1
                if tries > 7:
                    raise RuntimeError("ChanDaily: cannot find prior-day bar in any symbol")
            logger.warning("ChanDaily: truncating today's bar %s -> prior day %s",
                            today.date(), cutoff.date())
            for sym in bars:
                bars[sym] = bars[sym][bars[sym].index <= cutoff]
            today = cutoff

        # Chan signals
        from ChanConfig import CChanConfig  # noqa
        chan_cfg = CChanConfig({
            "trigger_step": True,
            "bi_strict": self.bt_cfg.chan_bi_strict,
            "divergence_rate": self.bt_cfg.chan_divergence_rate,
            "macd_algo": self.bt_cfg.chan_macd_algo,
            "bs_type": self.bt_cfg.chan_bs_type,
            "min_zs_cnt": self.bt_cfg.chan_min_zs_cnt,
            "bsp2_follow_1": self.bt_cfg.chan_bsp2_follow_1,
            "bsp3_follow_1": self.bt_cfg.chan_bsp3_follow_1,
            "bsp3_peak": self.bt_cfg.chan_bsp3_peak,
            "strict_bsp3": self.bt_cfg.chan_strict_bsp3,
            "bsp3a_max_zs_cnt": self.bt_cfg.chan_bsp3a_max_zs_cnt,
            "print_warning": False,
            "zs_algo": self.bt_cfg.chan_zs_algo,
        })
        signals = bt._preload_signals(list(bars.keys()), begin_str, today_str, chan_cfg)

        # Donchian (close-breakout, strictly prior 30 closes)
        d_high, d_low = {}, {}
        for sym in bars:
            d_high[sym] = bars[sym]["close"].shift(1).rolling(window=self.bt_cfg.donchian_period).max()
            d_low[sym] = bars[sym]["close"].shift(1).rolling(window=self.bt_cfg.donchian_period).min()

        # Momentum ranking
        ret_df = pd.DataFrame({
            s: bars[s]["close"].pct_change(self.bt_cfg.momentum_rank_lookback) for s in bars
        })
        rank_df = ret_df.rank(axis=1, method="min", ascending=False)
        ranks = {}
        if today in rank_df.index:
            for sym in bars:
                r = rank_df.loc[today].get(sym)
                if pd.notna(r):
                    ranks[sym] = float(r)

        return {
            "today": today,
            "bars": bars,
            "signals": signals,
            "d_high": d_high,
            "d_low": d_low,
            "ranks": ranks,
        }

    # ------------------------------------------------------------------
    # Strategy decisions
    # ------------------------------------------------------------------

    def _check_exits(self, positions: List[Position], account: Account, sigs: Dict) -> List[Dict]:
        """Per held position: exit on Chan T1 sell signal OR time stop ≥100 bars.

        The structural stop is handled broker-side (bracket SL leg) — fires intraday
        without us re-checking.
        """
        if not positions:
            return []
        out = []
        today = sigs["today"]
        for pos in positions:
            sym = pos.symbol
            if sym not in self.universe:
                logger.warning("ChanDaily: held %s not in universe, skipping exit check", sym)
                continue
            sig = sigs["signals"].get(sym, {}).get(today) or {}

            if sig.get("sell"):
                res = self._execute_signal(
                    symbol=sym, action="SELL",
                    confidence=0.7,
                    reasoning=f"chan_t1_sell:{sig['sell'].get('types_str', 'T1')}",
                    account=account, positions=positions,
                )
                out.append(res)
                continue

            # Time stop check via local position_states (entry_date)
            try:
                pos_state = self.db.get_position_state(sym)
                if pos_state and pos_state.get("entry_date"):
                    entry_dt = pd.Timestamp(pos_state["entry_date"])
                    if sym in sigs["bars"]:
                        bars_held = len(sigs["bars"][sym].loc[entry_dt:today])
                        if bars_held >= self.bt_cfg.time_stop_bars:
                            res = self._execute_signal(
                                symbol=sym, action="SELL",
                                confidence=0.6,
                                reasoning=f"time_stop_{bars_held}_bars",
                                account=account, positions=positions,
                            )
                            out.append(res)
                            continue
            except Exception as e:
                logger.warning("Time-stop check failed for %s: %s", sym, e)
        return out

    def _check_entries(self, account: Account, positions: List[Position], sigs: Dict) -> List[Dict]:
        """Build entry candidates with momentum priority, place bracket orders."""
        held = {p.symbol for p in positions}
        free_slots = max(0, self.bt_cfg.max_positions - len(held))
        if free_slots == 0:
            logger.info("ChanDaily: no free position slots (max=%d held=%d)",
                         self.bt_cfg.max_positions, len(held))
            return []

        today = sigs["today"]
        cands = []
        for sym in self.universe:
            if sym in held:
                continue
            if sym not in sigs["bars"] or today not in sigs["bars"][sym].index:
                continue
            close_today = float(sigs["bars"][sym].loc[today, "close"])
            sig = sigs["signals"].get(sym, {}).get(today) or {}

            cand = None
            # Donchian first (with momentum gate)
            if today in sigs["d_high"][sym].index:
                d_high = sigs["d_high"][sym].loc[today]
                d_low = sigs["d_low"][sym].loc[today]
                if pd.notna(d_high) and close_today > float(d_high):
                    rank = sigs["ranks"].get(sym, 9999.0)
                    if rank <= self.bt_cfg.momentum_rank_top_k:
                        cand = {
                            "symbol": sym, "type": "donchian", "rank": rank,
                            "ref_price": float(d_low) if pd.notna(d_low) else 0.0,
                            "close": close_today,
                        }
            # Then seg_bsp (bypasses momentum)
            if cand is None and sig.get("seg_buy"):
                cand = {
                    "symbol": sym, "type": "seg:" + sig["seg_buy"]["types_str"],
                    "rank": sigs["ranks"].get(sym, 9999.0),
                    "ref_price": sig["seg_buy"]["bi_low"],
                    "close": close_today,
                }
            if cand:
                cands.append(cand)

        # Priority: best momentum rank first
        cands.sort(key=lambda c: c["rank"])
        cands = cands[:free_slots]

        out = []
        for c in cands:
            sym = c["symbol"]
            atr_today = self._atr_today(sigs["bars"][sym])
            atr_stop = c["close"] - self.bt_cfg.stop_atr_mult * atr_today
            ref_price = c["ref_price"]
            if self.bt_cfg.structural_stop and ref_price > 0:
                stop_price = max(ref_price, atr_stop)
            else:
                stop_price = atr_stop
            if stop_price >= c["close"]:
                continue
            risk_per_share = c["close"] - stop_price
            tp_price = c["close"] + 2.5 * risk_per_share

            # Convert to stop_loss_pct + take_profit_pct for sizer/_execute_signal
            stop_pct = (c["close"] - stop_price) / c["close"]
            tp_pct = (tp_price - c["close"]) / c["close"]
            res = self._execute_signal(
                symbol=sym, action="BUY",
                confidence=0.7,
                reasoning=f"{c['type']}_mom_rank_{int(c['rank'])}",
                account=account, positions=positions,
                stop_loss_pct=stop_pct,
                take_profit_pct=tp_pct,
                signal_metadata={
                    "signal_type": c["type"],
                    "structural_stop": stop_price,
                    "momentum_rank": c["rank"],
                    "ref_price": ref_price,
                    "atr_20": atr_today,
                },
            )
            out.append(res)
        return out

    @staticmethod
    def _atr_today(df: pd.DataFrame, period: int = 20) -> float:
        """Wilder ATR last value."""
        h, l, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        return float(tr.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1])

    # ------------------------------------------------------------------
    # Order execution (mirrors ChanOrchestrator._execute_signal)
    # ------------------------------------------------------------------

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
        signal_metadata: Optional[Dict] = None,
    ) -> Dict:
        """Size, risk-check, and execute a signal via Alpaca."""
        signal = {
            "symbol": symbol, "action": action, "confidence": confidence,
            "reasoning": reasoning, "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "timeframe": "swing", "source": "chan_daily",
        }

        metadata_json = None
        if signal_metadata:
            try:
                import json as _json

                def _clean(v):
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        return None
                    return v

                metadata_json = _json.dumps(
                    {k: _clean(v) for k, v in signal_metadata.items()}, default=str
                )
            except Exception as e:
                logger.warning("chan_daily signal_metadata serialize failed: %s", e)

        signal_id = self.db.log_signal(
            symbol=symbol, action=action, confidence=confidence,
            reasoning=reasoning, stop_loss=stop_loss_pct,
            take_profit=take_profit_pct, timeframe="swing",
            signal_metadata=metadata_json,
        )

        try:
            current_price = self.broker.get_latest_price(symbol)
        except Exception as e:
            logger.error("Latest price fetch failed for %s: %s", symbol, e)
            self.db.mark_signal_rejected(signal_id, f"latest_price_failed: {e}")
            return {"symbol": symbol, "action": action, "traded": False, "error": str(e)}

        current_position = next((p for p in positions if p.symbol == symbol), None)
        total_pos_value = sum(p.market_value for p in positions)

        if action == "BUY":
            signal["stop_loss"] = current_price * (1 - stop_loss_pct)

        order_request = self.sizer.calculate(
            signal=signal, account=account, current_price=current_price,
            current_position=current_position, total_position_value=total_pos_value,
        )
        if order_request is None:
            logger.info("%s: No trade needed (sizer returned None)", symbol)
            return {"symbol": symbol, "action": action, "traded": False}

        if order_request.side == "buy":
            existing = self._find_existing_open_order(symbol, "buy")
            if existing is not None:
                reason = f"Existing open buy order {existing.order_id} [{existing.status}]"
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

        logger.info("%s: %s %d shares @ ~$%.2f",
                     symbol, order_request.side, order_request.qty, current_price)

        if order_request.side == "buy":
            sl_price = round(current_price * (1 - stop_loss_pct), 2)
            tp_price = round(current_price * (1 + take_profit_pct), 2)
            order_request.time_in_force = "gtc"
            order_result = self.broker.submit_bracket_order(
                order_request, stop_loss_price=sl_price,
                take_profit_price=tp_price, anchor_price=current_price,
            )
            sl_price = order_result.effective_stop_price or sl_price
            tp_price = order_result.effective_take_profit_price or tp_price
            logger.info("%s: Bracket SL=$%.2f TP=$%.2f", symbol, sl_price, tp_price)
        else:
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
            symbol=symbol, side=order_request.side, qty=order_request.qty,
            order_type="bracket" if order_request.side == "buy" else order_request.order_type,
            status=order_result.status, filled_qty=order_result.filled_qty,
            filled_price=order_result.filled_avg_price,
            order_id=order_result.order_id, signal_id=signal_id, reasoning=reasoning,
        )
        self.db.mark_signal_executed(signal_id)

        if order_request.side == "buy" and "fill" in str(order_result.status).lower():
            state_payload = {
                "entry_price": order_result.filled_avg_price or current_price,
                "entry_date": date.today().isoformat(),
                "highest_close": order_result.filled_avg_price or current_price,
                "current_stop": sl_price,
                "partial_taken": False,
                "variant": self.config.get("variant_name", "chan_daily"),
            }
            if signal_metadata and signal_metadata.get("signal_type"):
                state_payload["base_pattern"] = signal_metadata["signal_type"]
            self.db.upsert_position_state(symbol, state_payload)

        if order_request.side == "sell" and "fill" in str(order_result.status).lower():
            try:
                if self.config.get("trade_outcome_live_hook_chan_enabled", True):
                    from tradingagents.automation.trade_outcome import log_closed_trade
                    pos_state = self.db.get_position_state(symbol)
                    if pos_state:
                        log_closed_trade(
                            db=self.db, symbol=symbol, pos_state=pos_state,
                            exit_price=float(order_result.filled_avg_price or current_price),
                            exit_reason=reasoning or action,
                            broker=self.broker,
                            excursion_enabled=self.config.get(
                                "trade_outcome_excursion_enabled", False
                            ),
                        )
                        self.db.delete_position_state(symbol)
            except Exception as e:
                logger.warning("chan_daily trade_outcome hook failed for %s: %s", symbol, e)

        return {"symbol": symbol, "action": action, "traded": True,
                "qty": order_request.qty, "order_id": order_result.order_id,
                "status": order_result.status}

    def _find_existing_open_order(self, symbol: str, side: Optional[str] = None):
        """Return any open BUY order for symbol — guards against double-entry."""
        try:
            open_orders = self.broker.get_live_orders(symbol=symbol)
        except TypeError:
            open_orders = self.broker.get_live_orders()
        for o in open_orders or []:
            if str(getattr(o, "symbol", "")).upper() != symbol.upper():
                continue
            if side and str(getattr(o, "side", "")).lower() != side:
                continue
            return o
        return None

    def _notify_summary(self, results: Dict):
        try:
            entries = results.get("entries", [])
            exits = results.get("exits", [])
            if not entries and not exits:
                return
            title = f"ChanDaily: {len(entries)} entries, {len(exits)} exits"
            body_lines = []
            for e in entries:
                if e.get("traded"):
                    body_lines.append(f"+ BUY {e['symbol']} ({e.get('qty', '?')})")
            for e in exits:
                if e.get("traded"):
                    body_lines.append(f"- SELL {e['symbol']}")
            body = "\n".join(body_lines) if body_lines else "(no fills)"
            self.notifier.send(title, body)
        except Exception as e:
            logger.warning("Notify failed: %s", e)

    # ------------------------------------------------------------------
    # Scheduler interface methods (mirror ChanOrchestrator)
    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        clock = self.broker.get_clock()

        pos_list = []
        for p in positions:
            pos_list.append({
                "symbol": p.symbol, "qty": float(p.qty),
                "entry": float(p.avg_entry_price), "current": float(p.current_price),
                "pl": float(p.unrealized_pl),
                "pl_pct": f"{float(p.unrealized_plpc):.2%}" if p.unrealized_plpc else "0.00%",
            })

        trades_today = self.db.get_recent_trades(limit=50)
        today_str = date.today().isoformat()
        today_trades = [t for t in trades_today if t.get("timestamp", "").startswith(today_str)]

        return {
            "account": {
                "equity": float(account.equity), "cash": float(account.cash),
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
            "watchlist": f"{len(self.universe)} ETFs (donchian+seg_bsp)",
            "paper_mode": self.config.get("paper_trading", True),
            "strategy": "chan_daily",
        }

    def generate_daily_report(self, save: bool = True) -> Dict:
        status = self.get_status()
        trades = self.db.get_recent_trades(limit=50)
        today_str = date.today().isoformat()
        today_trades = [t for t in trades if t.get("timestamp", "").startswith(today_str)]

        report = {
            "date": today_str, "strategy": "chan_daily",
            "paper_mode": self.config.get("paper_trading", True),
            "account": status["account"],
            "trade_summary": {
                "total_orders": len(today_trades),
                "filled_orders": sum(1 for t in today_trades if t.get("status") == "filled"),
                "buy_orders": sum(1 for t in today_trades if t.get("side") == "buy"),
                "sell_orders": sum(1 for t in today_trades if t.get("side") == "sell"),
                "symbols": list({t["symbol"] for t in today_trades}),
            },
            "position_summary": status["positions"],
            "performance": status["performance"],
        }
        if save:
            report_dir = Path("results/chan_daily_reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"report_{today_str}.json"
            import json
            report_path.write_text(json.dumps(report, indent=2, default=str))
            logger.info("ChanDaily report saved to %s", report_path)
        return report

    def run_daily_reflection(self) -> Dict:
        """Placeholder — chan_daily doesn't use AI reflection."""
        return {}

    def take_market_snapshot(self) -> Dict:
        logger.info("ChanDaily: capturing scheduled market snapshot...")
        return self.tracker.take_daily_snapshot()

    def run_daily_trade_review(self) -> Dict:
        """Per-trade post-mortems for trades that closed today."""
        from tradingagents.automation.trade_review import run_daily_review
        return run_daily_review(
            db=self.db, broker=self.broker,
            variant_name=self.config.get("variant_name", "chan_daily"),
            config=self.config,
        )

    def run_held_position_review(self) -> Dict:
        """Health-check per held position."""
        from tradingagents.automation.trade_review import run_held_position_review
        return run_held_position_review(
            db=self.db, broker=self.broker,
            variant_name=self.config.get("variant_name", "chan_daily"),
            config=self.config,
        )

    def reconcile_orders(self) -> Dict:
        """Sync local trades vs broker (Track P-SYNC)."""
        from .reconciler import OrderReconciler
        reconciler = OrderReconciler(
            broker=self.broker, db=self.db,
            variant=self.config.get("variant_name"),
            notifier=self.notifier, config=self.config,
        )
        return reconciler.reconcile_once()

    def run_exit_check_pass(self, **kwargs) -> Dict:
        """Mid-day exit check pass — for chan_daily this is a no-op since
        we only act on end-of-day decisions; structural stops are already
        on the broker as bracket SL legs and fire intraday autonomously.

        Accepts **kwargs so the shared ab_runner caller (which passes
        ai_review_enabled=False to disable LLM exits on rule-based variants)
        doesn't error on chan_daily's narrower interface.
        """
        return {"status": "noop_chan_daily_runs_once_per_day"}
