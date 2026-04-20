"""Chan v2 backtester — extends v1 with exit and sizing improvements.

New features (all off by default for v1 parity):
- sell_cooldown_bars: ignore sell BSPs for first N bars after entry
- breakeven_atr, trail_tighten_atr, stop_grace_bars: already in v1, just wire CLI
- macd_algo: already in v1, just wire CLI
- zs_oscillation_exit: exit when ZS has N+ oscillations (Chan theory exhaustion)
- divergence_sizing: scale position size by divergence strength
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .chan_backtester import (
    CChan,
    CChanConfig,
    AUTYPE,
    KL_TYPE,
    ChanBacktestConfig,
    ChanPortfolioResult,
    DuckDBIntradayAPI,
    PortfolioChanBacktester,
    _PendingEntry,
    _PendingExit,
    _Position,
)

log = logging.getLogger("chan_v2")


@dataclass
class ChanV2BacktestConfig(ChanBacktestConfig):
    # --- Exit improvements ---
    sell_cooldown_bars: int = 0       # ignore sell BSP for first N bars (0=off)
    zs_oscillation_exit: bool = False # exit when inside exhausted ZS
    zs_max_oscillations: int = 5     # N oscillations = exhaustion

    # --- Sizing improvements ---
    divergence_sizing: bool = False       # scale size by divergence_rate
    divergence_sizing_base: float = 0.6   # div_rate at which sizing = 1.0x
    divergence_sizing_max_boost: float = 1.5  # max sizing multiplier


@dataclass
class _PendingEntryV2(_PendingEntry):
    divergence_rate: float | None = None


class PortfolioChanV2Backtester(PortfolioChanBacktester):
    """Chan v2 backtester with exit and sizing enhancements.

    Inherits all v1 helpers (stop logic, buy signal detection, daily filters,
    RS ranking, SEPA, regime, etc.) and only overrides the event loop and
    data extraction where v2 features need changes.
    """

    def __init__(self, config: ChanV2BacktestConfig | None = None):
        super().__init__(config or ChanV2BacktestConfig())

    def backtest_portfolio(
        self,
        symbols: list[str],
        begin: str,
        end: str,
        db_path: str,
        regime_df: pd.DataFrame | None = None,
    ) -> ChanPortfolioResult:
        cfg: ChanV2BacktestConfig = self.config
        DuckDBIntradayAPI.DB_PATH = db_path
        rng = random.Random(cfg.random_seed)

        chan_cfg = CChanConfig({
            "trigger_step": True,
            "bi_strict": cfg.chan_bi_strict,
            "skip_step": 0,
            "divergence_rate": cfg.chan_divergence_rate,
            "bsp2_follow_1": False,
            "bsp3_follow_1": False,
            "min_zs_cnt": cfg.chan_min_zs_cnt,
            "bs1_peak": False,
            "macd_algo": cfg.chan_macd_algo,
            "bs_type": cfg.chan_bs_type,
            "max_bs2_rate": cfg.chan_max_bs2_rate,
            "print_warning": False,
            "zs_algo": "normal",
            "mean_metrics": [20, 60] if cfg.ma_trend_filter else [],
        })

        buy_types_set = set(cfg.buy_types)

        daily_directions: dict[str, dict[str, str]] = {}
        need_daily = cfg.daily_filter or cfg.daily_zs_filter or cfg.daily_bsp_confirm
        if need_daily:
            log.info("Pre-loading daily Chan direction for %d symbols...", len(symbols))
            daily_directions = self._preload_daily_direction(symbols, begin, end, cfg)

        sepa_data: dict[str, dict[str, bool]] = {}
        if cfg.sepa_filter:
            log.info("Pre-loading SEPA trend template for %d symbols...", len(symbols))
            t0s = time.perf_counter()
            sepa_data = self._preload_sepa_trend(symbols, begin, end, cfg)
            loaded = sum(1 for v in sepa_data.values() for x in v.values() if x)
            log.info("SEPA pass-days: %d in %.1fs", loaded, time.perf_counter() - t0s)

        rs_data: dict[str, set[str]] = {}
        if cfg.rs_filter:
            log.info("Pre-loading RS ranking for %d symbols...", len(symbols))
            t0r = time.perf_counter()
            rs_data = self._preload_rs_ranking(symbols, begin, end, cfg)
            log.info("RS ranking loaded in %.1fs", time.perf_counter() - t0r)

        log.info("Pre-loading %d symbols via step_load()...", len(symbols))
        t0 = time.perf_counter()
        events = self._preload_events_v2(symbols, begin, end, chan_cfg, cfg)
        log.info("Pre-loaded %d events in %.1fs", len(events), time.perf_counter() - t0)

        regime_dates = None
        regime_values = None
        regime_scores = None
        if regime_df is not None and not regime_df.empty:
            regime_dates = list(regime_df.index)
            regime_values = list(regime_df["market_regime"])
            if "market_score" in regime_df.columns:
                regime_scores = list(regime_df["market_score"])

        cash = cfg.initial_cash
        positions: dict[str, _Position] = {}
        all_trades: list[dict] = []
        equity_curve: list[dict] = []
        daily_state: list[dict] = []
        seen_buy: set = set()
        seen_sell: dict[str, set] = {s: set() for s in symbols}
        last_prices: dict[str, float] = {}
        last_day: str | None = None

        pending_entries: dict[str, _PendingEntryV2] = {}
        pending_exits: dict[str, _PendingExit] = {}

        events = self._shuffle_ties(events, rng)

        for ts, symbol, snapshot_data in events:
            cur_open = snapshot_data["open"]
            cur_close = snapshot_data["close"]
            cur_high = snapshot_data["high"]
            cur_low = snapshot_data["low"]
            cur_time = snapshot_data["time_str"]
            bsp_list = snapshot_data["bsp_list"]
            zs_list = snapshot_data["zs_list"]
            seg_bsp_list = snapshot_data.get("seg_bsp_list", [])
            atr = snapshot_data["atr"]

            last_prices[symbol] = cur_close
            cur_day = ts.strftime("%Y-%m-%d")

            # --- Execute pending sell-signal exit at this bar's open ---
            if symbol in pending_exits and symbol in positions:
                pe = pending_exits.pop(symbol)
                pos = positions[symbol]
                actual_exit = cur_open * (1 - cfg.slippage_bps / 10000)
                self._close_position(
                    pos, actual_exit, cur_time, pe.reason, cfg,
                    all_trades, positions,
                )
                cash += pos.shares * actual_exit - abs(pos.shares * actual_exit * cfg.commission_bps / 10000)

            # --- Execute pending buy-signal entry at this bar's open ---
            if symbol in pending_entries and symbol not in positions:
                pe = pending_entries.pop(symbol)
                if len(positions) < cfg.max_positions:
                    regime = self._get_regime(ts, regime_dates, regime_values)
                    target_exp = self._target_exposure(regime, cfg)
                    market_value = sum(
                        last_prices.get(s, p.entry_price) * p.shares
                        for s, p in positions.items()
                    )
                    portfolio_value = cash + market_value
                    current_exposure = market_value / portfolio_value if portfolio_value > 0 else 0

                    if current_exposure < target_exp:
                        entry = self._size_entry_v2(
                            pe, cur_open, cur_high, cfg, cash, portfolio_value, ts,
                        )
                        if entry:
                            positions[symbol] = entry
                            cost = entry.shares * entry.entry_price
                            commission = entry.shares * entry.entry_price * cfg.commission_bps / 10000
                            cash -= cost + commission

            # --- Discard stale pending orders ---
            pending_entries.pop(symbol, None)
            pending_exits.pop(symbol, None)

            # --- Process existing position: stop-based exits (intrabar) ---
            if symbol in positions:
                pos = positions[symbol]
                pos.bars_held += 1
                if cur_high > pos.high_since_entry:
                    pos.high_since_entry = cur_high

                stop_reason, stop_price = self._check_stop_exit(
                    pos, cur_open, cur_close, cur_low, atr, zs_list, cfg,
                )
                if stop_reason:
                    actual_exit = stop_price * (1 - cfg.slippage_bps / 10000)
                    self._close_position(
                        pos, actual_exit, cur_time, stop_reason, cfg,
                        all_trades, positions,
                    )
                    cash += pos.shares * actual_exit - abs(pos.shares * actual_exit * cfg.commission_bps / 10000)
                else:
                    # --- V2: Sell signal cooldown ---
                    # Only check sell BSP if past the cooldown period
                    sell_sig = False
                    if cfg.sell_cooldown_bars <= 0 or pos.bars_held >= cfg.sell_cooldown_bars:
                        sell_sig = self._check_sell_signal(
                            bsp_list, seen_sell.get(symbol, set()), cfg,
                        )

                    if sell_sig:
                        pending_exits[symbol] = _PendingExit(
                            reason="sell_signal", signal_time=cur_time,
                        )
                    # --- V2: ZS oscillation exhaustion exit ---
                    elif cfg.zs_oscillation_exit:
                        if self._check_zs_exhaustion(
                            pos, cur_close, zs_list, cfg,
                        ):
                            pending_exits[symbol] = _PendingExit(
                                reason="zs_exhaustion", signal_time=cur_time,
                            )
                    if symbol not in pending_exits:
                        if pos.bars_held >= cfg.max_hold_bars:
                            pending_exits[symbol] = _PendingExit(
                                reason="time_stop", signal_time=cur_time,
                            )
                        elif (
                            cfg.dead_money_bars > 0
                            and pos.bars_held >= cfg.dead_money_bars
                            and (cur_close - pos.entry_price) / pos.entry_price < cfg.dead_money_min_gain
                        ):
                            pending_exits[symbol] = _PendingExit(
                                reason="dead_money", signal_time=cur_time,
                            )

            # --- Check buy signals → queue for NEXT bar execution ---
            if symbol not in positions and symbol not in pending_entries:
                daily_ok = True

                if cfg.regime_gate and regime_scores is not None:
                    score = self._get_regime_score(ts, regime_dates, regime_scores)
                    if score is not None and score <= cfg.regime_min_score:
                        daily_ok = False

                if daily_ok and daily_directions:
                    daily_dir = self._get_daily_direction(daily_directions, symbol, cur_day)
                    if cfg.daily_filter_mode == "bullish_only":
                        if daily_dir != "bullish":
                            daily_ok = False
                    else:
                        if daily_dir == "bearish":
                            daily_ok = False

                if daily_ok and sepa_data:
                    if not self._check_sepa_pass(sepa_data, symbol, cur_day):
                        daily_ok = False

                if daily_ok and rs_data:
                    if not self._check_rs_pass(rs_data, symbol, cur_day):
                        daily_ok = False

                if daily_ok and cfg.ma_trend_filter:
                    _ma20 = snapshot_data.get("ma20")
                    _ma60 = snapshot_data.get("ma60")
                    if _ma20 is not None and _ma60 is not None:
                        if not (cur_close > _ma20 > _ma60):
                            daily_ok = False

                if daily_ok and cfg.daily_zs_filter and daily_directions:
                    dd = self._get_daily_data(daily_directions, symbol, cur_day)
                    if isinstance(dd, dict):
                        d_zs_list = dd.get("daily_zs", [])
                        if d_zs_list:
                            nearest_dzs = d_zs_list[-1]
                            if cur_close < nearest_dzs["high"]:
                                daily_ok = False

                if daily_ok:
                    buy_sig = self._check_buy_signal_v2(
                        bsp_list, buy_types_set, seen_buy, cfg,
                        seg_bsp_list=seg_bsp_list,
                    )
                    if buy_sig and cfg.filter_zs_min_bi_cnt > 0:
                        nearest_zs = None
                        for zs in reversed(zs_list):
                            if zs["low"] < cur_close:
                                nearest_zs = zs
                                break
                        if nearest_zs and nearest_zs.get("bi_cnt", 0) < cfg.filter_zs_min_bi_cnt:
                            buy_sig = None
                    if buy_sig and cfg.min_stop_pct > 0:
                        stop = self._estimate_stop(
                            buy_sig["bi_low"], zs_list, cur_close, atr,
                        )
                        if stop > 0 and (cur_close - stop) / cur_close < cfg.min_stop_pct:
                            buy_sig = None
                    if buy_sig:
                        _daily_bsp = False
                        if cfg.daily_bsp_confirm and daily_directions:
                            dd = self._get_daily_data(daily_directions, symbol, cur_day)
                            if isinstance(dd, dict) and dd.get("has_daily_buy"):
                                _daily_bsp = True

                        # V2: Use _PendingEntryV2 with divergence_rate
                        pending_entries[symbol] = _PendingEntryV2(
                            symbol=symbol,
                            bsp_types=buy_sig["types_str"],
                            bi_low=buy_sig["bi_low"],
                            zs_list=zs_list,
                            atr=atr,
                            signal_time=cur_time,
                            seg_confirmed=buy_sig.get("seg_confirmed", False),
                            daily_bsp_confirmed=_daily_bsp,
                            divergence_rate=buy_sig.get("divergence_rate"),
                        )

            # --- Daily snapshot ---
            if cur_day != last_day:
                market_value = sum(
                    last_prices.get(s, p.entry_price) * p.shares
                    for s, p in positions.items()
                )
                portfolio_value = cash + market_value
                equity_curve.append({
                    "date": cur_day,
                    "equity": round(portfolio_value, 2),
                    "cash": round(cash, 2),
                    "positions": len(positions),
                })
                regime = self._get_regime(ts, regime_dates, regime_values) if regime_dates else "unknown"
                daily_state.append({
                    "date": cur_day,
                    "regime": regime,
                    "n_positions": len(positions),
                    "exposure": round(market_value / portfolio_value, 4) if portfolio_value > 0 else 0,
                })
                last_day = cur_day

        # Close remaining positions at last known price
        for sym, pos in list(positions.items()):
            price = last_prices.get(sym, pos.entry_price)
            actual_exit = price * (1 - cfg.slippage_bps / 10000)
            pnl = (actual_exit - pos.entry_price) * pos.shares
            all_trades.append({
                "symbol": sym,
                "entry_time": pos.entry_time.isoformat(),
                "exit_time": "end_of_data",
                "entry_price": round(pos.entry_price, 4),
                "exit_price": round(actual_exit, 4),
                "shares": pos.shares,
                "pnl": round(pnl, 2),
                "return_pct": round((actual_exit / pos.entry_price) - 1.0, 4),
                "bars_held": pos.bars_held,
                "exit_reason": "end_of_data",
                "bsp_types": pos.entry_bsp_types,
            })

        return self._build_result(all_trades, equity_curve, daily_state, cfg)

    # --- V2 preload: extends v1 with oscillation count ---

    def _preload_events_v2(
        self, symbols: list[str], begin: str, end: str, chan_cfg: CChanConfig,
        cfg: ChanV2BacktestConfig,
    ) -> list[tuple[datetime, str, dict]]:
        """Preload events with v2-specific data (oscillation count, divergence_rate)."""
        events = []
        for sym in symbols:
            try:
                chan = CChan(
                    code=sym, begin_time=begin, end_time=end,
                    data_src="custom:DuckDBAPI.DuckDB30mAPI",
                    lv_list=[KL_TYPE.K_30M], config=chan_cfg, autype=AUTYPE.QFQ,
                )
                hlc_history: list[tuple[float, float, float]] = []
                bar_idx = 0
                for snapshot in chan.step_load():
                    bar_idx += 1
                    try:
                        lvl = snapshot[0]
                        if not lvl.lst:
                            continue
                        ckline = lvl.lst[-1]
                        cur_klu = ckline.lst[-1]
                        cur_open = float(cur_klu.open)
                        cur_close = float(cur_klu.close)
                        cur_high = float(cur_klu.high)
                        cur_low = float(cur_klu.low)
                        cur_time_str = str(cur_klu.time)
                    except Exception:
                        continue

                    hlc_history.append((cur_high, cur_low, cur_close))
                    atr = self._true_atr(hlc_history, 14)

                    try:
                        bsp_list = snapshot.get_latest_bsp(idx=0, number=50)
                    except Exception:
                        bsp_list = []

                    zs_data = []
                    try:
                        for zs in lvl.zs_list:
                            zsd = {
                                "low": float(zs.low),
                                "high": float(zs.high),
                                "bi_cnt": len(zs.bi_lst),
                            }
                            if cfg.hub_peak_exit:
                                zsd["peak_high"] = float(zs.peak_high)
                                zsd["peak_low"] = float(zs.peak_low)
                            # V2: oscillation count from sub_zs_lst
                            if cfg.zs_oscillation_exit:
                                zsd["oscillation_cnt"] = len(zs.sub_zs_lst) if hasattr(zs, 'sub_zs_lst') and zs.sub_zs_lst else 0
                            zs_data.append(zsd)
                    except Exception:
                        pass

                    bsp_data = []
                    for bsp in bsp_list:
                        try:
                            _macd_dif = float(bsp.klu.macd.DIF)
                            _macd_dea = float(bsp.klu.macd.DEA)
                        except Exception:
                            _macd_dif = None
                            _macd_dea = None
                        try:
                            _div_rate = float(bsp.features['divergence_rate'])
                        except (KeyError, TypeError):
                            _div_rate = None
                        _vol_metric = None
                        if cfg.vol_divergence_filter:
                            try:
                                from Common.CEnum import MACD_ALGO
                                _vol_metric = float(bsp.bi.cal_macd_metric(MACD_ALGO.VOLUMN, is_reverse=bsp.is_buy))
                            except Exception:
                                pass
                        bsp_data.append({
                            "is_buy": bsp.is_buy,
                            "is_sure": bsp.bi.is_sure,
                            "types": tuple(sorted(t.name for t in bsp.type)),
                            "klu_idx": bsp.klu.idx,
                            "bi_low": float(bsp.bi._low()),
                            "macd_dif": _macd_dif,
                            "macd_dea": _macd_dea,
                            "bi_klu_cnt": bsp.bi.get_klu_cnt(),
                            "divergence_rate": _div_rate,
                            "vol_metric": _vol_metric,
                        })

                    seg_bsp_data = []
                    if cfg.seg_bsp_boost:
                        try:
                            _seg_bsp_obj = lvl.seg_bs_point_lst
                            for seg_bsp in _seg_bsp_obj.bsp1_list:
                                seg_bsp_data.append({
                                    "is_buy": seg_bsp.is_buy,
                                    "klu_idx": seg_bsp.klu.idx,
                                })
                            for klu_idx, seg_bsp in _seg_bsp_obj.bsp_store_flat_dict.items():
                                if klu_idx not in {s["klu_idx"] for s in seg_bsp_data}:
                                    seg_bsp_data.append({
                                        "is_buy": seg_bsp.is_buy,
                                        "klu_idx": seg_bsp.klu.idx,
                                    })
                        except Exception:
                            pass

                    # Extract MA trend values (always include for v1 parity)
                    ma20 = None
                    ma60 = None
                    try:
                        from Common.CEnum import TREND_TYPE
                        trend = getattr(cur_klu, "trend", {})
                        mean_dict = trend.get(TREND_TYPE.MEAN, {})
                        ma20 = mean_dict.get(20)
                        ma60 = mean_dict.get(60)
                    except Exception:
                        pass

                    ct = cur_klu.time
                    ts = datetime(ct.year, ct.month, ct.day, ct.hour, ct.minute)

                    events.append((ts, sym, {
                        "open": cur_open,
                        "close": cur_close,
                        "high": cur_high,
                        "low": cur_low,
                        "time_str": cur_time_str,
                        "bsp_list": bsp_data,
                        "zs_list": zs_data,
                        "seg_bsp_list": seg_bsp_data,
                        "atr": atr,
                        "bar_idx": bar_idx,
                        "ma20": ma20,
                        "ma60": ma60,
                    }))
            except Exception as e:
                log.warning("step_load failed for %s: %s", sym, e)

        events.sort(key=lambda x: (x[0], x[1]))
        return events

    # --- V2 buy signal: pass divergence_rate through ---

    @staticmethod
    def _check_buy_signal_v2(
        bsp_list: list[dict], buy_types_set: set, seen_buy: set,
        cfg: ChanV2BacktestConfig,
        seg_bsp_list: list[dict] | None = None,
    ) -> dict | None:
        """Check buy signal and include divergence_rate for v2 sizing."""
        result = PortfolioChanBacktester._check_buy_signal(
            bsp_list, buy_types_set, seen_buy, cfg,
            seg_bsp_list=seg_bsp_list,
        )
        if result is None:
            return None

        # Find the matching BSP to extract divergence_rate
        if cfg.divergence_sizing:
            for bsp in bsp_list:
                if not bsp["is_buy"]:
                    continue
                types_str = ",".join(sorted(set(bsp["types"])))
                if types_str == result["types_str"]:
                    result["divergence_rate"] = bsp.get("divergence_rate")
                    break
        return result

    # --- V2 sizing: divergence-based scaling ---

    def _size_entry_v2(
        self, pending: _PendingEntryV2, cur_open: float, cur_high: float,
        cfg: ChanV2BacktestConfig, cash: float, portfolio_value: float,
        ts: datetime,
    ) -> _Position | None:
        """Position sizing with divergence-based scaling."""
        actual_entry = cur_open * (1 + cfg.slippage_bps / 10000)
        commission_per_share = actual_entry * cfg.commission_bps / 10000

        stop_price = pending.bi_low
        if stop_price <= 0 or stop_price >= actual_entry:
            for zs in reversed(pending.zs_list):
                if zs["low"] < actual_entry:
                    stop_price = zs["low"]
                    break
            else:
                if pending.atr > 0:
                    stop_price = actual_entry - 3.0 * pending.atr
                else:
                    stop_price = actual_entry * 0.94

        risk_per_share = max(actual_entry - stop_price, actual_entry * 0.01)
        per_position_cap = portfolio_value * cfg.max_position_pct
        if cfg.vol_sizing and pending.atr > 0:
            atr_pct = pending.atr / actual_entry
            vol_scale = min(1.0, cfg.vol_target_atr_pct / atr_pct)
            per_position_cap *= vol_scale
        risk_budget = portfolio_value * cfg.risk_per_trade

        # V1 boosts (inherited)
        if cfg.seg_bsp_boost and pending.seg_confirmed:
            per_position_cap *= cfg.seg_bsp_boost_factor
            risk_budget *= cfg.seg_bsp_boost_factor
        if cfg.daily_bsp_confirm and pending.daily_bsp_confirmed:
            per_position_cap *= 1.5
            risk_budget *= 1.5

        # V2: Divergence-based sizing
        if cfg.divergence_sizing and pending.divergence_rate is not None:
            div_rate = max(pending.divergence_rate, 0.01)
            # Lower divergence_rate = stronger signal = bigger position
            boost = min(
                cfg.divergence_sizing_max_boost,
                cfg.divergence_sizing_base / div_rate,
            )
            boost = max(boost, 0.5)  # don't size DOWN below 0.5x
            per_position_cap *= boost
            risk_budget *= boost

        max_shares_cap = int(per_position_cap / actual_entry)
        max_shares_risk = int(risk_budget / risk_per_share)
        max_shares_cash = int((cash - 100) / (actual_entry + commission_per_share))

        shares = max(1, min(max_shares_cap, max_shares_risk, max_shares_cash))
        total_cost = shares * (actual_entry + commission_per_share)

        if total_cost > cash or shares < 1:
            return None

        return _Position(
            symbol=pending.symbol,
            shares=shares,
            entry_price=actual_entry,
            entry_time=ts,
            entry_bar_idx=0,
            entry_bsp_types=pending.bsp_types,
            entry_bi_low=pending.bi_low,
            high_since_entry=cur_high,
        )

    # --- V2 ZS exhaustion check ---

    @staticmethod
    def _check_zs_exhaustion(
        pos: _Position, cur_close: float, zs_list: list[dict],
        cfg: ChanV2BacktestConfig,
    ) -> bool:
        """Check if position is inside an exhausted ZS (too many oscillations)."""
        for zs in reversed(zs_list):
            # Position must be inside the ZS range
            if zs["low"] <= cur_close <= zs.get("peak_high", zs["high"]):
                osc = zs.get("oscillation_cnt", 0)
                if osc >= cfg.zs_max_oscillations:
                    return True
        return False
