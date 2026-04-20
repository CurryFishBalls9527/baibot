"""Portfolio-level Chan theory backtester with shared capital management.

Bias-corrected design:
- Entries/exits on NEXT bar's open (no same-bar lookahead)
- Gap-through stop fills at bar open, not stop level
- Proper True Range ATR (high/low/close, not close-to-close)
- Randomized event ordering at same-timestamp ties
"""

from __future__ import annotations

import logging
import random
import sys
import time
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

CHAN_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "chan.py"
if str(CHAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CHAN_ROOT))

from Chan import CChan  # noqa: E402
from ChanConfig import CChanConfig  # noqa: E402
from Common.CEnum import AUTYPE, KL_TYPE  # noqa: E402

from Common.CEnum import BI_DIR  # noqa: E402

from .chan_adapter import DuckDBDailyAPI, DuckDBIntradayAPI  # noqa: E402

log = logging.getLogger("chan_portfolio")


@dataclass
class ChanBacktestConfig:
    initial_cash: float = 100_000.0
    intraday_interval_minutes: int = 30
    max_positions: int = 8
    max_position_pct: float = 0.12
    risk_per_trade: float = 0.01
    max_hold_bars: int = 200
    slippage_bps: float = 5.0
    commission_bps: float = 1.0
    exit_mode: str = "zs_structural"
    min_stop_atr: float = 0.0
    zs_stop_confirm_close: bool = True
    breakeven_atr: float = 0.0    # raise stop to entry after +N ATR (0=off)
    trail_tighten_atr: float = 0.0  # switch to tight trail after +N ATR (0=off)
    trail_tighten_mult: float = 2.0  # tight trailing stop multiplier (N × ATR from high)

    # Regime-based exposure scaling (1.0 = no throttling).
    # Use regime_gate for binary entry gating instead (matches live system).
    target_exposure_confirmed_uptrend: float = 1.0
    target_exposure_uptrend_under_pressure: float = 1.0
    target_exposure_market_correction: float = 1.0

    chan_divergence_rate: float = 0.8
    chan_macd_algo: str = "area"
    chan_bs_type: str = "1,1p,2,2s"
    chan_min_zs_cnt: int = 1
    chan_max_bs2_rate: float = 0.618
    chan_bi_strict: bool = True

    require_sure: bool = True
    buy_types: tuple = ("T1", "T2", "T2S")
    min_stop_pct: float = 0.0  # minimum stop distance as % of price (e.g. 0.015 = 1.5%)
    stop_grace_bars: int = 0  # bars before structural stop activates (use safety ATR during grace)
    random_seed: int = 42

    daily_filter: bool = False
    daily_filter_mode: str = "bullish_only"
    daily_filter_use_seg: bool = False
    daily_db_path: str = "research_data/market_data.duckdb"

    filter_macd_zero_axis: bool = True
    filter_bi_min_klu_cnt: int = 0
    filter_divergence_max: float = 0.6
    filter_zs_min_bi_cnt: int = 0

    sepa_filter: bool = False
    vol_sizing: bool = False
    vol_target_atr_pct: float = 0.02  # target ATR% for full position; above this scales down

    rs_filter: bool = False
    rs_top_pct: float = 0.30          # only trade symbols in top N% RS ranking
    rs_lookback_days: int = 126       # ~6 months for RS calculation
    rs_rebalance_days: int = 21       # re-rank every ~1 month

    dead_money_bars: int = 0          # exit if held >= N bars with < min_gain (0=off)
    dead_money_min_gain: float = 0.05

    ma_trend_filter: bool = False     # only buy when close > MA20 > MA60 on the intraday bar set

    regime_gate: bool = False         # suppress new entries when market_score <= regime_min_score
    regime_min_score: int = 4

    # P1: Segment BSP boost — size up when bi buy is confirmed by segment buy
    seg_bsp_boost: bool = False
    seg_bsp_boost_factor: float = 1.5  # multiply position size by this factor

    # P1: Volume divergence — require volume divergence for T1 buys
    vol_divergence_filter: bool = False

    # P1: Hub peak exit — tighten stop to zs.high when price exceeds zs.peak_high
    hub_peak_exit: bool = False

    # P2: Enhanced daily filter — use daily ZS structure instead of simple bi direction
    daily_zs_filter: bool = False     # require price above nearest daily ZS low
    daily_bsp_confirm: bool = False   # boost when intraday buy aligns with daily buy BSP


@dataclass
class ChanPortfolioResult:
    summary: Dict
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    daily_state: pd.DataFrame
    symbol_summary: pd.DataFrame


@dataclass
class _Position:
    symbol: str
    shares: int
    entry_price: float
    entry_time: datetime
    entry_bar_idx: int
    entry_bsp_types: str
    entry_bi_low: float
    high_since_entry: float
    bars_held: int = 0


@dataclass
class _PendingEntry:
    symbol: str
    bsp_types: str
    bi_low: float
    zs_list: list
    atr: float
    signal_time: str
    seg_confirmed: bool = False
    daily_bsp_confirmed: bool = False


@dataclass
class _PendingExit:
    reason: str
    signal_time: str


class PortfolioChanBacktester:

    def __init__(self, config: ChanBacktestConfig | None = None):
        self.config = config or ChanBacktestConfig()

    @staticmethod
    def _intraday_k_type(interval_minutes: int) -> KL_TYPE:
        mapping = {
            5: KL_TYPE.K_5M,
            15: KL_TYPE.K_15M,
            30: KL_TYPE.K_30M,
        }
        try:
            return mapping[int(interval_minutes)]
        except (KeyError, ValueError):
            raise ValueError(
                f"Unsupported intraday interval: {interval_minutes}. Expected one of {tuple(mapping)}."
            )

    def backtest_portfolio(
        self,
        symbols: list[str],
        begin: str,
        end: str,
        db_path: str,
        regime_df: pd.DataFrame | None = None,
    ) -> ChanPortfolioResult:
        cfg = self.config
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
            t0d = time.perf_counter()
            daily_directions = self._preload_daily_direction(symbols, begin, end, cfg)
            loaded = sum(len(v) for v in daily_directions.values())
            log.info("Daily directions: %d symbol-days in %.1fs", loaded, time.perf_counter() - t0d)

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
        events = self._preload_events(symbols, begin, end, chan_cfg, cfg)
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

        pending_entries: dict[str, _PendingEntry] = {}
        pending_exits: dict[str, _PendingExit] = {}

        # Shuffle events at same-timestamp boundaries to remove ordering bias
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
                        entry = self._size_entry(
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
                    # Check sell signal → queue for NEXT bar execution
                    sell_sig = self._check_sell_signal(
                        bsp_list, seen_sell.get(symbol, set()), cfg,
                    )
                    if sell_sig:
                        pending_exits[symbol] = _PendingExit(
                            reason="sell_signal", signal_time=cur_time,
                        )
                    elif pos.bars_held >= cfg.max_hold_bars:
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

                # P2: Daily ZS filter — require price above nearest daily ZS low
                if daily_ok and cfg.daily_zs_filter and daily_directions:
                    dd = self._get_daily_data(daily_directions, symbol, cur_day)
                    if isinstance(dd, dict):
                        d_zs_list = dd.get("daily_zs", [])
                        if d_zs_list:
                            # Price must be above the high of the nearest daily ZS
                            # (i.e., broken out above the daily consolidation zone)
                            nearest_dzs = d_zs_list[-1]  # most recent daily ZS
                            if cur_close < nearest_dzs["high"]:
                                daily_ok = False

                if daily_ok:
                    buy_sig = self._check_buy_signal(
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
                        # P2: Check daily BSP confirmation
                        _daily_bsp = False
                        if cfg.daily_bsp_confirm and daily_directions:
                            dd = self._get_daily_data(daily_directions, symbol, cur_day)
                            if isinstance(dd, dict) and dd.get("has_daily_buy"):
                                _daily_bsp = True

                        pending_entries[symbol] = _PendingEntry(
                            symbol=symbol,
                            bsp_types=buy_sig["types_str"],
                            bi_low=buy_sig["bi_low"],
                            zs_list=zs_list,
                            atr=atr,
                            signal_time=cur_time,
                            seg_confirmed=buy_sig.get("seg_confirmed", False),
                            daily_bsp_confirmed=_daily_bsp,
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

    def _preload_events(
        self, symbols: list[str], begin: str, end: str, chan_cfg: CChanConfig,
        cfg: ChanBacktestConfig | None = None,
    ) -> list[tuple[datetime, str, dict]]:
        if cfg is None:
            cfg = ChanBacktestConfig()
        intraday_k_type = self._intraday_k_type(cfg.intraday_interval_minutes)
        events = []
        for sym in symbols:
            try:
                chan = CChan(
                    code=sym, begin_time=begin, end_time=end,
                    data_src="custom:DuckDBAPI.DuckDB30mAPI",
                    lv_list=[intraday_k_type], config=chan_cfg, autype=AUTYPE.QFQ,
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
                        # P1: Volume divergence metric (only compute when needed)
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

                    # P1: Extract segment-level BSPs (only when needed)
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

                    # Extract MA trend values if computed
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
                log.warning("Failed to load %s: %s", sym, e)

        events.sort(key=lambda x: (x[0], x[1]))
        return events

    @staticmethod
    def _shuffle_ties(events: list, rng: random.Random) -> list:
        """Shuffle events sharing the same timestamp to remove ordering bias."""
        result = []
        i = 0
        while i < len(events):
            j = i
            while j < len(events) and events[j][0] == events[i][0]:
                j += 1
            block = list(events[i:j])
            rng.shuffle(block)
            result.extend(block)
            i = j
        return result

    def _check_stop_exit(
        self, pos: _Position, cur_open: float, cur_close: float,
        cur_low: float, atr: float, zs_list: list[dict],
        cfg: ChanBacktestConfig,
    ) -> tuple[str | None, float]:
        """Check stop-based exits. These execute intrabar (pre-placed orders)."""

        if cfg.exit_mode == "zs_structural":
            in_grace = cfg.stop_grace_bars > 0 and pos.bars_held <= cfg.stop_grace_bars

            if in_grace:
                if atr > 0:
                    safety = pos.entry_price - 5.0 * atr
                    if cur_low <= safety:
                        fill = min(safety, cur_open) if cur_open < safety else safety
                        return "safety_stop", fill
            else:
                # --- Structural stop (floor) ---
                zs_stop = None
                for zs in reversed(zs_list):
                    if zs["low"] < pos.entry_price:
                        # P1: Hub peak exit — if price has exceeded peak_high,
                        # tighten stop to zs.high (hub ceiling) instead of zs.low
                        if (cfg.hub_peak_exit
                                and "peak_high" in zs
                                and pos.high_since_entry > zs["peak_high"]):
                            zs_stop = zs["high"]
                        else:
                            zs_stop = zs["low"]
                        break
                bi_stop = pos.entry_bi_low if pos.entry_bi_low > 0 else None

                structural_stop = None
                if zs_stop is not None and bi_stop is not None:
                    structural_stop = max(zs_stop, bi_stop)
                elif zs_stop is not None:
                    structural_stop = zs_stop
                elif bi_stop is not None:
                    structural_stop = bi_stop

                if structural_stop is not None and atr > 0:
                    min_stop = pos.entry_price - cfg.min_stop_atr * atr
                    if structural_stop > min_stop:
                        structural_stop = min_stop

                # --- Profit-lock: raise effective stop based on unrealized gain ---
                profit_stop = None
                if atr > 0:
                    gain_atr = (pos.high_since_entry - pos.entry_price) / atr
                    if cfg.trail_tighten_atr > 0 and gain_atr >= cfg.trail_tighten_atr:
                        profit_stop = pos.high_since_entry - cfg.trail_tighten_mult * atr
                    elif cfg.breakeven_atr > 0 and gain_atr >= cfg.breakeven_atr:
                        profit_stop = pos.entry_price

                # Effective stop = highest of structural and profit-lock
                effective_stop = structural_stop
                if profit_stop is not None:
                    if effective_stop is None or profit_stop > effective_stop:
                        effective_stop = profit_stop

                if effective_stop is not None and cur_low <= effective_stop:
                    if profit_stop is not None and profit_stop >= (structural_stop or 0):
                        reason = "profit_lock"
                        if cfg.zs_stop_confirm_close and cur_close > effective_stop:
                            pass
                        else:
                            fill = min(effective_stop, cur_open) if cur_open < effective_stop else effective_stop
                            return reason, fill
                    else:
                        if cfg.zs_stop_confirm_close and cur_close > effective_stop:
                            pass
                        else:
                            fill = min(effective_stop, cur_open) if cur_open < effective_stop else effective_stop
                            return "zs_break", fill

                if effective_stop is None and atr > 0:
                    safety = pos.high_since_entry - 5.0 * atr
                    if cur_low <= safety:
                        fill = min(safety, cur_open) if cur_open < safety else safety
                        return "safety_stop", fill
        else:
            if atr > 0:
                trail = pos.high_since_entry - 3.0 * atr
                if cur_low <= trail:
                    fill = min(trail, cur_open) if cur_open < trail else trail
                    return "trail_stop", fill

        return None, 0.0

    @staticmethod
    def _estimate_stop(
        bi_low: float, zs_list: list[dict], cur_price: float, atr: float,
    ) -> float:
        """Estimate the structural stop level for a potential entry."""
        zs_stop = None
        for zs in reversed(zs_list):
            if zs["low"] < cur_price:
                zs_stop = zs["low"]
                break
        bi_stop = bi_low if bi_low > 0 else None

        if zs_stop is not None and bi_stop is not None:
            return max(zs_stop, bi_stop)
        if zs_stop is not None:
            return zs_stop
        if bi_stop is not None:
            return bi_stop
        if atr > 0:
            return cur_price - 3.0 * atr
        return 0.0

    @staticmethod
    def _check_sell_signal(
        bsp_list: list[dict], seen_sell: set, cfg: ChanBacktestConfig,
    ) -> bool:
        for bsp in bsp_list:
            if bsp["is_buy"]:
                continue
            if cfg.require_sure and not bsp["is_sure"]:
                continue
            sig_id = (bsp["klu_idx"], False, bsp["types"])
            if sig_id in seen_sell:
                continue
            seen_sell.add(sig_id)
            return True
        return False

    @staticmethod
    def _check_buy_signal(
        bsp_list: list[dict], buy_types_set: set, seen_buy: set,
        cfg: ChanBacktestConfig,
        seg_bsp_list: list[dict] | None = None,
    ) -> dict | None:
        for bsp in bsp_list:
            if not bsp["is_buy"]:
                continue
            if cfg.require_sure and not bsp["is_sure"]:
                continue
            types_set = set(bsp["types"])
            if not types_set.intersection(buy_types_set):
                continue
            sig_id = (bsp["klu_idx"], True, bsp["types"])
            if sig_id in seen_buy:
                continue
            seen_buy.add(sig_id)

            is_t1 = "T1" in types_set or "T1P" in types_set

            if cfg.filter_macd_zero_axis and is_t1:
                dif = bsp.get("macd_dif")
                dea = bsp.get("macd_dea")
                if dif is not None and dea is not None and not (dif < 0 and dea < 0):
                    continue

            if cfg.filter_bi_min_klu_cnt > 0:
                if bsp.get("bi_klu_cnt", 0) < cfg.filter_bi_min_klu_cnt:
                    continue

            if cfg.filter_divergence_max > 0 and is_t1:
                div = bsp.get("divergence_rate")
                if div is not None and div > cfg.filter_divergence_max:
                    continue

            # P1: Volume divergence — for T1, require vol_metric < 0
            # (volume declining = bearish exhaustion = valid reversal)
            if cfg.vol_divergence_filter and is_t1:
                vol = bsp.get("vol_metric")
                if vol is not None and vol >= 0:
                    continue

            # P1: Check if a segment buy BSP confirms this bi buy
            seg_confirmed = False
            if seg_bsp_list:
                bsp_klu_idx = bsp["klu_idx"]
                for seg_bsp in seg_bsp_list:
                    if seg_bsp["is_buy"] and abs(seg_bsp["klu_idx"] - bsp_klu_idx) <= 5:
                        seg_confirmed = True
                        break

            return {
                "types_str": ",".join(sorted(types_set)),
                "bi_low": bsp["bi_low"],
                "seg_confirmed": seg_confirmed,
                "vol_metric": bsp.get("vol_metric"),
            }
        return None

    def _size_entry(
        self, pending: _PendingEntry, cur_open: float, cur_high: float,
        cfg: ChanBacktestConfig, cash: float, portfolio_value: float,
        ts: datetime,
    ) -> _Position | None:
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

        # P1: Segment BSP boost — increase allocation when confirmed
        if cfg.seg_bsp_boost and pending.seg_confirmed:
            per_position_cap *= cfg.seg_bsp_boost_factor
            risk_budget *= cfg.seg_bsp_boost_factor

        # P2: Daily BSP confirmation boost — 1.5x when daily BSP aligns
        if cfg.daily_bsp_confirm and pending.daily_bsp_confirmed:
            per_position_cap *= 1.5
            risk_budget *= 1.5

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

    @staticmethod
    def _close_position(
        pos: _Position, actual_exit: float, cur_time: str,
        exit_reason: str, cfg: ChanBacktestConfig,
        all_trades: list[dict], positions: dict,
    ):
        commission = abs(pos.shares * actual_exit * cfg.commission_bps / 10000)
        pnl = (actual_exit - pos.entry_price) * pos.shares - commission
        return_pct = (actual_exit / pos.entry_price) - 1.0

        all_trades.append({
            "symbol": pos.symbol,
            "entry_time": pos.entry_time.isoformat(),
            "exit_time": cur_time,
            "entry_price": round(pos.entry_price, 4),
            "exit_price": round(actual_exit, 4),
            "shares": pos.shares,
            "pnl": round(pnl, 2),
            "return_pct": round(return_pct, 4),
            "bars_held": pos.bars_held,
            "exit_reason": exit_reason,
            "bsp_types": pos.entry_bsp_types,
        })
        del positions[pos.symbol]

    @staticmethod
    def _get_regime(
        ts: datetime, regime_dates: list | None, regime_values: list | None,
    ) -> str:
        if regime_dates is None or regime_values is None:
            return "confirmed_uptrend"
        date = pd.Timestamp(ts.date())
        idx = bisect_right(regime_dates, date) - 1
        if idx < 0:
            return "uptrend_under_pressure"
        return regime_values[idx]

    @staticmethod
    def _get_regime_score(
        ts: datetime, regime_dates: list | None, regime_scores: list | None,
    ) -> int | None:
        if regime_dates is None or regime_scores is None:
            return None
        date = pd.Timestamp(ts.date())
        idx = bisect_right(regime_dates, date) - 1
        if idx < 0:
            return None
        val = regime_scores[idx]
        return int(val) if pd.notna(val) else None

    @staticmethod
    def _target_exposure(regime: str, cfg: ChanBacktestConfig) -> float:
        mapping = {
            "confirmed_uptrend": cfg.target_exposure_confirmed_uptrend,
            "uptrend_under_pressure": cfg.target_exposure_uptrend_under_pressure,
            "market_correction": cfg.target_exposure_market_correction,
        }
        return mapping.get(regime, cfg.target_exposure_uptrend_under_pressure)

    def _preload_daily_direction(
        self, symbols: list[str], begin: str, end: str, cfg: ChanBacktestConfig,
    ) -> dict[str, dict[str, str]]:
        """Run daily Chan per symbol, return {symbol: {date_str: "bullish"|"bearish"|"neutral"}}."""
        DuckDBDailyAPI.DB_PATH = cfg.daily_db_path
        if not Path(cfg.daily_db_path).exists():
            log.warning("Daily DB not found: %s — skipping daily filter", cfg.daily_db_path)
            return {}

        daily_cfg = CChanConfig({
            "trigger_step": True,
            "bi_strict": cfg.chan_bi_strict,
            "divergence_rate": cfg.chan_divergence_rate,
            "macd_algo": cfg.chan_macd_algo,
            "print_warning": False,
            "zs_algo": "normal",
        })

        extract_zs = cfg.daily_zs_filter or cfg.daily_bsp_confirm

        result: dict[str, dict[str, str]] = {}
        for sym in symbols:
            try:
                chan = CChan(
                    code=sym, begin_time=begin, end_time=end,
                    data_src="custom:DuckDBDailyAPI.DuckDBDailyAPI",
                    lv_list=[KL_TYPE.K_DAY], config=daily_cfg, autype=AUTYPE.QFQ,
                )
                sym_dir: dict[str, str] = {}
                for snapshot in chan.step_load():
                    lvl = snapshot[0]
                    if not lvl.lst:
                        continue
                    ckline = lvl.lst[-1]
                    cur_klu = ckline.lst[-1]
                    ct = cur_klu.time
                    date_str = f"{ct.year:04d}-{ct.month:02d}-{ct.day:02d}"

                    last_sure_dir = None
                    if cfg.daily_filter_use_seg:
                        for seg in reversed(list(lvl.seg_list)):
                            if seg.is_sure:
                                last_sure_dir = seg.dir
                                break
                    else:
                        for bi in reversed(list(lvl.bi_list)):
                            if bi.is_sure:
                                last_sure_dir = bi.dir
                                break

                    if last_sure_dir == BI_DIR.UP:
                        direction = "bullish"
                    elif last_sure_dir == BI_DIR.DOWN:
                        direction = "bearish"
                    else:
                        direction = "neutral"

                    if extract_zs:
                        # Extract daily ZS levels and BSPs
                        daily_zs = []
                        for zs in lvl.zs_list:
                            daily_zs.append({
                                "low": float(zs.low),
                                "high": float(zs.high),
                            })
                        close = float(cur_klu.close)
                        # Check if latest BSP is a buy (within last 5 daily bars)
                        has_daily_buy = False
                        try:
                            bsp_list = snapshot.get_latest_bsp(idx=0, number=10)
                            for bsp in bsp_list:
                                if bsp.is_buy and bsp.bi.is_sure:
                                    has_daily_buy = True
                                    break
                        except Exception:
                            pass
                        sym_dir[date_str] = {
                            "direction": direction,
                            "daily_zs": daily_zs,
                            "close": close,
                            "has_daily_buy": has_daily_buy,
                        }
                    else:
                        sym_dir[date_str] = direction

                result[sym] = sym_dir
            except Exception as e:
                log.warning("Daily Chan failed for %s: %s", sym, e)

        return result

    @staticmethod
    def _get_daily_data(
        daily_directions: dict, symbol: str, bar_date: str,
    ):
        """Look up daily data for a symbol on a given date.

        Returns either a string ("bullish"/"bearish"/"neutral") or a dict
        with rich daily structure, depending on the preload mode.
        """
        sym_dir = daily_directions.get(symbol)
        if not sym_dir:
            return "neutral"
        if bar_date in sym_dir:
            return sym_dir[bar_date]
        dates = sorted(sym_dir.keys())
        idx = bisect_right(dates, bar_date) - 1
        if idx < 0:
            return "neutral"
        return sym_dir[dates[idx]]

    @staticmethod
    def _get_daily_direction(
        daily_directions: dict, symbol: str, bar_date: str,
    ) -> str:
        """Look up daily direction string for a symbol on a given date."""
        data = PortfolioChanBacktester._get_daily_data(
            daily_directions, symbol, bar_date,
        )
        if isinstance(data, dict):
            return data.get("direction", "neutral")
        return data

    def _preload_sepa_trend(
        self, symbols: list[str], begin: str, end: str, cfg: ChanBacktestConfig,
    ) -> dict[str, dict[str, bool]]:
        """Precompute daily SEPA trend template pass/fail for each symbol-day.

        Checks: close > SMA50 > SMA150 > SMA200, SMA200 trending up,
        price ≥30% above 52w low, price within 25% of 52w high.
        """
        import duckdb

        db_path = cfg.daily_db_path
        if not Path(db_path).exists():
            log.warning("Daily DB not found: %s — skipping SEPA filter", db_path)
            return {}

        lookback_begin = pd.Timestamp(begin) - pd.Timedelta(days=400)
        lb_str = lookback_begin.strftime("%Y-%m-%d")

        conn = duckdb.connect(db_path, read_only=True)
        result: dict[str, dict[str, bool]] = {}

        for sym in symbols:
            try:
                rows = conn.execute(
                    "SELECT trade_date, close, high, low FROM daily_bars "
                    "WHERE symbol = ? AND trade_date >= ? AND trade_date <= ? "
                    "ORDER BY trade_date",
                    [sym, lb_str, end],
                ).fetchdf()
                if rows.empty or len(rows) < 252:
                    continue

                rows["trade_date"] = pd.to_datetime(rows["trade_date"])
                rows = rows.set_index("trade_date").sort_index()
                c = rows["close"]
                h = rows["high"]
                l = rows["low"]

                sma50 = c.rolling(50).mean()
                sma150 = c.rolling(150).mean()
                sma200 = c.rolling(200).mean()
                sma200_20ago = sma200.shift(20)
                high_52w = h.rolling(252).max()
                low_52w = l.rolling(252).min()

                passes = (
                    (c > sma50) & (sma50 > sma150) & (sma150 > sma200)
                    & (sma200 > sma200_20ago)
                    & (c >= low_52w * 1.30)
                    & (c >= high_52w * 0.75)
                )

                mask = rows.index >= pd.Timestamp(begin)
                sym_result = {}
                for dt, val in passes[mask].items():
                    date_str = dt.strftime("%Y-%m-%d")
                    sym_result[date_str] = bool(val)
                result[sym] = sym_result
            except Exception as e:
                log.warning("SEPA preload failed for %s: %s", sym, e)

        conn.close()
        return result

    @staticmethod
    def _check_sepa_pass(
        sepa_data: dict[str, dict[str, bool]], symbol: str, bar_date: str,
    ) -> bool:
        sym_data = sepa_data.get(symbol)
        if not sym_data:
            return True
        if bar_date in sym_data:
            return sym_data[bar_date]
        dates = sorted(sym_data.keys())
        idx = bisect_right(dates, bar_date) - 1
        if idx < 0:
            return True
        return sym_data[dates[idx]]

    def _preload_rs_ranking(
        self, symbols: list[str], begin: str, end: str, cfg: ChanBacktestConfig,
    ) -> dict[str, set[str]]:
        """Precompute which symbols pass the RS percentile filter on each rebalance date.

        Returns {date_str: set_of_qualifying_symbols}. RS = stock's 6-month
        return / SPY's 6-month return, ranked across the universe.
        """
        import duckdb
        import numpy as np

        db_path = cfg.daily_db_path
        if not Path(db_path).exists():
            log.warning("Daily DB not found: %s — skipping RS filter", db_path)
            return {}

        lookback_begin = pd.Timestamp(begin) - pd.Timedelta(days=cfg.rs_lookback_days + 30)
        lb_str = lookback_begin.strftime("%Y-%m-%d")

        conn = duckdb.connect(db_path, read_only=True)

        spy_df = conn.execute(
            "SELECT trade_date, close FROM daily_bars "
            "WHERE symbol = 'SPY' AND trade_date >= ? AND trade_date <= ? "
            "ORDER BY trade_date",
            [lb_str, end],
        ).fetchdf()
        if spy_df.empty:
            log.warning("No SPY data in daily DB — skipping RS filter")
            conn.close()
            return {}
        spy_df["trade_date"] = pd.to_datetime(spy_df["trade_date"])
        spy_df = spy_df.set_index("trade_date").sort_index()
        spy_ret = spy_df["close"].pct_change(cfg.rs_lookback_days)

        sym_returns: dict[str, pd.Series] = {}
        for sym in symbols:
            try:
                rows = conn.execute(
                    "SELECT trade_date, close FROM daily_bars "
                    "WHERE symbol = ? AND trade_date >= ? AND trade_date <= ? "
                    "ORDER BY trade_date",
                    [sym, lb_str, end],
                ).fetchdf()
                if rows.empty or len(rows) < cfg.rs_lookback_days:
                    continue
                rows["trade_date"] = pd.to_datetime(rows["trade_date"])
                rows = rows.set_index("trade_date").sort_index()
                sym_returns[sym] = rows["close"].pct_change(cfg.rs_lookback_days)
            except Exception:
                continue
        conn.close()

        trade_dates = sorted(spy_ret.dropna().index)
        trade_dates = [d for d in trade_dates if d >= pd.Timestamp(begin)]

        rebalance_dates = trade_dates[::cfg.rs_rebalance_days]
        if not rebalance_dates:
            return {}

        result: dict[str, set[str]] = {}
        for rdate in rebalance_dates:
            spy_r = spy_ret.get(rdate)
            if spy_r is None or np.isnan(spy_r):
                continue
            rs_scores: list[tuple[str, float]] = []
            for sym, ret_series in sym_returns.items():
                sym_r = ret_series.get(rdate)
                if sym_r is not None and not np.isnan(sym_r) and spy_r != 0:
                    rs_scores.append((sym, sym_r / spy_r if spy_r > 0 else sym_r - spy_r))
            if not rs_scores:
                continue
            rs_scores.sort(key=lambda x: x[1], reverse=True)
            cutoff = max(1, int(len(rs_scores) * cfg.rs_top_pct))
            qualifying = {sym for sym, _ in rs_scores[:cutoff]}
            result[rdate.strftime("%Y-%m-%d")] = qualifying

        log.info("RS ranking: %d rebalance dates, avg %.0f qualifying symbols",
                 len(result),
                 sum(len(v) for v in result.values()) / max(1, len(result)))
        return result

    @staticmethod
    def _check_rs_pass(
        rs_data: dict[str, set[str]], symbol: str, bar_date: str,
    ) -> bool:
        if not rs_data:
            return True
        dates = sorted(rs_data.keys())
        idx = bisect_right(dates, bar_date) - 1
        if idx < 0:
            return True
        return symbol in rs_data[dates[idx]]

    @staticmethod
    def _true_atr(hlc_history: list[tuple[float, float, float]], period: int) -> float:
        """Proper ATR using True Range: max(H-L, |H-prevC|, |L-prevC|)."""
        n = len(hlc_history)
        if n < period + 1:
            return 0.0
        trs = []
        for i in range(n - period, n):
            h, l, c = hlc_history[i]
            prev_c = hlc_history[i - 1][2]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            trs.append(tr)
        return sum(trs) / len(trs)

    def _build_result(
        self, trades: list[dict], equity_curve: list[dict],
        daily_state: list[dict], cfg: ChanBacktestConfig,
    ) -> ChanPortfolioResult:
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        eq_df = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame()
        ds_df = pd.DataFrame(daily_state) if daily_state else pd.DataFrame()

        total = len(trades)
        wins = sum(1 for t in trades if t["return_pct"] > 0)
        losses = total - wins

        final_equity = eq_df["equity"].iloc[-1] if not eq_df.empty else cfg.initial_cash
        total_return = (final_equity / cfg.initial_cash - 1) * 100

        max_dd = 0.0
        if not eq_df.empty:
            peak = eq_df["equity"].expanding().max()
            dd = (eq_df["equity"] - peak) / peak
            max_dd = round(float(dd.min()) * 100, 2)

        by_exit = {}
        by_type = {}
        for t in trades:
            reason = t["exit_reason"]
            by_exit.setdefault(reason, []).append(t["return_pct"])
            for tp in t["bsp_types"].split(","):
                by_type.setdefault(tp, []).append(t["return_pct"])

        summary = {
            "initial_cash": cfg.initial_cash,
            "final_equity": round(final_equity, 2),
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": max_dd,
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total, 4) if total else 0,
            "avg_return": round(sum(t["return_pct"] for t in trades) / total, 6) if total else 0,
            "avg_win": round(sum(t["return_pct"] for t in trades if t["return_pct"] > 0) / wins, 6) if wins else 0,
            "avg_loss": round(sum(t["return_pct"] for t in trades if t["return_pct"] <= 0) / losses, 6) if losses else 0,
            "avg_bars_held": round(sum(t["bars_held"] for t in trades) / total, 1) if total else 0,
            "by_exit_reason": {
                k: {"count": len(v), "avg_ret": round(sum(v) / len(v), 6),
                     "win_rate": round(sum(1 for x in v if x > 0) / len(v), 4)}
                for k, v in sorted(by_exit.items())
            },
            "by_entry_type": {
                k: {"count": len(v), "avg_ret": round(sum(v) / len(v), 6),
                     "win_rate": round(sum(1 for x in v if x > 0) / len(v), 4)}
                for k, v in sorted(by_type.items())
            },
        }

        sym_stats = {}
        for t in trades:
            sym = t["symbol"]
            sym_stats.setdefault(sym, []).append(t)
        sym_rows = []
        for sym, tlist in sorted(sym_stats.items()):
            w = sum(1 for t in tlist if t["return_pct"] > 0)
            sym_rows.append({
                "symbol": sym,
                "trades": len(tlist),
                "wins": w,
                "win_rate": round(w / len(tlist), 4),
                "total_pnl": round(sum(t["pnl"] for t in tlist), 2),
                "avg_return": round(sum(t["return_pct"] for t in tlist) / len(tlist), 6),
            })
        sym_df = pd.DataFrame(sym_rows) if sym_rows else pd.DataFrame()

        return ChanPortfolioResult(
            summary=summary,
            trades=trades_df,
            equity_curve=eq_df,
            daily_state=ds_df,
            symbol_summary=sym_df,
        )
