"""Daily-bar Chan strategy for liquid macro/index ETFs.

Fresh design — does not inherit chan_v1/chan_v2 intraday assumptions.
Signals are computed on daily bars; entries fill at next-day open;
exits are stop-hit (structural bi-low capped at N×ATR), T1 SELL
signal, or time stop.

Reuses:
  - DuckDBDailyAPI from chan_adapter for daily bar feeding
  - CChan engine for bi/seg/ZS/BSP detection
  - ChanPortfolioResult shape for downstream tooling

See plans/i-think-for-chan-bright-pearl.md for design rationale.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

# chan engine path injection (matches chan_backtester.py pattern)
_CHAN_REPO = Path(__file__).resolve().parents[2] / "third_party" / "chan.py"
if str(_CHAN_REPO) not in sys.path:
    sys.path.insert(0, str(_CHAN_REPO))

from Chan import CChan  # noqa: E402
from ChanConfig import CChanConfig  # noqa: E402
from Common.CEnum import AUTYPE, KL_TYPE  # noqa: E402

from .chan_adapter import DuckDBDailyAPI, DuckDBWeeklyAPI  # noqa: E402
from .chan_backtester import ChanPortfolioResult  # noqa: E402

log = logging.getLogger("chan_daily")


@dataclass
class ChanDailyBacktestConfig:
    """Daily-bar Chan strategy parameters (MVP — no v2 overlays)."""

    # Capital / portfolio
    initial_cash: float = 100_000.0
    max_positions: int = 4
    sizing_mode: str = "fixed"   # "fixed" | "atr_parity"
    position_pct: float = 0.10   # in fixed mode: notional per position. In atr_parity: max cap.
    risk_per_trade: float = 0.005  # atr_parity only: equity at risk per trade (0.5%)

    # Chan signal generation
    # bs_type values: '1', '1p', '2', '2s', '3a', '3b' (engine config)
    # buy_types names: 'T1', 'T1P', 'T2', 'T2S', 'T3A', 'T3B' (BSP_TYPE.name)
    chan_bs_type: str = "1,2,2s,3a,3b"
    buy_types: tuple = ("T1", "T2", "T2S", "T3A", "T3B")
    sell_types: tuple = ("T1", "T2", "T2S", "T3A", "T3B")  # sell BSPs that trigger shorts
    chan_macd_algo: str = "area"
    chan_divergence_rate: float = 0.6
    chan_min_zs_cnt: int = 1
    chan_bi_strict: bool = True
    require_sure: bool = True   # set False to act on BSPs on the live (unsure) bi

    # Direction
    enable_longs: bool = True
    enable_shorts: bool = True

    # Risk
    atr_period: int = 20
    stop_atr_mult: float = 2.0
    structural_stop: bool = True   # use bi-low/bi-high if closer than ATR cap
    time_stop_bars: int = 60       # ~3 months daily
    # Per-signal-type stops (Test 3): when > 0, override stop_atr_mult / time_stop_bars
    # for seg-bsp branch entries (entry types_str starts with "seg:").
    # 0 = use main values for both branches.
    stop_atr_mult_seg: float = 0.0
    time_stop_bars_seg: int = 0

    # Vol-adaptive exit overlay. When ATR(today) / ATR(entry) >= vol_expansion_ratio,
    # take action. Modes:
    #   "off"          — no-op
    #   "tighten_stop" — set stop to max(current_stop, close - vol_tightened_atr_mult * atr_today)
    #   "exit"         — close at next open immediately
    vol_adaptive_exit_mode: str = "off"
    vol_expansion_ratio: float = 1.5
    vol_tightened_atr_mult: float = 1.0

    # Re-entry after stop overlay. After a stop-loss exit, if price recovers above
    # the original entry-trigger level within reentry_window_bars, allow ONE re-entry
    # without requiring a fresh BSP/Donchian signal. Theory: noise stops in trends
    # leave alpha on the table — the trend isn't broken, only intraday noise hit stop.
    reentry_after_stop_enabled: bool = False
    reentry_window_bars: int = 5
    reentry_max_count: int = 1   # max re-entries per symbol within the window

    # Portfolio-level vol target. Compute realized vol of equity curve over the
    # last lookback days. If realized vol exceeds target, scale new-entry sizing
    # down proportionally (target/actual, clipped to min/max). 0.0 = disabled.
    portfolio_vol_target: float = 0.0   # annualized; 0.12 = 12%
    portfolio_vol_lookback: int = 30    # trading days
    portfolio_vol_scale_min: float = 0.5
    portfolio_vol_scale_max: float = 1.0

    # Trailing stop overlay (off by default).
    # When profit reaches breakeven_at_r * initial_risk, raise stop to entry price.
    # When profit reaches trail_at_r * initial_risk, trail stop at high - trail_atr_mult * ATR_today.
    trailing_stop_enabled: bool = False
    trail_breakeven_r: float = 2.0     # at +2R profit, move stop to breakeven
    trail_at_r: float = 4.0             # at +4R profit, start ATR-trailing
    trail_atr_mult: float = 1.5         # ATR multiplier for trail distance

    # Partial profit-taking overlay (off by default).
    # When profit reaches partial_at_r * initial_risk, exit partial_pct of the position.
    partial_exit_enabled: bool = False
    partial_at_r: float = 2.0           # at +2R profit, take partial off
    partial_pct: float = 0.5            # fraction of position to exit

    # Trend-overlay gate: only take Chan signals when aligned with longer-term trend.
    # 0 = off; 200 = require close > SMA(200) for longs (and < SMA(200) for shorts).
    trend_sma_period: int = 0

    # Cross-sectional momentum overlay: only take entries on top-K symbols by N-day return.
    # 0 = off; 63 = use 3-month rolling return ranking, top_k controls how many qualify.
    momentum_rank_lookback: int = 0
    momentum_rank_top_k: int = 4

    # Entry mode:
    #   "chan_bsp"        — bi-level Chan BSPs (default; matches all prior runs)
    #   "donchian"        — Donchian-N close breakout
    #   "seg_bsp"         — SEG-level Chan BSPs (lvl.seg_bs_point_lst); test of "段-level meaning"
    #   "any_bsp"         — fire on either bi-level or seg-level BSP (union)
    #   "donchian_or_seg" — fire on EITHER Donchian breakout OR seg-level BSP
    #   "zs_boundary"     — buy when bar retraces into a confirmed ZS while higher-level segseg
    #                        is up (canonical Chan ZS-edge trade). Auto-applies segseg gate.
    # Sell exits still use bi-level Chan T1 sell signals regardless.
    entry_mode: str = "chan_bsp"
    donchian_period: int = 20
    # In "donchian_or_seg" mode, whether the cross-sectional momentum filter applies
    # to BOTH branches. When False, momentum filter only gates Donchian entries
    # (seg-level BSPs bypass momentum — they're already structurally selective).
    momentum_filter_seg_branch: bool = False

    # Segseg gating (chan.py-internal higher-level structure; same data source as base level).
    # When True, long entries require the last segseg.dir == "up", short entries == "down".
    segseg_filter_enabled: bool = False
    # ZS-state gating: only allow long entries when the most recent ZS is in divergence
    # (i.e., MACD divergence within the central hub — quality filter for T1-type reversals).
    zs_divergence_required: bool = False
    # Use ZS being broken (end_bi_break) as an additional EXIT signal for longs.
    exit_long_on_zs_broken: bool = False

    # Exits
    exit_long_on_sell_signal: bool = True   # close long when sell BSP fires
    exit_short_on_buy_signal: bool = True   # cover short when buy BSP fires
    exit_on_sell_signal: bool = True        # legacy alias kept for compat (maps to exit_long_on_sell_signal)

    # Future-blanked probe knob: extra days between signal and entry execution.
    # 0 = signal on day D, enter at D+1 open (default).
    # 1 = signal on day D, enter at D+2 open (lag-1 probe).
    entry_lag_extra_days: int = 0

    # Bar level: "daily" (default) or "weekly". Weekly resamples daily bars
    # on the fly via DuckDBWeeklyAPI; CChan operates at K_WEEK.
    # Time stop, momentum lookback, Donchian period are interpreted in BARS
    # at whichever level is selected — caller scales them.
    kline_level: str = "daily"

    # Volume-confirmation gate for Donchian breakouts: require today's volume
    # to exceed `volume_confirm_lookback`-day average × `volume_confirm_mult`.
    # Applies to Donchian branch only (seg_bsp bypasses — already structurally
    # selective). Default off; canonical CTA volume-confirm setting is 1.5×.
    volume_confirm_enabled: bool = False
    volume_confirm_lookback: int = 20
    volume_confirm_mult: float = 1.5

    # Donchian breakout strength filter: require close > donchian_high × (1 + this).
    # 0.0 = current behavior (any margin above breakout). 0.005 = require 0.5% above.
    # Filters "marginal" breakouts that often reverse immediately.
    donchian_breakout_min_pct: float = 0.0

    # Pyramid scale-in: add to winners as MFE grows. Classic CTA technique.
    # Adds at each R-multiple threshold; first add moves stop to breakeven.
    pyramid_enabled: bool = False
    pyramid_thresholds_r: tuple = (1.5, 3.0)
    pyramid_add_fractions: tuple = (0.5, 0.33)
    pyramid_breakeven_after_first_add: bool = True
    # Conditional pyramid: only add when entry-time context matches (Test 1).
    # All three default OFF — enable individually or combined.
    pyramid_donchian_only: bool = False              # only pyramid Donchian-branch entries
    pyramid_require_up_segseg_sure: bool = False     # only pyramid when entry segseg up + sure
    pyramid_require_zs_broken: bool = False          # only pyramid when entry zs_broken == True

    # Equity-curve gate: pause new entries when account equity is in DD vs
    # high-water mark by more than `equity_dd_threshold_pct`. Resume when DD
    # recovers to within `equity_dd_resume_pct`. 0 = off.
    equity_dd_threshold_pct: float = 0.0
    equity_dd_resume_pct: float = 0.0

    # Sector cap: limit concurrent positions in "equity-correlated" group.
    # When set > 0, at most N of the held positions can be in this group.
    # Group symbols are passed via `equity_sector_symbols`.
    equity_sector_max_positions: int = 0
    equity_sector_symbols: tuple = (
        "SPY", "QQQ", "IWM", "DIA",
        "XLF", "XLE", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU",
    )

    # Same-level decomposition (同级别分解): only enter when most recent bi is
    # confirmed UP — i.e., a pullback has bottomed and reversed. Avoids "chasing"
    # entries mid-decline. This is the canonical Chan workflow: signal at level X
    # tells you direction; sub-structure (latest confirmed up-bi) gives timing.
    require_bi_up_at_entry: bool = False

    # Trend-type gate (走势类型): refines segseg gate with ZS-state condition.
    # Modes:
    #   "off"           — no filter (default)
    #   "trend_only"    — block entries when last ZS is active (i.e., in
    #                     consolidation 盘整, not in trend)
    #   "up_segseg_only" — require segseg.dir == UP AND segseg.is_sure
    #                      (confirmed higher-level uptrend)
    #   "up_trend_strict" — both: confirmed up segseg AND no active ZS
    trend_type_filter_mode: str = "off"

    # VIX-based regime gate: when VIX exceeds vix_block_threshold on decision day,
    # block new entries. Optionally also scale sizing inversely (vix_scale_mode).
    vix_filter_enabled: bool = False
    vix_symbol: str = "^VIX"
    vix_block_threshold: float = 30.0  # block new entries if VIX > this
    vix_scale_mode: str = "off"        # "off" | "linear_above_20"

    # Credit-spread regime gate: HYG/LQD ratio (or HYG/TLT) as risk-on/off proxy.
    # When ratio is below its lookback SMA OR has fallen by > drop_pct over lookback,
    # block new entries. Theory: credit spreads are a leading risk indicator,
    # detect bear regimes (2014-16, 2018Q4) without killing V-recoveries (2020-22).
    credit_spread_filter_enabled: bool = False
    credit_spread_numerator: str = "HYG"   # high-yield ETF
    credit_spread_denominator: str = "LQD" # investment-grade ETF (or "TLT" for flight-to-quality)
    credit_spread_lookback: int = 60       # SMA / delta window
    credit_spread_block_mode: str = "below_sma"  # "below_sma" | "negative_delta" | "z_below"
    credit_spread_z_threshold: float = -0.5  # z-score threshold for "z_below" mode
    credit_spread_drop_pct: float = 0.0    # additional buffer (block when below SMA × (1 - drop_pct))

    # Calendar / seasonality gates.
    # sell_in_may: block entries May-Oct (months 5-10), allow Nov-Apr.
    # santa_only: only allow entries in Nov-Jan (extreme: trade only Q4 + Jan).
    # block_months: explicit list of months to block (e.g., [9] = block September).
    # block_pre_fomc_days: block entries within N trading days BEFORE FOMC meetings
    #   (FOMC meeting dates are passed in fomc_dates).
    # block_earnings_days: block entries within N trading days BEFORE/AFTER earnings;
    #   earnings_dates is a dict[symbol, list[date]] (currently no auto-loader; null gate when empty).
    calendar_filter_mode: str = "off"  # "off" | "sell_in_may" | "santa_only" | "block_months"
    calendar_block_months: tuple[int, ...] = ()
    fomc_block_days: int = 0
    fomc_dates: tuple = ()  # tuple of pd.Timestamp or string dates
    # Realized-vol sizing scaler: scale risk_per_trade inversely to recent ATR%/price.
    # When enabled, position sizing reduces in high-vol regimes.
    vol_scale_enabled: bool = False
    vol_scale_target_pct: float = 0.02  # target 2% ATR/price; size scales target/actual

    # Multi-level: gate entries by a higher-timeframe Chan direction.
    # When enabled (and kline_level=="daily"), runs an additional CChan instance at
    # K_WEEK per symbol and only allows long entries when the most recent weekly seg
    # is up (and short entries when down).
    weekly_filter_enabled: bool = False
    # "seg" uses last seg dir; "bi" uses last bi dir (more responsive, noisier).
    # "confirmed_seg" uses last is_sure seg only (laggier).
    weekly_filter_mode: str = "seg"

    # Entry priority when cash binds and multiple signals compete:
    #   "fifo"     — process in queue order (alphabetical-ish; original behavior)
    #   "momentum" — strongest momentum (lowest rank) first (default — order-robust)
    #   "rank_random" — fixed shuffle (deterministic via random_seed)
    entry_priority_mode: str = "momentum"

    # Futures support: per-symbol contract multiplier (e.g., ES=50, NQ=20).
    # When a symbol has a multiplier here, sizing/P&L/MTM treat shares as # contracts.
    # Notional = shares × price × multiplier; P&L = shares × multiplier × (exit - entry).
    contract_multipliers: dict = field(default_factory=dict)
    # Initial margin per contract for futures sizing (e.g., ES=13000).
    # When set, sizing uses min(risk-based, margin-based) instead of notional cap.
    initial_margins: dict = field(default_factory=dict)

    # Data + execution
    daily_db_path: str = "research_data/market_data.duckdb"
    slippage_bps: float = 5.0
    commission_bps: float = 1.0
    random_seed: int = 42


@dataclass
class _DailyPosition:
    symbol: str
    shares: int
    entry_price: float
    entry_date: datetime
    entry_bar_idx: int
    bsp_types: str
    stop_price: float
    high_since_entry: float    # for longs: tracks high; for shorts: also tracks high (informational)
    direction: str = "long"    # for longs: "long"; for shorts: "short"
    low_since_entry: float = 0.0  # for shorts: tracks low (informational)
    bars_held: int = 0
    # Trailing/partial bookkeeping (only used when corresponding overlay enabled)
    initial_risk_per_share: float = 0.0   # entry_price - stop_price (long) or vice versa
    partial_taken: bool = False
    breakeven_set: bool = False
    # Pyramid bookkeeping
    initial_shares: int = 0          # shares at first entry (for pyramid sizing reference)
    pyramid_level: int = 0           # which threshold we've crossed (0 = no add yet)
    # Entry-time context (used by conditional pyramid logic)
    entry_segseg_dir: Optional[str] = None
    entry_segseg_is_sure: Optional[bool] = None
    entry_zs_broken: Optional[bool] = None
    # ATR at entry — used by vol-adaptive exit overlay
    entry_atr: float = 0.0


class PortfolioChanDailyBacktester:
    """Daily Chan portfolio backtester.

    Per-symbol pipeline:
      1. CChan with K_DAY level + DuckDBDailyAPI yields one snapshot per
         daily bar via step_load().
      2. Each snapshot's BSPs are scanned for T1/T3 buy and T1 sell signals.
      3. Entries observed at snapshot D fill at next trading day's open.
      4. Stops checked intraday against next day's high/low.
      5. Sell signals at snapshot D exit at next day's open.
    """

    def __init__(self, config: Optional[ChanDailyBacktestConfig] = None):
        self.config = config or ChanDailyBacktestConfig()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_bars(self, symbols: list[str], begin: str, end: str) -> dict[str, pd.DataFrame]:
        """Load bars per symbol from market_data.duckdb at configured kline_level."""
        cfg = self.config
        if not Path(cfg.daily_db_path).exists():
            raise FileNotFoundError(f"Daily DB not found: {cfg.daily_db_path}")
        conn = duckdb.connect(cfg.daily_db_path, read_only=True)
        out: dict[str, pd.DataFrame] = {}
        try:
            if cfg.kline_level == "weekly":
                # Resample daily bars to weekly via the same logic as DuckDBWeeklyAPI
                # so backtester bars and CChan-fed bars are bit-identical.
                sql = """
                    WITH d AS (
                        SELECT trade_date, open, high, low, close, volume,
                               date_trunc('week', trade_date) AS week_start
                        FROM daily_bars
                        WHERE symbol = ? AND trade_date BETWEEN ? AND ?
                    )
                    SELECT MAX(trade_date) AS trade_date,
                           arg_min(open, trade_date) AS open,
                           MAX(high) AS high,
                           MIN(low) AS low,
                           arg_max(close, trade_date) AS close,
                           SUM(COALESCE(volume, 0)) AS volume
                    FROM d
                    GROUP BY week_start
                    ORDER BY trade_date
                """
            elif cfg.kline_level == "daily":
                sql = (
                    "SELECT trade_date, open, high, low, close, volume "
                    "FROM daily_bars WHERE symbol = ? AND trade_date BETWEEN ? AND ? "
                    "ORDER BY trade_date"
                )
            else:
                raise ValueError(f"Unsupported kline_level: {cfg.kline_level!r}")

            for sym in symbols:
                df = conn.execute(sql, [sym, begin, end]).fetchdf()
                if df.empty:
                    log.warning("No %s bars for %s in [%s, %s]", cfg.kline_level, sym, begin, end)
                    continue
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.set_index("trade_date")
                out[sym] = df
        finally:
            conn.close()
        return out

    @staticmethod
    def _wilder_atr(df: pd.DataFrame, period: int) -> pd.Series:
        """Wells Wilder ATR on daily OHLC."""
        h, l, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([
            (h - l),
            (h - prev_c).abs(),
            (l - prev_c).abs(),
        ], axis=1).max(axis=1)
        # Wilder smoothing = EMA with alpha=1/period
        return tr.ewm(alpha=1.0 / period, adjust=False).mean()

    # ------------------------------------------------------------------
    # Signal extraction
    # ------------------------------------------------------------------

    def _preload_signals(
        self, symbols: list[str], begin: str, end: str, chan_cfg: CChanConfig,
    ) -> dict[str, dict[pd.Timestamp, dict]]:
        """Run daily Chan per symbol, return per-symbol per-date signal dict.

        signals_by_sym[sym][date] = {
            "buy":      {"types_str", "bi_low"}  | None,   # bi-level buy BSP
            "sell":     {"types_str", "bi_high"} | None,   # bi-level sell BSP (T1 by default)
            "seg_buy":  {"types_str", "bi_low"}  | None,   # seg-level buy BSP
            "seg_sell": {"types_str", "bi_high"} | None,   # seg-level sell BSP
            "segseg_dir":     "up" | "down" | None,        # 段套段 direction (higher-level trend)
            "zs_divergence":  bool | None,                  # last ZS has divergence (T1 quality)
            "zs_broken":      bool | None,                  # last ZS end_bi_break (中枢被破)
        }
        """
        cfg = self.config
        if cfg.kline_level == "weekly":
            DuckDBWeeklyAPI.DB_PATH = cfg.daily_db_path
            data_src = "custom:DuckDBWeeklyAPI.DuckDBWeeklyAPI"
            lv = KL_TYPE.K_WEEK
        else:
            DuckDBDailyAPI.DB_PATH = cfg.daily_db_path
            data_src = "custom:DuckDBDailyAPI.DuckDBDailyAPI"
            lv = KL_TYPE.K_DAY
        buy_types_set = set(cfg.buy_types)
        sell_types_set = set(cfg.sell_types)
        out: dict[str, dict[pd.Timestamp, dict]] = {}

        for sym in symbols:
            seen_buy_klu: set[int] = set()
            seen_sell_klu: set[int] = set()
            seen_seg_buy_klu: set[int] = set()
            seen_seg_sell_klu: set[int] = set()
            sym_signals: dict[pd.Timestamp, dict] = {}
            try:
                chan = CChan(
                    code=sym, begin_time=begin, end_time=end,
                    data_src=data_src,
                    lv_list=[lv], config=chan_cfg, autype=AUTYPE.QFQ,
                )
                for snapshot in chan.step_load():
                    try:
                        lvl = snapshot[0]
                        if not lvl.lst:
                            continue
                        ckline = lvl.lst[-1]
                        cur_klu = ckline.lst[-1]
                        ct = cur_klu.time
                        date = pd.Timestamp(year=ct.year, month=ct.month, day=ct.day)
                    except Exception:
                        continue

                    # ----- Bi-level BSPs (existing) -----
                    try:
                        bsp_list = snapshot.get_latest_bsp(idx=0, number=30)
                    except Exception:
                        bsp_list = []

                    buy_match = None
                    sell_match = None
                    for bsp in bsp_list:
                        try:
                            if cfg.require_sure and not bsp.bi.is_sure:
                                continue
                            types_set = {t.name for t in bsp.type}
                            klu_idx = bsp.klu.idx
                        except Exception:
                            continue

                        if bsp.is_buy:
                            if klu_idx in seen_buy_klu:
                                continue
                            if not types_set.intersection(buy_types_set):
                                continue
                            try:
                                bi_low = float(bsp.bi._low())
                            except Exception:
                                continue
                            seen_buy_klu.add(klu_idx)
                            if buy_match is None:
                                buy_match = {
                                    "types_str": ",".join(sorted(types_set)),
                                    "bi_low": bi_low,
                                }
                        else:
                            if klu_idx in seen_sell_klu:
                                continue
                            if not types_set.intersection(sell_types_set):
                                continue
                            try:
                                bi_high = float(bsp.bi._high())
                            except Exception:
                                continue
                            seen_sell_klu.add(klu_idx)
                            if sell_match is None:
                                sell_match = {
                                    "types_str": ",".join(sorted(types_set)),
                                    "bi_high": bi_high,
                                }

                    # ----- Seg-level BSPs (NEW) -----
                    seg_buy_match = None
                    seg_sell_match = None
                    try:
                        seg_bsp_list = lvl.seg_bs_point_lst.getSortedBspList()[-30:]
                    except Exception:
                        seg_bsp_list = []
                    for bsp in seg_bsp_list:
                        try:
                            if cfg.require_sure and not bsp.bi.is_sure:
                                continue
                            types_set = {t.name for t in bsp.type}
                            klu_idx = bsp.klu.idx
                        except Exception:
                            continue
                        if bsp.is_buy:
                            if klu_idx in seen_seg_buy_klu:
                                continue
                            if not types_set.intersection(buy_types_set):
                                continue
                            try:
                                bi_low = float(bsp.bi._low())
                            except Exception:
                                continue
                            seen_seg_buy_klu.add(klu_idx)
                            if seg_buy_match is None:
                                seg_buy_match = {
                                    "types_str": ",".join(sorted(types_set)),
                                    "bi_low": bi_low,
                                }
                        else:
                            if klu_idx in seen_seg_sell_klu:
                                continue
                            if not types_set.intersection(sell_types_set):
                                continue
                            try:
                                bi_high = float(bsp.bi._high())
                            except Exception:
                                continue
                            seen_seg_sell_klu.add(klu_idx)
                            if seg_sell_match is None:
                                seg_sell_match = {
                                    "types_str": ",".join(sorted(types_set)),
                                    "bi_high": bi_high,
                                }

                    # ----- Segseg direction (段套段) + sure flag -----
                    segseg_dir: Optional[str] = None
                    segseg_is_sure: Optional[bool] = None
                    try:
                        if lvl.segseg_list and len(lvl.segseg_list) > 0:
                            ss = lvl.segseg_list[-1]
                            segseg_dir = "up" if ss.dir.name == "UP" else "down"
                            segseg_is_sure = bool(getattr(ss, "is_sure", False))
                    except Exception:
                        segseg_dir = None
                        segseg_is_sure = None

                    # ----- Last bi direction + sure (同级别分解 timing signal) -----
                    last_bi_dir: Optional[str] = None
                    last_bi_sure: Optional[bool] = None
                    try:
                        if lvl.bi_list and len(lvl.bi_list) > 0:
                            lb = lvl.bi_list[-1]
                            last_bi_dir = "up" if lb.dir.name == "UP" else "down"
                            last_bi_sure = bool(getattr(lb, "is_sure", False))
                    except Exception:
                        pass

                    # ----- Last ZS state (中枢 metadata) -----
                    zs_div: Optional[bool] = None
                    zs_broken: Optional[bool] = None
                    zs_high: Optional[float] = None
                    zs_low: Optional[float] = None
                    zs_sure: Optional[bool] = None
                    try:
                        if lvl.zs_list and len(lvl.zs_list) > 0:
                            # Find the most recent CONFIRMED ZS (for boundary trading)
                            last_sure = next((zs for zs in reversed(lvl.zs_list) if zs.is_sure), None)
                            if last_sure is not None:
                                try:
                                    zs_high = float(last_sure.high)
                                    zs_low = float(last_sure.low)
                                    zs_sure = True
                                except Exception:
                                    pass
                            last_zs = lvl.zs_list[-1]
                            try:
                                # is_divergence(config, out_bi) returns (bool, ratio)
                                bsp_buy_cfg = chan_cfg.bs_point_conf.b_conf
                                div_result = last_zs.is_divergence(bsp_buy_cfg)
                                zs_div = bool(div_result[0]) if isinstance(div_result, tuple) else bool(div_result)
                            except Exception:
                                zs_div = None
                            try:
                                zs_broken = bool(last_zs.end_bi_break())
                            except Exception:
                                zs_broken = None
                    except Exception:
                        pass

                    # zs_active: a confirmed (sure) ZS exists and has not been broken yet.
                    # This is the chan-canonical "consolidation" (盘整) marker — when True,
                    # the market is range-bound; when False, in trend.
                    zs_active = bool(zs_sure and zs_broken is False)

                    if (buy_match or sell_match or seg_buy_match or seg_sell_match
                            or segseg_dir is not None or zs_div is not None or zs_broken is not None
                            or zs_high is not None or last_bi_dir is not None):
                        sym_signals[date] = {
                            "buy": buy_match, "sell": sell_match,
                            "seg_buy": seg_buy_match, "seg_sell": seg_sell_match,
                            "segseg_dir": segseg_dir,
                            "segseg_is_sure": segseg_is_sure,
                            "zs_divergence": zs_div,
                            "zs_broken": zs_broken,
                            "zs_high": zs_high,
                            "zs_low": zs_low,
                            "zs_sure": zs_sure,
                            "zs_active": zs_active,
                            "last_bi_dir": last_bi_dir,
                            "last_bi_sure": last_bi_sure,
                        }
            except Exception as e:
                log.warning("Daily Chan failed for %s: %s", sym, e)
                continue

            out[sym] = sym_signals

        n_buy = sum(sum(1 for s in syms.values() if s.get("buy")) for syms in out.values())
        n_sell = sum(sum(1 for s in syms.values() if s.get("sell")) for syms in out.values())
        n_seg_buy = sum(sum(1 for s in syms.values() if s.get("seg_buy")) for syms in out.values())
        n_seg_sell = sum(sum(1 for s in syms.values() if s.get("seg_sell")) for syms in out.values())
        log.info("Pre-loaded signals: bi=%d/%d buy/sell, seg=%d/%d buy/sell across %d symbols",
                 n_buy, n_sell, n_seg_buy, n_seg_sell, len(out))
        return out

    def _preload_weekly_directions(
        self, symbols: list[str], begin: str, end: str, chan_cfg: CChanConfig,
    ) -> dict[str, list[tuple[pd.Timestamp, str]]]:
        """Run weekly Chan per symbol, return per-symbol sorted (week_end, dir) list.

        dir is "up" or "down" derived from the most recent seg/bi at the snapshot
        per cfg.weekly_filter_mode. The returned list is monotonic in week_end so
        callers can bisect by date to find the direction valid as of a target day.
        """
        cfg = self.config
        DuckDBWeeklyAPI.DB_PATH = cfg.daily_db_path
        mode = cfg.weekly_filter_mode
        out: dict[str, list[tuple[pd.Timestamp, str]]] = {}

        for sym in symbols:
            sym_dirs: list[tuple[pd.Timestamp, str]] = []
            try:
                chan = CChan(
                    code=sym, begin_time=begin, end_time=end,
                    data_src="custom:DuckDBWeeklyAPI.DuckDBWeeklyAPI",
                    lv_list=[KL_TYPE.K_WEEK], config=chan_cfg, autype=AUTYPE.QFQ,
                )
                for snapshot in chan.step_load():
                    try:
                        lvl = snapshot[0]
                        if not lvl.lst:
                            continue
                        ckline = lvl.lst[-1]
                        cur_klu = ckline.lst[-1]
                        ct = cur_klu.time
                        date = pd.Timestamp(year=ct.year, month=ct.month, day=ct.day)
                    except Exception:
                        continue

                    direction: Optional[str] = None
                    if mode == "bi":
                        if lvl.bi_list and len(lvl.bi_list) > 0:
                            d = lvl.bi_list[-1].dir
                            direction = "up" if d.name == "UP" else "down"
                    elif mode == "confirmed_seg":
                        for seg in reversed(lvl.seg_list or []):
                            if getattr(seg, "is_sure", False):
                                direction = "up" if seg.dir.name == "UP" else "down"
                                break
                    else:  # "seg" — most recent seg, sure-or-not
                        if lvl.seg_list and len(lvl.seg_list) > 0:
                            d = lvl.seg_list[-1].dir
                            direction = "up" if d.name == "UP" else "down"
                        elif lvl.bi_list and len(lvl.bi_list) > 0:
                            d = lvl.bi_list[-1].dir
                            direction = "up" if d.name == "UP" else "down"

                    if direction is None:
                        continue
                    # Only append when direction or week_end changes (the snapshots
                    # iterate per weekly bar, so dates are monotonic increasing).
                    if not sym_dirs or sym_dirs[-1][0] != date:
                        sym_dirs.append((date, direction))
                    elif sym_dirs[-1][1] != direction:
                        sym_dirs[-1] = (date, direction)
            except Exception as e:
                log.warning("Weekly Chan direction failed for %s: %s", sym, e)
                continue
            out[sym] = sym_dirs

        n_obs = sum(len(v) for v in out.values())
        log.info("Pre-loaded weekly directions: %d observations across %d symbols",
                 n_obs, len(out))
        return out

    @staticmethod
    def _weekly_dir_as_of(
        weekly_dirs: list[tuple[pd.Timestamp, str]], today: pd.Timestamp,
    ) -> Optional[str]:
        """Return the most recent weekly direction with date <= today (None if no data)."""
        if not weekly_dirs:
            return None
        # weekly_dirs is sorted by date asc. Bisect for last entry with date <= today.
        import bisect
        # bisect_right on dates: returns idx of first date > today
        dates = [d for d, _ in weekly_dirs]
        idx = bisect.bisect_right(dates, today) - 1
        if idx < 0:
            return None
        return weekly_dirs[idx][1]

    # ------------------------------------------------------------------
    # Main backtest loop
    # ------------------------------------------------------------------

    def backtest_portfolio(
        self, symbols: list[str], begin: str, end: str,
    ) -> ChanPortfolioResult:
        cfg = self.config
        chan_cfg = CChanConfig({
            "trigger_step": True,
            "bi_strict": cfg.chan_bi_strict,
            "divergence_rate": cfg.chan_divergence_rate,
            "macd_algo": cfg.chan_macd_algo,
            "bs_type": cfg.chan_bs_type,
            "min_zs_cnt": cfg.chan_min_zs_cnt,
            "print_warning": False,
            "zs_algo": "normal",
        })

        bars = self._load_bars(symbols, begin, end)
        symbols = [s for s in symbols if s in bars]  # drop empties
        if not symbols:
            raise ValueError("No symbols had daily bars in the requested range")

        atr_per_sym = {s: self._wilder_atr(bars[s], cfg.atr_period) for s in symbols}
        sma_per_sym: dict[str, pd.Series] = {}
        if cfg.trend_sma_period > 0:
            for s in symbols:
                sma_per_sym[s] = bars[s]["close"].rolling(window=cfg.trend_sma_period).mean()

        # Cross-sectional momentum ranking: per-day rank of each symbol by N-day return.
        # Built as a single aligned DataFrame so we can apply rank-per-row.
        momentum_rank_df: Optional[pd.DataFrame] = None
        if cfg.momentum_rank_lookback > 0:
            ret_frames = {
                s: bars[s]["close"].pct_change(cfg.momentum_rank_lookback)
                for s in symbols
            }
            ret_df = pd.DataFrame(ret_frames)
            # Higher return = higher rank; rank 1 = best
            momentum_rank_df = ret_df.rank(axis=1, method="min", ascending=False)

        # Donchian breakout reference: max of the prior N closes (excludes today).
        donchian_high_per_sym: dict[str, pd.Series] = {}
        donchian_low_per_sym: dict[str, pd.Series] = {}
        if cfg.entry_mode in ("donchian", "donchian_or_seg", "donchian_seg_zs"):
            for s in symbols:
                donchian_high_per_sym[s] = bars[s]["close"].shift(1).rolling(window=cfg.donchian_period).max()
                donchian_low_per_sym[s] = bars[s]["close"].shift(1).rolling(window=cfg.donchian_period).min()

        # Volume confirmation: rolling N-day avg of PRIOR volume (excludes today)
        volume_avg_per_sym: dict[str, pd.Series] = {}
        if cfg.volume_confirm_enabled:
            for s in symbols:
                volume_avg_per_sym[s] = (
                    bars[s]["volume"].shift(1)
                    .rolling(window=cfg.volume_confirm_lookback).mean()
                )

        signals = self._preload_signals(symbols, begin, end, chan_cfg)

        weekly_dirs: dict[str, list[tuple[pd.Timestamp, str]]] = {}
        if cfg.weekly_filter_enabled and cfg.kline_level == "daily":
            weekly_dirs = self._preload_weekly_directions(symbols, begin, end, chan_cfg)
        elif cfg.weekly_filter_enabled and cfg.kline_level != "daily":
            log.warning("weekly_filter_enabled is only meaningful for kline_level=daily; ignoring")

        # Load VIX series if vix gate enabled
        vix_series: Optional[pd.Series] = None
        if cfg.vix_filter_enabled or cfg.vix_scale_mode != "off":
            try:
                conn = duckdb.connect(cfg.daily_db_path, read_only=True)
                df = conn.execute(
                    "SELECT trade_date, close FROM daily_bars WHERE symbol = ? ORDER BY trade_date",
                    [cfg.vix_symbol],
                ).fetchdf()
                conn.close()
                if not df.empty:
                    df["trade_date"] = pd.to_datetime(df["trade_date"])
                    vix_series = df.set_index("trade_date")["close"]
                    log.info("Loaded VIX series: %d obs, range %.1f-%.1f",
                             len(vix_series), vix_series.min(), vix_series.max())
                else:
                    log.warning("VIX symbol %s not found in DB; gate will be no-op", cfg.vix_symbol)
            except Exception as e:
                log.warning("VIX load failed: %s", e)

        # Load credit-spread proxy series (numerator / denominator ratio + rolling SMA)
        credit_ratio: Optional[pd.Series] = None
        credit_sma: Optional[pd.Series] = None
        credit_delta: Optional[pd.Series] = None
        credit_z: Optional[pd.Series] = None
        if cfg.credit_spread_filter_enabled:
            try:
                conn = duckdb.connect(cfg.daily_db_path, read_only=True)
                df_n = conn.execute(
                    "SELECT trade_date, close FROM daily_bars WHERE symbol = ? ORDER BY trade_date",
                    [cfg.credit_spread_numerator],
                ).fetchdf()
                df_d = conn.execute(
                    "SELECT trade_date, close FROM daily_bars WHERE symbol = ? ORDER BY trade_date",
                    [cfg.credit_spread_denominator],
                ).fetchdf()
                conn.close()
                if df_n.empty or df_d.empty:
                    log.warning("Credit-spread series missing (%s or %s not in DB); gate no-op",
                                cfg.credit_spread_numerator, cfg.credit_spread_denominator)
                else:
                    df_n["trade_date"] = pd.to_datetime(df_n["trade_date"])
                    df_d["trade_date"] = pd.to_datetime(df_d["trade_date"])
                    sn = df_n.set_index("trade_date")["close"]
                    sd = df_d.set_index("trade_date")["close"]
                    common = sn.index.intersection(sd.index)
                    credit_ratio = (sn.loc[common] / sd.loc[common]).sort_index()
                    credit_sma = credit_ratio.rolling(cfg.credit_spread_lookback, min_periods=cfg.credit_spread_lookback).mean()
                    credit_delta = credit_ratio - credit_ratio.shift(cfg.credit_spread_lookback)
                    rstd = credit_ratio.rolling(cfg.credit_spread_lookback, min_periods=cfg.credit_spread_lookback).std()
                    credit_z = (credit_ratio - credit_sma) / rstd.replace(0, pd.NA)
                    log.info("Loaded credit-spread series %s/%s: %d obs, lookback=%d, mode=%s",
                             cfg.credit_spread_numerator, cfg.credit_spread_denominator,
                             len(credit_ratio), cfg.credit_spread_lookback, cfg.credit_spread_block_mode)
            except Exception as e:
                log.warning("Credit-spread load failed: %s", e)

        all_dates = sorted({d for s in symbols for d in bars[s].index})
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        cash = cfg.initial_cash
        # Equity-curve high-water mark + DD-pause flag (for equity_dd gate)
        equity_high_water = cfg.initial_cash
        equity_dd_paused = False
        equity_sector_set = set(cfg.equity_sector_symbols)
        positions: dict[str, _DailyPosition] = {}
        pending_entries: list[dict] = []
        pending_exits: list[dict] = []  # sell-signal exits decided yesterday
        trades: list[dict] = []
        equity_curve: list[dict] = []
        # Re-entry tracking: per-symbol list of recent stops (for reentry_after_stop overlay).
        # Each entry: {"stop_bar_idx": int, "entry_price": float, "bsp_types": str,
        #              "ref_price": float, "count": int, "entry_ctx": dict}
        recently_stopped: dict[str, dict] = {}

        slip_buy = 1.0 + cfg.slippage_bps / 10_000.0
        slip_sell = 1.0 - cfg.slippage_bps / 10_000.0
        commission_rate = cfg.commission_bps / 10_000.0

        # Helpers for closing positions in either direction.
        # `return` is symmetric: positive means profit. pnl_dollars = ret × entry × shares
        # works for both directions because long ret=(exit-entry)/entry and
        # short ret=(entry-exit)/entry both use entry as denominator.
        def _mult(sym: str) -> float:
            """Contract multiplier for futures, 1.0 for ETFs/stocks."""
            return cfg.contract_multipliers.get(sym, 1.0)

        def _close_position(sym: str, pos: _DailyPosition, fill_price: float,
                            exit_date, reason: str):
            nonlocal cash
            mult = _mult(sym)
            if pos.direction == "long":
                exit_p = fill_price * slip_sell
                # Cash effect: for ETFs we sold shares (cash in). For futures, cash
                # change = P&L only (no notional ownership transfer).
                if mult == 1.0:
                    cash += pos.shares * exit_p * (1.0 - commission_rate)
                else:
                    pnl = (exit_p - pos.entry_price) * pos.shares * mult
                    commission = pos.shares * exit_p * mult * commission_rate
                    cash += pnl - commission
                ret = (exit_p - pos.entry_price) / pos.entry_price
            else:  # short
                exit_p = fill_price * slip_buy
                if mult == 1.0:
                    cash -= pos.shares * exit_p * (1.0 + commission_rate)
                else:
                    pnl = (pos.entry_price - exit_p) * pos.shares * mult
                    commission = pos.shares * exit_p * mult * commission_rate
                    cash += pnl - commission
                ret = (pos.entry_price - exit_p) / pos.entry_price
            trades.append({
                "symbol": sym, "entry_date": pos.entry_date, "exit_date": exit_date,
                "entry_price": pos.entry_price, "exit_price": exit_p,
                "shares": pos.shares, "direction": pos.direction,
                "return": ret, "bars_held": pos.bars_held,
                "exit_reason": reason, "bsp_types": pos.bsp_types,
            })

        def _partial_exit(sym: str, pos: _DailyPosition, fill_price: float,
                          exit_date, fraction: float):
            """Close `fraction` of position at fill_price; record as a trade row.
            Position remains open with reduced shares. Only longs supported here
            since current candidate is long-only — short logic mirrors trivially.
            """
            nonlocal cash
            if pos.direction != "long":
                return
            exit_shares = int(pos.shares * fraction)
            if exit_shares <= 0:
                return
            exit_p = fill_price * slip_sell
            cash += exit_shares * exit_p * (1.0 - commission_rate)
            ret = (exit_p - pos.entry_price) / pos.entry_price
            trades.append({
                "symbol": sym, "entry_date": pos.entry_date, "exit_date": exit_date,
                "entry_price": pos.entry_price, "exit_price": exit_p,
                "shares": exit_shares, "direction": pos.direction,
                "return": ret, "bars_held": pos.bars_held,
                "exit_reason": "partial", "bsp_types": pos.bsp_types,
            })
            pos.shares -= exit_shares
            pos.partial_taken = True

        for d_idx, today in enumerate(all_dates):
            # ----- 1. Execute pending entries at today's open -----
            # When cash binds, the order matters. By default fifo (alphabetical-ish).
            # Other modes rank entries by quality so the strongest signals get fully
            # funded first instead of whoever happened to be early in the symbol list.
            if cfg.entry_priority_mode == "momentum" and len(pending_entries) > 1:
                pending_entries.sort(key=lambda e: e.get("momentum_rank", 9999.0))
            elif cfg.entry_priority_mode == "rank_random" and len(pending_entries) > 1:
                import random
                rng = random.Random(cfg.random_seed + d_idx)
                pending_entries.sort(key=lambda e: rng.random())
            still_pending: list[dict] = []
            for entry in pending_entries:
                # Lag-probe: hold the entry until ready_after_idx is reached
                if entry.get("ready_after_idx", 0) >= d_idx:
                    still_pending.append(entry)
                    continue
                sym = entry["symbol"]
                direction = entry["direction"]
                if sym not in bars or today not in bars[sym].index:
                    continue
                if sym in positions:
                    continue
                if len(positions) >= cfg.max_positions:
                    continue
                today_bar = bars[sym].loc[today]
                open_p = float(today_bar["open"])

                atr_today = float(atr_per_sym[sym].loc[today]) if today in atr_per_sym[sym].index else 0.0

                # Per-signal-type stop multiplier (Test 3): seg-bsp branch can use
                # a different ATR multiplier (typically tighter — reversal signals
                # should fail/work fast).
                stop_mult_eff = cfg.stop_atr_mult
                if cfg.stop_atr_mult_seg > 0 and str(entry.get("types_str", "")).startswith("seg:"):
                    stop_mult_eff = cfg.stop_atr_mult_seg

                if direction == "long":
                    fill = open_p * slip_buy
                    atr_stop = fill - stop_mult_eff * atr_today
                    if cfg.structural_stop and entry["ref_price"] > 0:
                        stop_price = max(entry["ref_price"], atr_stop)  # tighter wins
                    else:
                        stop_price = atr_stop
                    if stop_price >= fill:
                        continue
                    risk_per_share = fill - stop_price
                else:  # short
                    fill = open_p * slip_sell  # we sell short at the bid-side
                    atr_stop = fill + stop_mult_eff * atr_today
                    if cfg.structural_stop and entry["ref_price"] > 0:
                        stop_price = min(entry["ref_price"], atr_stop)  # tighter (closer above) wins
                    else:
                        stop_price = atr_stop
                    if stop_price <= fill:
                        continue
                    risk_per_share = stop_price - fill

                # Sizing
                # Realized-vol regime scaler: dampens risk_per_trade in high-vol regimes.
                # Scale = target_pct / current_atr_pct, clipped to [0.5, 2.0]
                size_scale = 1.0
                if cfg.vol_scale_enabled and atr_today > 0 and fill > 0:
                    atr_pct = atr_today / fill
                    if atr_pct > 0:
                        size_scale = max(0.5, min(2.0, cfg.vol_scale_target_pct / atr_pct))

                # Portfolio-level vol target: scale down when book vol > target
                if cfg.portfolio_vol_target > 0 and len(equity_curve) >= cfg.portfolio_vol_lookback + 1:
                    eq_vals = [r["equity"] for r in equity_curve[-(cfg.portfolio_vol_lookback + 1):]]
                    rets = []
                    for i in range(1, len(eq_vals)):
                        if eq_vals[i - 1] > 0:
                            rets.append(eq_vals[i] / eq_vals[i - 1] - 1.0)
                    if len(rets) >= 5:
                        import statistics
                        daily_std = statistics.pstdev(rets)
                        ann_vol = daily_std * (252 ** 0.5)
                        if ann_vol > 0:
                            scale = cfg.portfolio_vol_target / ann_vol
                            scale = max(cfg.portfolio_vol_scale_min,
                                        min(cfg.portfolio_vol_scale_max, scale))
                            size_scale *= scale

                mult = _mult(sym)
                # Per-contract risk = risk-per-share × multiplier (futures point P&L)
                contract_risk = risk_per_share * mult
                if cfg.sizing_mode == "atr_parity":
                    risk_dollars = cash * cfg.risk_per_trade * size_scale
                    shares = int(risk_dollars // contract_risk) if contract_risk > 0 else 0
                    if mult == 1.0:
                        # ETF: cap by % notional of cash
                        max_shares_cap = int((cash * cfg.position_pct) // fill)
                        if max_shares_cap > 0:
                            shares = min(shares, max_shares_cap)
                    else:
                        # Futures: cap by initial margin (default $5k if not set)
                        margin_per = cfg.initial_margins.get(sym, 5000.0)
                        # Use up to position_pct of cash for margin per position
                        max_contracts = int((cash * cfg.position_pct) // margin_per)
                        if max_contracts > 0:
                            shares = min(shares, max_contracts)
                else:
                    notional = cash * cfg.position_pct * size_scale
                    if mult == 1.0:
                        shares = int(notional // fill)
                    else:
                        # For futures fixed-mode: use position_pct as fraction of cash for margin
                        margin_per = cfg.initial_margins.get(sym, 5000.0)
                        shares = int(notional // margin_per)
                if shares <= 0:
                    continue

                if mult == 1.0:
                    # ETF cash mechanics
                    if direction == "long":
                        cost = shares * fill * (1.0 + commission_rate)
                        if cost > cash:
                            continue
                        cash -= cost
                    else:
                        proceeds = shares * fill * (1.0 - commission_rate)
                        cash += proceeds
                else:
                    # Futures: post initial margin, no notional cash transfer
                    margin_per = cfg.initial_margins.get(sym, 5000.0)
                    margin_required = shares * margin_per
                    commission = shares * fill * mult * commission_rate
                    if margin_required + commission > cash:
                        continue
                    cash -= commission  # margin posted but tracked separately conceptually

                ectx = entry.get("entry_ctx") or {}
                positions[sym] = _DailyPosition(
                    symbol=sym, shares=shares, entry_price=fill,
                    entry_date=today, entry_bar_idx=d_idx,
                    bsp_types=entry["types_str"], stop_price=stop_price,
                    high_since_entry=float(today_bar["high"]),
                    direction=direction,
                    low_since_entry=float(today_bar["low"]),
                    initial_risk_per_share=risk_per_share,
                    initial_shares=shares,
                    entry_segseg_dir=ectx.get("segseg_dir"),
                    entry_segseg_is_sure=ectx.get("segseg_is_sure"),
                    entry_zs_broken=ectx.get("zs_broken"),
                    entry_atr=atr_today,
                )
            pending_entries = still_pending

            # ----- 2. Execute pending signal-based exits at today's open -----
            for ex in pending_exits:
                sym = ex["symbol"]
                if sym not in positions or today not in bars[sym].index:
                    continue
                pos = positions[sym]
                # Make sure the exit signal applies to the current direction:
                # sell-signal exits longs; buy-signal covers shorts.
                if ex["kind"] == "sell" and pos.direction != "long":
                    continue
                if ex["kind"] == "buy" and pos.direction != "short":
                    continue
                _close_position(sym, pos, float(bars[sym].loc[today, "open"]), today,
                                "sell_signal" if pos.direction == "long" else "buy_signal_cover")
                del positions[sym]
            pending_exits = []

            # ----- 3. For remaining positions, check stops / time stop intraday -----
            to_remove: list[str] = []
            for sym, pos in positions.items():
                if today not in bars[sym].index:
                    continue
                today_bar = bars[sym].loc[today]
                pos.bars_held += 1
                pos.high_since_entry = max(pos.high_since_entry, float(today_bar["high"]))
                pos.low_since_entry = min(pos.low_since_entry or float(today_bar["low"]), float(today_bar["low"]))

                # Trailing-stop / partial-exit overlays (longs only — current candidate)
                if pos.direction == "long" and pos.initial_risk_per_share > 0:
                    profit_per_share = pos.high_since_entry - pos.entry_price
                    r_multiple = profit_per_share / pos.initial_risk_per_share

                    if cfg.partial_exit_enabled and not pos.partial_taken \
                            and r_multiple >= cfg.partial_at_r:
                        # Take partial off at today's open (signal computed end-of-yesterday;
                        # for simplicity execute at today's open)
                        _partial_exit(sym, pos, float(today_bar["open"]), today, cfg.partial_pct)

                    if cfg.trailing_stop_enabled:
                        # Move stop to breakeven once profit >= breakeven_at_r
                        if not pos.breakeven_set and r_multiple >= cfg.trail_breakeven_r:
                            pos.stop_price = max(pos.stop_price, pos.entry_price)
                            pos.breakeven_set = True
                        # ATR-trail stop once profit >= trail_at_r
                        if r_multiple >= cfg.trail_at_r:
                            atr_today = float(atr_per_sym[sym].loc[today]) if today in atr_per_sym[sym].index else 0.0
                            new_stop = pos.high_since_entry - cfg.trail_atr_mult * atr_today
                            pos.stop_price = max(pos.stop_price, new_stop)

                    # Vol-adaptive exit overlay (long only — short side ignored for now).
                    # Detect ATR expansion vs entry; tighten stop or exit immediately.
                    if (cfg.vol_adaptive_exit_mode != "off"
                            and pos.direction == "long"
                            and pos.entry_atr > 0
                            and sym in atr_per_sym):
                        atr_today = float(atr_per_sym[sym].loc[today]) if today in atr_per_sym[sym].index else 0.0
                        if atr_today > 0 and atr_today / pos.entry_atr >= cfg.vol_expansion_ratio:
                            if cfg.vol_adaptive_exit_mode == "tighten_stop":
                                close_today = float(today_bar["close"])
                                new_stop = close_today - cfg.vol_tightened_atr_mult * atr_today
                                pos.stop_price = max(pos.stop_price, new_stop)
                            elif cfg.vol_adaptive_exit_mode == "exit":
                                # Schedule exit at next bar's open via sell-signal queue
                                pending_exits.append({"symbol": sym, "kind": "sell"})

                    # Pyramid scale-in: add to position when MFE crosses thresholds
                    # Conditional gates (Test 1): skip pyramid if entry-time context
                    # doesn't match enabled requirements.
                    pyramid_ctx_ok = True
                    if cfg.pyramid_enabled:
                        if cfg.pyramid_donchian_only:
                            # bsp_types is "donchian" for Donchian-branch entries; "seg:T2S" etc for seg-bsp
                            if not str(pos.bsp_types).startswith("donchian"):
                                pyramid_ctx_ok = False
                        if pyramid_ctx_ok and cfg.pyramid_require_up_segseg_sure:
                            if not (pos.entry_segseg_dir == "up" and bool(pos.entry_segseg_is_sure)):
                                pyramid_ctx_ok = False
                        if pyramid_ctx_ok and cfg.pyramid_require_zs_broken:
                            if not bool(pos.entry_zs_broken):
                                pyramid_ctx_ok = False

                    if cfg.pyramid_enabled and pyramid_ctx_ok and pos.initial_shares > 0:
                        # Check next pyramid threshold (if any remaining)
                        while pos.pyramid_level < len(cfg.pyramid_thresholds_r):
                            threshold_r = cfg.pyramid_thresholds_r[pos.pyramid_level]
                            if r_multiple < threshold_r:
                                break
                            # Add!
                            add_frac = cfg.pyramid_add_fractions[pos.pyramid_level]
                            add_shares = int(pos.initial_shares * add_frac)
                            if add_shares <= 0:
                                pos.pyramid_level += 1
                                continue
                            add_price = float(today_bar["open"]) * slip_buy
                            cost = add_shares * add_price * (1.0 + commission_rate)
                            if cost > cash:
                                # Not enough cash for the add — skip but advance level so we don't retry
                                pos.pyramid_level += 1
                                continue
                            cash -= cost
                            # Recompute weighted avg entry
                            new_total = pos.shares + add_shares
                            pos.entry_price = (pos.entry_price * pos.shares + add_price * add_shares) / new_total
                            pos.shares = new_total
                            pos.pyramid_level += 1
                            # Move stop to breakeven on first add
                            if pos.pyramid_level == 1 and cfg.pyramid_breakeven_after_first_add:
                                pos.stop_price = max(pos.stop_price, pos.entry_price)
                                pos.breakeven_set = True

                if pos.direction == "long":
                    if float(today_bar["low"]) <= pos.stop_price:
                        open_p = float(today_bar["open"])
                        # Gap below stop fills at open; else at stop
                        fill_price = min(pos.stop_price, open_p)
                        if cfg.reentry_after_stop_enabled:
                            recently_stopped[sym] = {
                                "stop_bar_idx": d_idx,
                                "entry_price": pos.entry_price,
                                "bsp_types": pos.bsp_types,
                                "count": 0,
                                "entry_ctx": {
                                    "segseg_dir": pos.entry_segseg_dir,
                                    "segseg_is_sure": pos.entry_segseg_is_sure,
                                    "zs_broken": pos.entry_zs_broken,
                                },
                            }
                        _close_position(sym, pos, fill_price, today, "stop")
                        to_remove.append(sym)
                        continue
                else:  # short
                    if float(today_bar["high"]) >= pos.stop_price:
                        open_p = float(today_bar["open"])
                        # Gap above stop covers at open; else at stop
                        fill_price = max(pos.stop_price, open_p)
                        _close_position(sym, pos, fill_price, today, "stop")
                        to_remove.append(sym)
                        continue

                # Time stop — exit at today's close. Per-signal-type override (Test 3):
                # seg-bsp branch may use a shorter time stop.
                ts_bars_eff = cfg.time_stop_bars
                if cfg.time_stop_bars_seg > 0 and str(pos.bsp_types).startswith("seg:"):
                    ts_bars_eff = cfg.time_stop_bars_seg
                if pos.bars_held >= ts_bars_eff:
                    _close_position(sym, pos, float(today_bar["close"]), today, "time_stop")
                    to_remove.append(sym)
            for sym in to_remove:
                del positions[sym]

            # ----- 4. Use today's snapshot to queue tomorrow's actions -----
            # NOTE: in donchian entry mode, the buy/sell signal source for ENTRIES is
            # the breakout, so we cannot short-circuit on "no Chan sig". The exit
            # decision still consults Chan T1 sell/buy signals.
            for sym in symbols:
                sig = signals.get(sym, {}).get(today) or {}
                # Exits queued for the position's opposite signal
                if sym in positions:
                    pos = positions[sym]
                    if pos.direction == "long" and sig.get("sell") and (
                        cfg.exit_long_on_sell_signal or cfg.exit_on_sell_signal
                    ):
                        pending_exits.append({"symbol": sym, "kind": "sell"})
                    elif pos.direction == "long" and cfg.exit_long_on_zs_broken and sig.get("zs_broken"):
                        # Treat ZS-broken as a "sell" signal-style exit at next open
                        pending_exits.append({"symbol": sym, "kind": "sell"})
                    elif pos.direction == "short" and sig.get("buy") and cfg.exit_short_on_buy_signal:
                        pending_exits.append({"symbol": sym, "kind": "buy"})
                    continue

                # Symbol is not currently held — evaluate entry.
                # Trend-overlay gate (if configured). Use day D's close and SMA — the
                # decision is made at end of day D and executes at D+1 open, so this
                # is not lookahead (the bar is already closed when we decide).
                trend_long_ok = True
                trend_short_ok = True
                if cfg.trend_sma_period > 0 and sym in sma_per_sym:
                    sma_today = sma_per_sym[sym].loc[today] if today in sma_per_sym[sym].index else None
                    close_today = float(bars[sym].loc[today, "close"]) if today in bars[sym].index else None
                    if sma_today is None or pd.isna(sma_today) or close_today is None:
                        trend_long_ok = trend_short_ok = False
                    else:
                        trend_long_ok = close_today > float(sma_today)
                        trend_short_ok = close_today < float(sma_today)

                # Cross-sectional momentum gate
                momentum_long_ok = True
                momentum_short_ok = True
                sym_rank_val: float = 9999.0  # default rank when no momentum data
                if momentum_rank_df is not None and today in momentum_rank_df.index:
                    rank_row = momentum_rank_df.loc[today]
                    sym_rank = rank_row.get(sym)
                    n_syms = int(rank_row.notna().sum())
                    if pd.isna(sym_rank):
                        momentum_long_ok = momentum_short_ok = False
                    else:
                        sym_rank_val = float(sym_rank)
                        momentum_long_ok = float(sym_rank) <= cfg.momentum_rank_top_k
                        momentum_short_ok = float(sym_rank) > (n_syms - cfg.momentum_rank_top_k)

                # Multi-level weekly Chan direction gate
                weekly_long_ok = True
                weekly_short_ok = True
                if cfg.weekly_filter_enabled and weekly_dirs:
                    wd = self._weekly_dir_as_of(weekly_dirs.get(sym, []), today)
                    if wd is None:
                        weekly_long_ok = weekly_short_ok = False
                    else:
                        weekly_long_ok = wd == "up"
                        weekly_short_ok = wd == "down"

                # Same-level decomposition gate: only enter when last bi is up + sure
                bi_timing_ok = True
                if cfg.require_bi_up_at_entry:
                    lbd = sig.get("last_bi_dir")
                    lbs = sig.get("last_bi_sure")
                    bi_timing_ok = (lbd == "up") and bool(lbs)

                # Equity-curve drawdown gate (block new entries while paused)
                equity_curve_ok = True
                if cfg.equity_dd_threshold_pct > 0 and equity_dd_paused:
                    equity_curve_ok = False

                # Sector cap: limit concurrent equity-correlated positions
                sector_cap_ok = True
                if cfg.equity_sector_max_positions > 0:
                    held_equity = sum(1 for s in positions if s in equity_sector_set)
                    if sym in equity_sector_set and held_equity >= cfg.equity_sector_max_positions:
                        sector_cap_ok = False

                # Trend-type gate (走势类型)
                trend_type_long_ok = True
                if cfg.trend_type_filter_mode != "off":
                    ssd = sig.get("segseg_dir")
                    ss_sure = sig.get("segseg_is_sure")
                    zs_active = sig.get("zs_active")
                    if cfg.trend_type_filter_mode == "trend_only":
                        # Block during active ZS (盘整 — consolidation)
                        if zs_active:
                            trend_type_long_ok = False
                    elif cfg.trend_type_filter_mode == "up_segseg_only":
                        # Require confirmed up-segseg
                        if not (ssd == "up" and ss_sure):
                            trend_type_long_ok = False
                    elif cfg.trend_type_filter_mode == "up_trend_strict":
                        # Both: confirmed up-segseg AND no active ZS
                        if not (ssd == "up" and ss_sure and not zs_active):
                            trend_type_long_ok = False

                # VIX regime gate
                vix_ok = True
                if cfg.vix_filter_enabled and vix_series is not None:
                    if today in vix_series.index:
                        vix_today = float(vix_series.loc[today])
                    else:
                        prior = vix_series.loc[vix_series.index <= today]
                        vix_today = float(prior.iloc[-1]) if not prior.empty else None
                    if vix_today is not None and vix_today > cfg.vix_block_threshold:
                        vix_ok = False

                # Calendar / seasonality gate
                cal_ok = True
                if cfg.calendar_filter_mode != "off":
                    m = today.month
                    if cfg.calendar_filter_mode == "sell_in_may":
                        # Block May-Oct
                        if m in (5, 6, 7, 8, 9, 10):
                            cal_ok = False
                    elif cfg.calendar_filter_mode == "santa_only":
                        # Allow only Nov-Jan
                        if m not in (11, 12, 1):
                            cal_ok = False
                    elif cfg.calendar_filter_mode == "block_months":
                        if m in cfg.calendar_block_months:
                            cal_ok = False
                # FOMC pre-meeting blackout
                if cal_ok and cfg.fomc_block_days > 0 and cfg.fomc_dates:
                    fdates = pd.to_datetime(list(cfg.fomc_dates))
                    today_ts = pd.Timestamp(today)
                    diffs = (fdates - today_ts).days
                    if any(0 <= d <= cfg.fomc_block_days for d in diffs):
                        cal_ok = False

                # Credit-spread regime gate. Use prior bar (today is decision day,
                # but ratio uses close — we read from the most recent index <= today
                # which on a trading day equals today's close. To stay strict no-look,
                # use ratio asof (today - 1 trading day).
                credit_ok = True
                if cfg.credit_spread_filter_enabled and credit_ratio is not None:
                    prior_idx = credit_ratio.index[credit_ratio.index < today]
                    if len(prior_idx) > 0:
                        ref_day = prior_idx[-1]
                        r_now = float(credit_ratio.loc[ref_day])
                        if cfg.credit_spread_block_mode == "below_sma":
                            sma_v = credit_sma.loc[ref_day] if ref_day in credit_sma.index else None
                            if sma_v is not None and not pd.isna(sma_v):
                                threshold = float(sma_v) * (1.0 - cfg.credit_spread_drop_pct)
                                if r_now < threshold:
                                    credit_ok = False
                        elif cfg.credit_spread_block_mode == "negative_delta":
                            d_v = credit_delta.loc[ref_day] if ref_day in credit_delta.index else None
                            if d_v is not None and not pd.isna(d_v) and float(d_v) < 0.0:
                                credit_ok = False
                        elif cfg.credit_spread_block_mode == "z_below":
                            z_v = credit_z.loc[ref_day] if ref_day in credit_z.index else None
                            if z_v is not None and not pd.isna(z_v) and float(z_v) < cfg.credit_spread_z_threshold:
                                credit_ok = False

                # Segseg direction gate (chan.py-internal 段套段, same data source as base)
                segseg_long_ok = True
                segseg_short_ok = True
                if cfg.segseg_filter_enabled:
                    ssd = sig.get("segseg_dir")
                    if ssd is None:
                        segseg_long_ok = segseg_short_ok = False
                    else:
                        segseg_long_ok = ssd == "up"
                        segseg_short_ok = ssd == "down"

                # ZS divergence gate
                zs_div_ok = True
                if cfg.zs_divergence_required:
                    zs_div_ok = bool(sig.get("zs_divergence"))

                # Determine entry signals based on entry_mode.
                long_signal = None
                short_signal = None
                if cfg.entry_mode == "donchian":
                    if sym in donchian_high_per_sym and today in donchian_high_per_sym[sym].index:
                        d_high = donchian_high_per_sym[sym].loc[today]
                        d_low = donchian_low_per_sym[sym].loc[today]
                        close_today = float(bars[sym].loc[today, "close"]) if today in bars[sym].index else None
                        if close_today is not None and not pd.isna(d_high) and close_today > float(d_high) * (1 + cfg.donchian_breakout_min_pct):
                            # Volume-confirm gate (Donchian branch only)
                            volume_ok = True
                            if cfg.volume_confirm_enabled and sym in volume_avg_per_sym:
                                vol_avg = volume_avg_per_sym[sym].loc[today] if today in volume_avg_per_sym[sym].index else None
                                vol_today = float(bars[sym].loc[today, "volume"]) if today in bars[sym].index else 0.0
                                if vol_avg is None or pd.isna(vol_avg) or vol_avg <= 0:
                                    volume_ok = False
                                else:
                                    volume_ok = vol_today >= float(vol_avg) * cfg.volume_confirm_mult
                            if volume_ok:
                                long_signal = {"types_str": "donchian", "ref_price": float(d_low) if not pd.isna(d_low) else 0.0}
                        if close_today is not None and not pd.isna(d_low) and close_today < float(d_low):
                            short_signal = {"types_str": "donchian", "ref_price": float(d_high) if not pd.isna(d_high) else 0.0}
                elif cfg.entry_mode == "seg_bsp":
                    if sig.get("seg_buy"):
                        long_signal = {"types_str": "seg:" + sig["seg_buy"]["types_str"], "ref_price": sig["seg_buy"]["bi_low"]}
                    if sig.get("seg_sell"):
                        short_signal = {"types_str": "seg:" + sig["seg_sell"]["types_str"], "ref_price": sig["seg_sell"]["bi_high"]}
                elif cfg.entry_mode == "any_bsp":
                    # Prefer seg-level (rarer, more meaningful); fall back to bi-level
                    if sig.get("seg_buy"):
                        long_signal = {"types_str": "seg:" + sig["seg_buy"]["types_str"], "ref_price": sig["seg_buy"]["bi_low"]}
                    elif sig.get("buy"):
                        long_signal = {"types_str": sig["buy"]["types_str"], "ref_price": sig["buy"]["bi_low"]}
                    if sig.get("seg_sell"):
                        short_signal = {"types_str": "seg:" + sig["seg_sell"]["types_str"], "ref_price": sig["seg_sell"]["bi_high"]}
                    elif sig.get("sell"):
                        short_signal = {"types_str": sig["sell"]["types_str"], "ref_price": sig["sell"]["bi_high"]}
                elif cfg.entry_mode == "zs_boundary":
                    # Long entry: in confirmed up-segseg, today's close has retraced
                    # into the most-recent confirmed ZS (between zs_low and zs_high).
                    # Stop is zs_low; this is the canonical "buy ZS edge" Chan trade.
                    zs_h = sig.get("zs_high")
                    zs_l = sig.get("zs_low")
                    sure = sig.get("zs_sure")
                    ssd = sig.get("segseg_dir")
                    if zs_h and zs_l and sure and ssd == "up" and today in bars[sym].index:
                        close_today = float(bars[sym].loc[today, "close"])
                        if zs_l <= close_today <= zs_h:
                            long_signal = {"types_str": "zs_edge", "ref_price": float(zs_l)}
                elif cfg.entry_mode == "donchian_or_seg":
                    # Try Donchian first, then seg-level BSP. Track which branch fired
                    # so we can conditionally bypass the momentum filter for seg.
                    branch = None
                    if sym in donchian_high_per_sym and today in donchian_high_per_sym[sym].index:
                        d_high = donchian_high_per_sym[sym].loc[today]
                        d_low = donchian_low_per_sym[sym].loc[today]
                        close_today = float(bars[sym].loc[today, "close"]) if today in bars[sym].index else None
                        if close_today is not None and not pd.isna(d_high) and close_today > float(d_high) * (1 + cfg.donchian_breakout_min_pct):
                            # Volume-confirm gate (Donchian branch only)
                            volume_ok = True
                            if cfg.volume_confirm_enabled and sym in volume_avg_per_sym:
                                vol_avg = volume_avg_per_sym[sym].loc[today] if today in volume_avg_per_sym[sym].index else None
                                vol_today = float(bars[sym].loc[today, "volume"]) if today in bars[sym].index else 0.0
                                if vol_avg is None or pd.isna(vol_avg) or vol_avg <= 0:
                                    volume_ok = False
                                else:
                                    volume_ok = vol_today >= float(vol_avg) * cfg.volume_confirm_mult
                            if volume_ok:
                                long_signal = {"types_str": "donchian", "ref_price": float(d_low) if not pd.isna(d_low) else 0.0}
                            branch = "donchian"
                        if close_today is not None and not pd.isna(d_low) and close_today < float(d_low):
                            short_signal = {"types_str": "donchian", "ref_price": float(d_high) if not pd.isna(d_high) else 0.0}
                    if long_signal is None and sig.get("seg_buy"):
                        long_signal = {"types_str": "seg:" + sig["seg_buy"]["types_str"], "ref_price": sig["seg_buy"]["bi_low"]}
                        branch = "seg"
                    if short_signal is None and sig.get("seg_sell"):
                        short_signal = {"types_str": "seg:" + sig["seg_sell"]["types_str"], "ref_price": sig["seg_sell"]["bi_high"]}
                    # Bypass momentum filter for seg-branch entries unless caller opts in
                    if branch == "seg" and not cfg.momentum_filter_seg_branch:
                        momentum_long_ok = True
                        momentum_short_ok = True
                elif cfg.entry_mode == "donchian_seg_zs":
                    # Triple union: Donchian → seg_bsp → zs_boundary, in priority order.
                    # zs_boundary needs its own segseg=up gate (handled inline).
                    branch = None
                    if sym in donchian_high_per_sym and today in donchian_high_per_sym[sym].index:
                        d_high = donchian_high_per_sym[sym].loc[today]
                        d_low = donchian_low_per_sym[sym].loc[today]
                        close_today = float(bars[sym].loc[today, "close"]) if today in bars[sym].index else None
                        if close_today is not None and not pd.isna(d_high) and close_today > float(d_high) * (1 + cfg.donchian_breakout_min_pct):
                            # Volume-confirm gate (Donchian branch only)
                            volume_ok = True
                            if cfg.volume_confirm_enabled and sym in volume_avg_per_sym:
                                vol_avg = volume_avg_per_sym[sym].loc[today] if today in volume_avg_per_sym[sym].index else None
                                vol_today = float(bars[sym].loc[today, "volume"]) if today in bars[sym].index else 0.0
                                if vol_avg is None or pd.isna(vol_avg) or vol_avg <= 0:
                                    volume_ok = False
                                else:
                                    volume_ok = vol_today >= float(vol_avg) * cfg.volume_confirm_mult
                            if volume_ok:
                                long_signal = {"types_str": "donchian", "ref_price": float(d_low) if not pd.isna(d_low) else 0.0}
                            branch = "donchian"
                        if close_today is not None and not pd.isna(d_low) and close_today < float(d_low):
                            short_signal = {"types_str": "donchian", "ref_price": float(d_high) if not pd.isna(d_high) else 0.0}
                    if long_signal is None and sig.get("seg_buy"):
                        long_signal = {"types_str": "seg:" + sig["seg_buy"]["types_str"], "ref_price": sig["seg_buy"]["bi_low"]}
                        branch = "seg"
                    if long_signal is None:
                        zs_h = sig.get("zs_high")
                        zs_l = sig.get("zs_low")
                        sure = sig.get("zs_sure")
                        ssd = sig.get("segseg_dir")
                        if zs_h and zs_l and sure and ssd == "up" and today in bars[sym].index:
                            close_today = float(bars[sym].loc[today, "close"])
                            if zs_l <= close_today <= zs_h:
                                long_signal = {"types_str": "zs_edge", "ref_price": float(zs_l)}
                                branch = "zs"
                    if short_signal is None and sig.get("seg_sell"):
                        short_signal = {"types_str": "seg:" + sig["seg_sell"]["types_str"], "ref_price": sig["seg_sell"]["bi_high"]}
                    # Bypass momentum filter for seg / zs branches
                    if branch in ("seg", "zs") and not cfg.momentum_filter_seg_branch:
                        momentum_long_ok = True
                        momentum_short_ok = True
                else:  # chan_bsp (default)
                    if sig.get("buy"):
                        long_signal = {"types_str": sig["buy"]["types_str"], "ref_price": sig["buy"]["bi_low"]}
                    if sig.get("sell"):
                        short_signal = {"types_str": sig["sell"]["types_str"], "ref_price": sig["sell"]["bi_high"]}

                # Capture entry-time signal context for conditional pyramid (Test 1)
                entry_ctx = {
                    "segseg_dir": sig.get("segseg_dir"),
                    "segseg_is_sure": sig.get("segseg_is_sure"),
                    "zs_broken": sig.get("zs_broken"),
                }
                if cfg.enable_longs and long_signal and trend_long_ok and momentum_long_ok and weekly_long_ok and segseg_long_ok and zs_div_ok and vix_ok and credit_ok and cal_ok and trend_type_long_ok and bi_timing_ok and equity_curve_ok and sector_cap_ok:
                    pending_entries.append({
                        "symbol": sym,
                        "direction": "long",
                        "ref_price": long_signal["ref_price"],
                        "types_str": long_signal["types_str"],
                        "ready_after_idx": d_idx + cfg.entry_lag_extra_days,
                        "momentum_rank": sym_rank_val,
                        "entry_ctx": entry_ctx,
                    })
                # Re-entry after stop overlay: if symbol was recently stopped and
                # today's close has reclaimed the original entry price, inject a
                # synthetic re-entry (without requiring a fresh BSP/Donchian signal).
                if (cfg.reentry_after_stop_enabled
                        and not long_signal  # only synthesize if no live signal already
                        and sym in recently_stopped
                        and cfg.enable_longs
                        and trend_long_ok and momentum_long_ok and weekly_long_ok
                        and segseg_long_ok and zs_div_ok and vix_ok and credit_ok and cal_ok
                        and trend_type_long_ok and bi_timing_ok and equity_curve_ok and sector_cap_ok):
                    rs = recently_stopped[sym]
                    bars_since = d_idx - rs["stop_bar_idx"]
                    if (bars_since <= cfg.reentry_window_bars
                            and rs["count"] < cfg.reentry_max_count
                            and today in bars[sym].index):
                        close_today = float(bars[sym].loc[today, "close"])
                        if close_today > rs["entry_price"]:
                            pending_entries.append({
                                "symbol": sym,
                                "direction": "long",
                                "ref_price": rs["entry_price"],
                                "types_str": str(rs["bsp_types"]) + ":reentry",
                                "ready_after_idx": d_idx + cfg.entry_lag_extra_days,
                                "momentum_rank": sym_rank_val,
                                "entry_ctx": rs["entry_ctx"],
                            })
                            rs["count"] += 1

                # Drop expired re-entry windows
                if cfg.reentry_after_stop_enabled and sym in recently_stopped:
                    if d_idx - recently_stopped[sym]["stop_bar_idx"] > cfg.reentry_window_bars:
                        del recently_stopped[sym]

                if cfg.enable_shorts and short_signal and trend_short_ok and momentum_short_ok and weekly_short_ok and segseg_short_ok and vix_ok and credit_ok and cal_ok:
                    pending_entries.append({
                        "symbol": sym,
                        "direction": "short",
                        "ref_price": short_signal["ref_price"],
                        "types_str": short_signal["types_str"],
                        "ready_after_idx": d_idx + cfg.entry_lag_extra_days,
                        "momentum_rank": sym_rank_val,
                        "entry_ctx": entry_ctx,
                    })

            # ----- 5. Mark-to-market equity -----
            # For longs: positive mtm contribution (we own shares).
            # For shorts: negative mtm contribution (we owe shares to broker).
            # Cash already reflects short proceeds collected at entry, so:
            #   equity = cash + sum(+shares × cur for longs) + sum(-shares × cur for shorts)
            # When today's bar is missing for a held symbol (data gap, holiday on
            # one exchange, etc.), forward-fill from the most recent prior close so
            # MTM doesn't silently zero out the position. Bug found 2026-04-27.
            mtm = 0.0
            for sym, pos in positions.items():
                if today in bars[sym].index:
                    cur = float(bars[sym].loc[today, "close"])
                else:
                    prior = bars[sym].loc[bars[sym].index <= today, "close"]
                    if prior.empty:
                        continue
                    cur = float(prior.iloc[-1])
                sign = 1.0 if pos.direction == "long" else -1.0
                mult = _mult(sym)
                if mult == 1.0:
                    # ETF: position market value
                    mtm += sign * pos.shares * cur
                else:
                    # Futures: cash already nets P&L on close, so MTM is unrealized
                    # P&L from entry to today, scaled by multiplier
                    unreal = sign * pos.shares * mult * (cur - pos.entry_price)
                    mtm += unreal
            equity_curve.append({
                "date": today, "cash": cash, "positions_value": mtm,
                "equity": cash + mtm, "n_positions": len(positions),
            })

            # Update equity-curve gate state
            if cfg.equity_dd_threshold_pct > 0:
                eq_today = cash + mtm
                equity_high_water = max(equity_high_water, eq_today)
                dd_now = (equity_high_water - eq_today) / equity_high_water
                if not equity_dd_paused and dd_now >= cfg.equity_dd_threshold_pct:
                    equity_dd_paused = True
                elif equity_dd_paused and dd_now <= cfg.equity_dd_resume_pct:
                    equity_dd_paused = False

        # ----- 6. Force-close any remaining at last available close -----
        for sym, pos in list(positions.items()):
            if not bars[sym].empty:
                exit_d = bars[sym].index[-1]
                _close_position(sym, pos, float(bars[sym].loc[exit_d, "close"]), exit_d, "end_of_data")
                del positions[sym]

        return self._build_result(cfg, trades, equity_curve)

    # ------------------------------------------------------------------
    # Result aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(
        cfg: ChanDailyBacktestConfig,
        trades: list[dict],
        equity_curve: list[dict],
    ) -> ChanPortfolioResult:
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame()

        if not equity_df.empty:
            initial = cfg.initial_cash
            final = float(equity_df["equity"].iloc[-1])
            total_return_pct = (final / initial - 1.0) * 100.0
            running_max = equity_df["equity"].cummax()
            dd = (equity_df["equity"] / running_max - 1.0) * 100.0
            max_drawdown_pct = float(dd.min())
        else:
            initial = cfg.initial_cash
            final = cfg.initial_cash
            total_return_pct = 0.0
            max_drawdown_pct = 0.0

        if not trades_df.empty:
            win = trades_df["return"] > 0
            wins = trades_df[win]
            losses = trades_df[~win]
            summary = {
                "initial_cash": initial,
                "final_equity": final,
                "total_return_pct": total_return_pct,
                "max_drawdown_pct": max_drawdown_pct,
                "total_trades": int(len(trades_df)),
                "win_rate": float(win.mean()),
                "avg_return": float(trades_df["return"].mean()),
                "avg_win": float(wins["return"].mean()) if not wins.empty else 0.0,
                "avg_loss": float(losses["return"].mean()) if not losses.empty else 0.0,
                "avg_bars_held": float(trades_df["bars_held"].mean()),
                "by_exit_reason": {
                    reason: {
                        "count": int(len(g)),
                        "win_rate": float((g["return"] > 0).mean()),
                        "avg_ret": float(g["return"].mean()),
                    }
                    for reason, g in trades_df.groupby("exit_reason")
                },
                "by_entry_type": {
                    bsp: {
                        "count": int(len(g)),
                        "win_rate": float((g["return"] > 0).mean()),
                        "avg_ret": float(g["return"].mean()),
                    }
                    for bsp, g in trades_df.groupby("bsp_types")
                },
            }
            sym_summary = (
                trades_df.assign(pnl=lambda d: d["return"] * d["entry_price"] * d["shares"])
                .groupby("symbol")
                .agg(
                    trades=("symbol", "size"),
                    win_rate=("return", lambda r: float((r > 0).mean())),
                    total_pnl=("pnl", "sum"),
                )
                .reset_index()
            )
        else:
            summary = {
                "initial_cash": initial,
                "final_equity": final,
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "avg_bars_held": 0.0,
                "by_exit_reason": {},
                "by_entry_type": {},
            }
            sym_summary = pd.DataFrame()

        return ChanPortfolioResult(
            summary=summary,
            trades=trades_df,
            equity_curve=equity_df,
            daily_state=pd.DataFrame(),  # not tracked in MVP
            symbol_summary=sym_summary,
        )
