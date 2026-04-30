"""Intraday cross-sectional mean-reversion backtester (Lehmann 1990 family).

Different *mechanism* from `intraday_backtester.py`:
  * intraday_backtester: per-symbol breakout signals (gap_reclaim, NR4, etc.).
    Long-only. Each symbol's frame is independent.
  * intraday_xsection_backtester (this module): walks the time axis. At each
    rebalance bar T, ranks the entire universe by trailing formation-window
    return, longs the bottom decile, shorts the top decile, dollar-neutral.
    Beta-zero so doesn't depend on bull regime.

Per memory `[Intraday alpha research plan]` (2026-04-29), this is the
top-pick experiment after 6+ accumulated nulls in the post-open-breakout
class. CLAUDE.md mandatory gates enforced by the runner script:
  * 4-period broad-universe test (2023_25 / 2020 / 2018; no 2015 intraday data)
  * Future-blanked probe — formation window is fully past-looking by
    construction; the `formation_lag_bars` knob delays the rank computation
    by N bars to verify no accidental forward leakage in the holding loop.
  * Slippage stress at 5/10/25 bps (dollar-neutral high-turnover is
    slippage-sensitive — bps cost compounds).

Cost model:
  * Half-spread + per-bar slippage on entry/exit.
  * Short-borrow proxy: configurable bps/day applied to short notional
    while held. Liquid names: ~0.5-2% annualized = 2-8 bps/day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Optional

import duckdb
import pandas as pd

from .market_context import build_market_context
from .warehouse import MarketDataWarehouse

logger = logging.getLogger(__name__)


@dataclass
class XSectionReversionConfig:
    """All knobs default to a Lehmann-1990-shaped baseline:
    rank by trailing 60min return, hold 60min, equal-weight 20×20 dollar-neutral.
    """
    initial_cash: float = 100_000.0

    # Universe + filtering
    universe: str = "broad250"  # path resolved by runner
    min_dollar_volume_avg: float = 5_000_000.0  # 20-day avg daily $-vol floor
    adv_lookback_days: int = 20
    max_position_pct_of_adv: Optional[float] = None

    # Sector neutrality (toggle for A/B). Pulls sector from fundamentals_snapshots.
    sector_neutral: bool = False

    # Prior-session regime gate. All enabled filters are combined with AND.
    market_context_min_score: Optional[int] = None
    market_context_max_score: Optional[int] = None
    allowed_market_regimes: tuple[str, ...] = field(default_factory=tuple)
    market_context_qqq_above_ema21_pct_min: Optional[float] = None
    market_context_qqq_above_ema21_pct_max: Optional[float] = None
    market_context_qqq_roc_5_min: Optional[float] = None
    market_context_qqq_roc_5_max: Optional[float] = None
    regime_filter_overrides: dict[str, dict[str, float | int | str | tuple[str, ...]]] = field(
        default_factory=dict
    )
    intraday_market_tilt_symbol: Optional[str] = None
    intraday_market_tilt_threshold: float = 0.0
    intraday_market_tilt_strength: float = 0.0
    intraday_market_tilt_strong_threshold: Optional[float] = None
    intraday_market_tilt_strong_strength: Optional[float] = None

    # Bar timing
    interval_minutes: int = 15
    formation_minutes: int = 60   # rank window
    hold_minutes: int = 60        # holding period
    tilt_refresh_minutes: Optional[int] = None  # optional bias-resize cadence
    formation_lag_bars: int = 0   # future-blanked probe (delays rank by N bars)

    # Cross-section selection
    signal_direction: str = "reversion"  # "reversion" or "momentum"
    n_long: int = 20    # bottom-N by formation return
    n_short: int = 20   # top-N by formation return
    dollar_neutral: bool = True  # equal $ long vs short notional

    # Position sizing
    # When dollar_neutral=True, 1.0 means 100% long + 100% short = 200% gross.
    # When dollar_neutral=False, 1.0 means 100% total gross split across names.
    target_gross_exposure: float = 1.0

    # Session bounds (CDT local times — matches intraday DB convention)
    earliest_rebalance_time: dtime = dtime(8, 30)   # earliest entry time
    latest_rebalance_time: dtime = dtime(15, 0)     # latest entry time
    flatten_at_close_time: dtime = dtime(15, 55)    # EOD flatten

    # Costs
    half_spread_bps: float = 1.0
    slippage_bps: float = 5.0           # tested at 5/10/25 in stress runs
    short_borrow_bps_per_day: float = 5.0  # ~1% annualized average


@dataclass
class XSectionTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    symbol: str
    side: str  # "long" or "short"
    qty: int
    entry_price: float
    exit_price: float
    gross_pnl: float       # before costs
    cost: float            # spread + slippage + borrow
    net_pnl: float
    formation_return: float


@dataclass
class XSectionResult:
    trades: list[XSectionTrade]
    equity_curve: pd.DataFrame
    summary: dict
    config: XSectionReversionConfig


def _load_universe(path: str) -> list[str]:
    import json
    from pathlib import Path
    d = json.loads(Path(path).read_text())
    return d["symbols"] if isinstance(d, dict) else d


def _load_sector_map(daily_db_path: str) -> dict[str, str]:
    """Pull most-recent sector per symbol from fundamentals_snapshots."""
    try:
        conn = duckdb.connect(daily_db_path, read_only=True)
        rows = conn.execute(
            "SELECT symbol, sector FROM fundamentals_snapshots "
            "WHERE sector IS NOT NULL "
            "QUALIFY row_number() OVER (PARTITION BY symbol ORDER BY snapshot_date DESC) = 1"
        ).fetchall()
        conn.close()
        return {sym: sector for sym, sector in rows}
    except Exception as exc:
        logger.warning("sector map load failed: %s — sector_neutral will be a no-op", exc)
        return {}


def _filter_session(df: pd.DataFrame) -> pd.DataFrame:
    """Same 8:30-15:00 CDT regular session as intraday_backtester."""
    if df.empty:
        return df
    minutes = df.index.hour * 60 + df.index.minute
    return df.loc[(minutes >= 8 * 60 + 30) & (minutes <= 15 * 60)].copy()


def _load_symbol_frames(
    db_path: str, symbols: list[str], begin: str, end: str, interval_minutes: int
) -> dict[str, pd.DataFrame]:
    table = f"bars_{interval_minutes}m"
    conn = duckdb.connect(db_path, read_only=True)
    out: dict[str, pd.DataFrame] = {}
    try:
        for sym in symbols:
            df = conn.execute(
                f"SELECT ts, open, high, low, close, volume FROM {table} "
                f"WHERE symbol=? AND ts >= ? AND ts <= ? ORDER BY ts",
                [sym, f"{begin} 00:00:00", f"{end} 23:59:59"],
            ).fetchdf()
            if df.empty:
                continue
            df["ts"] = pd.to_datetime(df["ts"], utc=False)
            df = df.set_index("ts").sort_index()
            df = _filter_session(df)
            if df.empty:
                continue
            out[sym] = df
    finally:
        conn.close()
    return out


def _compute_dollar_volume(frame: pd.DataFrame, lookback_bars: int) -> pd.Series:
    """Rolling avg dollar-volume for liquidity filter."""
    dv = frame["close"] * frame["volume"]
    return dv.rolling(lookback_bars, min_periods=max(1, lookback_bars // 4)).mean()


def _load_daily_adv(
    daily_db_path: str,
    symbols: list[str],
    begin: str,
    end: str,
    lookback_days: int,
) -> pd.DataFrame:
    """Load prior-day rolling ADV by symbol for liquidity-aware sizing."""
    if lookback_days <= 0:
        return pd.DataFrame()

    buffer_begin = (pd.Timestamp(begin) - pd.Timedelta(days=lookback_days * 30)).strftime(
        "%Y-%m-%d"
    )
    warehouse = MarketDataWarehouse(daily_db_path, read_only=True)
    series_map: dict[str, pd.Series] = {}
    try:
        for symbol in symbols:
            df = warehouse.get_daily_bars(symbol, buffer_begin, end)
            if df is None or df.empty:
                continue
            daily = df.copy().sort_index()
            adv = (
                (daily["close"] * daily["volume"])
                .rolling(lookback_days, min_periods=lookback_days)
                .mean()
                .shift(1)
            )
            adv.index = pd.to_datetime(adv.index).normalize()
            series_map[symbol] = adv
    finally:
        warehouse.close()

    if not series_map:
        return pd.DataFrame()
    return pd.DataFrame(series_map).sort_index()


def _allocate_equal_with_caps(max_allocs: dict[str, float], target_total: float) -> dict[str, float]:
    """Allocate equal-weight dollars across names subject to per-name caps."""
    allocs = {symbol: 0.0 for symbol in max_allocs}
    positive_caps = {symbol: cap for symbol, cap in max_allocs.items() if cap > 0}
    if not positive_caps or target_total <= 0:
        return allocs

    remaining = set(positive_caps.keys())
    remaining_budget = min(float(target_total), sum(positive_caps.values()))

    while remaining and remaining_budget > 1e-9:
        equal_share = remaining_budget / len(remaining)
        capped_now = {symbol for symbol in remaining if positive_caps[symbol] <= equal_share + 1e-12}
        if not capped_now:
            for symbol in remaining:
                allocs[symbol] = equal_share
            return allocs
        for symbol in capped_now:
            allocs[symbol] = positive_caps[symbol]
            remaining_budget -= allocs[symbol]
            remaining.remove(symbol)

    return allocs


class XSectionReversionBacktester:
    """Walk-the-time-axis backtester for cross-sectional reversion."""

    def __init__(self, config: XSectionReversionConfig):
        self.config = config

    def _intraday_market_tilt(
        self,
        t: pd.Timestamp,
        close_df: pd.DataFrame,
        session_open_df: pd.DataFrame,
        regime_label: Optional[str],
    ) -> float:
        """Return a bounded same-day directional tilt from the proxy's session trend.

        Positive values lean long, negative values lean short. The signal only
        uses data available at the current rebalance bar.
        """
        cfg = self.config
        override = cfg.regime_filter_overrides.get(regime_label or "", {})
        symbol = override.get("intraday_market_tilt_symbol", cfg.intraday_market_tilt_symbol)
        base_strength = float(
            override.get("intraday_market_tilt_strength", cfg.intraday_market_tilt_strength)
        )
        if not symbol or base_strength <= 0:
            return 0.0
        if symbol not in close_df.columns or symbol not in session_open_df.columns:
            return 0.0

        px = close_df.at[t, symbol]
        session_open = session_open_df.at[t, symbol]
        if pd.isna(px) or pd.isna(session_open) or float(session_open) <= 0:
            return 0.0

        session_ret = float(px) / float(session_open) - 1.0
        base_threshold = float(
            override.get("intraday_market_tilt_threshold", cfg.intraday_market_tilt_threshold)
        )
        strong_threshold_raw = override.get(
            "intraday_market_tilt_strong_threshold",
            cfg.intraday_market_tilt_strong_threshold,
        )
        strong_strength_raw = override.get(
            "intraday_market_tilt_strong_strength",
            cfg.intraday_market_tilt_strong_strength,
        )
        if strong_threshold_raw is not None and strong_strength_raw is not None:
            strong_threshold = float(strong_threshold_raw)
            strong_strength = min(float(strong_strength_raw), 1.0)
            if strong_strength > 0 and session_ret > strong_threshold:
                return strong_strength
            if strong_strength > 0 and session_ret < -strong_threshold:
                return -strong_strength
        if session_ret > base_threshold:
            return min(base_strength, 1.0)
        if session_ret < -base_threshold:
            return -min(base_strength, 1.0)
        return 0.0

    def _load_market_context(self, begin: str, end: str, daily_db_path: str) -> pd.DataFrame:
        """Load prior-session regime context for the requested interval."""
        if (
            self.config.market_context_min_score is None
            and self.config.market_context_max_score is None
            and not self.config.allowed_market_regimes
            and self.config.market_context_qqq_above_ema21_pct_min is None
            and self.config.market_context_qqq_above_ema21_pct_max is None
            and self.config.market_context_qqq_roc_5_min is None
            and self.config.market_context_qqq_roc_5_max is None
            and not self.config.regime_filter_overrides
        ):
            return pd.DataFrame()

        buffer_begin = (pd.Timestamp(begin) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
        warehouse = MarketDataWarehouse(daily_db_path, read_only=True)
        frames: dict[str, pd.DataFrame] = {}
        try:
            for symbol in ("SPY", "QQQ", "IWM", "SMH", "^VIX"):
                df = warehouse.get_daily_bars(symbol, buffer_begin, end)
                if df is not None and not df.empty:
                    frames[symbol] = df.copy().sort_index()
        finally:
            warehouse.close()

        context = build_market_context(frames)
        if context.empty:
            return pd.DataFrame()
        context = context.copy()
        context.index = pd.to_datetime(context.index).normalize()
        return context

    def _regime_allows_entry(
        self,
        context: pd.DataFrame,
        session_date: pd.Timestamp,
    ) -> tuple[bool, Optional[str], Optional[int]]:
        if context.empty:
            return True, None, None

        valid = context.loc[context.index < session_date]
        if valid.empty:
            return False, None, None

        latest = valid.iloc[-1]
        regime = (
            str(latest["market_regime"])
            if "market_regime" in latest and pd.notna(latest["market_regime"])
            else None
        )
        override = self.config.regime_filter_overrides.get(regime or "", {})
        score = (
            int(latest["market_score"])
            if "market_score" in latest and pd.notna(latest["market_score"])
            else None
        )

        effective_allowed_regimes = override.get("allowed_market_regimes")
        if effective_allowed_regimes is None:
            effective_allowed_regimes = self.config.allowed_market_regimes
        if effective_allowed_regimes and regime not in set(effective_allowed_regimes):
            return False, regime, score
        min_score = override.get("market_context_min_score", self.config.market_context_min_score)
        if (
            min_score is not None
            and (score is None or score < int(min_score))
        ):
            return False, regime, score
        max_score = override.get("market_context_max_score", self.config.market_context_max_score)
        if (
            max_score is not None
            and (score is None or score > int(max_score))
        ):
            return False, regime, score

        qqq_above_ema21_pct = (
            float(latest["qqq_above_ema21_pct"])
            if "qqq_above_ema21_pct" in latest and pd.notna(latest["qqq_above_ema21_pct"])
            else None
        )
        qqq_above_ema21_pct_min = override.get(
            "market_context_qqq_above_ema21_pct_min",
            self.config.market_context_qqq_above_ema21_pct_min,
        )
        if (
            qqq_above_ema21_pct_min is not None
            and (
                qqq_above_ema21_pct is None
                or qqq_above_ema21_pct < float(qqq_above_ema21_pct_min)
            )
        ):
            return False, regime, score
        qqq_above_ema21_pct_max = override.get(
            "market_context_qqq_above_ema21_pct_max",
            self.config.market_context_qqq_above_ema21_pct_max,
        )
        if (
            qqq_above_ema21_pct_max is not None
            and (
                qqq_above_ema21_pct is None
                or qqq_above_ema21_pct > float(qqq_above_ema21_pct_max)
            )
        ):
            return False, regime, score

        qqq_roc_5 = (
            float(latest["qqq_roc_5"])
            if "qqq_roc_5" in latest and pd.notna(latest["qqq_roc_5"])
            else None
        )
        qqq_roc_5_min = override.get(
            "market_context_qqq_roc_5_min",
            self.config.market_context_qqq_roc_5_min,
        )
        if (
            qqq_roc_5_min is not None
            and (qqq_roc_5 is None or qqq_roc_5 < float(qqq_roc_5_min))
        ):
            return False, regime, score
        qqq_roc_5_max = override.get(
            "market_context_qqq_roc_5_max",
            self.config.market_context_qqq_roc_5_max,
        )
        if (
            qqq_roc_5_max is not None
            and (qqq_roc_5 is None or qqq_roc_5 > float(qqq_roc_5_max))
        ):
            return False, regime, score
        return True, regime, score

    def backtest(
        self,
        symbols: list[str],
        begin: str,
        end: str,
        intraday_db_path: str,
        daily_db_path: str = "research_data/market_data.duckdb",
    ) -> XSectionResult:
        cfg = self.config
        bars_per_form = max(1, cfg.formation_minutes // cfg.interval_minutes)
        bars_per_hold = max(1, cfg.hold_minutes // cfg.interval_minutes)
        bars_per_tilt_refresh = (
            max(1, int(cfg.tilt_refresh_minutes) // cfg.interval_minutes)
            if cfg.tilt_refresh_minutes
            else None
        )

        # Liquidity filter lookback ~20 sessions ≈ 26 bars/day × 20 = 520 bars
        # but at 15m we have 26 bars/day; use 20-day lookback in BAR units.
        bars_per_day = (
            int((dtime(15, 0).hour * 60 + dtime(15, 0).minute
                 - dtime(8, 30).hour * 60 - dtime(8, 30).minute)
                / cfg.interval_minutes) + 1
        )
        liquidity_lookback_bars = bars_per_day * 20

        logger.info(
            "xsection: loading %d symbols / %s..%s / %dm bars",
            len(symbols), begin, end, cfg.interval_minutes,
        )
        frames = _load_symbol_frames(intraday_db_path, symbols, begin, end, cfg.interval_minutes)
        logger.info("xsection: %d symbols loaded with non-empty frames", len(frames))

        if not frames:
            return XSectionResult([], pd.DataFrame(), {"error": "no_data"}, cfg)

        loaded_symbols = sorted(frames.keys())
        sector_map = _load_sector_map(daily_db_path) if cfg.sector_neutral else {}
        market_context = self._load_market_context(begin, end, daily_db_path)
        adv_df = (
            _load_daily_adv(
                daily_db_path,
                loaded_symbols,
                begin,
                end,
                cfg.adv_lookback_days,
            )
            if cfg.max_position_pct_of_adv is not None
            else pd.DataFrame()
        )
        if cfg.signal_direction not in {"reversion", "momentum"}:
            raise ValueError(
                f"Unsupported signal_direction={cfg.signal_direction!r}; "
                "expected 'reversion' or 'momentum'"
            )

        # Master time index = sorted union of all bar times where ANY symbol has data.
        master_index = sorted(set().union(*[df.index for df in frames.values()]))
        master_index = pd.DatetimeIndex(master_index)

        # Pre-build open/close + dollar-volume DataFrames for vectorized ranking.
        open_df = pd.DataFrame(
            {sym: df["open"] for sym, df in frames.items()}, index=master_index
        )
        close_df = pd.DataFrame(
            {sym: df["close"] for sym, df in frames.items()}, index=master_index
        )
        session_open_df = open_df.groupby(master_index.normalize(), sort=False).transform("first")
        dv_df = pd.DataFrame(
            {sym: _compute_dollar_volume(df, liquidity_lookback_bars) for sym, df in frames.items()},
            index=master_index,
        )

        equity = cfg.initial_cash
        equity_curve: list[dict] = []
        trades: list[XSectionTrade] = []

        # Track open positions: dict[symbol] -> {side, qty, entry_price, entry_time}
        open_positions: dict[str, dict] = {}
        pending_rebalance: Optional[dict] = None
        rebalance_count = 0
        tilt_refresh_count = 0
        regime_allowed_rebalances = 0
        regime_blocked_rebalances = 0
        liquidity_capped_rebalances = 0

        # Walk every bar; act only at rebalance bars.
        # Rebalance schedule: every hold_minutes inside the session, between
        # earliest_rebalance and latest_rebalance.
        last_rebalance_idx: Optional[int] = None
        last_tilt_refresh_idx: Optional[int] = None
        for i, t in enumerate(master_index):
            t_local = t.time()
            next_ts = master_index[i + 1] if i + 1 < len(master_index) else None
            is_last_bar_of_session = (
                next_ts is None or next_ts.normalize() != t.normalize()
            )

            # Execute the prior bar's rebalance decision on the next bar open.
            if pending_rebalance is not None:
                for sym in list(open_positions.keys()):
                    pos = open_positions[sym]
                    px = open_df.at[t, sym] if sym in open_df.columns else None
                    if px is None or pd.isna(px):
                        continue
                    exit_px = float(px)
                    side = pos["side"]
                    qty = pos["qty"]
                    entry_px = pos["entry_price"]
                    if side == "long":
                        gross = (exit_px - entry_px) * qty
                    else:
                        gross = (entry_px - exit_px) * qty

                    # Costs: half-spread + slippage on entry+exit, plus borrow on shorts.
                    notional = abs(entry_px * qty) + abs(exit_px * qty)
                    cost_bps = (cfg.half_spread_bps + cfg.slippage_bps) * 2  # entry+exit
                    cost = notional * cost_bps / 1e4 / 2  # /2 because notional summed twice
                    if side == "short":
                        days_held = max(
                            1.0, (t - pos["entry_time"]).total_seconds() / 86400
                        )
                        cost += abs(entry_px * qty) * cfg.short_borrow_bps_per_day / 1e4 * days_held
                    net = gross - cost

                    equity += net
                    trades.append(XSectionTrade(
                        entry_time=pos["entry_time"], exit_time=t, symbol=sym,
                        side=side, qty=qty, entry_price=entry_px, exit_price=exit_px,
                        gross_pnl=gross, cost=cost, net_pnl=net,
                        formation_return=pos["formation_return"],
                    ))
                    del open_positions[sym]

                for sym in pending_rebalance["longs"]:
                    px = open_df.at[t, sym] if sym in open_df.columns else None
                    if px is None or pd.isna(px):
                        continue
                    px = float(px)
                    qty = int(pending_rebalance["long_allocs"].get(sym, 0.0) / px)
                    if qty <= 0:
                        continue
                    open_positions[sym] = {
                        "side": "long",
                        "qty": qty,
                        "entry_price": px,
                        "entry_time": t,
                        "formation_return": pending_rebalance["ret"].get(sym, 0.0),
                        "regime": pending_rebalance["regime_label"],
                        "market_score": pending_rebalance["regime_score"],
                        "applied_tilt": pending_rebalance.get("applied_tilt", 0.0),
                    }
                for sym in pending_rebalance["shorts"]:
                    px = open_df.at[t, sym] if sym in open_df.columns else None
                    if px is None or pd.isna(px):
                        continue
                    px = float(px)
                    qty = int(pending_rebalance["short_allocs"].get(sym, 0.0) / px)
                    if qty <= 0:
                        continue
                    open_positions[sym] = {
                        "side": "short",
                        "qty": qty,
                        "entry_price": px,
                        "entry_time": t,
                        "formation_return": pending_rebalance["ret"].get(sym, 0.0),
                        "regime": pending_rebalance["regime_label"],
                        "market_score": pending_rebalance["regime_score"],
                        "applied_tilt": pending_rebalance.get("applied_tilt", 0.0),
                    }
                pending_rebalance = None

            # Force-close at EOD flatten time on each session date. The last
            # bar-of-session fallback keeps the book flat even when the session
            # filter ends before the configured wall-clock flatten time.
            is_eod = t_local >= cfg.flatten_at_close_time or is_last_bar_of_session
            in_rebalance_window = (
                cfg.earliest_rebalance_time <= t_local <= cfg.latest_rebalance_time
            )
            since_last_rebalance = (
                None if last_rebalance_idx is None else (i - last_rebalance_idx)
            )
            should_rebalance = (
                in_rebalance_window
                and (last_rebalance_idx is None or since_last_rebalance >= bars_per_hold)
            )
            since_last_tilt_refresh = (
                None if last_tilt_refresh_idx is None else (i - last_tilt_refresh_idx)
            )
            should_refresh_tilt = (
                bars_per_tilt_refresh is not None
                and open_positions
                and in_rebalance_window
                and not should_rebalance
                and pending_rebalance is None
                and since_last_tilt_refresh is not None
                and since_last_tilt_refresh >= bars_per_tilt_refresh
                and next_ts is not None
                and next_ts.normalize() == t.normalize()
                and not is_eod
            )

            # 1. Force-close at EOD using the bar close.
            if open_positions and is_eod:
                for sym in list(open_positions.keys()):
                    pos = open_positions[sym]
                    px = close_df.at[t, sym] if sym in close_df.columns else None
                    if px is None or pd.isna(px):
                        continue
                    exit_px = float(px)
                    side = pos["side"]
                    qty = pos["qty"]
                    entry_px = pos["entry_price"]
                    if side == "long":
                        gross = (exit_px - entry_px) * qty
                    else:
                        gross = (entry_px - exit_px) * qty

                    # Costs: half-spread + slippage on entry+exit, plus borrow on shorts.
                    notional = abs(entry_px * qty) + abs(exit_px * qty)
                    cost_bps = (cfg.half_spread_bps + cfg.slippage_bps) * 2  # entry+exit
                    cost = notional * cost_bps / 1e4 / 2  # /2 because notional summed twice
                    if side == "short":
                        days_held = max(
                            1.0, (t - pos["entry_time"]).total_seconds() / 86400
                        )
                        cost += abs(entry_px * qty) * cfg.short_borrow_bps_per_day / 1e4 * days_held
                    net = gross - cost

                    equity += net
                    trades.append(XSectionTrade(
                        entry_time=pos["entry_time"], exit_time=t, symbol=sym,
                        side=side, qty=qty, entry_price=entry_px, exit_price=exit_px,
                        gross_pnl=gross, cost=cost, net_pnl=net,
                        formation_return=pos["formation_return"],
                    ))
                    del open_positions[sym]

            def _schedule_rebalance(
                longs: list[str],
                shorts: list[str],
                ret: pd.Series,
                regime_label: Optional[str],
                regime_score: Optional[int],
            ) -> Optional[dict]:
                nonlocal liquidity_capped_rebalances
                total_names = len(longs) + len(shorts)
                if total_names == 0:
                    return None

                intraday_tilt = self._intraday_market_tilt(
                    t,
                    close_df,
                    session_open_df,
                    regime_label,
                )
                if cfg.dollar_neutral:
                    long_side_multiple = max(0.0, 1.0 + intraday_tilt)
                    short_side_multiple = max(0.0, 1.0 - intraday_tilt)
                    long_target_per_name = (
                        equity * cfg.target_gross_exposure * long_side_multiple / len(longs)
                        if longs
                        else 0.0
                    )
                    short_target_per_name = (
                        equity * cfg.target_gross_exposure * short_side_multiple / len(shorts)
                        if shorts
                        else 0.0
                    )
                else:
                    shared_dollar_per_name = (
                        equity * cfg.target_gross_exposure / total_names
                    )
                    long_target_per_name = shared_dollar_per_name
                    short_target_per_name = shared_dollar_per_name

                long_max_allocs: dict[str, float] = {}
                short_max_allocs: dict[str, float] = {}
                session_date = pd.Timestamp(t.normalize())

                def _adv_cap(symbol: str, target_dollars: float) -> float:
                    if cfg.max_position_pct_of_adv is None or adv_df.empty:
                        return target_dollars
                    if session_date not in adv_df.index or symbol not in adv_df.columns:
                        return 0.0
                    adv_value = adv_df.at[session_date, symbol]
                    if pd.isna(adv_value) or float(adv_value) <= 0:
                        return 0.0
                    return min(
                        target_dollars,
                        float(adv_value) * float(cfg.max_position_pct_of_adv),
                    )

                for sym in longs:
                    long_max_allocs[sym] = _adv_cap(sym, long_target_per_name)
                for sym in shorts:
                    short_max_allocs[sym] = _adv_cap(sym, short_target_per_name)

                if cfg.dollar_neutral:
                    desired_long_total = long_target_per_name * len(longs)
                    desired_short_total = short_target_per_name * len(shorts)
                    if abs(intraday_tilt) < 1e-12:
                        actual_side_budget = min(
                            desired_long_total,
                            desired_short_total,
                            sum(long_max_allocs.values()),
                            sum(short_max_allocs.values()),
                        )
                        if actual_side_budget + 1e-9 < min(desired_long_total, desired_short_total):
                            liquidity_capped_rebalances += 1
                        long_allocs = _allocate_equal_with_caps(long_max_allocs, actual_side_budget)
                        short_allocs = _allocate_equal_with_caps(short_max_allocs, actual_side_budget)
                    else:
                        capped_long_total = sum(long_max_allocs.values())
                        capped_short_total = sum(short_max_allocs.values())
                        if (
                            capped_long_total + 1e-9 < desired_long_total
                            or capped_short_total + 1e-9 < desired_short_total
                        ):
                            liquidity_capped_rebalances += 1
                        long_allocs = _allocate_equal_with_caps(long_max_allocs, desired_long_total)
                        short_allocs = _allocate_equal_with_caps(short_max_allocs, desired_short_total)
                else:
                    capped_total = sum(long_max_allocs.values()) + sum(short_max_allocs.values())
                    desired_total = long_target_per_name * len(longs) + short_target_per_name * len(shorts)
                    if capped_total + 1e-9 < desired_total:
                        liquidity_capped_rebalances += 1
                    long_allocs = long_max_allocs
                    short_allocs = short_max_allocs

                return {
                    "longs": longs,
                    "shorts": shorts,
                    "long_allocs": long_allocs,
                    "short_allocs": short_allocs,
                    "ret": {sym: float(ret[sym]) for sym in ret.index},
                    "regime_label": regime_label,
                    "regime_score": regime_score,
                    "applied_tilt": intraday_tilt,
                }

            # 2. Generate a selection rebalance decision on signal bars. It executes on the
            # next available bar open, not on the same bar close.
            if (
                should_rebalance
                and not is_eod
                and i >= bars_per_form
                and next_ts is not None
                and next_ts.normalize() == t.normalize()
            ):
                rebalance_count += 1
                session_date = pd.Timestamp(t.normalize())
                regime_ok, regime_label, regime_score = self._regime_allows_entry(
                    market_context,
                    session_date,
                )
                if not regime_ok:
                    regime_blocked_rebalances += 1
                    last_rebalance_idx = i
                    continue
                regime_allowed_rebalances += 1
                form_idx = i - bars_per_form - cfg.formation_lag_bars
                if form_idx < 0:
                    last_rebalance_idx = i
                    continue
                # Formation return per symbol, only where both endpoints have valid data.
                ret = (close_df.iloc[i] / close_df.iloc[form_idx]) - 1.0
                # Liquidity filter
                liq_ok = dv_df.iloc[i] >= cfg.min_dollar_volume_avg
                ret = ret.where(liq_ok)
                ret = ret.dropna()

                if ret.empty:
                    last_rebalance_idx = i
                    continue

                if cfg.sector_neutral and sector_map:
                    # Demean within sector before ranking, so ranks reflect sector-relative move.
                    sector_series = pd.Series({s: sector_map.get(s) for s in ret.index})
                    ret_demeaned = ret - ret.groupby(sector_series).transform("mean")
                    rank_score = ret_demeaned
                else:
                    rank_score = ret

                rank_score = rank_score.sort_values()
                if cfg.signal_direction == "reversion":
                    longs = list(rank_score.head(cfg.n_long).index)
                    shorts = list(rank_score.tail(cfg.n_short).index)
                else:
                    longs = list(rank_score.tail(cfg.n_long).index)
                    shorts = list(rank_score.head(cfg.n_short).index)

                pending_rebalance = _schedule_rebalance(
                    longs,
                    shorts,
                    ret,
                    regime_label,
                    regime_score,
                )

                last_rebalance_idx = i
                last_tilt_refresh_idx = i

            elif should_refresh_tilt:
                longs = sorted(
                    [sym for sym, pos in open_positions.items() if pos["side"] == "long"]
                )
                shorts = sorted(
                    [sym for sym, pos in open_positions.items() if pos["side"] == "short"]
                )
                ret = pd.Series(
                    {
                        sym: float(pos.get("formation_return", 0.0))
                        for sym, pos in open_positions.items()
                    }
                )
                sample_pos = next(iter(open_positions.values()))
                regime_label = sample_pos.get("regime")
                regime_score = sample_pos.get("market_score")
                current_tilt = float(sample_pos.get("applied_tilt", 0.0))
                candidate_rebalance = _schedule_rebalance(
                    longs,
                    shorts,
                    ret,
                    regime_label,
                    regime_score,
                )
                if (
                    candidate_rebalance is not None
                    and abs(float(candidate_rebalance.get("applied_tilt", 0.0)) - current_tilt) > 1e-12
                ):
                    pending_rebalance = candidate_rebalance
                    tilt_refresh_count += 1
                last_tilt_refresh_idx = i

            # 3. Equity tick (mark to market open positions).
            mtm = 0.0
            gross_notional = 0.0
            net_notional = 0.0
            long_count = 0
            short_count = 0
            for sym, pos in open_positions.items():
                px = close_df.at[t, sym] if sym in close_df.columns else None
                if px is None or pd.isna(px):
                    continue
                position_notional = float(px) * pos["qty"]
                gross_notional += abs(position_notional)
                if pos["side"] == "long":
                    long_count += 1
                    net_notional += position_notional
                    mtm += (float(px) - pos["entry_price"]) * pos["qty"]
                else:
                    short_count += 1
                    net_notional -= position_notional
                    mtm += (pos["entry_price"] - float(px)) * pos["qty"]
            marked_equity = equity + mtm
            gross_exposure = gross_notional / marked_equity if marked_equity > 0 else 0.0
            net_exposure = net_notional / marked_equity if marked_equity > 0 else 0.0
            equity_curve.append(
                {
                    "trade_date": t,
                    "equity": marked_equity,
                    "gross_exposure": gross_exposure,
                    "net_exposure": net_exposure,
                    "long_count": long_count,
                    "short_count": short_count,
                }
            )

        eq_df = pd.DataFrame(equity_curve)
        if not eq_df.empty:
            running_peak = eq_df["equity"].cummax()
            max_dd = ((running_peak - eq_df["equity"]) / running_peak).max()
        else:
            max_dd = 0.0

        total_return_pct = ((equity / cfg.initial_cash) - 1.0) * 100.0
        win_rate = (
            sum(1 for t in trades if t.net_pnl > 0) / len(trades) if trades else 0.0
        )
        summary = {
            "total_return_pct": round(total_return_pct, 4),
            "max_drawdown_pct": round(max_dd * 100, 4),
            "total_trades": len(trades),
            "win_rate": round(win_rate, 4),
            "avg_gross_exposure": round(float(eq_df["gross_exposure"].mean()), 4)
            if not eq_df.empty
            else 0.0,
            "avg_net_exposure": round(float(eq_df["net_exposure"].mean()), 4)
            if not eq_df.empty
            else 0.0,
            "avg_long_count": round(float(eq_df["long_count"].mean()), 2)
            if not eq_df.empty
            else 0.0,
            "avg_short_count": round(float(eq_df["short_count"].mean()), 2)
            if not eq_df.empty
            else 0.0,
            "signal_direction": cfg.signal_direction,
            "rebalance_count": rebalance_count,
            "tilt_refresh_count": tilt_refresh_count,
            "regime_allowed_rebalances": regime_allowed_rebalances,
            "regime_blocked_rebalances": regime_blocked_rebalances,
            "liquidity_capped_rebalances": liquidity_capped_rebalances,
            "n_long_avg": cfg.n_long,
            "n_short_avg": cfg.n_short,
            "final_equity": round(equity, 2),
        }
        return XSectionResult(trades=trades, equity_curve=eq_df, summary=summary, config=cfg)
