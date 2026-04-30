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

    # Sector neutrality (toggle for A/B). Pulls sector from fundamentals_snapshots.
    sector_neutral: bool = False

    # Bar timing
    interval_minutes: int = 15
    formation_minutes: int = 60   # rank window
    hold_minutes: int = 60        # holding period
    formation_lag_bars: int = 0   # future-blanked probe (delays rank by N bars)

    # Cross-section selection
    n_long: int = 20    # bottom-N by formation return
    n_short: int = 20   # top-N by formation return
    dollar_neutral: bool = True  # equal $ long vs short notional

    # Position sizing
    target_gross_exposure: float = 1.0  # 1.0 = 100% long + 100% short = 200% gross
    # If False, each name gets target_gross_exposure / (n_long+n_short) of equity.

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


class XSectionReversionBacktester:
    """Walk-the-time-axis backtester for cross-sectional reversion."""

    def __init__(self, config: XSectionReversionConfig):
        self.config = config

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

        sector_map = _load_sector_map(daily_db_path) if cfg.sector_neutral else {}

        # Master time index = sorted union of all bar times where ANY symbol has data.
        master_index = sorted(set().union(*[df.index for df in frames.values()]))
        master_index = pd.DatetimeIndex(master_index)

        # Pre-build close + dollar-volume DataFrames for vectorized ranking.
        close_df = pd.DataFrame(
            {sym: df["close"] for sym, df in frames.items()}, index=master_index
        )
        dv_df = pd.DataFrame(
            {sym: _compute_dollar_volume(df, liquidity_lookback_bars) for sym, df in frames.items()},
            index=master_index,
        )

        equity = cfg.initial_cash
        equity_curve: list[dict] = []
        trades: list[XSectionTrade] = []

        # Track open positions: dict[symbol] -> {side, qty, entry_price, entry_time}
        open_positions: dict[str, dict] = {}

        # Walk every bar; act only at rebalance bars.
        # Rebalance schedule: every hold_minutes inside the session, between
        # earliest_rebalance and latest_rebalance.
        last_rebalance_idx: Optional[int] = None
        for i, t in enumerate(master_index):
            t_local = t.time()

            # Force-close at EOD flatten time on each session date.
            is_eod = t_local >= cfg.flatten_at_close_time
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

            # 1. Close existing positions if we're rebalancing OR at EOD.
            if open_positions and (should_rebalance or is_eod):
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

            # 2. Open new positions on rebalance bars (skip if EOD).
            if should_rebalance and not is_eod and i >= bars_per_form:
                form_idx = i - bars_per_form - cfg.formation_lag_bars
                if form_idx < 0:
                    last_rebalance_idx = i
                    continue
                t_formation = master_index[form_idx]
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
                longs = list(rank_score.head(cfg.n_long).index)
                shorts = list(rank_score.tail(cfg.n_short).index)

                # Sizing: equal $ per name. Use equity * gross_exposure / total_names.
                total_names = len(longs) + len(shorts)
                if total_names == 0:
                    last_rebalance_idx = i
                    continue
                target_dollar_per_name = (
                    equity * cfg.target_gross_exposure / total_names
                )

                for sym in longs:
                    px = float(close_df.at[t, sym])
                    qty = int(target_dollar_per_name / px)
                    if qty <= 0:
                        continue
                    open_positions[sym] = {
                        "side": "long",
                        "qty": qty,
                        "entry_price": px,
                        "entry_time": t,
                        "formation_return": float(ret[sym]),
                    }
                for sym in shorts:
                    px = float(close_df.at[t, sym])
                    qty = int(target_dollar_per_name / px)
                    if qty <= 0:
                        continue
                    open_positions[sym] = {
                        "side": "short",
                        "qty": qty,
                        "entry_price": px,
                        "entry_time": t,
                        "formation_return": float(ret[sym]),
                    }

                last_rebalance_idx = i

            # 3. Equity tick (mark to market open positions).
            mtm = 0.0
            for sym, pos in open_positions.items():
                px = close_df.at[t, sym] if sym in close_df.columns else None
                if px is None or pd.isna(px):
                    continue
                if pos["side"] == "long":
                    mtm += (float(px) - pos["entry_price"]) * pos["qty"]
                else:
                    mtm += (pos["entry_price"] - float(px)) * pos["qty"]
            equity_curve.append({"trade_date": t, "equity": equity + mtm})

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
            "n_long_avg": cfg.n_long,
            "n_short_avg": cfg.n_short,
            "final_equity": round(equity, 2),
        }
        return XSectionResult(trades=trades, equity_curve=eq_df, summary=summary, config=cfg)
