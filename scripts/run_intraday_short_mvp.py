#!/usr/bin/env python3
"""MVP backtest: do mirror short-side intraday signals have gross alpha?

Tests two short-direction signals on broad250 / 15m intraday data:
  * gap_reject_short  — mirror of gap_reclaim_long.
    Setup: prior session close. Today opens UP (gap up). Within first ~6 bars,
    the price REJECTS — i.e., closes back BELOW open. Short on that confirmation
    bar. Exit at end of session or when stop hits.
  * nr4_breakdown_short — mirror of nr4_breakout.
    Setup: prior session was Narrow-Range-4 (today's range vs trailing 3 days).
    During session, close BELOW prior session low with volume confirmation.
    Short on confirmation bar.

Decision criteria:
  * Gross-only test (zero costs) on 2018 + 2020 (the periods where shorts
    theoretically help most — gap_reclaim_long fails 2020 per memory).
  * If gross R/DD > +1.5 on at least one period AND trade count > 50, escalate
    to full backtester integration. Otherwise null, drop the thread.

This is a STANDALONE simulator — does NOT touch intraday_backtester.py. It
re-implements only the minimum needed to compute gross P&L per signal, so we
can decide whether to invest in proper integration.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import time as dtime
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────── config


@dataclass
class ShortMVPConfig:
    interval_minutes: int = 15
    # Gap-reject short
    gap_min_pct: float = 0.005          # at least 0.5% gap up
    gap_max_pct: float = 0.05           # at most 5% (avoid earnings flyers)
    gap_reject_earliest_bar: int = 1     # first eligible confirmation bar
    gap_reject_latest_bar: int = 6      # last eligible
    gap_reject_min_volume_ratio: float = 1.2
    gap_reject_fraction: float = 0.5    # how far back below open before "reject"
    # NR4 breakdown
    nr4_lookback: int = 3                # range vs trailing 3 days
    nr4_earliest_bar: int = 1
    nr4_latest_bar: int = 12
    nr4_min_volume_ratio: float = 1.0
    nr4_min_breakdown_pct: float = 0.0
    # Position management
    stop_pct: float = 0.03              # stop above entry for shorts
    hold_to_eod: bool = True            # else use fixed bars-held
    bars_held: int = 12                 # if not hold_to_eod
    # Sizing (paper-style, equal-weight per signal)
    position_pct: float = 0.10          # of equity per name
    initial_cash: float = 100_000.0
    # Costs (gross test = 0; net test = realistic)
    half_spread_bps: float = 0.0
    slippage_bps: float = 0.0
    short_borrow_bps_per_day: float = 0.0


# ─────────────────────────────────────────────────────── data


def _load_universe(path: str) -> list[str]:
    payload = json.loads(Path(path).read_text())
    return payload["symbols"] if isinstance(payload, dict) else payload


def _load_symbol_frame(con, symbol: str, table: str, begin: str, end: str) -> pd.DataFrame:
    df = con.execute(
        f"SELECT ts, open, high, low, close, volume FROM {table} "
        f"WHERE symbol=? AND ts >= ? AND ts <= ? ORDER BY ts",
        [symbol, f"{begin} 00:00:00", f"{end} 23:59:59"],
    ).fetchdf()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()
    minutes = df.index.hour * 60 + df.index.minute
    df = df[(minutes >= 8 * 60 + 30) & (minutes <= 15 * 60)].copy()
    return df


def _enrich_session_features(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Add session_open, session_date, prior_session_close/low/high/is_nr,
    bar_in_session, volume_ratio, etc."""
    if df.empty:
        return df
    df = df.copy()
    df["session_date"] = df.index.normalize()
    grouped = df.groupby("session_date", sort=False)
    df["session_open"] = grouped["open"].transform("first")
    df["bar_in_session"] = grouped.cumcount()

    # Prior-session aggregates
    daily = grouped.agg(
        s_close=("close", "last"),
        s_low=("low", "min"),
        s_high=("high", "max"),
    )
    daily["range"] = daily["s_high"] - daily["s_low"]
    daily["is_nr"] = (
        daily["range"] < daily["range"].rolling(lookback_days).min().shift(1)
    ).fillna(False)
    daily["prior_session_close"] = daily["s_close"].shift(1)
    daily["prior_session_low"] = daily["s_low"].shift(1)
    daily["prior_session_high"] = daily["s_high"].shift(1)
    daily["prior_session_is_nr"] = daily["is_nr"].shift(1).fillna(False)

    df = df.join(
        daily[["prior_session_close", "prior_session_low",
               "prior_session_high", "prior_session_is_nr"]],
        on="session_date",
    )

    # Volume ratio = current bar / mean of past 20 bars in same time-of-day
    minute_of_day = df.index.hour * 60 + df.index.minute
    avg_vol = df.groupby(minute_of_day)["volume"].transform(
        lambda s: s.rolling(20, min_periods=5).mean().shift(1)
    )
    df["volume_ratio"] = df["volume"] / avg_vol.replace(0, np.nan)
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)
    return df


# ─────────────────────────────────────────────────────── signals


def detect_gap_reject_short(df: pd.DataFrame, cfg: ShortMVPConfig) -> pd.Series:
    """Gap-up + close-below-open within early bars + volume confirmation.

    Mirror of gap_reclaim_long: buyers vs sellers vote on whether gap closes.
    """
    if df.empty or "session_open" not in df.columns:
        return pd.Series(False, index=df.index)
    prior = df["prior_session_close"]
    valid = prior.notna() & (prior > 0) & df["session_open"].notna()
    gap_pct = (df["session_open"] - prior) / prior.where(valid)
    gap_ok = valid & (gap_pct >= cfg.gap_min_pct) & (gap_pct <= cfg.gap_max_pct)

    # "Reject" threshold: price has fallen back FROM session_open TOWARDS prior_close
    # by at least gap_reject_fraction of the gap.
    # session_open + reject_frac * (prior_close - session_open)
    reject_threshold = (
        df["session_open"] + cfg.gap_reject_fraction * (prior - df["session_open"])
    )
    rejected = valid & (df["close"] <= reject_threshold)
    prior_bar_close = df["close"].shift(1)
    same_session = df["session_date"] == df["session_date"].shift(1)
    prior_above_threshold = (
        prior_bar_close.notna() & same_session & (prior_bar_close > reject_threshold)
    )

    bar_ok = (df["bar_in_session"] >= cfg.gap_reject_earliest_bar) & (
        df["bar_in_session"] <= cfg.gap_reject_latest_bar
    )
    vol_ok = df["volume_ratio"] >= cfg.gap_reject_min_volume_ratio
    below_open = df["close"] < df["session_open"]
    return (gap_ok & rejected & prior_above_threshold & bar_ok & vol_ok & below_open).fillna(False)


def detect_nr4_breakdown_short(df: pd.DataFrame, cfg: ShortMVPConfig) -> pd.Series:
    if df.empty or "prior_session_low" not in df.columns:
        return pd.Series(False, index=df.index)
    prior_low = df["prior_session_low"]
    prior_is_nr = df["prior_session_is_nr"].astype(bool)
    valid = prior_low.notna() & (prior_low > 0)

    breakdown_dist_pct = (prior_low - df["close"]) / prior_low.where(valid)
    breakdown_ok = (
        valid
        & prior_is_nr
        & (df["close"] < prior_low)
        & (breakdown_dist_pct >= cfg.nr4_min_breakdown_pct)
    )

    prior_bar_close = df["close"].shift(1)
    same_session = df["session_date"] == df["session_date"].shift(1)
    prior_above_break = (
        prior_bar_close.notna() & same_session & (prior_bar_close >= prior_low)
    )

    vol_ok = df["volume_ratio"] >= cfg.nr4_min_volume_ratio
    bar_ok = (df["bar_in_session"] >= cfg.nr4_earliest_bar) & (
        df["bar_in_session"] <= cfg.nr4_latest_bar
    )
    return (breakdown_ok & prior_above_break & vol_ok & bar_ok).fillna(False)


# ─────────────────────────────────────────────────────── simulation


@dataclass
class Trade:
    symbol: str
    setup: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    gross_pnl: float
    cost: float
    net_pnl: float
    exit_reason: str


def simulate_symbol_shorts(
    df: pd.DataFrame,
    signals: dict[str, pd.Series],  # setup_name → bool series
    cfg: ShortMVPConfig,
) -> list[Trade]:
    """One-position-per-session-per-symbol short simulation.

    Entry: bar where any signal is True (first one in session wins).
    Exit: first to fire of (a) stop_price hit, (b) end of session.
    """
    trades: list[Trade] = []
    if df.empty:
        return trades
    in_position = False
    pos: Optional[dict] = None
    df = df.sort_index()
    sessions_seen: set = set()
    fired_in_session: dict = {}

    bars_per_day = sum(
        cfg.interval_minutes * 0 + 1
        for _ in pd.date_range(
            "2024-01-01 08:30:00", "2024-01-01 15:00:00",
            freq=f"{cfg.interval_minutes}min",
        )
    )
    for ts, row in df.iterrows():
        sess = row["session_date"]
        # Manage open position first
        if in_position:
            high = float(row["high"])
            close = float(row["close"])
            # Stop: shorts get hurt by HIGH
            if high >= pos["stop_price"]:
                exit_price = pos["stop_price"]  # assume stop fills exactly
                pos["exit_reason"] = "stop"
                pos["exit_ts"] = ts
                pos["exit_price"] = exit_price
                _close_position(pos, trades, cfg)
                in_position = False
                pos = None
            else:
                # End of session?
                next_idx = df.index.get_loc(ts) + 1
                next_session = (
                    df.iloc[next_idx]["session_date"] if next_idx < len(df) else None
                )
                if next_session is None or next_session != sess:
                    pos["exit_reason"] = "eod"
                    pos["exit_ts"] = ts
                    pos["exit_price"] = close
                    _close_position(pos, trades, cfg)
                    in_position = False
                    pos = None

        # Try to open new position
        if not in_position and not fired_in_session.get(sess, False):
            for setup_name, sig in signals.items():
                if ts in sig.index and bool(sig.loc[ts]):
                    entry_price = float(row["close"])  # enter at signal-bar close (proxy for next-bar open)
                    if entry_price <= 0:
                        continue
                    shares = int(cfg.initial_cash * cfg.position_pct / entry_price)
                    if shares <= 0:
                        continue
                    # Apply entry-side cost: short = sell, paid bid (lower)
                    fill_price = entry_price * (
                        1.0 - (cfg.half_spread_bps + cfg.slippage_bps) / 1e4
                    )
                    stop_price = fill_price * (1.0 + cfg.stop_pct)
                    pos = {
                        "symbol": None,  # filled by caller
                        "setup": setup_name,
                        "entry_ts": ts,
                        "entry_price": fill_price,
                        "shares": shares,
                        "stop_price": stop_price,
                    }
                    in_position = True
                    fired_in_session[sess] = True
                    break
    return trades


def _close_position(pos: dict, trades: list[Trade], cfg: ShortMVPConfig) -> None:
    entry = pos["entry_price"]
    exit_p = pos["exit_price"]
    shares = pos["shares"]
    # short P&L = (entry - exit) * shares
    gross = (entry - exit_p) * shares
    # Exit-side cost: short close = buy at ask (higher)
    exit_fill = exit_p * (1.0 + (cfg.half_spread_bps + cfg.slippage_bps) / 1e4)
    cost_from_exit = (exit_fill - exit_p) * shares
    # Borrow over hold (intraday → max 1 day)
    days_held = max(1.0, (pos["exit_ts"] - pos["entry_ts"]).total_seconds() / 86400)
    borrow_cost = entry * shares * cfg.short_borrow_bps_per_day / 1e4 * days_held
    cost = cost_from_exit + borrow_cost
    net = gross - cost
    trades.append(Trade(
        symbol=pos.get("symbol", "?"),
        setup=pos["setup"],
        entry_ts=pos["entry_ts"],
        exit_ts=pos["exit_ts"],
        entry_price=entry,
        exit_price=exit_p,
        shares=shares,
        gross_pnl=gross,
        cost=cost,
        net_pnl=net,
        exit_reason=pos["exit_reason"],
    ))


def run_period(label: str, db_path: str, begin: str, end: str,
               universe: list[str], cfg: ShortMVPConfig) -> dict:
    con = duckdb.connect(db_path, read_only=True)
    table = f"bars_{cfg.interval_minutes}m"
    all_trades: list[Trade] = []
    skipped = 0
    try:
        for sym in universe:
            df = _load_symbol_frame(con, sym, table, begin, end)
            if df.empty:
                skipped += 1
                continue
            df = _enrich_session_features(df, cfg.nr4_lookback)
            signals = {
                "gap_reject_short": detect_gap_reject_short(df, cfg),
                "nr4_breakdown_short": detect_nr4_breakdown_short(df, cfg),
            }
            trades = simulate_symbol_shorts(df, signals, cfg)
            for t in trades:
                t.symbol = sym
            all_trades.extend(trades)
    finally:
        con.close()

    # Aggregate
    if not all_trades:
        return {"label": label, "trades": 0, "gross": 0, "net": 0}
    gross = sum(t.gross_pnl for t in all_trades)
    net = sum(t.net_pnl for t in all_trades)
    wins = sum(1 for t in all_trades if t.net_pnl > 0)
    by_setup: dict = {}
    for t in all_trades:
        s = by_setup.setdefault(t.setup, {"n": 0, "gross": 0, "net": 0, "wins": 0})
        s["n"] += 1
        s["gross"] += t.gross_pnl
        s["net"] += t.net_pnl
        s["wins"] += int(t.net_pnl > 0)
    return {
        "label": label,
        "trades": len(all_trades),
        "wins": wins,
        "gross_pnl": round(gross, 2),
        "net_pnl": round(net, 2),
        "gross_pct": round(gross / cfg.initial_cash * 100, 2),
        "net_pct": round(net / cfg.initial_cash * 100, 2),
        "win_rate": round(wins / len(all_trades), 3),
        "by_setup": {
            k: {
                "n": v["n"],
                "gross_pct": round(v["gross"] / cfg.initial_cash * 100, 2),
                "net_pct": round(v["net"] / cfg.initial_cash * 100, 2),
                "wr": round(v["wins"] / v["n"], 3),
            }
            for k, v in by_setup.items()
        },
    }


def main() -> int:
    universe = _load_universe("research_data/intraday_top250_universe.json")
    print(f"Universe: {len(universe)} symbols")

    PERIODS = [
        ("2018",      "2018-01-01", "2018-12-31", "research_data/intraday_15m_2018.duckdb"),
        ("2020",      "2020-01-01", "2020-12-31", "research_data/intraday_15m_2020.duckdb"),
        ("2023_2025", "2023-01-01", "2025-12-30", "research_data/intraday_15m.duckdb"),
    ]

    print("\n=== GROSS (zero cost — does the mechanism have alpha?) ===")
    cfg_gross = ShortMVPConfig()
    for label, b, e, db in PERIODS:
        result = run_period(label, db, b, e, universe, cfg_gross)
        print(f"\n{label}:")
        print(f"  trades={result['trades']}  wins={result.get('wins',0)}  "
              f"gross={result['gross_pct']:+.2f}%  win_rate={result.get('win_rate',0):.2%}")
        for setup, s in result.get("by_setup", {}).items():
            print(f"    {setup:>22}: n={s['n']:>4}  gross={s['gross_pct']:+.2f}%  wr={s['wr']:.2%}")

    print("\n=== NET (realistic costs: 1+5 bps + 5 bps/day borrow) ===")
    cfg_net = ShortMVPConfig(
        half_spread_bps=1.0, slippage_bps=5.0, short_borrow_bps_per_day=5.0,
    )
    for label, b, e, db in PERIODS:
        result = run_period(label, db, b, e, universe, cfg_net)
        print(f"\n{label}:")
        print(f"  trades={result['trades']}  net={result['net_pct']:+.2f}%  win_rate={result.get('win_rate',0):.2%}")
        for setup, s in result.get("by_setup", {}).items():
            print(f"    {setup:>22}: n={s['n']:>4}  net={s['net_pct']:+.2f}%  wr={s['wr']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
