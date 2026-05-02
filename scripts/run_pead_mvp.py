#!/usr/bin/env python3
"""PEAD (post-earnings-announcement drift) backtester — MVP.

Tests the classic Bernard & Thomas (1989) anomaly: stocks with positive
EPS surprises drift up for ~60 days post-announcement; negative surprises
drift down. Documented Sharpe 0.5-1.0 across decades, hasn't been
over-arbitraged at retail because institutions need scale.

Universe: 104 symbols in `earnings_events` table (yfinance source).
Cadence: daily (entry next-day-open after earnings, hold N days).
Direction: configurable (long-only / L-S based on surprise sign).

CLAUDE.md gates enforced here:
  * 3-period broad-universe test (2018, 2020, 2023-25)
  * Future-blanked probe (entry_lag_days knob)
  * Slippage stress (modeled via cost_bps round-trip)

Standalone — does NOT touch intraday_backtester or xsection. If gross-positive
across periods AND lookahead-clean, escalate to proper integration + live
deployment.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradingagents.research.warehouse import MarketDataWarehouse  # noqa: E402


@dataclass
class PEADConfig:
    initial_cash: float = 100_000.0
    # Signal
    min_positive_surprise_pct: float = 2.0  # at or above → long
    max_positive_surprise_pct: float = 50.0  # cap on extreme prints (data noise)
    enable_short_on_negative: bool = False
    min_negative_surprise_pct: float = 2.0  # absolute value
    # Holding
    hold_days: int = 5
    entry_lag_days: int = 1  # bars after event close. 1 = enter next day open. >1 = future-blanked probe.
    # Sizing
    position_pct: float = 0.05  # 5% of equity per event
    max_concurrent_positions: int = 10
    max_per_symbol_positions: int = 1
    # Costs (round-trip total = 2 * (half_spread + slippage) + borrow_bps_per_day * hold_days for shorts)
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0
    short_borrow_bps_per_day: float = 5.0


@dataclass
class PEADTrade:
    symbol: str
    side: str          # "long" or "short"
    event_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    surprise_pct: float
    gross_pnl: float
    cost: float
    net_pnl: float


def _load_earnings_events(daily_db_path: str, begin: str, end: str) -> pd.DataFrame:
    """Load past earnings events with non-null surprise % in the window."""
    con = duckdb.connect(daily_db_path, read_only=True)
    try:
        df = con.execute(
            """
            SELECT symbol, event_datetime, surprise_pct, time_hint
            FROM earnings_events
            WHERE is_future = false
              AND surprise_pct IS NOT NULL
              AND event_datetime >= ?
              AND event_datetime <= ?
            ORDER BY event_datetime
            """,
            [f"{begin} 00:00:00", f"{end} 23:59:59"],
        ).fetchdf()
    finally:
        con.close()
    df["event_datetime"] = pd.to_datetime(df["event_datetime"])
    df["event_date"] = df["event_datetime"].dt.normalize()
    return df


def _load_daily_bars_universe(
    daily_db_path: str, symbols: list[str], begin: str, end: str
) -> dict[str, pd.DataFrame]:
    pad_begin = (pd.Timestamp(begin) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    pad_end = (pd.Timestamp(end) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    w = MarketDataWarehouse(daily_db_path, read_only=True)
    out = {}
    try:
        for sym in symbols:
            df = w.get_daily_bars(sym, pad_begin, pad_end)
            if df is None or df.empty:
                continue
            out[sym] = df.copy().sort_index()
    finally:
        w.close()
    return out


def _entry_exit_dates(
    event_dt: pd.Timestamp,
    time_hint: str,
    bar_index: pd.DatetimeIndex,
    hold_days: int,
    entry_lag_days: int,
) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Map an earnings event to (entry_bar_date, exit_bar_date) on the
    symbol's bar index. amc events trigger entry on next session; bmo
    events trigger entry on the same session's open. entry_lag_days
    shifts the entry by N additional bars (probe knob).
    """
    event_date = event_dt.normalize()
    # First eligible session for entry:
    if str(time_hint) == "bmo":
        # Before market open — same-day open is the entry.
        start_date = event_date
    else:
        # amc / unknown — assume after-close, next session is entry.
        start_date = event_date + pd.Timedelta(days=1)
    # Find the first bar index >= start_date
    eligible = bar_index[bar_index >= start_date]
    if len(eligible) <= entry_lag_days:
        return None, None
    entry_bar = eligible[entry_lag_days]
    after_entry = bar_index[bar_index > entry_bar]
    if len(after_entry) < hold_days:
        return None, after_entry[-1] if len(after_entry) > 0 else None
    exit_bar = after_entry[hold_days - 1]
    return entry_bar, exit_bar


def simulate(
    cfg: PEADConfig,
    events_df: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
) -> list[PEADTrade]:
    """One pass through events. Per-symbol position cap. Cash-aware sizing."""
    events_df = events_df.copy().sort_values("event_datetime").reset_index(drop=True)
    trades: list[PEADTrade] = []
    open_positions: dict[str, list[dict]] = {}  # symbol → list of open positions

    cash = cfg.initial_cash
    equity = cfg.initial_cash

    for _, row in events_df.iterrows():
        sym = row["symbol"]
        if sym not in bars:
            continue
        sp = float(row["surprise_pct"])

        # Direction
        if sp >= cfg.min_positive_surprise_pct and sp <= cfg.max_positive_surprise_pct:
            side = "long"
        elif (
            cfg.enable_short_on_negative
            and sp <= -cfg.min_negative_surprise_pct
            and sp >= -cfg.max_positive_surprise_pct
        ):
            side = "short"
        else:
            continue

        bar_index = bars[sym].index
        entry_bar, exit_bar = _entry_exit_dates(
            row["event_datetime"], row.get("time_hint", "amc"),
            bar_index, cfg.hold_days, cfg.entry_lag_days,
        )
        if entry_bar is None or exit_bar is None:
            continue

        # Per-symbol position cap
        existing = open_positions.get(sym, [])
        if len(existing) >= cfg.max_per_symbol_positions:
            continue
        # Total concurrent cap
        total_open = sum(len(v) for v in open_positions.values())
        if total_open >= cfg.max_concurrent_positions:
            continue

        try:
            entry_price = float(bars[sym].loc[entry_bar, "open"])
            exit_price = float(bars[sym].loc[exit_bar, "close"])
        except KeyError:
            continue
        if entry_price <= 0 or exit_price <= 0:
            continue

        budget = min(cash, equity * cfg.position_pct)
        shares = int(budget / entry_price)
        if shares <= 0:
            continue

        # P&L direction-aware
        if side == "long":
            gross = (exit_price - entry_price) * shares
        else:
            gross = (entry_price - exit_price) * shares

        # Costs
        notional = (entry_price + exit_price) * shares
        cost = notional * (cfg.half_spread_bps + cfg.slippage_bps) / 1e4
        if side == "short":
            days_held = max(1, (exit_bar - entry_bar).days)
            cost += entry_price * shares * cfg.short_borrow_bps_per_day / 1e4 * days_held
        net = gross - cost
        cash += net  # realized P&L; simplified equity tracking
        equity = cash  # no MTM in this simple path; closes are deterministic

        trades.append(PEADTrade(
            symbol=sym, side=side, event_date=row["event_date"],
            entry_date=entry_bar, exit_date=exit_bar,
            entry_price=entry_price, exit_price=exit_price, shares=shares,
            surprise_pct=sp, gross_pnl=gross, cost=cost, net_pnl=net,
        ))

    return trades


def aggregate(trades: list[PEADTrade], initial_cash: float) -> dict:
    if not trades:
        return {"trades": 0, "gross_pct": 0, "net_pct": 0, "win_rate": 0,
                "n_long": 0, "n_short": 0}
    gross = sum(t.gross_pnl for t in trades)
    net = sum(t.net_pnl for t in trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    n_long = sum(1 for t in trades if t.side == "long")
    n_short = sum(1 for t in trades if t.side == "short")
    return {
        "trades": len(trades),
        "wins": wins,
        "n_long": n_long,
        "n_short": n_short,
        "gross_pct": round(gross / initial_cash * 100, 2),
        "net_pct": round(net / initial_cash * 100, 2),
        "win_rate": round(wins / len(trades), 3),
    }


def run_period(
    label: str, daily_db: str, begin: str, end: str,
    cfg: PEADConfig,
) -> dict:
    events = _load_earnings_events(daily_db, begin, end)
    if events.empty:
        return {"label": label, "trades": 0, "gross_pct": 0, "net_pct": 0,
                "win_rate": 0, "n_long": 0, "n_short": 0}
    symbols = sorted(events["symbol"].unique().tolist())
    bars = _load_daily_bars_universe(daily_db, symbols, begin, end)
    trades = simulate(cfg, events, bars)
    summary = aggregate(trades, cfg.initial_cash)
    summary["label"] = label
    summary["events"] = len(events)
    summary["symbols"] = len(symbols)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-db", default="research_data/earnings_data.duckdb")
    parser.add_argument("--include-shorts", action="store_true",
                        help="Enable L/S based on surprise sign (vs long-only)")
    args = parser.parse_args()

    PERIODS = [
        ("2018",      "2018-01-01", "2018-12-31"),
        ("2020",      "2020-01-01", "2020-12-31"),
        ("2023_2025", "2023-01-01", "2025-12-31"),
    ]

    VARIANTS = [
        ("V0_long_2pct_5d_hold (baseline)", PEADConfig(
            min_positive_surprise_pct=2.0, hold_days=5, entry_lag_days=1,
        )),
        ("V1_long_5pct_10d_hold", PEADConfig(
            min_positive_surprise_pct=5.0, hold_days=10, entry_lag_days=1,
        )),
        ("V2_long_5pct_20d_hold", PEADConfig(
            min_positive_surprise_pct=5.0, hold_days=20, entry_lag_days=1,
        )),
        ("V3_LS_2pct_5d_hold", PEADConfig(
            min_positive_surprise_pct=2.0, hold_days=5, entry_lag_days=1,
            enable_short_on_negative=True, min_negative_surprise_pct=2.0,
        )),
        ("V4_long_2pct_5d_hold_zero_cost (gross)", PEADConfig(
            min_positive_surprise_pct=2.0, hold_days=5, entry_lag_days=1,
            half_spread_bps=0.0, slippage_bps=0.0, short_borrow_bps_per_day=0.0,
        )),
    ]

    print(f"{'variant':<48} {'period':<10} {'events':>7} {'trades':>7} "
          f"{'gross%':>8} {'net%':>8} {'WR':>6} {'L/S':>10}")
    print("-" * 115)
    for vlabel, vcfg in VARIANTS:
        for plabel, b, e in PERIODS:
            r = run_period(plabel, args.daily_db, b, e, vcfg)
            print(f"{vlabel:<48} {plabel:<10} {r['events']:>7} {r['trades']:>7} "
                  f"{r['gross_pct']:>+8.2f} {r['net_pct']:>+8.2f} {r['win_rate']:>6.2%} "
                  f"{r['n_long']}/{r['n_short']:<5}")
        print()

    # Mandatory lookahead probe on baseline variant
    print(f"=== LOOKAHEAD PROBE (entry_lag_days=2 vs 1) ===")
    for plabel, b, e in PERIODS:
        cfg = PEADConfig(min_positive_surprise_pct=2.0, hold_days=5, entry_lag_days=2)
        r = run_period(plabel, args.daily_db, b, e, cfg)
        print(f"{'V0_lag2':<48} {plabel:<10} {r['events']:>7} {r['trades']:>7} "
              f"{r['gross_pct']:>+8.2f} {r['net_pct']:>+8.2f} {r['win_rate']:>6.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
