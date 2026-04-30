"""Regime-gated variant of nr4_breakdown_short.

Question: does a prior-day macro regime filter (SPY trend / VIX) kill the
2023_25 cost-bind without sacrificing the 2018+2020 alpha?

Tests 4 regime gates on the 2023_25 / 2020 / 2018 standard periods:
  G0: no gate (baseline from MVP)
  G1: SPY close < SPY 50-DMA (prior-day) — short-term down-regime
  G2: SPY close < SPY 200-DMA (prior-day) — long-term bear
  G3: VIX > 20 (prior-day)
  G4: SPY trailing 20-day return < 0 (prior-day) — momentum-down regime

All gates use prior-day data (shift(1)) — no lookahead.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_intraday_short_mvp import (  # noqa: E402
    ShortMVPConfig,
    Trade,
    _enrich_session_features,
    _load_symbol_frame,
    _load_universe,
    detect_nr4_breakdown_short,
    simulate_symbol_shorts,
)
from tradingagents.research.warehouse import MarketDataWarehouse  # noqa: E402


# ─────────────────────────────────────────────────────── regime indicators


def build_regime_table(begin: str, end: str) -> pd.DataFrame:
    """Build a session-date-indexed DataFrame with prior-day regime indicators.

    Padding: load 250 calendar days before `begin` so the 200-DMA has data.
    """
    pad = (pd.Timestamp(begin) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    w = MarketDataWarehouse("research_data/market_data.duckdb", read_only=True)
    try:
        spy = w.get_daily_bars("SPY", pad, end)
        vix = w.get_daily_bars("^VIX", pad, end)
    finally:
        w.close()
    if spy is None or spy.empty:
        return pd.DataFrame()

    spy = spy.copy()
    spy["sma50"] = spy["close"].rolling(50).mean()
    spy["sma200"] = spy["close"].rolling(200).mean()
    spy["roc20"] = spy["close"].pct_change(20)
    spy["session_date"] = pd.to_datetime(spy.index).normalize()

    df = spy[["close", "sma50", "sma200", "roc20", "session_date"]].rename(
        columns={"close": "spy_close"}
    )
    if vix is not None and not vix.empty:
        df["vix_close"] = vix["close"]
    else:
        df["vix_close"] = np.nan
    # Use PRIOR-DAY indicators (shift(1)) so the gate applied to today's
    # session is fully prior-data.
    df["g1_spy_below_50dma"] = (df["spy_close"] < df["sma50"]).shift(1).fillna(False)
    df["g2_spy_below_200dma"] = (df["spy_close"] < df["sma200"]).shift(1).fillna(False)
    df["g3_vix_above_20"] = (df["vix_close"] > 20).shift(1).fillna(False)
    df["g4_spy_roc20_neg"] = (df["roc20"] < 0).shift(1).fillna(False)
    df = df.set_index("session_date")
    return df


def apply_gate(signal: pd.Series, regime_df: pd.DataFrame, gate_col: str) -> pd.Series:
    """Mask out signals on session-dates where the gate is False."""
    if regime_df.empty or gate_col not in regime_df.columns:
        return signal
    if signal.empty:
        return signal
    session_dates = pd.to_datetime(signal.index).normalize()
    gate_lookup = regime_df[gate_col].reindex(session_dates).fillna(False).values
    return signal & pd.Series(gate_lookup, index=signal.index)


# ─────────────────────────────────────────────────────── main


def run_period(label: str, db_path: str, begin: str, end: str,
               universe: list[str], cfg: ShortMVPConfig,
               gate_col: str | None) -> dict:
    regime_df = build_regime_table(begin, end)
    con = duckdb.connect(db_path, read_only=True)
    table = f"bars_{cfg.interval_minutes}m"
    all_trades: list[Trade] = []
    try:
        for sym in universe:
            df = _load_symbol_frame(con, sym, table, begin, end)
            if df.empty:
                continue
            df = _enrich_session_features(df, cfg.nr4_lookback)
            sig = detect_nr4_breakdown_short(df, cfg)
            if gate_col is not None:
                sig = apply_gate(sig, regime_df, gate_col)
            signals = {"nr4_breakdown_short": sig}
            trades = simulate_symbol_shorts(df, signals, cfg)
            for t in trades:
                t.symbol = sym
            all_trades.extend(trades)
    finally:
        con.close()

    if not all_trades:
        return {"label": label, "trades": 0, "gross": 0, "net": 0, "wins": 0,
                "gross_pct": 0, "net_pct": 0, "win_rate": 0}
    gross = sum(t.gross_pnl for t in all_trades)
    net = sum(t.net_pnl for t in all_trades)
    wins = sum(1 for t in all_trades if t.net_pnl > 0)
    return {
        "label": label,
        "trades": len(all_trades),
        "wins": wins,
        "gross_pct": round(gross / cfg.initial_cash * 100, 2),
        "net_pct": round(net / cfg.initial_cash * 100, 2),
        "win_rate": round(wins / len(all_trades), 3),
    }


def main() -> int:
    universe = _load_universe("research_data/intraday_top250_universe.json")
    print(f"Universe: {len(universe)} symbols\n")

    PERIODS = [
        ("2018",      "2018-01-01", "2018-12-31", "research_data/intraday_15m_2018.duckdb"),
        ("2020",      "2020-01-01", "2020-12-31", "research_data/intraday_15m_2020.duckdb"),
        ("2023_2025", "2023-01-01", "2025-12-30", "research_data/intraday_15m.duckdb"),
    ]

    GATES = [
        ("G0_no_gate",          None),
        ("G1_spy<50dma",        "g1_spy_below_50dma"),
        ("G2_spy<200dma",       "g2_spy_below_200dma"),
        ("G3_vix>20",           "g3_vix_above_20"),
        ("G4_spy_roc20<0",      "g4_spy_roc20_neg"),
    ]

    # Realistic Alpaca-paper costs (closer to what we'd actually pay).
    cfg_paper = ShortMVPConfig(
        half_spread_bps=0.5, slippage_bps=1.0, short_borrow_bps_per_day=1.0,
    )

    print(f"{'gate':<22} {'period':<10} {'trades':>7} {'gross%':>8} {'net%':>8} {'WR':>6}")
    print("-" * 70)
    for gate_label, gate_col in GATES:
        for label, b, e, db in PERIODS:
            r = run_period(label, db, b, e, universe, cfg_paper, gate_col)
            print(f"{gate_label:<22} {r['label']:<10} {r['trades']:>7} "
                  f"{r['gross_pct']:>+8.2f} {r['net_pct']:>+8.2f} "
                  f"{r['win_rate']:>6.2%}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
