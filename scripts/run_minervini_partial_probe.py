#!/usr/bin/env python3
"""Phase A1 future-blanked probe — lag the partial trigger by 1 bar.

Re-runs the Phase A1 winner (`trig15_frac50_trail10`) with
`partial_trigger_lag_bars=1` across all three regimes and compares against
the lag=0 baseline. If R/DD drops by > 0.10 when the trigger is shifted to
strictly-past data, the "edge" was same-bar lookahead and should not ship.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tradingagents.backtesting.screener import LARGE_CAP_UNIVERSE
from tradingagents.research import (
    MarketDataWarehouse,
    MinerviniBacktester,
    MinerviniConfig,
    MinerviniScreener,
)
from tradingagents.research.backtester import BacktestConfig

REGIMES = {
    "bull":   {"data_start": "2022-04-01", "trade_start": "2023-01-01", "end": "2025-12-31"},
    "chop":   {"data_start": "2017-01-01", "trade_start": "2018-01-01", "end": "2018-12-31"},
    "crisis": {"data_start": "2019-01-01", "trade_start": "2020-02-01", "end": "2020-06-30"},
}


def _winner_config(lag: int) -> BacktestConfig:
    return BacktestConfig(
        progressive_entries=True,
        initial_entry_fraction=1.0,
        add_on_fraction_1=0.0,
        add_on_fraction_2=0.0,
        stop_loss_pct=0.08,
        trail_stop_pct=0.10,
        breakeven_trigger_pct=0.08,
        breakeven_lock_offset_pct=0.01,
        min_template_score=7,
        require_volume_surge=False,
        require_market_regime=False,
        partial_profit_trigger_pct=0.15,
        partial_profit_fraction=0.50,
        partial_trigger_lag_bars=lag,
    )


def run(regime: str, lag: int, warehouse: MarketDataWarehouse) -> dict:
    cfg = REGIMES[regime]
    symbols = [
        s for s in LARGE_CAP_UNIVERSE
        if not warehouse.get_daily_bars(s, cfg["data_start"], cfg["end"]).empty
    ]
    data_by_symbol = {
        s: warehouse.get_daily_bars(s, cfg["data_start"], cfg["end"]) for s in symbols
    }
    benchmark_df = warehouse.get_daily_bars("SPY", cfg["data_start"], cfg["end"])

    screener = MinerviniScreener(
        MinerviniConfig(require_fundamentals=False, require_market_uptrend=False)
    )
    bt = MinerviniBacktester(screener=screener, config=_winner_config(lag))
    results = bt.backtest_universe(
        data_by_symbol, benchmark_df=benchmark_df, trade_start_date=cfg["trade_start"]
    )
    summary = results["summary"]
    trades = results["trades"]
    active = summary[summary["total_trades"] > 0] if not summary.empty else summary
    avg_ret = float(active["total_return"].mean()) if not active.empty else 0.0
    avg_dd = float(summary["max_drawdown"].mean()) if not summary.empty else 0.0
    r_dd = avg_ret / avg_dd if avg_dd > 1e-6 else float("nan")
    partials = int((trades["exit_reason"] == "partial_profit").sum()) if not trades.empty else 0

    return {
        "regime": regime,
        "lag_bars": lag,
        "total_trades": int(len(trades)),
        "partials": partials,
        "avg_ret": round(avg_ret, 4),
        "avg_dd": round(avg_dd, 4),
        "r_dd": round(r_dd, 4) if r_dd == r_dd else None,
    }


def main():
    results_dir = ROOT / "results" / "partial_exit_sweep"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()

    warehouse = MarketDataWarehouse(str(ROOT / "research_data" / "market_data.duckdb"), read_only=True)
    rows = []
    try:
        t0 = time.time()
        for lag in (0, 1):
            for regime in REGIMES:
                row = run(regime, lag, warehouse)
                rows.append(row)
                print(
                    f"  lag={lag} {regime:6s}: trades={row['total_trades']:3d} partials={row['partials']:3d} "
                    f"avg_ret={row['avg_ret']:+.4f} avg_dd={row['avg_dd']:.4f} R/DD={row['r_dd']} "
                    f"({time.time()-t0:.0f}s)"
                )
    finally:
        warehouse.close()

    df = pd.DataFrame(rows)
    # Pivot for easy read
    pivot = df.pivot(index="regime", columns="lag_bars", values="r_dd")
    pivot.columns = [f"lag{c}" for c in pivot.columns]
    pivot["delta_rdd"] = pivot["lag1"] - pivot["lag0"]
    pivot["pass"] = pivot["delta_rdd"].abs() <= 0.10
    print("\n=== Probe result ===")
    print(pivot.to_string())
    out_path = results_dir / f"partial_probe_{stamp}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    if pivot["delta_rdd"].abs().max() > 0.10:
        print(
            "\nFAIL: lag-1 trigger diverges from lag-0 by > 0.10 R/DD on at "
            "least one regime. Same-bar close may be lookahead."
        )
    else:
        print("\nPASS: lag-1 trigger within 0.10 R/DD of lag-0 on all regimes.")


if __name__ == "__main__":
    main()
