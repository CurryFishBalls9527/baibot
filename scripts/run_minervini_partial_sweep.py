#!/usr/bin/env python3
"""Phase A1 — Partial-exit + runner sweep for Minervini mechanical_v2.

Sweeps (trigger pct, partial fraction, trail pct) across bull / chop / crisis
regimes. Produces per-config metrics + a rollup CSV so we can see whether
partial-exit delivers edge vs the `partial_profit_fraction=0.0` baseline.

Uses `progressive_entries=True` to activate the partial-exit code path at
`tradingagents/research/backtester.py:604-622`. Add-ons are zeroed
(`add_on_fraction_1/2 = 0.0`) so the sweep isolates the partial effect and
preserves parity with live mechanical_v2 (which doesn't pyramid).

Regimes:
    bull   : 2023-01-01 → 2025-12-31   (post-COVID trending)
    chop   : 2018-01-01 → 2018-12-31   (late-cycle chop)
    crisis : 2020-02-01 → 2020-06-30   (COVID crash + recovery)

Usage:
    python scripts/run_minervini_partial_sweep.py
    python scripts/run_minervini_partial_sweep.py --workers 4
    python scripts/run_minervini_partial_sweep.py --regimes bull
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import json

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


# Regime windows. Data start backs off 250 bars for 200-DMA warmup.
REGIMES = {
    "bull":   {"data_start": "2022-04-01", "trade_start": "2023-01-01", "end": "2025-12-31"},
    "chop":   {"data_start": "2017-01-01", "trade_start": "2018-01-01", "end": "2018-12-31"},
    "crisis": {"data_start": "2019-01-01", "trade_start": "2020-02-01", "end": "2020-06-30"},
}


def _base_config(**overrides) -> BacktestConfig:
    """Progressive-path config with add-ons zeroed (parity with live mechanical_v2)."""
    cfg = BacktestConfig(
        progressive_entries=True,
        initial_entry_fraction=1.0,
        add_on_fraction_1=0.0,
        add_on_fraction_2=0.0,
        # Live mechanical_v2 parity
        stop_loss_pct=0.08,
        trail_stop_pct=0.10,
        breakeven_trigger_pct=0.08,
        breakeven_lock_offset_pct=0.01,
        # Relaxed entry gates so we get enough trades in chop/crisis
        min_template_score=7,
        require_volume_surge=False,
        require_market_regime=False,
        # Partial OFF in base; overridden by sweep
        partial_profit_fraction=0.0,
    )
    return replace(cfg, **overrides)


def build_grid() -> list[tuple[str, BacktestConfig]]:
    """12 experimental configs + 1 baseline control.

    Axes:
      - trigger pct: 0.08, 0.10, 0.12, 0.15
      - fraction:    0.33, 0.50
      - trail pct:   0.10 (default) for the main 8, plus a 0.08 variant at fraction=0.33
                     for 4 more configs. (Regime-aware trail is an ExitManagerV2
                     feature and isn't in the backtester; skipped here.)
    """
    grid: list[tuple[str, BacktestConfig]] = [
        ("baseline_no_partial", _base_config(partial_profit_fraction=0.0, trail_stop_pct=0.10)),
    ]

    triggers = [0.08, 0.10, 0.12, 0.15]
    for trig in triggers:
        for frac in [0.33, 0.50]:
            name = f"trig{int(trig*100):02d}_frac{int(frac*100):02d}_trail10"
            grid.append((
                name,
                _base_config(
                    partial_profit_trigger_pct=trig,
                    partial_profit_fraction=frac,
                    trail_stop_pct=0.10,
                ),
            ))

    for trig in triggers:
        name = f"trig{int(trig*100):02d}_frac33_trail08"
        grid.append((
            name,
            _base_config(
                partial_profit_trigger_pct=trig,
                partial_profit_fraction=0.33,
                trail_stop_pct=0.08,
            ),
        ))

    return grid


def compute_metrics(
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    portfolio_summary: dict,
) -> dict:
    active = summary[summary["total_trades"] > 0] if not summary.empty else summary
    total_trades = int(len(trades))
    trade_win_rate = float((trades["pnl"] > 0).mean()) if total_trades > 0 else 0.0
    avg_trade_return = float(trades["return_pct"].mean()) if total_trades > 0 else 0.0
    avg_active_return = float(active["total_return"].mean()) if not active.empty else 0.0
    avg_max_dd = float(summary["max_drawdown"].mean()) if not summary.empty else 0.0
    # R/DD at universe-avg level: avg_return / avg_max_dd. Guard against zero.
    r_dd = avg_active_return / avg_max_dd if avg_max_dd > 1e-6 else float("nan")
    partial_trades = int((trades["exit_reason"] == "partial_profit").sum()) if total_trades > 0 else 0

    return {
        "total_trades": total_trades,
        "partial_trades": partial_trades,
        "symbols_with_trades": int((summary["total_trades"] > 0).sum()) if not summary.empty else 0,
        "trade_win_rate": round(trade_win_rate, 4),
        "avg_trade_return": round(avg_trade_return, 4),
        "avg_active_symbol_return": round(avg_active_return, 4),
        "avg_max_drawdown": round(avg_max_dd, 4),
        "r_dd": round(r_dd, 4) if r_dd == r_dd else None,  # filter NaN
    }


def run_one(
    config_name: str,
    config: BacktestConfig,
    regime: str,
    data_by_symbol: dict,
    benchmark_df: pd.DataFrame,
    trade_start: str,
) -> dict:
    screener = MinerviniScreener(
        MinerviniConfig(require_fundamentals=False, require_market_uptrend=False)
    )
    backtester = MinerviniBacktester(screener=screener, config=config)
    results = backtester.backtest_universe(
        data_by_symbol, benchmark_df=benchmark_df, trade_start_date=trade_start
    )
    summary = results["summary"]
    trades = results["trades"]
    metrics = compute_metrics(summary, trades, results["portfolio_summary"])
    return {
        "config": config_name,
        "regime": regime,
        "partial_profit_trigger_pct": config.partial_profit_trigger_pct,
        "partial_profit_fraction": config.partial_profit_fraction,
        "trail_stop_pct": config.trail_stop_pct,
        **metrics,
    }, summary, trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "research_data" / "market_data.duckdb"))
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "partial_exit_sweep"))
    parser.add_argument("--workers", type=int, default=3, help="Parallel backtests (cap 3-4)")
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES.keys()))
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument(
        "--universe-file",
        default=None,
        help="Path to a JSON universe file with a 'symbols' list. Defaults to built-in LARGE_CAP_UNIVERSE.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Optional config-name filter (subset of the 13 built configs).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()

    grid = build_grid()
    if args.configs:
        wanted = set(args.configs)
        grid = [(n, c) for n, c in grid if n in wanted]
        missing = wanted - {n for n, _ in grid}
        if missing:
            print(f"WARN: unknown configs ignored: {sorted(missing)}")

    if args.universe_file:
        with open(args.universe_file) as f:
            data = json.load(f)
        universe = data.get("symbols") if isinstance(data, dict) else data
        print(f"Universe: {args.universe_file} ({len(universe)} symbols)")
    else:
        universe = LARGE_CAP_UNIVERSE
        print(f"Universe: built-in LARGE_CAP_UNIVERSE ({len(universe)} symbols)")

    print(f"Grid: {len(grid)} configs × {len(args.regimes)} regimes = {len(grid) * len(args.regimes)} backtests")

    # Preload all data once per regime (DuckDB read-only; no contention).
    warehouse = MarketDataWarehouse(args.db, read_only=True)
    preloaded: dict[str, tuple[dict, pd.DataFrame, str]] = {}
    try:
        for regime in args.regimes:
            cfg = REGIMES[regime]
            symbols = [
                s for s in universe
                if not warehouse.get_daily_bars(s, cfg["data_start"], cfg["end"]).empty
            ]
            data_by_symbol = {
                s: warehouse.get_daily_bars(s, cfg["data_start"], cfg["end"])
                for s in symbols
            }
            benchmark_df = warehouse.get_daily_bars(args.benchmark, cfg["data_start"], cfg["end"])
            preloaded[regime] = (data_by_symbol, benchmark_df, cfg["trade_start"])
            print(f"  {regime}: {len(symbols)} symbols, {len(benchmark_df)} benchmark bars")
    finally:
        warehouse.close()

    jobs = []
    for regime in args.regimes:
        data_by_symbol, benchmark_df, trade_start = preloaded[regime]
        for config_name, config in grid:
            jobs.append((config_name, config, regime, data_by_symbol, benchmark_df, trade_start))

    metric_rows: list[dict] = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, *job): (job[0], job[2])  # (config_name, regime)
            for job in jobs
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            config_name, regime = futures[fut]
            try:
                metrics, summary, trades = fut.result()
            except Exception as e:
                print(f"  [{i}/{len(jobs)}] {config_name} × {regime} FAILED: {e}")
                continue
            metric_rows.append(metrics)
            prefix = f"{config_name}_{regime}_{stamp}"
            summary.to_csv(results_dir / f"{prefix}_summary.csv", index=False)
            if not trades.empty:
                trades.to_csv(results_dir / f"{prefix}_trades.csv", index=False)
            elapsed = time.time() - t0
            print(
                f"  [{i}/{len(jobs)}] {config_name} × {regime}: "
                f"trades={metrics['total_trades']} partials={metrics['partial_trades']} "
                f"avg_ret={metrics['avg_active_symbol_return']:.4f} "
                f"maxDD={metrics['avg_max_drawdown']:.4f} R/DD={metrics['r_dd']} "
                f"({elapsed:.0f}s elapsed)"
            )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = results_dir / f"partial_sweep_{stamp}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Pivot: rows=config, cols=regime, values=R/DD and avg_return for quick read.
    print("\n=== R/DD by config × regime ===")
    pivot_rdd = metrics_df.pivot(index="config", columns="regime", values="r_dd")
    print(pivot_rdd.to_string())
    print("\n=== avg_active_symbol_return by config × regime ===")
    pivot_ret = metrics_df.pivot(index="config", columns="regime", values="avg_active_symbol_return")
    print(pivot_ret.to_string())
    print(f"\nFull metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
