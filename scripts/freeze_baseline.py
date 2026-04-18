#!/usr/bin/env python3
"""Freeze the current live strategy config as baseline B0.

Runs the current BacktestConfig (reflecting live mechanical variant behavior)
through all mandatory test periods and writes research_data/baselines/baseline_B0.json.

Every subsequent strategy change compares its treatment against this file.
Re-run only on explicit revision (bump to B1, B2, ... rather than overwriting B0).

Usage:
    python scripts/freeze_baseline.py
    python scripts/freeze_baseline.py --out research_data/baselines/baseline_B0.json
    python scripts/freeze_baseline.py --periods 2023_2025 2020
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradingagents.research.backtester import BacktestConfig
from tradingagents.research.seed_universe import load_seed_universe
from tradingagents.research.strategy_ab_runner import (
    DEFAULT_PERIODS,
    PeriodSpec,
    StrategyABRunner,
    write_json,
)
from tradingagents.research.walk_forward import WalkForwardConfig
from tradingagents.research.warehouse import MarketDataWarehouse


def build_current_live_config() -> BacktestConfig:
    """Mirror the defaults used by the live mechanical variant.

    Keep this in sync with tradingagents/automation/config.py and
    tradingagents/portfolio/exit_manager.py. If the live defaults change,
    this baseline must be re-frozen (bump to B1).
    """
    return BacktestConfig(
        initial_cash=100_000.0,
        max_position_pct=0.10,
        risk_per_trade=0.012,
        stop_loss_pct=0.05,
        trail_stop_pct=0.10,
        max_hold_days=60,
        breakeven_trigger_pct=0.05,
        partial_profit_trigger_pct=0.12,
        partial_profit_fraction=0.33,
        use_50dma_exit=True,
        max_positions=6,
    )


def build_research_config() -> BacktestConfig:
    """Looser config for statistical power — produces many more trades per period.

    Used as a SECOND baseline so every change is tested twice: once against the
    live baseline (safety check), once against this (signal check via paired t-test).
    Does NOT modify the live strategy in any way.
    """
    return BacktestConfig(
        initial_cash=100_000.0,
        max_position_pct=0.08,
        risk_per_trade=0.02,
        stop_loss_pct=0.05,
        trail_stop_pct=0.10,
        max_hold_days=60,
        breakeven_trigger_pct=0.05,
        partial_profit_trigger_pct=0.12,
        partial_profit_fraction=0.33,
        use_50dma_exit=True,
        max_positions=12,
    )


def build_wf_config(flavor: str = "live") -> WalkForwardConfig:
    if flavor == "research":
        return WalkForwardConfig(
            rebalance_frequency="weekly",
            min_template_score=5,
            min_rs_percentile=60.0,
            max_screen_candidates=80,
        )
    return WalkForwardConfig(
        rebalance_frequency="weekly",
        min_template_score=6,
        min_rs_percentile=70.0,
    )


def filter_periods(names):
    if not names:
        return list(DEFAULT_PERIODS)
    keep = {n for n in names}
    return [p for p in DEFAULT_PERIODS if p.name in keep]


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze current live strategy config as baseline B0.")
    parser.add_argument(
        "--out",
        default="research_data/baselines/baseline_B0.json",
        help="Output JSON path (default: research_data/baselines/baseline_B0.json)",
    )
    parser.add_argument(
        "--db",
        default="research_data/market_data.duckdb",
        help="DuckDB warehouse path",
    )
    parser.add_argument(
        "--universe",
        default="research_data/seed_universe.json",
        help="Seed universe JSON",
    )
    parser.add_argument(
        "--periods",
        nargs="*",
        default=None,
        help=f"Subset of period names. Default: all. Available: {[p.name for p in DEFAULT_PERIODS]}",
    )
    parser.add_argument("--log", default="INFO")
    parser.add_argument(
        "--flavor",
        choices=["live", "research"],
        default="live",
        help="'live' = mirror live mechanical config; 'research' = looser for statistical power",
    )
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        help="Override BacktestConfig field (e.g. use_dead_money_stop=true). Used to freeze B1, B2, ...",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log), format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    out_path = Path(args.out)
    if out_path.exists():
        logger.error(
            "Refusing to overwrite existing baseline at %s. "
            "Bump to B1 (e.g. baseline_B1.json) on explicit revision.",
            out_path,
        )
        return 2

    if not Path(args.db).exists():
        logger.error("Warehouse not found: %s", args.db)
        return 2

    seed_symbols = load_seed_universe(args.universe)
    logger.info("Loaded %d seed symbols from %s", len(seed_symbols), args.universe)

    warehouse = MarketDataWarehouse(db_path=args.db, read_only=True)

    periods = filter_periods(args.periods)
    logger.info("Running %d period(s): %s", len(periods), [p.name for p in periods])

    runner = StrategyABRunner(
        warehouse=warehouse,
        seed_symbols=seed_symbols,
        wf_config=build_wf_config(args.flavor),
    )

    config = build_research_config() if args.flavor == "research" else build_current_live_config()
    if args.sets:
        from dataclasses import asdict as _asdict
        from tradingagents.research.backtester import BacktestConfig as _BC
        data = _asdict(config)
        for s in args.sets:
            k, v = s.split("=", 1)
            vl = v.strip().lower()
            if vl in ("true", "false"):
                data[k.strip()] = vl == "true"
            else:
                try:
                    data[k.strip()] = float(v) if "." in v else int(v)
                except ValueError:
                    data[k.strip()] = v.strip()
        config = _BC(**data)
    baseline = runner.run_baseline(config, periods)
    baseline["flavor"] = args.flavor
    baseline["overrides"] = {s.split("=", 1)[0]: s.split("=", 1)[1] for s in (args.sets or [])}
    baseline["__meta__"] = {
        "universe_path": args.universe,
        "n_seed_symbols": len(seed_symbols),
        "warehouse_db": args.db,
    }
    write_json(baseline, out_path)
    logger.info("Baseline written to %s", out_path)
    for p in baseline["periods"]:
        s = p["summary"]
        logger.info(
            "  %s: return=%+.2f%% DD=%.2f%% sharpe=%.2f trades=%d winrate=%.1f%%",
            p["period"]["name"],
            s["total_return"] * 100,
            s["max_drawdown"] * 100,
            s["sharpe_ratio"],
            s["total_trades"],
            s["trade_win_rate"] * 100,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
