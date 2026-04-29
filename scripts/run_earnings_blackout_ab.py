#!/usr/bin/env python3
"""Sweep earnings-report blackout (entry × exit) for Minervini.

Grid:
  * earnings_blackout_entry_days in {0, 3, 5, 7}
  * earnings_flatten_days_before in {0, 1, 2, 3}
  * periods from DEFAULT_PERIODS (2023_2025 IS + 2020 + 2018 OOS typical)

For each cell, run through the same walk-forward machinery as
scripts/run_strategy_ab.py (live flavor by default) and emit one CSV with
metrics plus one JSON per cell for deeper inspection.

Example:
  python scripts/run_earnings_blackout_ab.py
  python scripts/run_earnings_blackout_ab.py --entry 0,3 --exit 0,1 --periods 2023_2025
  python scripts/run_earnings_blackout_ab.py --flavor research
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradingagents.research.backtester import BacktestConfig  # noqa: E402
from tradingagents.research.seed_universe import load_seed_universe  # noqa: E402
from tradingagents.research.strategy_ab_runner import (  # noqa: E402
    DEFAULT_PERIODS,
    PeriodSpec,
    run_single_period,
)
from tradingagents.research.walk_forward import WalkForwardConfig  # noqa: E402
from tradingagents.research.warehouse import MarketDataWarehouse  # noqa: E402

from scripts.freeze_baseline import (  # noqa: E402
    build_current_live_config,
    build_research_config,
    build_wf_config,
)
from scripts.run_strategy_ab import (  # noqa: E402
    ACCEPTED_CHANGES,
    apply_overrides,
    apply_wf_overrides,
    pick_db,
    split_overrides,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("er_blackout_ab")


def _metric_row(
    period: PeriodSpec,
    entry_days: int,
    exit_days: int,
    result,
    er_exit_count: int,
) -> dict:
    summary = result.portfolio_result.summary or {}
    trades = result.portfolio_result.trades
    worst_trade_return = None
    if trades is not None and not trades.empty and "return_pct" in trades.columns:
        worst_trade_return = float(trades["return_pct"].min())
    return {
        "period": period.name,
        "entry_blackout_days": entry_days,
        "exit_flatten_days": exit_days,
        "total_return": summary.get("total_return"),
        "max_drawdown": summary.get("max_drawdown"),
        "sharpe_ratio": summary.get("sharpe_ratio"),
        "r_over_dd": (
            summary.get("total_return") / summary.get("max_drawdown")
            if summary.get("max_drawdown")
            else None
        ),
        "total_trades": summary.get("total_trades"),
        "trade_win_rate": summary.get("trade_win_rate"),
        "avg_trade_return": summary.get("avg_trade_return"),
        "worst_trade_return": worst_trade_return,
        "avg_exposure_pct": summary.get("avg_exposure_pct"),
        "er_driven_exit_count": er_exit_count,
    }


def _count_er_exits(result) -> int:
    trades = result.portfolio_result.trades
    if trades is None or trades.empty or "exit_reason" not in trades.columns:
        return 0
    return int((trades["exit_reason"] == "earnings_flatten").sum())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entry", default="0,3,5,7",
                   help="Comma-separated earnings_blackout_entry_days values")
    p.add_argument("--exit", default="0,1,2,3",
                   help="Comma-separated earnings_flatten_days_before values")
    p.add_argument("--last-bar-exit", action="store_true",
                   help="Use just-in-time last-safe-bar flatten; ignores --exit")
    p.add_argument("--periods", nargs="*", default=None,
                   help="Subset of period names (default: all DEFAULT_PERIODS)")
    p.add_argument("--flavor", choices=["live", "research"], default="live")
    p.add_argument("--out-dir", default="results/earnings_blackout_ab")
    p.add_argument("--csv", default="results/earnings_blackout_ab.csv")
    args = p.parse_args()

    entry_values = [int(x) for x in args.entry.split(",") if x.strip() != ""]
    if args.last_bar_exit:
        # Encode the last-bar-only mode as sentinel -1 for the grid label.
        exit_values = [-1]
    else:
        exit_values = [int(x) for x in args.exit.split(",") if x.strip() != ""]

    periods = [p for p in DEFAULT_PERIODS if not args.periods or p.name in args.periods]
    if not periods:
        logger.error("No periods selected (available: %s)",
                     [p.name for p in DEFAULT_PERIODS])
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_symbols = load_seed_universe(
        os.environ.get("AB_SEED_UNIVERSE", "research_data/seed_universe.json")
    )
    raw_wf = build_wf_config(args.flavor)
    raw_base = (
        build_research_config()
        if args.flavor == "research"
        else build_current_live_config()
    )

    accepted_bt, accepted_wf = split_overrides(ACCEPTED_CHANGES)
    base_config = apply_overrides(raw_base, accepted_bt)
    base_wf = apply_wf_overrides(raw_wf, accepted_wf)

    # One warehouse per DB, shared across cells.
    wh_cache: dict[str, MarketDataWarehouse] = {}

    rows: list[dict] = []
    grid = list(itertools.product(entry_values, exit_values))
    total_cells = len(grid) * len(periods)
    done = 0
    for (entry_days, exit_days) in grid:
        if exit_days == -1:
            cell_overrides = {
                "earnings_blackout_entry_days": entry_days,
                "earnings_flatten_days_before": 0,
                "earnings_flatten_last_bar_only": True,
            }
            cell_label = f"e{entry_days}_xLAST"
        else:
            cell_overrides = {
                "earnings_blackout_entry_days": entry_days,
                "earnings_flatten_days_before": exit_days,
                "earnings_flatten_last_bar_only": False,
            }
            cell_label = f"e{entry_days}_x{exit_days}"
        cell_cfg = apply_overrides(base_config, cell_overrides)
        cell_json = {"cell": cell_label, "flavor": args.flavor, "periods": []}

        for period in periods:
            done += 1
            db_path = pick_db(period.name)
            warehouse = wh_cache.setdefault(
                db_path,
                MarketDataWarehouse(db_path=db_path, read_only=True),
            )
            logger.info(
                "[%d/%d] cell=%s period=%s db=%s",
                done, total_cells, cell_label, period.name, db_path,
            )
            result = run_single_period(
                warehouse, seed_symbols, period, cell_cfg, base_wf,
            )
            er_exits = _count_er_exits(result)
            row = _metric_row(period, entry_days, exit_days, result, er_exits)
            rows.append(row)
            cell_json["periods"].append(row)

        (out_dir / f"cell_{cell_label}.json").write_text(
            json.dumps(cell_json, default=str, indent=2)
        )

    df = pd.DataFrame(rows)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    logger.info("Wrote %d rows to %s", len(df), args.csv)

    # Pretty summary
    for period_name in [p.name for p in periods]:
        sub = df[df["period"] == period_name]
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            index="entry_blackout_days",
            columns="exit_flatten_days",
            values="r_over_dd",
        )
        print(f"\n=== {period_name} — R/DD by (entry_days, exit_days) ===")
        print(pivot.to_string(float_format=lambda v: f"{v:.3f}" if v is not None else "na"))
        pivot_ret = sub.pivot_table(
            index="entry_blackout_days",
            columns="exit_flatten_days",
            values="total_return",
        )
        print(f"\n=== {period_name} — total_return by (entry_days, exit_days) ===")
        print((pivot_ret * 100).to_string(float_format=lambda v: f"{v:+5.1f}%"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
