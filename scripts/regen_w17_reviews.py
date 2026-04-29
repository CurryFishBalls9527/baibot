#!/usr/bin/env python3
"""Regenerate W17 (2026-04-20 → 2026-04-24) daily and weekly reviews.

Why: post-W17 we shipped three review-pipeline fixes that the original
reports didn't see:
  1. Benchmark off-by-one (analytics/benchmark.py — was dropping the
     strategy's first-day return when trimming to min(strat,bench) length)
  2. Prompt clarification distinguishing sum_trade_returns_pct (per-trade
     sum) vs strategy_return (portfolio equity-curve compounded)
  3. setup_candidates fallback in trade_outcome.log_closed_trade — swing
     variant trade_outcomes now have regime_at_entry / base_pattern /
     stage / RS instead of NULL (backfilled in DB earlier this session)

Plus: per-trade reviews for 4/20-4/21 never ran (cron started 4/22).
This script fills those in and re-runs 4/22-4/24 with the fixes applied.

Existing reports were backed up to /tmp/w17_review_backup/ first.
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from tradingagents.testing.ab_config import load_experiment
from tradingagents.testing.ab_runner import ABRunner
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.automation.config import AutomationConfig
from tradingagents.automation.trade_review import run_daily_review
from tradingagents.automation.weekly_review import run_weekly_review


W17_DAYS = [
    date(2026, 4, 20),
    date(2026, 4, 21),
    date(2026, 4, 22),
    date(2026, 4, 23),
    date(2026, 4, 24),
]
W17_WEEKLY_REVIEW_DATE = date(2026, 4, 25)  # Saturday — picks up Mon-Fri window


def main() -> int:
    config = dict(DEFAULT_CONFIG)
    config.update(AutomationConfig().to_dict())
    # Force-enable both pipelines for the regen — the live config kill-switches
    # are about cron scheduling, not module-level capability.
    config["daily_trade_review_enabled"] = True
    config["weekly_strategy_review_enabled"] = True
    # Dry-run sends output to results/{daily,weekly}_reviews_dryrun/ so we
    # don't overwrite the originals AND we don't double-insert proposals.
    config["daily_trade_review_dry_run"] = True
    config["weekly_review_dry_run"] = True

    experiment_path = ROOT / "experiments" / "paper_launch_v2.yaml"
    experiment = load_experiment(str(experiment_path))
    config["experiment_config_path"] = str(experiment_path)
    runner = ABRunner(experiment, config)

    print(f"Loaded {len(runner.orchestrators)} variants: "
          f"{', '.join(runner.orchestrators.keys())}")

    # ---- Pass 1: per-trade daily reviews ----
    daily_summary: dict = {}
    for d in W17_DAYS:
        daily_summary[d.isoformat()] = {}
        for name, orch in runner.orchestrators.items():
            try:
                # Each orchestrator's run_daily_trade_review hardcodes today;
                # call the underlying function directly with review_date.
                merged_cfg = dict(orch.config)
                merged_cfg["daily_trade_review_enabled"] = True
                res = run_daily_review(
                    db=orch.db,
                    broker=orch.broker,
                    variant_name=name,
                    config=merged_cfg,
                    review_date=d,
                )
                analyzed = res.get("analyzed", 0) if isinstance(res, dict) else 0
                closed = res.get("closed_trades", 0) if isinstance(res, dict) else 0
                daily_summary[d.isoformat()][name] = (analyzed, closed)
                if closed:
                    print(f"  {d} [{name}]: analyzed={analyzed}/{closed}")
            except Exception as e:
                daily_summary[d.isoformat()][name] = ("ERROR", str(e))
                print(f"  {d} [{name}]: ERROR {e}")

    # ---- Pass 2: weekly review for W17 ----
    print(f"\nWeekly review for W17 (review_date={W17_WEEKLY_REVIEW_DATE})...")
    weekly_summary = run_weekly_review(
        ab_runner=runner,
        config=config,
        review_date=W17_WEEKLY_REVIEW_DATE,
    )
    for variant, info in (weekly_summary.get("variants") or {}).items():
        status = info.get("status") if isinstance(info, dict) else info
        path = info.get("path") if isinstance(info, dict) else None
        print(f"  weekly[{variant}]: {status}  {path or ''}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
