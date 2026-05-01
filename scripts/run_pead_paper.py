#!/usr/bin/env python3
"""PEAD live paper trader CLI — daily one-shot.

Required env vars (per --account-prefix, default PEAD):
    ALPACA_PEAD_API_KEY
    ALPACA_PEAD_SECRET_KEY

Default behavior: DRY-RUN. Pass --live-submit to actually send orders.

Recommended cron / launchd cadence: once per trading day around 08:35 CDT
(5 minutes after market open). Catches both:
  * AMC events from prior day → first eligible session is today
  * BMO events from today → first eligible session is today

Skip non-trading-days automatically via Alpaca calendar.

Example dry-run with universe gate:

    set -a; source .env; set +a
    ./.venv/bin/python scripts/run_pead_paper.py \
        --universe research_data/intraday_top250_universe.json \
        --min-surprise-pct 5.0 \
        --hold-days 20 \
        --position-pct 0.05 \
        --max-concurrent 10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Pre-warm automation package to dodge the broker.__init__ circular import.
import tradingagents.automation.config  # noqa: E402,F401

from tradingagents.broker.alpaca_broker import AlpacaBroker  # noqa: E402
from tradingagents.research.pead_trader import (  # noqa: E402
    PEADConfig,
    PEADTrader,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "PEAD live paper trader. One-shot — invoke daily after market open "
            "(e.g. 08:35 CDT). Defaults to DRY-RUN."
        ),
    )
    p.add_argument("--account-prefix", default="PEAD",
                   help="Env var prefix: ALPACA_<PREFIX>_API_KEY/SECRET_KEY")
    p.add_argument("--daily-db", default="research_data/market_data.duckdb")
    p.add_argument("--universe",
                   default=None,
                   help="Optional universe filter JSON path. None = use all "
                        "symbols in earnings_events table.")
    p.add_argument("--min-surprise-pct", type=float, default=5.0)
    p.add_argument("--max-surprise-pct", type=float, default=50.0,
                   help="Cap to filter out data-noise prints (e.g. 500%% surprise)")
    p.add_argument("--hold-days", type=int, default=20)
    p.add_argument("--position-pct", type=float, default=0.05)
    p.add_argument("--max-concurrent", type=int, default=10)
    p.add_argument("--max-gross-exposure", type=float, default=0.5,
                   help="Safety cap on total exposure (no single position above this)")
    p.add_argument("--log-dir", default="results/pead/paper")
    p.add_argument("--live-submit", action="store_true",
                   help="DANGER — actually submit orders. Default is dry-run.")
    return p.parse_args()


def build_broker(prefix: str) -> AlpacaBroker:
    api_key = os.environ.get(f"ALPACA_{prefix}_API_KEY")
    secret = os.environ.get(f"ALPACA_{prefix}_SECRET_KEY")
    if not api_key or not secret:
        raise SystemExit(
            f"ALPACA_{prefix}_API_KEY and ALPACA_{prefix}_SECRET_KEY must be set."
        )
    return AlpacaBroker(api_key=api_key, secret_key=secret, paper=True)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    args = parse_args()
    broker = build_broker(args.account_prefix)
    cfg = PEADConfig(
        min_positive_surprise_pct=args.min_surprise_pct,
        max_positive_surprise_pct=args.max_surprise_pct,
        hold_days=args.hold_days,
        position_pct=args.position_pct,
        max_concurrent_positions=args.max_concurrent,
        max_gross_exposure=args.max_gross_exposure,
        universe_path=args.universe,
        daily_db_path=args.daily_db,
        log_dir=Path(args.log_dir),
        dry_run=not args.live_submit,
    )
    trader = PEADTrader(cfg, broker)
    mode = "LIVE-SUBMIT" if args.live_submit else "DRY-RUN"
    logging.warning(
        "PEAD %s | min_surprise=%.1f%% | hold=%dd | pos_pct=%.1f%% | "
        "max_concurrent=%d | universe=%s",
        mode, cfg.min_positive_surprise_pct, cfg.hold_days,
        cfg.position_pct * 100, cfg.max_concurrent_positions,
        args.universe or "ALL_EARNINGS_SYMBOLS",
    )
    summary = trader.run_once()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
