#!/usr/bin/env python3
"""Cross-asset rotation paper-trader CLI.

One-shot runner. Invoke once per day (cron / launchd). Acts only on the
first trading day of each new month — otherwise no-op.

Required env vars:
    ALPACA_XSECTION_API_KEY  (or any paper-account key — set ALPACA_XASSET_API_KEY
                              if running on a separate account)
    ALPACA_XSECTION_SECRET_KEY
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
from tradingagents.research.xasset_rotation_trader import (  # noqa: E402
    XAssetRotationConfig,
    XAssetRotationTrader,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Cross-asset rotation paper trader. One-shot — invoke daily. "
            "Defaults to DRY-RUN; pass --live-submit to send orders."
        ),
    )
    p.add_argument("--universe", default="SPY,QQQ,TLT,IEF,GLD",
                   help="Comma-separated ETF symbols")
    p.add_argument("--lookback-days", type=int, default=20)
    p.add_argument("--top-n", type=int, default=2)
    p.add_argument("--max-gross-exposure", type=float, default=0.5)
    p.add_argument("--daily-db", default="research_data/market_data.duckdb")
    p.add_argument("--log-dir", default="results/xasset_rotation/paper")
    p.add_argument("--live-submit", action="store_true",
                   help="DANGER — actually submit orders. Default is dry-run.")
    p.add_argument("--account-prefix", default="XSECTION",
                   help="Env var prefix: ALPACA_<PREFIX>_API_KEY/SECRET_KEY")
    p.add_argument("--force", action="store_true",
                   help="Run even if today isn't a rebalance day (for testing)")
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
    cfg = XAssetRotationConfig(
        universe=tuple(s.strip().upper() for s in args.universe.split(",") if s.strip()),
        lookback_days=args.lookback_days,
        top_n=args.top_n,
        daily_db_path=args.daily_db,
        log_dir=Path(args.log_dir),
        dry_run=not args.live_submit,
        max_gross_exposure=args.max_gross_exposure,
    )
    trader = XAssetRotationTrader(cfg, broker)
    mode = "LIVE-SUBMIT" if args.live_submit else "DRY-RUN"
    logging.warning(
        "xasset rotation %s | universe=%s | lookback=%dd | top-%d | gross_cap=%.2f",
        mode, list(cfg.universe), cfg.lookback_days, cfg.top_n, cfg.max_gross_exposure,
    )

    if args.force:
        # Bypass rebalance-day check by directly invoking the rebalance path
        from datetime import datetime
        asof = datetime.now()
        signal = trader.compute_signal(asof)
        target_symbols = trader.select_basket(signal)
        target_dollars = trader.compute_target_dollars()
        summary = {
            "action": "forced_rebalance",
            "rebalance_date": asof.date().isoformat(),
            "target_symbols": target_symbols,
            "signal": {s: round(v, 6) for s, v in signal.items()},
            "dry_run": cfg.dry_run,
        }
        trader.append_signal_summary(summary)
        records = trader.rebalance_to(
            target_symbols, target_dollars, signal,
            asof.date().isoformat(),
        )
        trader.append_records(records)
        summary["close_count"] = sum(1 for r in records if r.action == "rebalance_close")
        summary["open_count"] = sum(1 for r in records if r.action == "open_long")
        print(summary)
    else:
        summary = trader.run_once()
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
