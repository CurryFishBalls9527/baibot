#!/usr/bin/env python3
"""CLI runner for the PEAD → dashboard SQLite bridge.

Mirrors PEAD's JSON state (positions.json, fills.jsonl) and Alpaca
account state into ``trading_pead.db`` so the unified dashboard
auto-discovers PEAD as the 7th tile.

Designed to be invoked from PEAD's launchd wrapper after
``run_pead_paper.py`` exits, AND nightly at EOD to capture the final
daily_snapshot.

Usage:
    ./.venv/bin/python scripts/sync_pead_to_dashboard.py
    ./.venv/bin/python scripts/sync_pead_to_dashboard.py \\
        --state-dir results/pead/paper \\
        --db trading_pead.db
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradingagents.automation.pead_ingest import sync_pead_to_sqlite  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--state-dir", default="results/pead/paper")
    p.add_argument("--db", default="trading_pead.db")
    args = p.parse_args()
    summary = sync_pead_to_sqlite(state_dir=args.state_dir, db_path=args.db)
    logging.warning("PEAD → SQLite sync done: %s", summary.as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
