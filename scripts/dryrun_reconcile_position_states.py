#!/usr/bin/env python3
"""Dry-run or apply _reconcile_position_states across all variants.

Exercises the new reconciler pass against live Alpaca + local SQLite.
Default is dry-run (no writes). Pass `--apply` to actually reconcile.

Usage:
    python scripts/dryrun_reconcile_position_states.py             # dry-run
    python scripts/dryrun_reconcile_position_states.py --apply     # apply
    python scripts/dryrun_reconcile_position_states.py --variants chan chan_v2
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from tradingagents.automation.reconciler import OrderReconciler
from tradingagents.broker.alpaca_broker import AlpacaBroker
from tradingagents.storage.database import TradingDatabase


VARIANTS = [
    ("mechanical",    "trading_mechanical.db",    "ALPACA_MECHANICAL_API_KEY",      "ALPACA_MECHANICAL_SECRET_KEY"),
    ("llm",           "trading_llm.db",           "ALPACA_LLM_API_KEY",             "ALPACA_LLM_SECRET_KEY"),
    ("chan",          "trading_chan.db",          "ALPACA_CHAN_API_KEY",            "ALPACA_CHAN_SECRET_KEY"),
    ("mechanical_v2", "trading_mechanical_v2.db", "ALPACA_MECHANICAL_V2_API_KEY",   "ALPACA_MECHANICAL_V2_SECRET_KEY"),
    ("chan_v2",       "trading_chan_v2.db",       "ALPACA_CHAN_V2_API_KEY",         "ALPACA_CHAN_V2_SECRET_KEY"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=[v[0] for v in VARIANTS])
    parser.add_argument("--apply", action="store_true", help="Actually write changes")
    args = parser.parse_args()
    wanted = set(args.variants)
    dry_run = not args.apply

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    grand_total = {"ghosts": 0, "id_fixes": 0, "orphans": 0, "errors": 0}

    for name, dbfile, kenv, senv in VARIANTS:
        if name not in wanted:
            continue
        api = os.environ.get(kenv)
        sec = os.environ.get(senv)
        if not api or not sec:
            print(f"{name}: missing env, skip")
            continue
        dbpath = ROOT / dbfile
        if not dbpath.exists():
            print(f"{name}: db {dbfile} missing, skip")
            continue

        conn = sqlite3.connect(str(dbpath))
        db = TradingDatabase.__new__(TradingDatabase)
        db.conn = conn
        db.conn.row_factory = sqlite3.Row

        broker = AlpacaBroker(api_key=api, secret_key=sec, paper=True)
        reconciler = OrderReconciler(broker=broker, db=db, variant=name)

        mode_label = "dry-run" if dry_run else "APPLY"
        print(f"\n=== {name} — {mode_label} ===")
        summary = reconciler._reconcile_position_states(dry_run=dry_run)
        verb = "would" if dry_run else "did"
        print(
            f"  {verb}: delete {summary['ghosts']} ghost, fix {summary['id_fixes']} IDs, "
            f"import {summary['orphans']} orphan, errors={summary['errors']}"
        )
        conn.close()

        for k in grand_total:
            grand_total[k] += summary[k]

    mode_label = "dry-run" if dry_run else "APPLIED"
    print(
        f"\nTOTAL ({mode_label}): delete {grand_total['ghosts']} ghost, "
        f"fix {grand_total['id_fixes']} IDs, "
        f"import {grand_total['orphans']} orphan, "
        f"errors={grand_total['errors']}"
    )


if __name__ == "__main__":
    main()
