#!/usr/bin/env python3
"""Push locally-ratcheted DB stops up to the broker for the 16 drifted positions.

Default is dry-run. Pass --apply to actually issue replace_order calls.

Exercises `OrderReconciler._sync_broker_stops` directly against live Alpaca.
The scheduler's reconciler will do the same thing on its next tick — this
script exists for (a) one-time catch-up on the 16 pre-existing drifts,
(b) observability before committing.

Usage:
    python scripts/sync_broker_stops.py               # dry-run
    python scripts/sync_broker_stops.py --apply       # actually submit
    python scripts/sync_broker_stops.py --variants mechanical_v2
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
    parser.add_argument("--apply", action="store_true", help="Actually submit replace_order")
    args = parser.parse_args()
    wanted = set(args.variants)
    dry_run = not args.apply

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    totals = {"synced": 0, "skipped": 0, "errors": 0}
    for name, dbfile, kenv, senv in VARIANTS:
        if name not in wanted:
            continue
        api = os.environ.get(kenv)
        sec = os.environ.get(senv)
        dbpath = ROOT / dbfile
        if not api or not sec or not dbpath.exists():
            print(f"{name}: skip (env or db missing)")
            continue

        conn = sqlite3.connect(str(dbpath))
        db = TradingDatabase.__new__(TradingDatabase)
        db.conn = conn
        db.conn.row_factory = sqlite3.Row

        broker = AlpacaBroker(api_key=api, secret_key=sec, paper=True)
        reconciler = OrderReconciler(broker=broker, db=db, variant=name)

        mode = "dry-run" if dry_run else "APPLY"
        print(f"\n=== {name} — {mode} ===")
        summary = reconciler._sync_broker_stops(dry_run=dry_run)
        verb = "would sync" if dry_run else "synced"
        print(
            f"  {verb} {summary['synced']} broker stops, "
            f"skipped {summary['skipped']}, errors {summary['errors']}"
        )
        conn.close()
        for k in totals:
            totals[k] += summary[k]

    label = "DRY-RUN" if dry_run else "APPLIED"
    print(
        f"\nTOTAL ({label}): "
        f"synced={totals['synced']} skipped={totals['skipped']} "
        f"errors={totals['errors']}"
    )


if __name__ == "__main__":
    main()
