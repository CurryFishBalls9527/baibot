#!/usr/bin/env python3
"""Backfill trade_outcomes.base_pattern for Chan variants.

The live Chan entry path only started persisting T-type in `base_pattern`
after the feedback-loop round-2 PR. Historical trade_outcomes rows for
chan / chan_v2 therefore have `base_pattern = NULL`, which makes the
Saturday weekly review's "By entry pattern" table unusable for Chan.

The T-type is still recoverable: the entry-side `trades` row stores it
in `reasoning` as "Chan buy signal: T2S at 2026/04/17 18:00" (or
"T1+T2S"). This script parses those strings and patches the matching
trade_outcomes row.

Idempotent — skips rows that already have a non-NULL base_pattern.

Usage:
    python scripts/backfill_chan_base_pattern.py              # dry-run
    python scripts/backfill_chan_base_pattern.py --apply
    python scripts/backfill_chan_base_pattern.py --variants chan
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tradingagents.storage.database import TradingDatabase


# Matches both "T2S", "T1", "T2", and combinations like "T1+T2S" or "T2+T2S".
CHAN_PATTERN_RE = re.compile(r"Chan buy signal:\s+((?:T[12](?:S)?(?:\+T[12](?:S)?)*))")


VARIANTS = [
    ("chan",    "trading_chan.db"),
    ("chan_v2", "trading_chan_v2.db"),
]


def extract_t_type(reasoning: str) -> str | None:
    """Parse the T-type out of Chan's entry reasoning string.

    Returns the canonical type string (e.g. "T2S", "T1+T2S") or None if
    the reasoning doesn't match the expected format — we log and move on.
    """
    if not reasoning:
        return None
    m = CHAN_PATTERN_RE.search(reasoning)
    return m.group(1) if m else None


def backfill_variant(name: str, dbfile: str, apply: bool) -> dict:
    dbpath = ROOT / dbfile
    if not dbpath.exists():
        print(f"  {name}: DB missing, skip")
        return {"rows": 0, "patched": 0, "unmatched": 0}

    db = TradingDatabase(str(dbpath))
    # Target only rows with NULL base_pattern — idempotent on re-run.
    rows = db.conn.execute(
        """SELECT id, symbol, entry_date FROM trade_outcomes
           WHERE base_pattern IS NULL
           ORDER BY entry_date"""
    ).fetchall()
    if not rows:
        db.close()
        print(f"  {name}: 0 rows with NULL base_pattern (already backfilled)")
        return {"rows": 0, "patched": 0, "unmatched": 0}

    patched = 0
    unmatched = 0
    print(f"\n=== {name}: {len(rows)} rows to check ===")
    for r in rows:
        outcome_id, symbol, entry_date = r
        # Find the matching entry-side BUY trade for (symbol, entry_date)
        trade = db.conn.execute(
            """SELECT reasoning FROM trades
               WHERE symbol = ? AND date(timestamp) = ? AND side = 'buy'
                 AND status LIKE '%filled%'
               ORDER BY timestamp ASC LIMIT 1""",
            (symbol, entry_date),
        ).fetchone()
        reasoning = trade[0] if trade else None
        t_type = extract_t_type(reasoning or "")
        if t_type is None:
            unmatched += 1
            snippet = (reasoning or "")[:80].replace("\n", " ")
            print(f"  ⚠  {symbol} {entry_date}: no T-type in reasoning "
                  f"[{snippet!r}]")
            continue
        print(f"  ✓ {symbol} {entry_date} → {t_type}")
        if apply:
            db.update_trade_outcome_base_pattern(outcome_id, t_type)
            patched += 1
    db.close()
    return {"rows": len(rows), "patched": patched, "unmatched": unmatched}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Write the patches. Default is dry-run.")
    parser.add_argument("--variants", nargs="+",
                        default=[v[0] for v in VARIANTS])
    args = parser.parse_args()

    wanted = set(args.variants)
    totals = {"rows": 0, "patched": 0, "unmatched": 0}
    for name, dbfile in VARIANTS:
        if name not in wanted:
            continue
        r = backfill_variant(name, dbfile, apply=args.apply)
        for k in totals:
            totals[k] += r[k]

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(
        f"\nTOTAL ({mode}): rows={totals['rows']} patched={totals['patched']} "
        f"unmatched={totals['unmatched']}"
    )


if __name__ == "__main__":
    main()
