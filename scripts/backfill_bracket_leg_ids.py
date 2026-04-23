#!/usr/bin/env python3
"""Backfill stop_order_id / tp_order_id for existing live positions.

The broker bug: Alpaca's initial bracket submit response has `legs=[]`, so
position_states was persisted with NULL leg IDs. ExitManagerV2's broker
stop ratchet silently no-ops on NULL IDs. This script walks every
variant's DB, finds open positions with missing leg IDs, queries the
matching Alpaca account for each symbol's BUY bracket parent, and patches
the DB with its child IDs.

Safe to run repeatedly — skips positions that already have IDs, reports a
clear summary. Use `--dry-run` for a no-write preview.

Usage:
    python scripts/backfill_bracket_leg_ids.py
    python scripts/backfill_bracket_leg_ids.py --dry-run
    python scripts/backfill_bracket_leg_ids.py --variants mechanical_v2
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus, OrderClass, OrderSide


VARIANTS = [
    ("mechanical",    "trading_mechanical.db",    "ALPACA_MECHANICAL_API_KEY",      "ALPACA_MECHANICAL_SECRET_KEY"),
    ("llm",           "trading_llm.db",           "ALPACA_LLM_API_KEY",             "ALPACA_LLM_SECRET_KEY"),
    ("chan",          "trading_chan.db",          "ALPACA_CHAN_API_KEY",            "ALPACA_CHAN_SECRET_KEY"),
    ("mechanical_v2", "trading_mechanical_v2.db", "ALPACA_MECHANICAL_V2_API_KEY",   "ALPACA_MECHANICAL_V2_SECRET_KEY"),
    ("chan_v2",       "trading_chan_v2.db",       "ALPACA_CHAN_V2_API_KEY",         "ALPACA_CHAN_V2_SECRET_KEY"),
]


def find_bracket_parent(client: TradingClient, symbol: str, qty: float, entry_price: float):
    """Find the filled BUY bracket parent for this position.

    Match by symbol + side=buy + order_class=bracket + status=filled, then
    disambiguate by qty and filled_avg_price (within 0.5%) if multiple.
    Returns the parent order with `.legs` populated, or None.
    """
    req = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        symbols=[symbol],
        side=OrderSide.BUY,
        # Look back 180 days — covers any position held since the variant launched.
        after=datetime.now(timezone.utc) - timedelta(days=180),
        limit=100,
    )
    candidates = [
        o for o in client.get_orders(filter=req)
        if o.order_class == OrderClass.BRACKET
        and str(o.status).lower().endswith("filled")
    ]
    if not candidates:
        return None
    # Disambiguate by qty
    candidates = [o for o in candidates if float(o.qty) == qty] or candidates
    # Disambiguate by fill price (within 50bps)
    def _price_match(o):
        fp = float(o.filled_avg_price) if o.filled_avg_price else 0.0
        return fp > 0 and abs(fp - entry_price) / entry_price < 0.005
    matched = [o for o in candidates if _price_match(o)]
    if matched:
        candidates = matched
    # Most recent first
    candidates.sort(key=lambda o: o.submitted_at or "", reverse=True)
    parent = candidates[0]
    # Refetch to ensure legs populated
    return client.get_order_by_id(str(parent.id))


def extract_leg_ids(parent):
    stop_id, tp_id = None, None
    for leg in (parent.legs or []):
        if leg.stop_price and stop_id is None:
            stop_id = str(leg.id)
        elif leg.limit_price and tp_id is None:
            tp_id = str(leg.id)
    return stop_id, tp_id


def backfill_variant(name: str, dbfile: str, api_key: str, secret_key: str, dry_run: bool):
    dbpath = ROOT / dbfile
    if not dbpath.exists():
        print(f"  {name}: DB {dbfile} missing — skip")
        return 0, 0, 0

    client = TradingClient(api_key, secret_key, paper=True)
    con = sqlite3.connect(str(dbpath))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT symbol, entry_price, current_stop, stop_order_id, tp_order_id,
               entry_order_id, (SELECT qty FROM trades
                                WHERE trades.symbol = position_states.symbol
                                  AND trades.side = 'buy'
                                ORDER BY id DESC LIMIT 1) AS last_buy_qty
        FROM position_states
        """
    ).fetchall()
    if not rows:
        con.close()
        print(f"  {name}: no open positions")
        return 0, 0, 0

    needs_fix = [r for r in rows if not r["stop_order_id"] or not r["tp_order_id"]]
    print(f"  {name}: {len(rows)} positions, {len(needs_fix)} need fix")
    fixed = 0
    failed = 0
    for r in needs_fix:
        sym = r["symbol"]
        entry_price = float(r["entry_price"])
        qty = float(r["last_buy_qty"]) if r["last_buy_qty"] is not None else 0.0
        try:
            parent = find_bracket_parent(client, sym, qty, entry_price)
        except Exception as e:
            print(f"     {sym}: lookup error — {e}")
            failed += 1
            continue
        if parent is None:
            print(f"     {sym}: no filled BUY bracket parent found at Alpaca — skip")
            failed += 1
            continue
        stop_id, tp_id = extract_leg_ids(parent)
        if not stop_id and not tp_id:
            print(f"     {sym}: parent {str(parent.id)[:8]} has no legs populated — skip")
            failed += 1
            continue
        # Preserve any existing IDs
        new_stop = r["stop_order_id"] or stop_id
        new_tp = r["tp_order_id"] or tp_id
        new_entry = r["entry_order_id"] or str(parent.id)
        action = "[DRY-RUN]" if dry_run else "[PATCH]"
        print(
            f"     {sym}: {action} parent={str(parent.id)[:8]} "
            f"stop_id={str(new_stop)[:8]} tp_id={str(new_tp)[:8]}"
        )
        if not dry_run:
            con.execute(
                """
                UPDATE position_states
                SET stop_order_id = ?, tp_order_id = ?, entry_order_id = ?
                WHERE symbol = ?
                """,
                (new_stop, new_tp, new_entry, sym),
            )
        fixed += 1
    if not dry_run:
        con.commit()
    con.close()
    return len(rows), fixed, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print changes, don't write")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[v[0] for v in VARIANTS],
        help="Subset of variants to backfill",
    )
    args = parser.parse_args()

    selected = [v for v in VARIANTS if v[0] in set(args.variants)]
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'WRITE'}")
    print(f"Variants: {[v[0] for v in selected]}")
    print()

    total_pos = 0
    total_fixed = 0
    total_failed = 0
    for name, dbfile, key_env, sec_env in selected:
        api_key = os.environ.get(key_env)
        sec_key = os.environ.get(sec_env)
        if not api_key or not sec_key:
            print(f"  {name}: env vars missing — skip")
            continue
        n, f, x = backfill_variant(name, dbfile, api_key, sec_key, args.dry_run)
        total_pos += n
        total_fixed += f
        total_failed += x

    print()
    print(f"Summary: {total_pos} positions, {total_fixed} patched, {total_failed} failed")


if __name__ == "__main__":
    main()
