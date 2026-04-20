"""Dry-run backfill of bracket-leg SELL rows into variant trades tables.

Local `trades` table only records orders the orchestrator submits. Bracket
stop-loss / take-profit legs fill broker-side under their own order_ids, so
they never land in SQLite — dashboard and trade history show BUY only.

This script walks each variant's recent BUY trades, asks Alpaca for the
parent order's bracket legs, and for any leg that's filled (or canceled, as
the losing side of an OCO) but not yet in the local DB, emits a planned
INSERT. Default is dry-run; pass --apply to actually write.

Usage:
    .venv/bin/python scripts/backfill_bracket_fills.py \\
        --experiment experiments/paper_launch_v2.yaml \\
        [--variant chan] [--lookback-days 30] [--apply]
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv

from tradingagents.broker.alpaca_broker import AlpacaBroker
from tradingagents.testing.ab_config import load_experiment
from tradingagents.testing.ab_models import ExperimentVariant


def _terminal(status: Optional[str]) -> bool:
    if not status:
        return False
    s = status.lower()
    return ("filled" in s and "partially" not in s) or any(
        t in s for t in ("canceled", "cancelled", "expired", "rejected")
    )


def _existing_leg_row(conn: sqlite3.Connection, order_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT id, status, filled_qty, filled_price FROM trades WHERE order_id = ?",
        (order_id,),
    ).fetchone()
    if not row:
        return None
    return {"id": row[0], "status": row[1], "filled_qty": row[2], "filled_price": row[3]}


def _scan_variant(
    variant: ExperimentVariant,
    lookback_days: int,
    apply: bool,
) -> dict:
    if not (variant.alpaca_api_key and variant.alpaca_secret_key):
        print(f"  [{variant.name}] skipped — missing Alpaca credentials in env")
        return {"checked": 0, "found": 0, "inserted": 0, "skipped": True}

    db_path = Path(variant.db_path)
    if not db_path.exists():
        print(f"  [{variant.name}] skipped — db not found at {db_path}")
        return {"checked": 0, "found": 0, "inserted": 0, "skipped": True}

    broker = AlpacaBroker(
        variant.alpaca_api_key, variant.alpaca_secret_key, paper=True
    )
    # Sanity: probe account; bail if creds are bad.
    try:
        broker.get_account()
    except Exception as e:
        print(f"  [{variant.name}] skipped — broker auth failed: {e}")
        return {"checked": 0, "found": 0, "inserted": 0, "skipped": True}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
    buys = conn.execute(
        """
        SELECT id, symbol, qty, filled_qty, filled_price, order_id, timestamp
        FROM trades
        WHERE side = 'buy' AND order_id IS NOT NULL AND timestamp >= ?
        ORDER BY timestamp DESC
        """,
        (cutoff,),
    ).fetchall()

    checked, found, inserted = 0, 0, 0
    for parent in buys:
        checked += 1
        try:
            remote_parent = broker.get_order(parent["order_id"])
        except Exception as e:
            print(
                f"  [{variant.name}] {parent['symbol']} parent "
                f"{parent['order_id'][:8]} get_order failed: {e}"
            )
            continue

        leg_pairs = [
            ("stop_loss", remote_parent.stop_order_id),
            ("take_profit", remote_parent.tp_order_id),
        ]
        for label, leg_id in leg_pairs:
            if not leg_id:
                continue
            if _existing_leg_row(conn, leg_id):
                continue
            try:
                leg = broker.get_order(leg_id)
            except Exception as e:
                print(
                    f"  [{variant.name}] {parent['symbol']} leg "
                    f"{leg_id[:8]} ({label}) get_order failed: {e}"
                )
                continue
            if not _terminal(leg.status):
                continue
            filled_qty = float(leg.filled_qty or 0)
            filled_price = (
                float(leg.filled_avg_price)
                if leg.filled_avg_price is not None
                else None
            )
            status_l = (leg.status or "").lower()
            # Losing side of an OCO (or orchestrator-cancelled leg) with zero
            # fill did not move shares — skip to keep sell rows meaningful.
            is_zero_cancel = (
                ("canceled" in status_l or "cancelled" in status_l)
                and filled_qty == 0
            )
            if is_zero_cancel:
                continue
            found += 1
            print(
                f"  [{variant.name}] MISSING  {parent['symbol']:<6} "
                f"{label:<11} leg={leg_id[:8]} status={leg.status} "
                f"qty={filled_qty} @ {filled_price}"
            )
            if apply:
                conn.execute(
                    """
                    INSERT INTO trades (
                        symbol, side, qty, notional, order_type, status,
                        filled_qty, filled_price, order_id, signal_id,
                        reasoning, variant
                    ) VALUES (?, 'sell', ?, NULL, ?, ?, ?, ?, ?, NULL, ?, ?)
                    """,
                    (
                        parent["symbol"],
                        float(leg.qty) if leg.qty else filled_qty,
                        "stop" if label == "stop_loss" else "limit",
                        str(leg.status),
                        filled_qty,
                        filled_price,
                        leg_id,
                        f"bracket_{label}",
                        variant.name,
                    ),
                )
                inserted += 1

    if apply:
        conn.commit()
    conn.close()
    return {
        "checked": checked,
        "found": found,
        "inserted": inserted,
        "skipped": False,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--experiment", default="experiments/paper_launch_v2.yaml")
    p.add_argument("--variant", help="Restrict to single variant by name")
    p.add_argument("--lookback-days", type=int, default=30)
    p.add_argument(
        "--apply",
        action="store_true",
        help="Write INSERTs. Default is dry-run (report only).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    exp = load_experiment(args.experiment)
    variants = exp.variants
    if args.variant:
        variants = [v for v in variants if v.name == args.variant]
        if not variants:
            print(f"No variant named {args.variant!r} in {args.experiment}")
            return 1

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== Bracket-fill backfill ({mode}) | lookback={args.lookback_days}d ===")
    totals = {"checked": 0, "found": 0, "inserted": 0}
    for v in variants:
        print(f"\n[{v.name}] db={v.db_path}")
        r = _scan_variant(v, args.lookback_days, args.apply)
        if r["skipped"]:
            continue
        print(
            f"  checked={r['checked']} missing_legs={r['found']} "
            f"inserted={r['inserted']}"
        )
        for k in totals:
            totals[k] += r[k]

    print(
        f"\nTOTAL: checked={totals['checked']} "
        f"missing_legs={totals['found']} inserted={totals['inserted']}"
    )
    if not args.apply and totals["found"] > 0:
        print("Re-run with --apply to write these rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
