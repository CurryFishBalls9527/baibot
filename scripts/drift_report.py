#!/usr/bin/env python3
"""Enumerate DB ↔ Alpaca drift across all 5 known failure modes.

Read-only. Exits non-zero if any drift is detected — suitable as a
pre-ship / post-reconcile assertion.

Five failure modes (per memory/project_bracket_leg_id_bug.md):
  1. NULL leg IDs in position_states (fixed forward-only 2026-04-22).
  2. DB current_stop drifting above broker SL (ratchet didn't propagate).
  3. Ghost position_states row — DB has row but Alpaca has no position.
  4. OCO-mismatch — DB leg IDs point to cancelled orders; live Alpaca
     has different SL/TP IDs (often from manual resubmission).
  5. Orphan Alpaca position — live position at broker with no DB row.

Usage:
    python scripts/drift_report.py
    python scripts/drift_report.py --variants mechanical_v2 chan
    python scripts/drift_report.py --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus


VARIANTS = [
    ("mechanical",    "trading_mechanical.db",    "ALPACA_MECHANICAL_API_KEY",      "ALPACA_MECHANICAL_SECRET_KEY"),
    ("llm",           "trading_llm.db",           "ALPACA_LLM_API_KEY",             "ALPACA_LLM_SECRET_KEY"),
    ("chan",          "trading_chan.db",          "ALPACA_CHAN_API_KEY",            "ALPACA_CHAN_SECRET_KEY"),
    ("mechanical_v2", "trading_mechanical_v2.db", "ALPACA_MECHANICAL_V2_API_KEY",   "ALPACA_MECHANICAL_V2_SECRET_KEY"),
    ("chan_v2",       "trading_chan_v2.db",       "ALPACA_CHAN_V2_API_KEY",         "ALPACA_CHAN_V2_SECRET_KEY"),
]

LIVE_STATUSES = {
    "new", "accepted", "pending_new", "accepted_for_bidding", "held",
    "replaced", "pending_replace",
}


@dataclass
class DriftRow:
    variant: str
    symbol: str
    issues: List[str] = field(default_factory=list)
    db_current_stop: Optional[float] = None
    broker_sl_price: Optional[float] = None
    broker_tp_price: Optional[float] = None
    broker_position_qty: Optional[float] = None
    db_stop_id: Optional[str] = None
    broker_sl_id: Optional[str] = None


def scan_variant(name: str, dbfile: str, api_key: str, secret_key: str) -> List[DriftRow]:
    dbpath = ROOT / dbfile
    if not dbpath.exists():
        return []

    client = TradingClient(api_key, secret_key, paper=True)
    orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500))
    positions_by_sym = {p.symbol: p for p in client.get_all_positions()}

    live_sl = {}  # symbol -> (id, stop_price)
    live_tp = {}  # symbol -> (id, limit_price)
    for o in orders:
        if o.status.value.lower() not in LIVE_STATUSES:
            continue
        if o.side.value != "sell":
            continue
        sym = o.symbol
        if o.order_type.value == "stop" and o.stop_price:
            # Prefer the most recent submission if multiple
            prev = live_sl.get(sym)
            if prev is None or (o.submitted_at and prev[2] and o.submitted_at > prev[2]):
                live_sl[sym] = (str(o.id), float(o.stop_price), o.submitted_at)
        elif o.order_type.value == "limit" and o.limit_price:
            prev = live_tp.get(sym)
            if prev is None or (o.submitted_at and prev[2] and o.submitted_at > prev[2]):
                live_tp[sym] = (str(o.id), float(o.limit_price), o.submitted_at)

    con = sqlite3.connect(str(dbpath))
    con.row_factory = sqlite3.Row
    db_rows = {
        r["symbol"]: dict(r)
        for r in con.execute(
            "SELECT symbol, entry_price, current_stop, stop_order_id, tp_order_id FROM position_states"
        ).fetchall()
    }
    con.close()

    all_syms = set(db_rows.keys()) | set(positions_by_sym.keys())
    out: List[DriftRow] = []
    for sym in sorted(all_syms):
        row = DriftRow(variant=name, symbol=sym)
        db = db_rows.get(sym)
        pos = positions_by_sym.get(sym)
        sl = live_sl.get(sym)
        tp = live_tp.get(sym)

        if db:
            row.db_current_stop = float(db["current_stop"]) if db["current_stop"] else None
            row.db_stop_id = db["stop_order_id"]
        if sl:
            row.broker_sl_id = sl[0]
            row.broker_sl_price = sl[1]
        if tp:
            row.broker_tp_price = tp[1]
        if pos:
            row.broker_position_qty = float(pos.qty)

        # Source #5: orphan — broker position, no DB row
        if pos and not db:
            row.issues.append("orphan")
        # Source #3: ghost — DB row, no broker position
        if db and not pos:
            row.issues.append("ghost")
        # Source #1: NULL leg IDs (only meaningful when DB + position both exist)
        if db and pos and not db["stop_order_id"]:
            row.issues.append("null_stop_id")
        if db and pos and not db["tp_order_id"]:
            row.issues.append("null_tp_id")
        # Source #4: OCO-mismatch — DB ID doesn't match live broker ID
        if db and pos and db["stop_order_id"] and sl and db["stop_order_id"] != sl[0]:
            row.issues.append("stop_id_mismatch")
        if db and pos and db["tp_order_id"] and tp and db["tp_order_id"] != tp[0]:
            row.issues.append("tp_id_mismatch")
        # Source #2: ratchet drift — DB current_stop > broker SL by > 0.5%
        if db and pos and sl and db["current_stop"]:
            drift_pct = (float(db["current_stop"]) - sl[1]) / sl[1] * 100
            if drift_pct > 0.5:
                row.issues.append(f"ratchet_drift_{drift_pct:+.1f}pct")
        # Broker position but no active SL at all — unprotected
        if pos and not sl:
            row.issues.append("unprotected_no_sl")
        if pos and not tp:
            row.issues.append("no_active_tp")

        if row.issues:
            out.append(row)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=[v[0] for v in VARIANTS])
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    args = parser.parse_args()

    selected = [v for v in VARIANTS if v[0] in set(args.variants)]
    all_issues: List[DriftRow] = []
    per_variant: Dict[str, List[DriftRow]] = {}
    for name, dbfile, kenv, senv in selected:
        api = os.environ.get(kenv)
        sec = os.environ.get(senv)
        if not api or not sec:
            continue
        rows = scan_variant(name, dbfile, api, sec)
        per_variant[name] = rows
        all_issues.extend(rows)

    if args.json:
        print(json.dumps(
            {v: [asdict(r) for r in rows] for v, rows in per_variant.items()},
            indent=2,
            default=str,
        ))
    else:
        print(f"{'VARIANT':15s} {'SYMBOL':7s} {'ISSUES':60s} {'DB_STOP':>9s} {'BRK_SL':>8s} {'BRK_TP':>8s}")
        print("-" * 112)
        for r in all_issues:
            issues_str = ",".join(r.issues)[:58]
            db_stop = f"${r.db_current_stop:.2f}" if r.db_current_stop else "     -"
            brk_sl = f"${r.broker_sl_price:.2f}" if r.broker_sl_price else "     -"
            brk_tp = f"${r.broker_tp_price:.2f}" if r.broker_tp_price else "     -"
            print(f"{r.variant:15s} {r.symbol:7s} {issues_str:60s} {db_stop:>9s} {brk_sl:>8s} {brk_tp:>8s}")

        # Summary
        from collections import Counter
        issue_counts: Counter = Counter()
        for r in all_issues:
            for i in r.issues:
                # Collapse ratchet_drift_X.Xpct to a single key
                key = "ratchet_drift" if i.startswith("ratchet_drift_") else i
                issue_counts[key] += 1
        print(f"\nSummary: {len(all_issues)} rows with at least one drift issue")
        for k, v in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {k:22s}: {v}")

    return 0 if not all_issues else 1


if __name__ == "__main__":
    sys.exit(main())
