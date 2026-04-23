#!/usr/bin/env python3
"""Reconstruct daily_snapshots rows for chan and chan_v2 from 2026-04-13.

Chan orchestrators never initialised a PortfolioTracker, so the
`daily_snapshots` table is empty for those variants — the dashboard
equity curve starts blank. This script walks each chan variant's
`trades` table forward from the launch date, applies each day's trades
to the cash balance, prices held positions against Alpaca historical
daily closes, and writes one snapshot row per trading day.

Assumptions:
  - Starting equity $100,000 on the launch date.
  - All bracket SL/TP leg sells have been backfilled into `trades` already
    by scripts/backfill_bracket_fills.py. Any that are missing will leave
    a phantom open position in the reconstruction.

Default is dry-run (prints the reconstructed rows). Pass --apply to write.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import json

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

VARIANTS = [
    ("chan",    "trading_chan.db",    "ALPACA_CHAN_API_KEY",    "ALPACA_CHAN_SECRET_KEY"),
    ("chan_v2", "trading_chan_v2.db", "ALPACA_CHAN_V2_API_KEY", "ALPACA_CHAN_V2_SECRET_KEY"),
]

LAUNCH_DATE = date(2026, 4, 13)
STARTING_EQUITY = 100_000.0


def _trading_days(start: date, end: date) -> List[date]:
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            days.append(d)
        d += timedelta(days=1)
    return days


def _parse_ts(ts: str) -> datetime:
    # DB timestamps come in two flavours: 'YYYY-MM-DD HH:MM:SS.ffffff'
    # (orchestrator inserts) and 'YYYY-MM-DDTHH:MM:SS' (reconciler bracket-leg
    # backfill). datetime.fromisoformat handles both.
    return datetime.fromisoformat(ts)


def _fetch_closes(
    client: StockHistoricalDataClient, symbols: List[str],
    start: date, end: date,
) -> Dict[str, Dict[date, float]]:
    """Return {symbol: {date: close_price}} across the full window."""
    if not symbols:
        return {}
    req = StockBarsRequest(
        symbol_or_symbols=sorted(set(symbols)),
        timeframe=TimeFrame.Day,
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end + timedelta(days=1), datetime.min.time()),
    )
    bars = client.get_stock_bars(req)
    out: Dict[str, Dict[date, float]] = {}
    for sym, series in bars.data.items():
        out[sym] = {b.timestamp.date(): float(b.close) for b in series}
    return out


def backfill_variant(
    name: str, dbfile: str, api_key: str, secret_key: str,
    apply: bool,
) -> None:
    dbpath = ROOT / dbfile
    if not dbpath.exists():
        print(f"  {name}: DB missing, skip")
        return

    con = sqlite3.connect(str(dbpath))
    con.row_factory = sqlite3.Row
    trades = [dict(r) for r in con.execute(
        "SELECT timestamp, symbol, side, qty, filled_qty, filled_price, status "
        "FROM trades ORDER BY timestamp"
    ).fetchall()]
    if not trades:
        print(f"  {name}: no trades, skip")
        con.close()
        return

    symbols = sorted({t["symbol"] for t in trades})
    today = date.today()
    client = StockHistoricalDataClient(api_key, secret_key)
    closes = _fetch_closes(client, symbols, LAUNCH_DATE, today)

    # Group trades by date (based on filled date — timestamp is submission;
    # assume same day for this reconstruction, which is correct for
    # market-hours fills).
    trades_by_date: Dict[date, List[dict]] = {}
    for t in trades:
        if not (t.get("filled_qty") and t.get("filled_price")):
            continue
        status = (t.get("status") or "").lower()
        if "filled" not in status:
            continue
        ts = _parse_ts(t["timestamp"])
        trades_by_date.setdefault(ts.date(), []).append(t)

    cash = STARTING_EQUITY
    positions: Dict[str, float] = {}  # symbol -> qty
    prev_equity = STARTING_EQUITY

    rows_to_write: List[dict] = []
    for day in _trading_days(LAUNCH_DATE, today):
        # Apply the day's trades
        for t in trades_by_date.get(day, []):
            qty = float(t["filled_qty"])
            price = float(t["filled_price"])
            side = t["side"].lower()
            if side == "buy":
                cash -= qty * price
                positions[t["symbol"]] = positions.get(t["symbol"], 0.0) + qty
            elif side == "sell":
                cash += qty * price
                positions[t["symbol"]] = positions.get(t["symbol"], 0.0) - qty
                if positions[t["symbol"]] <= 1e-6:
                    positions.pop(t["symbol"], None)

        # Price remaining positions at day's close. If the date is a
        # non-trading day for that symbol (corporate action, halt), walk
        # back to the most recent available close.
        pos_list = []
        market_value = 0.0
        for sym, qty in positions.items():
            sym_closes = closes.get(sym, {})
            close_price = sym_closes.get(day)
            if close_price is None:
                # Walk back up to 5 days to find a usable close.
                probe = day - timedelta(days=1)
                for _ in range(5):
                    if probe in sym_closes:
                        close_price = sym_closes[probe]
                        break
                    probe -= timedelta(days=1)
            if close_price is None:
                # Last resort: use most recent filled_price for this symbol.
                close_price = next(
                    (
                        float(t["filled_price"])
                        for t in reversed(trades)
                        if t["symbol"] == sym and t.get("filled_price")
                    ),
                    0.0,
                )
            pos_list.append({
                "symbol": sym,
                "qty": qty,
                "current_price": close_price,
                "market_value": qty * close_price,
            })
            market_value += qty * close_price

        equity = cash + market_value
        daily_pl = equity - prev_equity
        daily_pl_pct = daily_pl / prev_equity if prev_equity > 0 else 0.0

        rows_to_write.append({
            "date": day.isoformat(),
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "buying_power": round(cash, 2),  # paper account approximation
            "portfolio_value": round(equity, 2),
            "positions_json": json.dumps(pos_list),
            "daily_pl": round(daily_pl, 2),
            "daily_pl_pct": round(daily_pl_pct, 6),
        })
        prev_equity = equity

    print(f"\n=== {name}: {len(rows_to_write)} trading days reconstructed ===")
    for r in rows_to_write:
        print(
            f"  {r['date']}  eq=${r['equity']:,.2f}  cash=${r['cash']:,.2f}  "
            f"pl=${r['daily_pl']:,.2f} ({r['daily_pl_pct']:.3%})"
        )

    if not apply:
        con.close()
        print(f"  (dry-run — pass --apply to write)")
        return

    # Skip any date already present (idempotent).
    existing = {
        r[0] for r in con.execute("SELECT date FROM daily_snapshots").fetchall()
    }
    inserted = 0
    for r in rows_to_write:
        if r["date"] in existing:
            continue
        con.execute(
            """INSERT INTO daily_snapshots
               (date, equity, cash, buying_power, portfolio_value,
                positions_json, daily_pl, daily_pl_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                r["date"], r["equity"], r["cash"], r["buying_power"],
                r["portfolio_value"], r["positions_json"],
                r["daily_pl"], r["daily_pl_pct"],
            ),
        )
        inserted += 1
    con.commit()
    con.close()
    print(f"  inserted {inserted} rows into {dbfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--variants", nargs="+",
                        default=[v[0] for v in VARIANTS])
    args = parser.parse_args()
    wanted = set(args.variants)

    for name, dbfile, kenv, senv in VARIANTS:
        if name not in wanted:
            continue
        api = os.environ.get(kenv)
        sec = os.environ.get(senv)
        if not api or not sec:
            print(f"{name}: env missing, skip")
            continue
        backfill_variant(name, dbfile, api, sec, apply=args.apply)


if __name__ == "__main__":
    main()
