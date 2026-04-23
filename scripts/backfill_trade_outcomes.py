#!/usr/bin/env python3
"""Backfill `trade_outcomes` rows from the `trades` table for closed positions.

The existing `_log_trade_outcome` hook only fires on ExitManager-path SELLs
(Minervini/swing). It does NOT fire when brackets close broker-side or when
Chan / intraday orchestrators exit via their own paths. As a result, the
`trade_outcomes` table is empty across every variant — the daily trade
review would have no input.

This script reads the already-reconciled `trades` table (bracket leg fills
included thanks to the reconciler), pairs buys with sells FIFO per symbol
per variant, computes return/hold-days/MFE/MAE, and inserts
`trade_outcomes` rows. Idempotent — skips pairs that already produced an
outcome row (matched on symbol + entry_date + entry_price).

Default is dry-run. Pass --apply to write.

Usage:
    python scripts/backfill_trade_outcomes.py
    python scripts/backfill_trade_outcomes.py --apply
    python scripts/backfill_trade_outcomes.py --variants chan chan_v2
    python scripts/backfill_trade_outcomes.py --skip-excursion    # no Alpaca call
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")


VARIANTS = [
    ("mechanical",          "trading_mechanical.db",          "ALPACA_MECHANICAL_API_KEY",      "ALPACA_MECHANICAL_SECRET_KEY"),
    ("llm",                 "trading_llm.db",                 "ALPACA_LLM_API_KEY",             "ALPACA_LLM_SECRET_KEY"),
    ("chan",                "trading_chan.db",                "ALPACA_CHAN_API_KEY",            "ALPACA_CHAN_SECRET_KEY"),
    ("mechanical_v2",       "trading_mechanical_v2.db",       "ALPACA_MECHANICAL_V2_API_KEY",   "ALPACA_MECHANICAL_V2_SECRET_KEY"),
    ("chan_v2",             "trading_chan_v2.db",             "ALPACA_CHAN_V2_API_KEY",         "ALPACA_CHAN_V2_SECRET_KEY"),
    ("intraday_mechanical", "trading_intraday_mechanical.db", "ALPACA_V2_INTRADAY_API_KEY",     "ALPACA_V2_INTRADAY_SECRET_KEY"),
]


@dataclass
class Fill:
    id: int
    ts: datetime
    qty: float
    price: float
    reasoning: Optional[str]
    order_type: Optional[str]


def _parse_ts(raw: str) -> datetime:
    # DB has 'YYYY-MM-DD HH:MM:SS[.ffffff]' from orchestrator inserts and
    # 'YYYY-MM-DDTHH:MM:SS' from reconciler bracket-leg inserts. Both parse
    # cleanly with fromisoformat.
    return datetime.fromisoformat(raw)


def _derive_exit_reason(reasoning: Optional[str]) -> str:
    if not reasoning:
        return "closed"
    low = reasoning.lower()
    if "bracket_stop_loss" in low:
        return "bracket_stop_loss"
    if "bracket_take_profit" in low:
        return "bracket_take_profit"
    if "partial profit" in low or "partial_profit" in low:
        return "partial_profit"
    if "trailing_stop" in low or "trailing stop" in low:
        return "trailing_stop"
    if "dead_money" in low or "dead money" in low:
        return "dead_money"
    if "eod" in low or "flatten" in low:
        return "eod_flatten"
    if "lost_50dma" in low:
        return "lost_50dma"
    if "max_hold" in low:
        return "max_hold_days"
    return reasoning[:64]


def _fetch_fills(conn: sqlite3.Connection, side: str) -> List[Fill]:
    rows = conn.execute(
        """
        SELECT id, timestamp, symbol, filled_qty, filled_price, reasoning, order_type
        FROM trades
        WHERE side = ? AND status LIKE '%filled%'
          AND filled_qty > 0 AND filled_price IS NOT NULL
        ORDER BY timestamp
        """,
        (side,),
    ).fetchall()
    fills_by_symbol: dict = {}
    for r in rows:
        try:
            ts = _parse_ts(r[1])
        except Exception:
            continue
        fills_by_symbol.setdefault(r[2], []).append(
            Fill(
                id=int(r[0]),
                ts=ts,
                qty=float(r[3]),
                price=float(r[4]),
                reasoning=r[5],
                order_type=r[6],
            )
        )
    return fills_by_symbol


def _pair_fifo(buys, sells):
    """Pair buys with sells FIFO within a symbol. Yields (entry, exit, qty)."""
    buys = list(buys)
    sells = list(sells)
    bi = 0
    for sell in sells:
        remaining = sell.qty
        while remaining > 1e-6 and bi < len(buys):
            buy = buys[bi]
            take = min(remaining, buy.qty)
            if take > 1e-6:
                yield (buy, sell, take)
                buy.qty -= take
                remaining -= take
            if buy.qty <= 1e-6:
                bi += 1


def _compute_excursion(
    symbol: str,
    entry_dt: datetime,
    exit_dt: datetime,
    entry_price: float,
    alpaca_client,
) -> tuple[Optional[float], Optional[float]]:
    if alpaca_client is None or entry_price <= 0:
        return (None, None)
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Hour,
            start=entry_dt,
            end=exit_dt + timedelta(hours=1),
        )
        bars = alpaca_client.get_stock_bars(req)
        series = getattr(bars, "data", {}).get(symbol) or []
        if not series:
            return (None, None)
        max_high = max(float(b.high) for b in series)
        min_low = min(float(b.low) for b in series)
        return (
            round((max_high - entry_price) / entry_price, 4),
            round((min_low - entry_price) / entry_price, 4),
        )
    except Exception as e:
        print(f"    excursion({symbol}): {e}")
        return (None, None)


def backfill_variant(name: str, dbfile: str, api_key: str, secret_key: str,
                     apply: bool, skip_excursion: bool) -> dict:
    dbpath = ROOT / dbfile
    if not dbpath.exists():
        print(f"  {name}: DB missing, skip")
        return {"skipped": True}

    conn = sqlite3.connect(str(dbpath))
    conn.row_factory = sqlite3.Row

    # Existing outcomes — idempotency key is (symbol, entry_date, entry_price).
    existing = {
        (r["symbol"], r["entry_date"], round(float(r["entry_price"] or 0), 2))
        for r in conn.execute(
            "SELECT symbol, entry_date, entry_price FROM trade_outcomes"
        ).fetchall()
    }

    buys_by_symbol = _fetch_fills(conn, "buy")
    sells_by_symbol = _fetch_fills(conn, "sell")

    alpaca_client = None
    if not skip_excursion and api_key and secret_key:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            alpaca_client = StockHistoricalDataClient(api_key, secret_key)
        except Exception as e:
            print(f"  {name}: Alpaca client init failed: {e}")

    proposed = []
    skipped_existing = 0
    for symbol in sorted(set(buys_by_symbol) | set(sells_by_symbol)):
        buys = buys_by_symbol.get(symbol, [])
        sells = sells_by_symbol.get(symbol, [])
        for buy, sell, qty in _pair_fifo(buys, sells):
            entry_date = buy.ts.date().isoformat()
            if (symbol, entry_date, round(buy.price, 2)) in existing:
                skipped_existing += 1
                continue
            exit_date = sell.ts.date().isoformat()
            return_pct = (sell.price - buy.price) / buy.price if buy.price > 0 else 0.0
            hold_days = (sell.ts.date() - buy.ts.date()).days
            exit_reason = _derive_exit_reason(sell.reasoning)
            mfe, mae = _compute_excursion(
                symbol, buy.ts, sell.ts, buy.price, alpaca_client
            ) if not skip_excursion else (None, None)
            proposed.append({
                "symbol": symbol,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": round(buy.price, 2),
                "exit_price": round(sell.price, 2),
                "return_pct": round(return_pct, 4),
                "hold_days": hold_days,
                "exit_reason": exit_reason,
                "qty": qty,
                "mfe": mfe,
                "mae": mae,
            })

    print(f"\n=== {name}: {len(proposed)} new outcomes, {skipped_existing} skipped (already present) ===")
    for p in proposed:
        mfe_s = f"{p['mfe']:+.1%}" if p["mfe"] is not None else " -  "
        mae_s = f"{p['mae']:+.1%}" if p["mae"] is not None else " -  "
        print(
            f"  {p['entry_date']} → {p['exit_date']}  {p['symbol']:6s} "
            f"qty={p['qty']:.0f}  ret={p['return_pct']:+.2%}  hold={p['hold_days']}d  "
            f"mfe={mfe_s} mae={mae_s}  reason={p['exit_reason']}"
        )

    if not apply:
        print("  (dry-run)")
        conn.close()
        return {"proposed": len(proposed), "skipped_existing": skipped_existing}

    inserted = 0
    for p in proposed:
        try:
            conn.execute(
                """INSERT INTO trade_outcomes
                   (symbol, entry_date, exit_date, entry_price, exit_price,
                    return_pct, hold_days, exit_reason,
                    max_favorable_excursion, max_adverse_excursion)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (p["symbol"], p["entry_date"], p["exit_date"],
                 p["entry_price"], p["exit_price"], p["return_pct"],
                 p["hold_days"], p["exit_reason"], p["mfe"], p["mae"]),
            )
            inserted += 1
        except Exception as e:
            print(f"    insert {p['symbol']} failed: {e}")
    conn.commit()
    conn.close()
    print(f"  inserted {inserted} outcomes")
    return {"proposed": len(proposed), "inserted": inserted,
            "skipped_existing": skipped_existing}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--variants", nargs="+", default=[v[0] for v in VARIANTS])
    parser.add_argument("--skip-excursion", action="store_true",
                        help="Don't call Alpaca for hourly bars; MFE/MAE stay NULL")
    args = parser.parse_args()

    wanted = set(args.variants)
    total = {"proposed": 0, "inserted": 0, "skipped_existing": 0}
    for name, dbfile, kenv, senv in VARIANTS:
        if name not in wanted:
            continue
        api = os.environ.get(kenv, "")
        sec = os.environ.get(senv, "")
        r = backfill_variant(name, dbfile, api, sec,
                             apply=args.apply,
                             skip_excursion=args.skip_excursion)
        for k in total:
            total[k] += r.get(k, 0)

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"\nTOTAL ({mode}): proposed={total['proposed']} "
          f"inserted={total['inserted']} skipped={total['skipped_existing']}")


if __name__ == "__main__":
    main()
