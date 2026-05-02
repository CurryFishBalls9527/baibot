#!/usr/bin/env python3
"""Yfinance fallback for recent-reporter earnings surprise ingest.

Companion to scripts/ingest_earnings_alphavantage.py. When AV is rate-
limited (or our key is exhausted at the IP level), this script can
populate `earnings_events.surprise_pct` from yfinance for symbols that
just reported.

Safety contract: NEVER overwrites an existing earnings_events row.
For each (symbol, event_datetime) pulled from yfinance, we look for an
existing row within +/-1 day of the same symbol; if found, we skip.
This means AV-derived rows always win — yfinance only fills holes.

Default behavior reads the recent-reporter list from the local
earnings_calendar table (same logic the AV ingest uses for prioritization).
Pass --symbols A,B,C to override.

Usage:
    ./.venv/bin/python scripts/ingest_earnings_yfinance_recent.py
    ./.venv/bin/python scripts/ingest_earnings_yfinance_recent.py --symbols META,MSFT
    ./.venv/bin/python scripts/ingest_earnings_yfinance_recent.py --lookback-days 7
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="",
                   help="Comma-separated symbols. Default: recent reporters from calendar.")
    p.add_argument("--db", default="research_data/earnings_data.duckdb")
    p.add_argument("--lookback-days", type=int, default=4,
                   help="How far back to look for recent reports (default 4d)")
    p.add_argument("--delay-ms", type=int, default=200,
                   help="Sleep between yfinance calls (default 200ms)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _load_recent_reporters(db_path: str, lookback_days: int) -> list[str]:
    today = datetime.now().date()
    start = today - timedelta(days=lookback_days)
    end = today + timedelta(days=1)
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            """
            SELECT symbol FROM earnings_calendar
            WHERE expected_report_date >= ?
              AND expected_report_date <= ?
              AND confirmed_reported_at IS NULL
            GROUP BY symbol
            ORDER BY MAX(expected_report_date) DESC, symbol
            """,
            [start, end],
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def _fetch_yfinance(symbol: str, lookback_days: int) -> list[dict]:
    """Pull yfinance.Ticker(symbol).get_earnings_dates(). Filter to past N days
    AND non-NULL surprise_pct. Returns rows ready for upsert."""
    import yfinance as yf

    try:
        t = yf.Ticker(symbol)
        ed = t.get_earnings_dates(limit=8)
    except Exception as exc:
        logging.warning("yfinance %s failed: %s", symbol, exc)
        return []
    if ed is None or ed.empty:
        return []

    today = datetime.now().date()
    start = today - timedelta(days=lookback_days)
    out = []
    for idx, row in ed.iterrows():
        try:
            d = idx.date() if hasattr(idx, "date") else None
        except Exception:
            continue
        if d is None or d < start or d > today + timedelta(days=1):
            continue
        # yfinance columns: "EPS Estimate", "Reported EPS", "Surprise(%)"
        surprise = row.get("Surprise(%)")
        if surprise is None or (isinstance(surprise, float) and surprise != surprise):
            continue  # NaN — yfinance hasn't received the surprise yet
        # Time hint from event hour (US/Eastern)
        try:
            et = idx.tz_convert("US/Eastern") if idx.tzinfo else idx
            h, m = et.hour, et.minute
            if h < 9 or (h == 9 and m < 30):
                time_hint, time_part = "bmo", "08:00:00"
            elif h >= 16:
                time_hint, time_part = "amc", "21:00:00"
            else:
                time_hint, time_part = "unknown", "12:00:00"
        except Exception:
            time_hint, time_part = "unknown", "12:00:00"

        eps_est = row.get("EPS Estimate")
        eps_rep = row.get("Reported EPS")

        def _f(v):
            if v is None:
                return None
            try:
                fv = float(v)
                if fv != fv:  # NaN check
                    return None
                return fv
            except Exception:
                return None

        out.append({
            "symbol": symbol,
            "event_datetime": f"{d} {time_part}",
            "eps_estimate": _f(eps_est),
            "reported_eps": _f(eps_rep),
            "surprise_pct": _f(surprise),
            "revenue_average": None,
            "is_future": False,
            "source": "yfinance",
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "time_hint": time_hint,
        })
    return out


def _upsert_safe(con: duckdb.DuckDBPyConnection, row: dict) -> str:
    """Insert only if no existing row within +/-1 day for same symbol.
    Never overwrites AV or any other source — yfinance is fill-only."""
    existing = con.execute(
        """
        SELECT event_datetime, source, surprise_pct
        FROM earnings_events
        WHERE symbol = ?
          AND ABS(EXTRACT(epoch FROM (event_datetime - ?::TIMESTAMP)) / 86400) <= 1
        ORDER BY ABS(EXTRACT(epoch FROM (event_datetime - ?::TIMESTAMP)))
        LIMIT 1
        """,
        [row["symbol"], row["event_datetime"], row["event_datetime"]],
    ).fetchone()
    if existing is not None:
        existing_dt, existing_source, existing_surp = existing
        # If existing row already has surprise_pct, leave it alone.
        if existing_surp is not None:
            return "kept_existing"
        # Existing row has NULL surprise_pct (likely a yfinance "future" row
        # that landed early) — fill it with our yfinance surprise.
        con.execute(
            """
            UPDATE earnings_events SET
              eps_estimate = COALESCE(?, eps_estimate),
              reported_eps = COALESCE(?, reported_eps),
              surprise_pct = ?,
              time_hint = ?,
              updated_at = ?,
              is_future = false
            WHERE symbol = ? AND event_datetime = ?
            """,
            [row["eps_estimate"], row["reported_eps"], row["surprise_pct"],
             row["time_hint"], row["updated_at"],
             row["symbol"], existing_dt],
        )
        return "filled_null"
    # No existing row — pure insert.
    con.execute(
        """
        INSERT INTO earnings_events
          (symbol, event_datetime, eps_estimate, reported_eps,
           surprise_pct, revenue_average, is_future, source,
           updated_at, time_hint)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [row["symbol"], row["event_datetime"], row["eps_estimate"],
         row["reported_eps"], row["surprise_pct"], row["revenue_average"],
         row["is_future"], row["source"], row["updated_at"],
         row["time_hint"]],
    )
    return "inserted"


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _load_recent_reporters(args.db, args.lookback_days)
    logging.warning("yfinance recent ingest %s | %d symbols | lookback=%dd",
                    "DRY-RUN" if args.dry_run else "LIVE",
                    len(symbols), args.lookback_days)

    counters = {"inserted": 0, "filled_null": 0, "kept_existing": 0,
                "no_data": 0, "errors": 0}

    con = duckdb.connect(args.db) if not args.dry_run else None
    try:
        for i, sym in enumerate(symbols, 1):
            try:
                rows = _fetch_yfinance(sym, args.lookback_days)
            except Exception as exc:
                logging.warning("[%d/%d] %s fetch failed: %s",
                                i, len(symbols), sym, exc)
                counters["errors"] += 1
                time.sleep(args.delay_ms / 1000.0)
                continue
            if not rows:
                counters["no_data"] += 1
                logging.info("[%d/%d] %s: no surprise data yet", i, len(symbols), sym)
                time.sleep(args.delay_ms / 1000.0)
                continue
            for row in rows:
                if args.dry_run:
                    logging.info("  [dry] %s %s surprise=%.2f%% time=%s",
                                 row["symbol"], row["event_datetime"][:10],
                                 row["surprise_pct"], row["time_hint"])
                    continue
                try:
                    result = _upsert_safe(con, row)
                    counters[result] = counters.get(result, 0) + 1
                    logging.info("[%d/%d] %s: %s (%s, surprise=%.2f%%)",
                                 i, len(symbols), sym, result,
                                 row["event_datetime"][:10], row["surprise_pct"])
                except Exception as exc:
                    logging.warning("upsert(%s) failed: %s", sym, exc)
                    counters["errors"] += 1
            time.sleep(args.delay_ms / 1000.0)
    finally:
        if con is not None:
            con.close()

    logging.warning("Done. Counters: %s", counters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
