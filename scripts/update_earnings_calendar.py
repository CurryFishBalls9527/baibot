#!/usr/bin/env python3
"""Maintain a local earnings calendar table.

Decouples "knowing WHO reports WHEN" (this script) from "fetching the
actual earnings data" (ingest_earnings_alphavantage.py). The ingest
script reads from this table to prioritize which symbols to fetch from
Alpha Vantage each night.

Schema (DuckDB):
    CREATE TABLE earnings_calendar (
        symbol VARCHAR,
        expected_report_date DATE,
        expected_report_time VARCHAR,   -- 'bmo' | 'amc' | 'unknown'
        confirmed_reported_at TIMESTAMP,  -- when we saw an actual event row
        source VARCHAR,                 -- 'yfinance' for now
        fetched_at TIMESTAMP,
        PRIMARY KEY (symbol, expected_report_date)
    );

Source: per-symbol yfinance.Ticker.earnings_dates. Free, no rate limit
issues at our scale (~250 symbols × 1-2 calls/sec ≈ 4-5 min runtime).
Returns historical + upcoming earnings dates; we keep the [-90d, +180d]
window — past for confirmation, future for forward planning.

Cadence: run weekly via launchd. Doesn't need to fire daily — yfinance
calendar entries don't change often (analyst date updates, occasional
postponements). Lower bound on freshness: every Sunday at 02:00 CDT.
But also fine to run more frequently — it's free.

Usage:
    ./.venv/bin/python scripts/update_earnings_calendar.py
    ./.venv/bin/python scripts/update_earnings_calendar.py --symbols META AMZN  # specific
    ./.venv/bin/python scripts/update_earnings_calendar.py --max-symbols 50      # cap
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ETFs and instruments without earnings reports — skip to save runtime.
NO_EARNINGS_TICKERS: set[str] = {
    "AGG", "IEF", "IWM", "QQQ", "SMH", "SPY", "TLT",
    "DIA", "GLD", "SLV", "VTI", "VOO", "VEA", "VWO",
    "HYG", "LQD", "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "XBI",
    "ARKK", "TQQQ", "SQQQ", "UVXY",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="*",
                   help="Specific symbols. Default: full broad250.")
    p.add_argument("--db", default="research_data/market_data.duckdb")
    p.add_argument("--universe", default="research_data/intraday_top250_universe.json")
    p.add_argument("--max-symbols", type=int, default=None,
                   help="Cap symbols processed. Default: all.")
    p.add_argument("--lookback-days", type=int, default=90,
                   help="Keep historical calendar entries this far back")
    p.add_argument("--lookahead-days", type=int, default=180,
                   help="Keep future calendar entries this far ahead")
    p.add_argument("--delay-ms", type=int, default=200,
                   help="Sleep between yfinance calls (default 200ms)")
    return p.parse_args()


def _ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS earnings_calendar (
            symbol VARCHAR,
            expected_report_date DATE,
            expected_report_time VARCHAR,
            confirmed_reported_at TIMESTAMP,
            source VARCHAR,
            fetched_at TIMESTAMP,
            PRIMARY KEY (symbol, expected_report_date)
        )
        """
    )


def _classify_time_hint(report_dt) -> str:
    """yfinance earnings_dates index is timezone-aware (US/Eastern).
    BMO if hour < 9:30 ET, AMC if hour > 16:00 ET, else unknown."""
    if report_dt is None:
        return "unknown"
    try:
        et = report_dt.tz_convert("US/Eastern") if report_dt.tzinfo else None
        if et is None:
            return "unknown"
        h, m = et.hour, et.minute
        if h < 9 or (h == 9 and m < 30):
            return "bmo"
        if h >= 16:
            return "amc"
        return "unknown"
    except Exception:
        return "unknown"


def _fetch_symbol_calendar(symbol: str, lookback: timedelta, lookahead: timedelta) -> list[dict]:
    """Pull yfinance.Ticker(symbol).earnings_dates. Filter to window.
    Returns list of rows for upsert."""
    import yfinance as yf

    try:
        t = yf.Ticker(symbol)
        ed = t.earnings_dates  # DataFrame indexed by report datetime (TZ-aware)
    except Exception as exc:
        logging.warning("yfinance %s failed: %s", symbol, exc)
        return []
    if ed is None or ed.empty:
        return []
    today = datetime.now().date()
    start = today - lookback
    end = today + lookahead
    out = []
    for idx, _row in ed.iterrows():
        try:
            d = idx.date() if hasattr(idx, "date") else None
        except Exception:
            continue
        if d is None or d < start or d > end:
            continue
        out.append({
            "symbol": symbol,
            "expected_report_date": d,
            "expected_report_time": _classify_time_hint(idx),
        })
    return out


def _upsert_calendar(con: duckdb.DuckDBPyConnection, row: dict, source: str = "yfinance") -> str:
    fetched_at = datetime.now(timezone.utc)
    existing = con.execute(
        """
        SELECT expected_report_time, confirmed_reported_at, source
        FROM earnings_calendar
        WHERE symbol = ? AND expected_report_date = ?
        """,
        [row["symbol"], row["expected_report_date"]],
    ).fetchone()
    if existing is None:
        con.execute(
            """
            INSERT INTO earnings_calendar
                (symbol, expected_report_date, expected_report_time,
                 confirmed_reported_at, source, fetched_at)
            VALUES (?, ?, ?, NULL, ?, ?)
            """,
            [row["symbol"], row["expected_report_date"],
             row["expected_report_time"], source, fetched_at],
        )
        return "inserted"
    # Update if time_hint changed (e.g. yfinance updated their schedule)
    existing_time, existing_confirmed, existing_source = existing
    if (existing_time or "unknown") == (row["expected_report_time"] or "unknown"):
        return "identical"
    con.execute(
        """
        UPDATE earnings_calendar
        SET expected_report_time = ?, fetched_at = ?
        WHERE symbol = ? AND expected_report_date = ?
        """,
        [row["expected_report_time"], fetched_at,
         row["symbol"], row["expected_report_date"]],
    )
    return "updated"


def _confirm_from_actuals(con: duckdb.DuckDBPyConnection) -> int:
    """For each calendar entry, if a matching event exists in
    earnings_events (within ±2 days for safety), set confirmed_reported_at.
    Returns count of newly-confirmed entries."""
    n = con.execute(
        """
        UPDATE earnings_calendar AS c
        SET confirmed_reported_at = sub.event_datetime
        FROM (
            SELECT
                ec.symbol AS sym,
                ec.expected_report_date AS d,
                ee.event_datetime AS event_datetime
            FROM earnings_calendar ec
            JOIN earnings_events ee
              ON ee.symbol = ec.symbol
             AND ABS(EXTRACT(epoch FROM (ee.event_datetime - ec.expected_report_date::TIMESTAMP)) / 86400) <= 2
            WHERE ec.confirmed_reported_at IS NULL
              AND ee.is_future = false
        ) AS sub
        WHERE c.symbol = sub.sym
          AND c.expected_report_date = sub.d
        """
    ).fetchone()
    # DuckDB UPDATE doesn't return rowcount easily — query before/after for diff if needed.
    # For now return 0; use SELECT COUNT for explicit check.
    confirmed_count = con.execute(
        "SELECT COUNT(*) FROM earnings_calendar WHERE confirmed_reported_at IS NOT NULL"
    ).fetchone()[0]
    return int(confirmed_count)


def _load_universe(path: str) -> list[str]:
    payload = json.loads(Path(path).read_text())
    return payload["symbols"] if isinstance(payload, dict) else payload


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = _load_universe(args.universe)
        symbols = [s for s in symbols if s not in NO_EARNINGS_TICKERS]
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]

    logging.warning("update_earnings_calendar: %d symbols, lookback=%dd lookahead=%dd",
                    len(symbols), args.lookback_days, args.lookahead_days)

    lookback = timedelta(days=args.lookback_days)
    lookahead = timedelta(days=args.lookahead_days)
    counts = {"inserted": 0, "updated": 0, "identical": 0, "errors": 0, "no_data": 0}

    con = duckdb.connect(args.db)
    try:
        _ensure_table(con)
        for i, sym in enumerate(symbols, 1):
            try:
                rows = _fetch_symbol_calendar(sym, lookback, lookahead)
            except Exception as exc:
                logging.warning("[%d/%d] %s failed: %s", i, len(symbols), sym, exc)
                counts["errors"] += 1
                time.sleep(args.delay_ms / 1000.0)
                continue
            if not rows:
                counts["no_data"] += 1
            else:
                for row in rows:
                    try:
                        result = _upsert_calendar(con, row)
                        counts[result] = counts.get(result, 0) + 1
                    except Exception as exc:
                        logging.warning("upsert %s/%s failed: %s",
                                        sym, row["expected_report_date"], exc)
                        counts["errors"] += 1
            if i % 25 == 0 or i == len(symbols):
                logging.info("[%d/%d] cumulative: %s", i, len(symbols),
                             {k: v for k, v in counts.items() if v > 0})
            time.sleep(args.delay_ms / 1000.0)

        confirmed = _confirm_from_actuals(con)
        logging.info("Confirmation pass: %d entries now have confirmed_reported_at",
                     confirmed)
    finally:
        con.close()

    logging.warning("Done. Counters: %s", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
