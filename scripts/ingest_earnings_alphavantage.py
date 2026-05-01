#!/usr/bin/env python3
"""Ingest historical earnings (with EPS surprise) from Alpha Vantage.

UPSERTs into the existing `earnings_events` DuckDB table. Where AV and
yfinance disagree on a (symbol, event_datetime) row, we keep BOTH unless
they're within 1 day of each other — in that case AV wins (cleaner data,
fewer "unknown" time_hints).

Free tier: 25 requests/day, 5/min. Default rate-limit delay 15s gives
~4/min comfortably under the per-minute cap. With 25/day, ingesting all
180 broad250 names not yet in our table takes ~7-8 days of nightly runs.
For one-time bulk ingest, upgrade to AV premium ($50/mo, 75/min).

Usage:
    # Default: prioritize broad250 symbols not yet in earnings_events
    ./.venv/bin/python scripts/ingest_earnings_alphavantage.py

    # Specific symbols
    ./.venv/bin/python scripts/ingest_earnings_alphavantage.py NVDA META TSLA

    # Cap requests (free tier)
    ./.venv/bin/python scripts/ingest_earnings_alphavantage.py --max-symbols 25

    # Tighter rate limit if we're on a higher tier
    ./.venv/bin/python scripts/ingest_earnings_alphavantage.py --delay-seconds 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


AV_BASE = "https://www.alphavantage.co/query"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("symbols", nargs="*",
                   help="Specific symbols to fetch. Default: broad250 names "
                        "not yet in earnings_events.")
    p.add_argument("--db", default="research_data/market_data.duckdb")
    p.add_argument("--universe", default="research_data/intraday_top250_universe.json",
                   help="Universe JSON for default symbol list")
    p.add_argument("--max-symbols", type=int, default=25,
                   help="Cap requests per run. Default 25 = free-tier daily limit.")
    p.add_argument("--delay-seconds", type=float, default=15.0,
                   help="Sleep between requests. 15s = ~4/min (under free 5/min cap).")
    p.add_argument("--dry-run", action="store_true",
                   help="Fetch and parse but do not write to DB.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ETFs and other instruments without earnings reports. Hardcoded so we
# don't waste AV's 25/day on AGG / SPY / etc., which would otherwise sit
# at the top of the alphabetical backfill list forever (always "missing
# from earnings_events" because there's nothing to insert).
NO_EARNINGS_TICKERS: set[str] = {
    "AGG", "IEF", "IWM", "QQQ", "SMH", "SPY", "TLT",
    "DIA", "GLD", "SLV", "VTI", "VOO", "VEA", "VWO",
    "HYG", "LQD", "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "XBI",
    "ARKK", "TQQQ", "SQQQ", "UVXY",
}


def _load_priority_symbols(
    universe_path: str, db_path: str, refresh_after_days: int = 60,
) -> list[str]:
    """Returns symbols in priority order for nightly ingest:

    1. BACKFILL — symbols in universe with NO earnings_events row at all
    2. REFRESH — symbols whose latest event is older than `refresh_after_days`
       (oldest-first, so stale-est data refreshes first)

    Quarterly EPS reports happen every ~90 days, so refresh_after_days=60
    means we re-fetch any symbol that hasn't been touched in 60+ days —
    this picks up new quarters within ~30 days of release at worst.

    Once backfill is exhausted (after ~10 days at AV free tier 25/day),
    the rotation naturally becomes refresh-only on a ~10-day cycle for
    a 250-symbol universe.
    """
    payload = json.loads(Path(universe_path).read_text())
    universe = payload["symbols"] if isinstance(payload, dict) else payload
    # Skip ETFs / non-earnings instruments to avoid wasting AV quota.
    universe = [s for s in universe if s not in NO_EARNINGS_TICKERS]
    con = duckdb.connect(db_path, read_only=True)
    try:
        # latest event per symbol
        rows = con.execute(
            "SELECT symbol, MAX(event_datetime) FROM earnings_events GROUP BY symbol"
        ).fetchall()
    finally:
        con.close()
    latest_by_sym = {sym: ts for sym, ts in rows}

    from datetime import datetime as _dt, timedelta as _td
    cutoff = _dt.now() - _td(days=refresh_after_days)

    backfill: list[str] = []
    refresh: list[tuple[str, object]] = []
    for sym in universe:
        if sym not in latest_by_sym:
            backfill.append(sym)
        elif latest_by_sym[sym] is not None and latest_by_sym[sym] < cutoff:
            refresh.append((sym, latest_by_sym[sym]))
    backfill.sort()
    refresh.sort(key=lambda x: x[1])  # oldest first
    return backfill + [s for s, _ in refresh]


def _av_fetch_earnings(symbol: str, key: str, timeout: int = 20) -> dict:
    r = requests.get(
        AV_BASE,
        params={"function": "EARNINGS", "symbol": symbol, "apikey": key},
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    data = r.json()
    if "Error Message" in data:
        raise RuntimeError(f"API error: {data['Error Message']}")
    if "Note" in data:
        raise RuntimeError(f"Rate limited: {data['Note']}")
    if "Information" in data:
        raise RuntimeError(f"AV info (often rate limit): {data['Information']}")
    return data


def _normalize_av_quarter(symbol: str, q: dict) -> dict | None:
    """AV quarterly record → our earnings_events row schema."""
    reported_date = q.get("reportedDate")
    if not reported_date or reported_date in ("None", "-"):
        return None
    try:
        # AV's reportTime is "pre-market" / "post-market" / sometimes None.
        rt = (q.get("reportTime") or "").strip().lower()
        if rt == "pre-market":
            time_hint = "bmo"
            time_part = "08:00:00"  # before open
        elif rt == "post-market":
            time_hint = "amc"
            time_part = "21:00:00"  # after close
        else:
            time_hint = "unknown"
            time_part = "12:00:00"
        event_dt = f"{reported_date} {time_part}"

        def _f(v):
            if v in (None, "None", "-", ""):
                return None
            try:
                return float(v)
            except Exception:
                return None

        return {
            "symbol": symbol,
            "event_datetime": event_dt,
            "eps_estimate": _f(q.get("estimatedEPS")),
            "reported_eps": _f(q.get("reportedEPS")),
            "surprise_pct": _f(q.get("surprisePercentage")),
            "revenue_average": None,  # not in EARNINGS endpoint
            "is_future": False,
            "source": "alphavantage",
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "time_hint": time_hint,
        }
    except Exception:
        return None


def _upsert(con: duckdb.DuckDBPyConnection, row: dict) -> str:
    """Returns 'inserted' / 'updated' / 'skipped' / 'identical' /
    'replaced_yfinance' / 'kept_yfinance_av_no_surprise'.
    """
    # Check for existing event within 1 day of this one for same symbol
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
        if existing_source == "alphavantage":
            # Already from AV. Only update if AV's new value carries data we
            # didn't have. Don't overwrite a populated surprise with NULL.
            if row["surprise_pct"] is None and existing_surp is not None:
                return "kept_existing_av"
            if existing_surp is not None and abs(existing_surp - (row["surprise_pct"] or 0)) < 0.001:
                return "identical"
            con.execute(
                """
                UPDATE earnings_events SET
                  eps_estimate = ?, reported_eps = ?, surprise_pct = ?,
                  time_hint = ?, updated_at = ?
                WHERE symbol = ? AND event_datetime = ?
                """,
                [row["eps_estimate"], row["reported_eps"], row["surprise_pct"],
                 row["time_hint"], row["updated_at"],
                 row["symbol"], existing_dt],
            )
            return "updated"
        else:
            # Existing yfinance row. Only replace with AV when AV is strictly
            # better — i.e. AV has a populated surprise_pct. Otherwise keep
            # yfinance (which may already have the surprise % we need).
            if row["surprise_pct"] is None and existing_surp is not None:
                return "kept_yfinance_av_no_surprise"
            con.execute(
                """
                DELETE FROM earnings_events
                WHERE symbol = ? AND event_datetime = ?
                """,
                [row["symbol"], existing_dt],
            )
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
            return "replaced_yfinance"
    # No existing event — pure insert
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
    api_key = (
        os.environ.get("ALPHA_VANTAGE_API_KEY")
        or os.environ.get("ALPHAVANTAGE_API_KEY")
    )
    if not api_key:
        raise SystemExit("Set ALPHA_VANTAGE_API_KEY in environment.")

    if args.symbols:
        symbols = args.symbols
    else:
        symbols = _load_priority_symbols(args.universe, args.db)
        logging.info("Default priority symbols (broad250 not yet in events): %d", len(symbols))

    if args.max_symbols:
        symbols = symbols[: args.max_symbols]

    logging.warning(
        "AV ingest %s | %d symbols | %.1fs delay between calls (~%.1f/min)",
        "DRY-RUN" if args.dry_run else "LIVE",
        len(symbols),
        args.delay_seconds,
        60.0 / max(args.delay_seconds, 0.1),
    )

    counters = {"inserted": 0, "updated": 0, "replaced_yfinance": 0,
                "identical": 0, "skipped": 0, "errors": 0}
    rows_total = 0

    con = duckdb.connect(args.db) if not args.dry_run else None
    try:
        for i, sym in enumerate(symbols, 1):
            try:
                data = _av_fetch_earnings(sym, api_key)
            except Exception as exc:
                logging.warning("[%d/%d] %s: FETCH FAILED — %s", i, len(symbols), sym, exc)
                counters["errors"] += 1
                # Common case: rate-limit hit → bail out cleanly
                if "rate" in str(exc).lower() or "Information" in str(exc):
                    logging.warning("Rate limit hit. Stopping ingest. Re-run tomorrow.")
                    break
                time.sleep(args.delay_seconds)
                continue
            quarters = data.get("quarterlyEarnings", []) or []
            sym_inserted = 0
            sym_updated = 0
            sym_skipped = 0
            for q in quarters:
                row = _normalize_av_quarter(sym, q)
                if row is None:
                    counters["skipped"] += 1
                    sym_skipped += 1
                    continue
                rows_total += 1
                if args.dry_run:
                    if args.verbose:
                        logging.info("  [dry] %s %s estimate=%.2f reported=%.2f surprise=%.2f%% time=%s",
                                     row["symbol"], row["event_datetime"][:10],
                                     row["eps_estimate"] or 0, row["reported_eps"] or 0,
                                     row["surprise_pct"] or 0, row["time_hint"])
                    continue
                try:
                    result = _upsert(con, row)
                    counters[result] = counters.get(result, 0) + 1
                    if result == "inserted":
                        sym_inserted += 1
                    elif result in ("updated", "replaced_yfinance"):
                        sym_updated += 1
                except Exception as exc:
                    logging.warning("upsert(%s, %s) failed: %s", sym, row["event_datetime"], exc)
                    counters["errors"] += 1
            logging.info("[%d/%d] %s: %d quarters → %d inserted, %d updated, %d skipped",
                         i, len(symbols), sym, len(quarters), sym_inserted, sym_updated, sym_skipped)
            if i < len(symbols):
                time.sleep(args.delay_seconds)
    finally:
        if con is not None:
            con.close()

    logging.warning("Done. Rows seen: %d. Counters: %s", rows_total, counters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
