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
    p.add_argument("--db", default="research_data/earnings_data.duckdb")
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


def _load_recent_reporters_from_calendar(
    db_path: str, lookback_days: int = 4,
) -> dict:
    """Read from the local earnings_calendar table.

    Returns dict {symbol: most_recent_expected_report_date} for symbols
    whose expected_report_date is in [today - lookback_days, today + 1]
    AND we don't yet have a confirmed actual event for that date.

    Decoupled from data fetching: maintenance of earnings_calendar is
    handled by scripts/update_earnings_calendar.py (separate cron).
    This function is a pure DB read — no network, no rate limits.

    Returns empty dict if the table doesn't exist or no matching rows.
    """
    from datetime import datetime as _dt, timedelta as _td

    today = _dt.now().date()
    start = today - _td(days=lookback_days)
    end = today + _td(days=1)
    try:
        con = duckdb.connect(db_path, read_only=True)
        try:
            con.execute("SELECT 1 FROM earnings_calendar LIMIT 1")
        except Exception:
            con.close()
            return {}
        rows = con.execute(
            """
            SELECT symbol, MAX(expected_report_date) AS latest_expected
            FROM earnings_calendar
            WHERE expected_report_date >= ?
              AND expected_report_date <= ?
              AND confirmed_reported_at IS NULL
            GROUP BY symbol
            """,
            [start, end],
        ).fetchall()
        con.close()
    except Exception:
        return {}
    return {r[0]: r[1] for r in rows}


def _load_priority_symbols(
    universe_path: str,
    db_path: str,
    refresh_after_days: int = 60,
    fresh_nan_days: int = 30,
    recent_reporter_dates: dict | None = None,
) -> list[str]:
    """Returns symbols in priority order for nightly ingest:

    1. RECENT_REPORTER — symbols that ACTUALLY REPORTED in the past few
                         days (per yfinance check). HIGHEST priority —
                         fresh signal data exists upstream.
    2. NAN_REFRESH — symbols whose most recent non-future event has
                     `surprise_pct = NULL` AND is < `fresh_nan_days` old.
                     The report exists in our DB but data lag means we
                     don't yet have the surprise %.
    3. STALE_REFRESH — symbols whose latest non-future event is >
                       `refresh_after_days` old. Likely a new quarter
                       to fetch.
    4. BACKFILL — symbols in universe with no earnings_events row at all.

    Background: AV free tier = 25 calls/day. Need to spend that budget
    on the symbols MOST LIKELY to have a fresh signal for PEAD.
    """
    payload = json.loads(Path(universe_path).read_text())
    universe = payload["symbols"] if isinstance(payload, dict) else payload
    universe = [s for s in universe if s not in NO_EARNINGS_TICKERS]
    universe_set = set(universe)
    con = duckdb.connect(db_path, read_only=True)
    try:
        # Latest NON-FUTURE event per symbol with its surprise status.
        # (Future events from yfinance — like BP's expected Aug report — should
        # not be treated as "latest" for refresh purposes.)
        rows = con.execute(
            """
            WITH latest AS (
                SELECT symbol, event_datetime, surprise_pct,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY event_datetime DESC) AS rn
                FROM earnings_events
                WHERE is_future = false
            )
            SELECT symbol, event_datetime, surprise_pct FROM latest WHERE rn = 1
            """
        ).fetchall()
    finally:
        con.close()
    latest_by_sym = {sym: (ts, surp) for sym, ts, surp in rows}

    from datetime import datetime as _dt, timedelta as _td
    now = _dt.now()
    cutoff_stale = now - _td(days=refresh_after_days)
    cutoff_recent = now - _td(days=fresh_nan_days)

    # Tier 1: Recent reporters (per local calendar table). Sort by URGENCY:
    #   Tier 1a: latest non-future event in DB is OLDER than the calendar's
    #            expected_report_date — we're missing the recent quarter
    #   Tier 1b: latest non-future event >= expected_report_date — we have
    #            the recent quarter (just may need surprise via NaN-refresh)
    # Within each tier: most-recent expected date first, then alphabetical.
    recent_priority: list[str] = []
    if recent_reporter_dates:
        scored: list[tuple] = []
        for sym, expected_date in recent_reporter_dates.items():
            if sym not in universe_set:
                continue
            latest_entry = latest_by_sym.get(sym)
            if latest_entry is None:
                # Not in DB at all — most urgent
                urgency = 0
            else:
                ts, _surp = latest_entry
                ts_date = ts.date() if hasattr(ts, "date") else ts
                if ts_date < expected_date:
                    urgency = 0   # missing the recent quarter
                else:
                    urgency = 1   # have the recent quarter (NaN-refresh tier handles missing surprise)
            # Sort key: urgency, then most-recent date first (negative ordinal),
            # then alphabetical. Smaller key = higher priority.
            scored.append((urgency, -expected_date.toordinal(), sym))
        scored.sort()
        recent_priority = [s for _, _, s in scored]

    on_recent = set(recent_priority)

    nan_refresh: list[tuple[str, object]] = []
    stale_refresh: list[tuple[str, object]] = []
    backfill: list[str] = []
    for sym in universe:
        if sym in on_recent:
            continue
        if sym not in latest_by_sym:
            backfill.append(sym)
            continue
        ts, surp = latest_by_sym[sym]
        if ts is None:
            backfill.append(sym)
            continue
        if surp is None and ts >= cutoff_recent:
            # Have the event row but missing the surprise % — refresh ASAP
            nan_refresh.append((sym, ts))
        elif ts < cutoff_stale:
            # Latest event is old, likely a new quarter exists upstream
            stale_refresh.append((sym, ts))

    nan_refresh.sort(key=lambda x: x[1], reverse=True)  # most recent first
    stale_refresh.sort(key=lambda x: x[1])              # oldest first
    backfill.sort()

    return (
        recent_priority
        + [s for s, _ in nan_refresh]
        + [s for s, _ in stale_refresh]
        + backfill
    )


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
    # Dedicated key for the nightly earnings ingest cron — kept separate
    # from the shared ALPHA_VANTAGE_API_KEY so other callers (fundamentals
    # tools, ad-hoc scripts) can't burn this 25/day budget.
    api_key = os.environ.get("ALPHA_VANTAGE_INGEST_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set ALPHA_VANTAGE_INGEST_API_KEY in environment "
            "(dedicated key for earnings ingest; do not reuse the shared key)."
        )

    if args.symbols:
        symbols = args.symbols
    else:
        # Read from local earnings_calendar table — pure DB query, no network.
        # Calendar maintenance is a separate concern: see
        # scripts/update_earnings_calendar.py (runs daily via launchd).
        # Returns dict {symbol: most_recent_expected_report_date} so the
        # priority logic can compare against latest non-future event in DB.
        recent_syms = _load_recent_reporters_from_calendar(args.db, lookback_days=4)
        if recent_syms:
            logging.info("Calendar table: %d symbols with unconfirmed expected reports in past 4d",
                         len(recent_syms))
        else:
            logging.warning(
                "earnings_calendar empty/missing — run update_earnings_calendar.py first. "
                "Falling back to refresh+backfill priorities only."
            )

        symbols = _load_priority_symbols(
            args.universe, args.db,
            refresh_after_days=60,
            fresh_nan_days=30,
            recent_reporter_dates=recent_syms,
        )
        logging.info("Total priority queue: %d (recent+nan+stale+backfill)", len(symbols))

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
