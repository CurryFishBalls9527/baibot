#!/usr/bin/env python3
"""One-shot migration: split earnings_events + earnings_calendar tables
out of market_data.duckdb into a new earnings_data.duckdb.

Why: market_data.duckdb is a process-exclusive file. The live scheduler
holds brief write windows on it during daily-bar refresh. Standalone
crons (PEAD ingest, AV ingest, calendar update) and ad-hoc reads
(pead_trader's read_only=True query) collide with those windows and
fail with `Could not set lock`. Splitting the earnings tables into a
separate file means scheduler and earnings-pipeline processes never
contend at the OS-lock level.

Cross-DB queries (specifically warehouse.get_latest_fundamentals'
JOIN with earnings_events) keep working via DuckDB ATTACH ... (READ_ONLY).

Idempotent: if target already has matching row counts, exits 0 with a
no-op log message. Source DB is left intact — cleanup happens in a
later phase once the rollout is proven.

Usage:
    ./.venv/bin/python scripts/migrate_split_earnings_db.py
    ./.venv/bin/python scripts/migrate_split_earnings_db.py --dry-run
    ./.venv/bin/python scripts/migrate_split_earnings_db.py --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb


SOURCE_DB = "research_data/market_data.duckdb"
TARGET_DB = "research_data/earnings_data.duckdb"

# (table_name, CREATE TABLE DDL with PRIMARY KEY)
TABLES = [
    (
        "earnings_events",
        """
        CREATE TABLE IF NOT EXISTS earnings_events (
            symbol VARCHAR NOT NULL,
            event_datetime TIMESTAMP NOT NULL,
            eps_estimate DOUBLE,
            reported_eps DOUBLE,
            surprise_pct DOUBLE,
            revenue_average DOUBLE,
            is_future BOOLEAN,
            source VARCHAR,
            updated_at TIMESTAMP,
            time_hint VARCHAR,
            PRIMARY KEY (symbol, event_datetime)
        )
        """,
    ),
    (
        "earnings_calendar",
        """
        CREATE TABLE IF NOT EXISTS earnings_calendar (
            symbol VARCHAR NOT NULL,
            expected_report_date DATE NOT NULL,
            expected_report_time VARCHAR,
            confirmed_reported_at TIMESTAMP,
            source VARCHAR,
            fetched_at TIMESTAMP,
            PRIMARY KEY (symbol, expected_report_date)
        )
        """,
    ),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=SOURCE_DB)
    p.add_argument("--target", default=TARGET_DB)
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would happen, but don't write")
    p.add_argument("--force", action="store_true",
                   help="Force re-copy even if target row counts match")
    return p.parse_args()


def _count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    try:
        return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except duckdb.CatalogException:
        return -1  # table doesn't exist yet


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    src = Path(args.source)
    tgt = Path(args.target)
    if not src.exists():
        logging.error("source DB missing: %s", src)
        return 1

    # Read source counts (read-only — won't contend with scheduler writes
    # except in the brief moment of the source's own lock window).
    src_counts: dict[str, int] = {}
    src_con = duckdb.connect(str(src), read_only=True)
    try:
        for table, _ddl in TABLES:
            src_counts[table] = _count(src_con, table)
            logging.info("source %s: %d rows", table, src_counts[table])
    finally:
        src_con.close()

    # Open target read-write, create tables, copy.
    if tgt.exists():
        logging.info("target exists: %s (will append/skip per idempotency)", tgt)
    else:
        logging.info("target will be created: %s", tgt)
        tgt.parent.mkdir(parents=True, exist_ok=True)

    tgt_con = duckdb.connect(str(tgt))
    try:
        # Idempotency check first
        skip_all = True
        for table, ddl in TABLES:
            tgt_con.execute(ddl)  # idempotent CREATE IF NOT EXISTS
            tgt_count = _count(tgt_con, table)
            src_count = src_counts[table]
            if args.force:
                skip_all = False
                continue
            if tgt_count == src_count and src_count > 0:
                logging.info("target %s already has %d rows — match, skip",
                             table, tgt_count)
            else:
                skip_all = False

        if skip_all and not args.force:
            logging.info("All tables already in sync — no-op exit.")
            return 0

        if args.dry_run:
            logging.warning("DRY-RUN: would copy missing rows now; exiting.")
            return 0

        # Copy: ATTACH source read-only, INSERT OR IGNORE (DuckDB equivalent).
        # Using anti-join because DuckDB's INSERT doesn't support OR IGNORE
        # but PRIMARY KEY + WHERE NOT EXISTS does the job idempotently.
        tgt_con.execute(f"ATTACH '{src}' AS src (READ_ONLY)")
        try:
            for table, _ddl in TABLES:
                if table == "earnings_events":
                    pk_cols = "symbol, event_datetime"
                else:
                    pk_cols = "symbol, expected_report_date"
                before = _count(tgt_con, table)
                tgt_con.execute(
                    f"""
                    INSERT INTO {table}
                    SELECT * FROM src.{table} s
                    WHERE NOT EXISTS (
                        SELECT 1 FROM {table} t
                        WHERE ({', '.join(f't.{c}' for c in pk_cols.split(', '))}) =
                              ({', '.join(f's.{c}' for c in pk_cols.split(', '))})
                    )
                    """
                )
                after = _count(tgt_con, table)
                logging.info("copied %s: %d → %d (+%d)", table, before, after,
                             after - before)
        finally:
            tgt_con.execute("DETACH src")

        # Final verification
        for table, _ddl in TABLES:
            tgt_count = _count(tgt_con, table)
            src_count = src_counts[table]
            if tgt_count != src_count:
                logging.warning(
                    "MISMATCH after copy %s: source=%d target=%d "
                    "(may indicate concurrent writes during migration)",
                    table, src_count, tgt_count,
                )
            else:
                logging.info("verified %s: %d rows match", table, tgt_count)
    finally:
        tgt_con.close()

    logging.warning("Migration complete: %s", tgt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
