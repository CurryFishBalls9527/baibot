#!/usr/bin/env python3
"""Compute morning_volume_ratio feature from intraday 30m bars.

For each (symbol, trade_date):
  morning_volume = sum of 30m-bar volume in [09:30, 13:00] ET start-time
                   (covers regular trading 09:30 → 13:30 ET, i.e. first 4 hours,
                   matching when pyramid add-ons fire at ~13:35 ET).
  baseline_20d   = trailing 20-trading-day mean of morning_volume per symbol
                   (excluding today; uses prior 20 trading days).
  ratio          = morning_volume / baseline_20d
                   (NaN until ≥10 prior days exist).

Writes to `market_data.duckdb` table `morning_volume_features`.

Idempotent: drops + recreates the table on each run. The Minervini feature
pipeline JOINs this table by (symbol, trade_date) when the
`min_morning_volume_ratio` config knob is non-zero.

Sources processed (concatenated):
  - research_data/intraday_30m_broad.duckdb       (2023-01 → present, broad)
  - research_data/intraday_30m_broad_2018.duckdb  (2017-12 → 2019-01)
  - research_data/intraday_30m_broad_2020.duckdb  (2019-12 → 2021-01)

Driver: W18 daily-review verification — leader_continuation entries had no
volume gate and the daily-bar `breakout_volume_ratio` was the wrong metric.
This metric captures volume context AT the live decision moment.
See `memory/project_volume_gate_null.md` for context.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("morning_volume_features")


INTRADAY_DBS = [
    "research_data/intraday_30m_broad.duckdb",
    "research_data/intraday_30m_broad_2018.duckdb",
    "research_data/intraday_30m_broad_2020.duckdb",
]

OUTPUT_DB = "research_data/market_data.duckdb"
TABLE_NAME = "morning_volume_features"


# Bars to include: those whose `ts` start time falls in the first 4 hours of
# the regular session, i.e. start ∈ {09:30, 10:00, 10:30, 11:00, 11:30, 12:00,
# 12:30, 13:00}. This is exactly the 8 30m-bars covering 09:30 → 13:30 ET.
# At 13:35 ET (pyramid add-on time) these are all completed; at 15:30 ET
# (swing-analysis time) it's a strict prefix of the trading day.
_MORNING_FILTER_SQL = """(
    (EXTRACT(hour FROM ts) = 9 AND EXTRACT(minute FROM ts) = 30)
    OR (EXTRACT(hour FROM ts) BETWEEN 10 AND 12)
    OR (EXTRACT(hour FROM ts) = 13 AND EXTRACT(minute FROM ts) = 0)
)"""


def aggregate_morning_volume(intraday_db: str) -> pd.DataFrame:
    """Return DataFrame with columns symbol, trade_date, morning_volume."""
    if not os.path.exists(intraday_db):
        log.warning("Skipping missing DB: %s", intraday_db)
        return pd.DataFrame(columns=["symbol", "trade_date", "morning_volume"])

    log.info("Aggregating morning volume from %s", intraday_db)
    con = duckdb.connect(intraday_db, read_only=True)
    df = con.execute(f"""
        SELECT
            symbol,
            DATE(ts) AS trade_date,
            SUM(volume) AS morning_volume
        FROM bars_30m
        WHERE {_MORNING_FILTER_SQL}
        GROUP BY symbol, DATE(ts)
    """).df()
    con.close()
    log.info("  rows: %d  symbols: %d  date range: %s → %s",
             len(df), df["symbol"].nunique(),
             df["trade_date"].min() if len(df) else "n/a",
             df["trade_date"].max() if len(df) else "n/a")
    return df


def compute_baseline_and_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add `baseline_20d` and `ratio` columns. Per-symbol trailing window.

    Uses 20 prior trading days (shift 1 then mean of last 20). `ratio` is
    `morning_volume / baseline_20d`; both NaN until 10+ priors exist.
    """
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    log.info("Computing 20-day rolling baseline per symbol (%d (sym,date) rows)...",
             len(df))
    df["baseline_20d"] = (
        df.groupby("symbol")["morning_volume"]
        .transform(lambda s: s.shift(1).rolling(window=20, min_periods=10).mean())
    )
    df["ratio"] = df["morning_volume"] / df["baseline_20d"]
    df.loc[df["baseline_20d"].isna() | (df["baseline_20d"] <= 0), "ratio"] = pd.NA
    n_with_ratio = df["ratio"].notna().sum()
    log.info("  rows with valid ratio: %d / %d (%.1f%%)",
             n_with_ratio, len(df), 100.0 * n_with_ratio / max(1, len(df)))
    return df


def write_to_market_data(df: pd.DataFrame) -> None:
    out_path = ROOT / OUTPUT_DB
    log.info("Writing %d rows to %s :: %s", len(df), out_path, TABLE_NAME)
    con = duckdb.connect(str(out_path))
    con.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    con.execute(f"""
        CREATE TABLE {TABLE_NAME} (
            symbol         VARCHAR NOT NULL,
            trade_date     DATE    NOT NULL,
            morning_volume DOUBLE,
            baseline_20d   DOUBLE,
            ratio          DOUBLE,
            PRIMARY KEY (symbol, trade_date)
        )
    """)
    con.register("df_view", df)
    con.execute(f"""
        INSERT INTO {TABLE_NAME}
        SELECT symbol, trade_date, morning_volume, baseline_20d, ratio FROM df_view
    """)
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_date ON {TABLE_NAME}(trade_date)")
    n = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    log.info("  table %s now has %d rows", TABLE_NAME, n)
    con.close()


def main() -> int:
    frames = []
    for db in INTRADAY_DBS:
        path = ROOT / db
        frames.append(aggregate_morning_volume(str(path)))
    if not any(len(f) for f in frames):
        log.error("No data aggregated from any intraday DB")
        return 1
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["symbol", "trade_date"], keep="last"
    )
    log.info("Combined: %d (sym,date) rows after dedup", len(combined))
    enriched = compute_baseline_and_ratio(combined)
    write_to_market_data(enriched)
    return 0


if __name__ == "__main__":
    sys.exit(main())
