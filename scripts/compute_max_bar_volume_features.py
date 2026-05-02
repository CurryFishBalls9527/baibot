#!/usr/bin/env python3
"""Bar-level intraday volume features for fake-breakout detection.

Computes TWO complementary metrics per (symbol, trade_date):

1. ``max_bar_rvol_20d`` — same-time-of-day cross-day baseline:
   For each slot s ∈ {09:30, 10:00, ..., 15:30}:
     baseline_s = trailing 20-trading-day mean of vol(symbol, slot=s)
     ratio_s    = today_vol(slot=s) / baseline_s
   Then max over s. Captures "this bar stands out vs typical days at
   this hour." Picks up regime shifts AND news spikes.

2. ``max_bar_rvol_intraday`` — within-day concentration:
   ratio = max(today_bar_volumes) / mean(today_bar_volumes)
   Captures "today had a concentrated moment vs the rest of today's
   session." Invariant to baseline drift — doesn't suffer the mature-
   leader problem the prior null `morning_volume_gate` hit (where a
   6-month leader's elevated 20-day baseline made every bar look
   "normal" so no single bar passed the 20-day cross-day gate).

Hypothesis: gating pyramid add-ons on EITHER metric ≥ threshold filters
out fake breakouts (drift through pivot without volume conviction)
while preserving real breakouts (single spike OR elevated-throughout
session OR concentrated-within-today).

Distinguishing this from the prior null `morning_volume_features`:
  * morning_volume aggregates 09:30-13:30 (8 bars summed) → dilutes
    bar-level spikes against ambient session volume.
  * Both metrics here keep bar-level resolution — different signal.

Writes to `market_data.duckdb` table `max_bar_volume_features` with
columns for both metrics so the A/B can test them independently.

Sources processed (concatenated):
  - research_data/intraday_30m_broad.duckdb       (2023-01 → present)
  - research_data/intraday_30m_broad_2018.duckdb  (2017-12 → 2019-01)
  - research_data/intraday_30m_broad_2020.duckdb  (2019-12 → 2021-01)

Driver: see `memory/project_morning_volume_gate_null.md` "Open question".
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
log = logging.getLogger("max_bar_volume_features")


INTRADAY_DBS = [
    "research_data/intraday_30m_broad.duckdb",
    "research_data/intraday_30m_broad_2018.duckdb",
    "research_data/intraday_30m_broad_2020.duckdb",
]
OUTPUT_DB = "research_data/market_data.duckdb"
TABLE_NAME = "max_bar_volume_features"

# Regular session 30m bars only: start times 09:30 → 15:30 ET (13 bars).
# Skip pre-market / after-hours bars (low volume noise).
_SESSION_FILTER_SQL = """(
    (EXTRACT(hour FROM ts) = 9 AND EXTRACT(minute FROM ts) = 30)
    OR (EXTRACT(hour FROM ts) BETWEEN 10 AND 14)
    OR (EXTRACT(hour FROM ts) = 15 AND EXTRACT(minute FROM ts) IN (0, 30))
)"""


def aggregate_per_bar(intraday_db: str) -> pd.DataFrame:
    """Return DataFrame with columns symbol, trade_date, bar_slot, volume.

    bar_slot is a string like '09:30', '10:00', ... '15:30'.
    """
    if not os.path.exists(intraday_db):
        log.warning("Skipping missing DB: %s", intraday_db)
        return pd.DataFrame(columns=["symbol", "trade_date", "bar_slot", "volume"])
    log.info("Aggregating per-bar volume from %s", intraday_db)
    con = duckdb.connect(intraday_db, read_only=True)
    df = con.execute(f"""
        SELECT
            symbol,
            DATE(ts) AS trade_date,
            LPAD(CAST(EXTRACT(hour FROM ts) AS VARCHAR), 2, '0') || ':' ||
            LPAD(CAST(EXTRACT(minute FROM ts) AS VARCHAR), 2, '0') AS bar_slot,
            SUM(volume) AS volume
        FROM bars_30m
        WHERE {_SESSION_FILTER_SQL}
        GROUP BY symbol, DATE(ts), bar_slot
    """).df()
    con.close()
    log.info("  rows: %d  symbols: %d  date range: %s → %s",
             len(df), df["symbol"].nunique() if len(df) else 0,
             df["trade_date"].min() if len(df) else "n/a",
             df["trade_date"].max() if len(df) else "n/a")
    return df


def compute_baseline_per_slot(df: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, bar_slot), compute 20-trading-day rolling baseline of
    that slot's volume. Returns df with added columns baseline_20d, ratio."""
    df = df.sort_values(["symbol", "bar_slot", "trade_date"]).reset_index(drop=True)
    log.info("Computing 20-day rolling baseline per (symbol, bar_slot) — "
             "%d rows...", len(df))
    df["baseline_20d"] = (
        df.groupby(["symbol", "bar_slot"])["volume"]
        .transform(lambda s: s.shift(1).rolling(window=20, min_periods=10).mean())
    )
    df["ratio"] = df["volume"] / df["baseline_20d"]
    df.loc[df["baseline_20d"].isna() | (df["baseline_20d"] <= 0), "ratio"] = pd.NA
    log.info("  per-bar rows with valid ratio: %d / %d",
             df["ratio"].notna().sum(), len(df))
    return df


def reduce_to_day_max(per_bar: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, trade_date), produce both metrics.

    Output schema:
      symbol, trade_date,
      max_bar_rvol_20d        — max over slots of (today / 20-day same-slot)
      max_bar_slot_20d        — slot of that argmax
      max_bar_rvol_intraday   — max(today_vol) / mean(today_vol)
      n_valid_bars_20d        — count of slots with 20-day baseline ready
      n_bars_today            — count of bars present today (for the within-
                                day metric; need ≥6 to be meaningful)
    """
    log.info("Reducing per-bar ratios to per-day metrics...")

    # --- Metric 1: cross-day same-slot 20-day ratio (max over slots) ---
    valid = per_bar.dropna(subset=["ratio"])
    if valid.empty:
        cross_day = pd.DataFrame(columns=[
            "symbol", "trade_date", "max_bar_rvol_20d", "max_bar_slot_20d",
            "n_valid_bars_20d",
        ])
    else:
        idx = valid.groupby(["symbol", "trade_date"])["ratio"].idxmax()
        cross_day = valid.loc[idx, ["symbol", "trade_date", "ratio", "bar_slot"]].rename(
            columns={"ratio": "max_bar_rvol_20d", "bar_slot": "max_bar_slot_20d"},
        )
        counts_20d = (
            valid.groupby(["symbol", "trade_date"]).size()
            .reset_index(name="n_valid_bars_20d")
        )
        cross_day = cross_day.merge(counts_20d, on=["symbol", "trade_date"], how="left")

    # --- Metric 2: within-day concentration max(today)/mean(today) ---
    # Use the raw `volume` column (not the baseline-scaled `ratio`). All bars
    # available today count, even ones that don't yet have a 20-day baseline
    # (a young symbol can still have a within-day metric).
    intraday_agg = (
        per_bar.groupby(["symbol", "trade_date"])["volume"]
        .agg(_today_max="max", _today_mean="mean", n_bars_today="count")
        .reset_index()
    )
    intraday_agg["max_bar_rvol_intraday"] = (
        intraday_agg["_today_max"] / intraday_agg["_today_mean"]
    )
    intraday_agg.loc[
        intraday_agg["_today_mean"] <= 0, "max_bar_rvol_intraday"
    ] = pd.NA
    # Need ≥6 bars (~half-day) for the within-day stat to be meaningful;
    # otherwise NaN.
    intraday_agg.loc[
        intraday_agg["n_bars_today"] < 6, "max_bar_rvol_intraday"
    ] = pd.NA
    intraday_agg = intraday_agg[
        ["symbol", "trade_date", "max_bar_rvol_intraday", "n_bars_today"]
    ]

    # --- Merge both metrics on (symbol, trade_date) ---
    out = intraday_agg.merge(
        cross_day, on=["symbol", "trade_date"], how="outer",
    )
    log.info("  per-day rows: %d", len(out))
    if len(out):
        for col in ("max_bar_rvol_20d", "max_bar_rvol_intraday"):
            ser = out[col].dropna()
            if len(ser):
                log.info(
                    "  %s distribution: p50=%.2f p75=%.2f p90=%.2f p99=%.2f",
                    col, ser.median(), ser.quantile(0.75),
                    ser.quantile(0.90), ser.quantile(0.99),
                )
    return out


def write_to_market_data(df: pd.DataFrame) -> None:
    out_path = ROOT / OUTPUT_DB
    log.info("Writing %d rows to %s :: %s", len(df), out_path, TABLE_NAME)
    con = duckdb.connect(str(out_path))
    con.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    con.execute(f"""
        CREATE TABLE {TABLE_NAME} (
            symbol                  VARCHAR NOT NULL,
            trade_date              DATE    NOT NULL,
            max_bar_rvol_20d        DOUBLE,
            max_bar_slot_20d        VARCHAR,
            n_valid_bars_20d        INTEGER,
            max_bar_rvol_intraday   DOUBLE,
            n_bars_today            INTEGER,
            PRIMARY KEY (symbol, trade_date)
        )
    """)
    con.register("df_view", df)
    con.execute(f"""
        INSERT INTO {TABLE_NAME}
        SELECT symbol, trade_date,
               max_bar_rvol_20d, max_bar_slot_20d, n_valid_bars_20d,
               max_bar_rvol_intraday, n_bars_today
        FROM df_view
    """)
    con.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_date "
        f"ON {TABLE_NAME}(trade_date)"
    )
    n = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    log.info("  table %s now has %d rows", TABLE_NAME, n)
    con.close()


def main() -> int:
    frames = []
    for db in INTRADAY_DBS:
        frames.append(aggregate_per_bar(str(ROOT / db)))
    if not any(len(f) for f in frames):
        log.error("No data aggregated from any intraday DB")
        return 1
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["symbol", "trade_date", "bar_slot"], keep="last"
    )
    log.info("Combined per-bar rows after dedup: %d", len(combined))
    enriched = compute_baseline_per_slot(combined)
    per_day = reduce_to_day_max(enriched)
    write_to_market_data(per_day)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
