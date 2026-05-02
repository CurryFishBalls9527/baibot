#!/usr/bin/env python3
"""Post-hoc analysis: what would an ER-blackout gate have done to live trades?

Iterates trade_outcomes across the live paper-trading SQLite DBs, joins to
earnings_events in the research warehouse, and classifies each closed trade
by proximity to the nearest ER during its hold window. Reports:

  - P&L bucketed by days_to_nearest_ER
  - aggregate return with vs. without a retroactive ER gate
  - count of trades that would have been skipped at each gate threshold

Not a new backtest — just an exposure audit on the historical live book.

Usage:
  python scripts/analyze_llm_er_exposure.py
  python scripts/analyze_llm_er_exposure.py --dbs trading_llm.db,trading_mechanical.db
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradingagents.research.warehouse import MarketDataWarehouse  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("llm_er_exposure")

DEFAULT_DBS = [
    "trading_llm.db",
    "trading_mechanical.db",
    "trading_mechanical_v2.db",
    "trading_chan.db",
    "trading_chan_v2.db",
    "trading_intraday_mechanical.db",
]


def _load_trade_outcomes(db_path: str) -> pd.DataFrame:
    if not Path(db_path).exists():
        logger.warning("Skipping missing DB: %s", db_path)
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT id, symbol, entry_date, exit_date, entry_price, exit_price,
                   return_pct, hold_days, exit_reason
            FROM trade_outcomes
            WHERE entry_date IS NOT NULL
            """,
            conn,
        )
    except Exception as exc:
        logger.warning("Failed reading trade_outcomes from %s: %s", db_path, exc)
        return pd.DataFrame()
    finally:
        conn.close()
    df["source_db"] = Path(db_path).name
    return df


def _nearest_er_during_hold(
    events_for_symbol: pd.DataFrame,
    entry: pd.Timestamp,
    exit_: pd.Timestamp,
) -> dict | None:
    """Return the ER event closest to, and inside OR adjacent to, the hold
    window. 'Adjacent' means within 5 cal days of entry/exit — useful for
    'entered right before ER' detection."""
    if events_for_symbol.empty:
        return None
    ev = events_for_symbol["event_datetime"]
    in_hold = events_for_symbol[(ev >= entry) & (ev <= exit_)]
    if not in_hold.empty:
        row = in_hold.iloc[0]
        return {
            "event_datetime": row["event_datetime"],
            "time_hint": row.get("time_hint"),
            "source": row.get("source"),
            "position": "within_hold",
        }
    # nearest outside-hold event
    deltas = (ev - entry).abs().combine((ev - exit_).abs(), min)
    idx = deltas.idxmin()
    row = events_for_symbol.loc[idx]
    delta_days = (pd.Timestamp(row["event_datetime"]) - entry).days
    return {
        "event_datetime": row["event_datetime"],
        "time_hint": row.get("time_hint"),
        "source": row.get("source"),
        "position": "outside_hold",
        "days_from_entry": delta_days,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dbs", default=",".join(DEFAULT_DBS),
                   help="Comma-separated SQLite paths")
    p.add_argument("--warehouse",
                   default="research_data/market_data.duckdb",
                   help="MarketDataWarehouse main DB. Earnings events are read "
                        "via the warehouse's attached earnings_data.duckdb.")
    p.add_argument("--entry-gate-days", default="1,3,5,7",
                   help="Sweep these entry-blackout thresholds")
    p.add_argument("--out", default="results/llm_er_exposure.csv")
    args = p.parse_args()

    dbs = [s.strip() for s in args.dbs.split(",") if s.strip()]
    frames = [_load_trade_outcomes(db) for db in dbs]
    trades = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if trades.empty:
        logger.warning("No closed trades found — nothing to analyze.")
        return 0

    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    logger.info("Loaded %d closed trades across %d DBs",
                len(trades), trades["source_db"].nunique())

    wh = MarketDataWarehouse(db_path=args.warehouse, read_only=True)

    # Per-symbol ER cache
    er_cache: dict[str, pd.DataFrame] = {}

    def _get_er(sym: str) -> pd.DataFrame:
        if sym not in er_cache:
            ev = wh.get_earnings_events(sym)
            if ev is None:
                ev = pd.DataFrame()
            elif not ev.empty:
                ev["event_datetime"] = pd.to_datetime(ev["event_datetime"])
            er_cache[sym] = ev
        return er_cache[sym]

    enriched = []
    for _, t in trades.iterrows():
        ev_frame = _get_er(t["symbol"])
        match = _nearest_er_during_hold(ev_frame, t["entry_date"], t["exit_date"])
        row = dict(t)
        if match:
            row.update({
                "er_event_datetime": match.get("event_datetime"),
                "er_time_hint": match.get("time_hint"),
                "er_source": match.get("source"),
                "er_position": match.get("position"),
                "er_days_from_entry": match.get("days_from_entry"),
            })
            if match.get("position") == "within_hold":
                ev_ts = pd.Timestamp(match["event_datetime"]).normalize()
                row["er_days_from_entry"] = (ev_ts - t["entry_date"].normalize()).days
                row["held_through_er"] = True
            else:
                row["held_through_er"] = False
        else:
            row.update({"er_event_datetime": None, "held_through_er": None})
        enriched.append(row)

    out = pd.DataFrame(enriched)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    logger.info("Wrote enriched trades CSV to %s", args.out)

    # Aggregates
    total = len(out)
    held = int((out["held_through_er"] == True).sum())  # noqa: E712
    logger.info("Total closed trades: %d", total)
    logger.info("Trades held THROUGH at least one ER: %d (%.1f%%)",
                held, 100.0 * held / max(1, total))

    if held > 0:
        held_df = out[out["held_through_er"] == True]  # noqa: E712
        logger.info("  Mean return of ER-held trades: %.2f%%",
                    100 * held_df["return_pct"].mean())
        logger.info("  Median return of ER-held trades: %.2f%%",
                    100 * held_df["return_pct"].median())
        logger.info("  Worst ER-held trade: %.2f%%",
                    100 * held_df["return_pct"].min())

    not_held = out[out["held_through_er"] != True]  # noqa: E712
    if not not_held.empty:
        logger.info("Mean return of non-ER-held trades: %.2f%%",
                    100 * not_held["return_pct"].mean())

    # Retroactive entry-gate sweep
    print("\n=== Retroactive entry-gate what-if ===")
    gate_values = [int(x) for x in args.entry_gate_days.split(",") if x.strip()]
    baseline_mean = out["return_pct"].mean()
    baseline_median = out["return_pct"].median()
    print(f"baseline (no gate): n={total}  mean={baseline_mean:+.3%}  "
          f"median={baseline_median:+.3%}")
    for g in gate_values:
        # A trade would have been SKIPPED under a g-day entry gate if an ER
        # fell within g days AFTER the entry date.
        def _blocked(row, g=g):
            sym = row["symbol"]
            ev = _get_er(sym)
            if ev.empty:
                return False
            entry = row["entry_date"].normalize()
            upper = entry + pd.Timedelta(days=g)
            within = ev[(ev["event_datetime"] >= entry) &
                        (ev["event_datetime"] <= upper)]
            return not within.empty
        mask = out.apply(_blocked, axis=1)
        kept = out[~mask]
        skipped = int(mask.sum())
        if len(kept) == 0:
            print(f"  gate={g}d: skipped={skipped}/{total} — ALL trades skipped")
            continue
        print(f"  gate={g}d: skipped={skipped}/{total}  "
              f"kept_mean={kept['return_pct'].mean():+.3%}  "
              f"kept_median={kept['return_pct'].median():+.3%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
