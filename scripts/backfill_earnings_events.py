"""One-shot earnings-events backfill.

Iterates the union of broad_universe.json + LARGE_CAP_UNIVERSE +
(optionally) a paper watchlist, and populates earnings_events from
yfinance + Alpha Vantage. Idempotent.

Example:
    python scripts/backfill_earnings_events.py --universe research_data/broad_universe.json
    python scripts/backfill_earnings_events.py --symbols AAPL,MSFT,NVDA

Emits a coverage report. Exits non-zero if >threshold-pct of symbols
have fewer than ``--min-events`` events (the whole point of this run).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Make the repo importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.backtesting.screener import LARGE_CAP_UNIVERSE  # noqa: E402
from tradingagents.research.warehouse import MarketDataWarehouse  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("backfill_earnings")


def _load_universe(path: Path) -> list[str]:
    with path.open() as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "symbols" in payload:
        return list(payload["symbols"])
    if isinstance(payload, list):
        return list(payload)
    raise ValueError(f"Unrecognized universe format: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default="research_data/market_data.duckdb",
        help="Warehouse DB path",
    )
    parser.add_argument(
        "--universe",
        default="research_data/broad_universe.json",
        help="JSON file with list or {symbols: [...]}",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols to use instead of --universe",
    )
    parser.add_argument(
        "--include-large-cap",
        action="store_true",
        help="Union with LARGE_CAP_UNIVERSE",
    )
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument(
        "--no-av-fallback",
        action="store_true",
        help="Skip the Alpha Vantage fallback",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=0,
        help="Debug: cap the symbol count",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=5.0,
        help="Fail if more than this %% of symbols have < min-events",
    )
    parser.add_argument(
        "--report-out",
        default="results/earnings_coverage.csv",
        help="Path to write the per-symbol coverage CSV",
    )
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _load_universe(Path(args.universe))

    if args.include_large_cap:
        symbols = sorted(set(symbols) | set(LARGE_CAP_UNIVERSE))

    if args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    logger.info("Backfilling %d symbols into %s", len(symbols), args.db)

    wh = MarketDataWarehouse(db_path=args.db, read_only=False)
    row_counts = wh.fetch_and_store_earnings_events(
        symbols,
        limit=args.limit,
        min_events=args.min_events,
        use_alpha_vantage_fallback=not args.no_av_fallback,
    )

    coverage_rows = []
    for sym in symbols:
        events = wh.get_earnings_events(sym)
        n = len(events)
        if n > 0:
            earliest = events["event_datetime"].min()
            latest = events["event_datetime"].max()
            sources = (
                events["source"].value_counts().to_dict()
                if "source" in events.columns
                else {}
            )
        else:
            earliest = None
            latest = None
            sources = {}
        coverage_rows.append(
            {
                "symbol": sym,
                "n_events": n,
                "earliest": earliest,
                "latest": latest,
                "sources": sources,
                "fetch_row_count": row_counts.get(sym, 0),
            }
        )

    coverage = pd.DataFrame(coverage_rows)
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.report_out, index=False)
    logger.info("Wrote coverage report to %s", args.report_out)

    # Summary
    with_data = coverage[coverage["n_events"] > 0]
    below = coverage[coverage["n_events"] < args.min_events]
    logger.info(
        "Summary: %d/%d symbols have events, %d below min_events=%d",
        len(with_data),
        len(coverage),
        len(below),
        args.min_events,
    )
    if len(with_data) > 0:
        logger.info(
            "Event-count percentiles: 25=%d 50=%d 75=%d max=%d",
            int(with_data["n_events"].quantile(0.25)),
            int(with_data["n_events"].quantile(0.50)),
            int(with_data["n_events"].quantile(0.75)),
            int(with_data["n_events"].max()),
        )
        earliest = coverage["earliest"].dropna().min()
        logger.info("Oldest event across all symbols: %s", earliest)

    below_pct = 100.0 * len(below) / max(1, len(coverage))
    if below_pct > args.threshold_pct:
        logger.error(
            "FAIL: %.1f%% of symbols below min_events (threshold %.1f%%)",
            below_pct,
            args.threshold_pct,
        )
        return 1
    logger.info(
        "PASS: %.1f%% below threshold (limit %.1f%%)",
        below_pct,
        args.threshold_pct,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
