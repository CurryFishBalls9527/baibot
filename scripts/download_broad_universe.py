#!/usr/bin/env python3
"""Download historical daily bars for a broad seed universe.

One-time data preparation for walk-forward backtesting.
Downloads 2-3 years of OHLCV data into the DuckDB warehouse.

Usage:
    python scripts/download_broad_universe.py --start 2022-01-01 --end 2025-12-31
    python scripts/download_broad_universe.py --universe research_data/seed_universe.json
    python scripts/download_broad_universe.py --start 2023-01-01 --end 2025-12-31 --batch-size 30
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradingagents.research.seed_universe import (
    build_seed_universe,
    load_seed_universe,
    save_seed_universe,
)
from tradingagents.research.warehouse import MarketDataWarehouse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Download broad universe data")
    parser.add_argument(
        "--start", type=str, default="2022-01-01",
        help="Start date for data download (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, default=datetime.now().strftime("%Y-%m-%d"),
        help="End date for data download (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--db", type=str, default="research_data/market_data.duckdb",
        help="Path to DuckDB warehouse",
    )
    parser.add_argument(
        "--universe", type=str, default=None,
        help="Path to existing seed universe JSON (skips building)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Symbols per yfinance batch download",
    )
    parser.add_argument(
        "--save-universe", type=str, default="research_data/seed_universe.json",
        help="Where to save the seed universe JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Build or load seed universe
    if args.universe:
        symbols = load_seed_universe(args.universe)
    else:
        symbols = build_seed_universe()
        save_seed_universe(symbols, args.save_universe, metadata={
            "start_date": args.start,
            "end_date": args.end,
        })

    # Always include benchmarks
    benchmarks = ["SPY", "QQQ", "IWM", "SMH"]
    all_symbols = sorted(set(symbols + benchmarks))

    logger.info(f"Universe: {len(all_symbols)} symbols")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Database: {args.db}")

    # 2. Initialize warehouse
    warehouse = MarketDataWarehouse(db_path=args.db)

    # 3. Check what's already downloaded
    existing = set(warehouse.available_symbols())
    needed = [s for s in all_symbols if s not in existing]
    already = len(all_symbols) - len(needed)
    if already > 0:
        logger.info(f"Already in warehouse: {already} symbols")

    # Download all (including existing to update)
    to_download = all_symbols
    total = len(to_download)
    logger.info(f"Downloading {total} symbols in batches of {args.batch_size}...")

    success = 0
    failed = []
    batch_size = args.batch_size

    for i in range(0, total, batch_size):
        batch = to_download[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        logger.info(
            f"  Batch {batch_num}/{total_batches}: "
            f"{batch[0]}..{batch[-1]} ({len(batch)} symbols)"
        )

        try:
            counts = warehouse.fetch_and_store_daily_bars(
                batch, start_date=args.start, end_date=args.end
            )
            batch_success = sum(1 for v in counts.values() if v > 0)
            batch_failed = [s for s, v in counts.items() if v == 0]
            success += batch_success
            failed.extend(batch_failed)
        except Exception as e:
            logger.warning(f"  Batch error: {e}")
            failed.extend(batch)

        # Rate limit courtesy
        if i + batch_size < total:
            time.sleep(1)

    # 4. Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Download complete")
    logger.info(f"  Successful: {success}/{total}")
    if failed:
        logger.warning(f"  Failed ({len(failed)}): {', '.join(failed[:20])}")
        if len(failed) > 20:
            logger.warning(f"    ... and {len(failed) - 20} more")

    # 5. Verify
    final_symbols = warehouse.available_symbols()
    logger.info(f"  Total symbols in warehouse: {len(final_symbols)}")
    warehouse.close()


if __name__ == "__main__":
    main()
