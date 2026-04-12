#!/usr/bin/env python3
"""One-shot: fetch sector/industry for all seed-universe symbols from yfinance.

Writes research_data/sector_map.json as a simple dict:
    {"NVDA": {"sector": "Technology", "industry": "Semiconductors"}, ...}

Re-run any time you want to refresh. Symbols that fail yfinance lookup
land under sector="Unknown", industry="Unknown".
"""
import json
import logging
import sys
from pathlib import Path

import yfinance as yf

from tradingagents.research.seed_universe import build_seed_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sector_map")
logging.getLogger("yfinance").setLevel(logging.ERROR)


def main():
    out_path = Path("research_data/sector_map.json")
    symbols = list(dict.fromkeys(build_seed_universe() + ["SPY"]))
    log.info("Fetching sector/industry for %d symbols", len(symbols))

    existing = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        log.info("Loaded %d existing entries; will skip those", len(existing))

    result = dict(existing)
    fetched = 0
    failed = 0
    for i, sym in enumerate(symbols, 1):
        if sym in result and result[sym].get("sector") not in (None, "Unknown"):
            continue
        try:
            info = yf.Ticker(sym).info
            result[sym] = {
                "sector": info.get("sector") or "Unknown",
                "industry": info.get("industry") or "Unknown",
            }
            fetched += 1
        except Exception as e:
            log.warning("%s failed: %s", sym, e)
            result[sym] = {"sector": "Unknown", "industry": "Unknown"}
            failed += 1
        if i % 25 == 0:
            log.info("Progress %d/%d (fetched=%d failed=%d)", i, len(symbols), fetched, failed)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    sectors = {}
    industries = {}
    for sym, meta in result.items():
        sectors[meta["sector"]] = sectors.get(meta["sector"], 0) + 1
        industries[meta["industry"]] = industries.get(meta["industry"], 0) + 1
    log.info("Done. %d symbols, %d sectors, %d industries", len(result), len(sectors), len(industries))
    log.info("Sectors: %s", dict(sorted(sectors.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
