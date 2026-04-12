#!/usr/bin/env python3
"""Fetch iShares Core S&P Small-Cap ETF (IJR) holdings and save as JSON.

Produces research_data/ijr_holdings.json with current constituents.
Re-run any time to refresh. Used by seed_universe.py to expand the
walk-forward backtest universe with liquid small-caps.
"""
import csv
import json
import logging
import urllib.request
from datetime import datetime
from pathlib import Path

IJR_URL = (
    "https://www.ishares.com/us/products/239774/"
    "ishares-core-sp-smallcap-etf/1467271812596.ajax"
    "?fileType=csv&fileName=IJR_holdings&dataType=fund"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ijr")


def fetch_ijr_holdings(min_price: float = 5.0) -> dict:
    req = urllib.request.Request(IJR_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8-sig", "replace")
    lines = raw.splitlines()

    fund_date = ""
    for ln in lines[:10]:
        if ln.startswith("Fund Holdings as of"):
            fund_date = ln.split(",", 1)[1].strip().strip('"')
            break

    header_idx = next(i for i, l in enumerate(lines) if l.startswith("Ticker,"))
    reader = csv.DictReader(lines[header_idx:])
    equities = [
        r for r in reader
        if r.get("Asset Class") == "Equity" and r.get("Ticker", "").strip().isalnum()
    ]

    entries = []
    for r in equities:
        try:
            price = float((r.get("Price") or "0").replace(",", ""))
        except ValueError:
            price = 0.0
        if price < min_price:
            continue
        entries.append({
            "ticker": r["Ticker"].strip(),
            "name": r.get("Name", "").strip(),
            "sector": r.get("Sector", "").strip(),
            "price": price,
        })

    entries.sort(key=lambda e: e["ticker"])
    return {
        "source": "iShares IJR (S&P SmallCap 600)",
        "fund_date": fund_date,
        "fetched_at": datetime.now().isoformat(),
        "min_price": min_price,
        "count": len(entries),
        "holdings": entries,
    }


def main():
    out = Path("research_data/ijr_holdings.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    data = fetch_ijr_holdings(min_price=5.0)
    out.write_text(json.dumps(data, indent=2))

    sectors: dict[str, int] = {}
    for e in data["holdings"]:
        sectors[e["sector"]] = sectors.get(e["sector"], 0) + 1
    log.info(
        "Saved %d holdings (fund date: %s) to %s",
        data["count"], data["fund_date"], out,
    )
    log.info("Sectors: %s", dict(sorted(sectors.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
