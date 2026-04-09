"""Large-cap stock screener — filter by market cap, volume, sector."""

import logging
from typing import List, Dict, Optional

import yfinance as yf

logger = logging.getLogger(__name__)

LARGE_CAP_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B",
    "AVGO", "JPM", "LLY", "V", "UNH", "MA", "XOM", "COST", "HD",
    "PG", "JNJ", "ABBV", "WMT", "NFLX", "CRM", "BAC", "KO", "ORCL",
    "CVX", "MRK", "AMD", "PEP", "TMO", "ADBE", "LIN", "ACN", "CSCO",
    "MCD", "ABT", "WFC", "DHR", "PM", "NOW", "INTC", "QCOM", "IBM",
    "INTU", "TXN", "AMGN", "GE", "CAT", "ISRG", "AMAT", "GS", "MS",
    "BLK", "PFE", "UNP", "SYK", "LOW", "BKNG", "VRTX", "ADP",
    "MDLZ", "T", "CB", "LRCX", "PANW", "SCHW", "MMC", "DE",
]


class LargeCapScreener:
    """Screen for large-cap stocks suitable for trading."""

    def __init__(self, min_market_cap_b: float = 30.0, min_avg_volume: int = 1_000_000):
        self.min_market_cap = min_market_cap_b * 1e9
        self.min_avg_volume = min_avg_volume

    def screen(self, candidates: List[str] = None) -> List[Dict]:
        """Screen stocks and return qualifying ones with metadata.

        Returns list of dicts sorted by market cap descending.
        """
        symbols = candidates or LARGE_CAP_UNIVERSE
        results = []

        logger.info(f"Screening {len(symbols)} candidates (min cap ${self.min_market_cap/1e9:.0f}B)...")

        for symbol in symbols:
            try:
                info = yf.Ticker(symbol).info
                mcap = info.get("marketCap", 0)
                avg_vol = info.get("averageVolume", 0)

                if mcap and mcap >= self.min_market_cap and avg_vol >= self.min_avg_volume:
                    results.append({
                        "symbol": symbol,
                        "name": info.get("shortName", symbol),
                        "market_cap": mcap,
                        "market_cap_b": round(mcap / 1e9, 1),
                        "avg_volume": avg_vol,
                        "sector": info.get("sector", "Unknown"),
                        "price": info.get("currentPrice") or info.get("previousClose", 0),
                        "pe_ratio": info.get("trailingPE"),
                        "beta": info.get("beta"),
                    })
            except Exception as e:
                logger.debug(f"Skipping {symbol}: {e}")
                continue

        results.sort(key=lambda x: x["market_cap"], reverse=True)
        logger.info(f"Screener found {len(results)} stocks with market cap >= ${self.min_market_cap/1e9:.0f}B")
        return results

    def get_top_n(self, n: int = 10, candidates: List[str] = None) -> List[str]:
        """Return top N symbols by market cap."""
        results = self.screen(candidates)
        return [r["symbol"] for r in results[:n]]
