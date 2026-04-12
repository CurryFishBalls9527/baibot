"""Broad seed universe for survivorship-bias-reduced backtesting.

Instead of hand-picking 88 known winners, we use index constituents +
liquid stocks as a broad starting point, then let the Minervini screener
dynamically select which stocks qualify at each point in time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# S&P 500 constituents (as of April 2026, ~500 tickers).
# Some survivorship bias remains (current membership), but dramatically less
# severe than hand-picking 88 growth leaders that already 10x'd.
SP500_CONSTITUENTS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE",
    "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALK",
    "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN",
    "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO",
    "ATVI", "AVB", "AVGO", "AVY", "AWK", "AXP", "AZO", "BA", "BAC", "BAX",
    "BBWI", "BBY", "BDX", "BEN", "BF-B", "BIO", "BK", "BKNG", "BKR", "BLK",
    "BMY", "BR", "BRK-B", "BRO", "BSX", "BWA", "BXP", "C", "CAG", "CAH",
    "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDAY", "CDNS", "CDW",
    "CE", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF", "CL",
    "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF",
    "COO", "COP", "COST", "CPB", "CPRT", "CPT", "CRL", "CRM", "CSCO", "CSGP",
    "CSX", "CTAS", "CTLT", "CTRA", "CTSH", "CTVA", "CVS", "CVX", "CZR", "D",
    "DAL", "DD", "DE", "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DISH",
    "DLR", "DLTR", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN",
    "DXC", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EIX", "EL", "EMN",
    "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS", "ETN",
    "ETR", "ETSY", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG",
    "FAST", "FBHS", "FCX", "FDS", "FDX", "FE", "FFIV", "FIS", "FISV", "FITB",
    "FLT", "FMC", "FOX", "FOXA", "FRC", "FRT", "FTNT", "FTV", "GD", "GE",
    "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN",
    "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", "HD", "HOLX", "HON",
    "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUM", "HWM", "IBM", "ICE",
    "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", "INTU", "INVH", "IP", "IPG",
    "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JCI",
    "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC", "KIM",
    "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "L", "LDOS", "LEN", "LH",
    "LHX", "LIN", "LKQ", "LLY", "LMT", "LNC", "LNT", "LOW", "LRCX", "LUMN",
    "LUV", "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD",
    "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK", "MKC",
    "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR",
    "MRK", "MRNA", "MRO", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH", "MTD",
    "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC",
    "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWL", "NWS",
    "NWSA", "NXPI", "O", "ODFL", "OGN", "OKE", "OMC", "ON", "ORCL", "ORLY",
    "OTIS", "OXY", "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEAK", "PEG", "PEP",
    "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PKI", "PLD", "PM",
    "PNC", "PNR", "PNW", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC",
    "PVH", "PWR", "PXD", "PYPL", "QCOM", "QRVO", "RCL", "RE", "REG", "REGN",
    "RF", "RHI", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG",
    "RTX", "SBAC", "SBNY", "SBUX", "SCHW", "SEE", "SHW", "SIVB", "SJM", "SLB",
    "SNA", "SNPS", "SO", "SPG", "SPGI", "SRE", "STE", "STT", "STX", "STZ",
    "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH",
    "TEL", "TER", "TFC", "TFX", "TGT", "TMO", "TMUS", "TPR", "TRGP", "TRMB",
    "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT", "TYL",
    "UAL", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V",
    "VFC", "VICI", "VLO", "VMC", "VNO", "VRSK", "VRSN", "VRTX", "VTR", "VTRS",
    "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WHR",
    "WM", "WMB", "WMT", "WRB", "WRK", "WST", "WTW", "WY", "WYNN", "XEL",
    "XOM", "XRAY", "XYL", "YUM", "ZBH", "ZBRA", "ZION", "ZTS",
]

# Nasdaq 100 constituents (adds growth/tech names not in S&P 500)
NASDAQ100_EXTRA = [
    "ABNB", "APP", "ARM", "AXON", "AZN", "BIIB", "BKNG", "CCEP", "COIN",
    "CPNG", "CRWD", "CSGP", "DASH", "DDOG", "DLTR", "DXCM", "EA", "ENPH",
    "FANG", "FAST", "FTNT", "GEHC", "GFS", "GILD", "HOOD", "IDXX", "ILMN",
    "KDP", "KHC", "LULU", "MCHP", "MDLZ", "MELI", "MNST", "MRNA", "MRVL",
    "NXPI", "ODFL", "ON", "PANW", "PAYX", "PCAR", "PLTR", "PYPL", "REGN",
    "RIVN", "ROST", "SIRI", "SMCI", "SNPS", "SPLK", "TEAM", "TMUS", "TTD",
    "VRSK", "WDAY", "ZS",
]

# Additional liquid growth stocks to broaden the universe
LIQUID_GROWTH_EXTRA = [
    "AFRM", "AI", "BILL", "CAVA", "CELH", "CFLT", "CRSP", "CYBR", "DOCS",
    "DOCU", "DUOL", "ELF", "ESTC", "FICO", "FIVE", "GDDY", "GLBE", "HIMS",
    "HUBS", "IOT", "LI", "MARA", "MDB", "MNDY", "NET", "NU", "NTNX",
    "OKTA", "ONON", "PATH", "PINS", "PODD", "PSTG", "RBLX", "RDDT", "RKLB",
    "ROKU", "SE", "SHOP", "SNOW", "SOFI", "SQ", "SYM", "TEM", "TOST",
    "TWLO", "U", "UBER", "VRT", "WIX", "XYZ", "YOU", "ZI",
]


@dataclass
class SeedUniverseConfig:
    include_sp500: bool = True
    include_nasdaq100: bool = True
    include_growth_extra: bool = True
    include_ijr: bool = False  # S&P SmallCap 600 — run scripts/fetch_ijr_holdings.py first
    ijr_holdings_path: str = "research_data/ijr_holdings.json"
    min_price: float = 10.0
    max_symbols: int = 1500
    cache_path: str = "research_data/seed_universe.json"


def _load_ijr_tickers(path: str, min_price: float) -> list[str]:
    p = Path(path)
    if not p.exists():
        logger.warning(
            "IJR holdings file %s missing; skipping. Run scripts/fetch_ijr_holdings.py first.",
            path,
        )
        return []
    data = json.loads(p.read_text())
    holdings = data.get("holdings", [])
    return [
        h["ticker"] for h in holdings
        if h.get("ticker") and float(h.get("price", 0) or 0) >= min_price
    ]


def build_seed_universe(config: SeedUniverseConfig | None = None) -> list[str]:
    """Build a broad, liquid universe for walk-forward backtesting.

    Returns deduplicated, sorted list of symbols.
    """
    config = config or SeedUniverseConfig()
    symbols: set[str] = set()

    if config.include_sp500:
        symbols.update(SP500_CONSTITUENTS)
    if config.include_nasdaq100:
        symbols.update(NASDAQ100_EXTRA)
    if config.include_growth_extra:
        symbols.update(LIQUID_GROWTH_EXTRA)
    if config.include_ijr:
        ijr = _load_ijr_tickers(config.ijr_holdings_path, config.min_price)
        symbols.update(ijr)
        logger.info(f"Added {len(ijr)} IJR (S&P SmallCap 600) tickers")

    result = sorted(symbols)[: config.max_symbols]
    logger.info(f"Built seed universe: {len(result)} symbols")
    return result


def save_seed_universe(
    symbols: list[str],
    path: str = "research_data/seed_universe.json",
    metadata: Optional[dict] = None,
) -> None:
    """Save seed universe to JSON with metadata for reproducibility."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "created_at": datetime.now().isoformat(),
        "count": len(symbols),
        "symbols": symbols,
        **(metadata or {}),
    }
    p.write_text(json.dumps(data, indent=2))
    logger.info(f"Saved {len(symbols)} symbols to {path}")


def load_seed_universe(path: str = "research_data/seed_universe.json") -> list[str]:
    """Load a previously saved seed universe."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Seed universe not found: {path}")
    data = json.loads(p.read_text())
    symbols = data["symbols"]
    logger.info(f"Loaded {len(symbols)} symbols from {path} (created {data.get('created_at', 'unknown')})")
    return symbols
