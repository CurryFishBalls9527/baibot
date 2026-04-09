"""Curated research universes for Minervini-style trend studies."""

from __future__ import annotations

from tradingagents.backtesting.screener import LARGE_CAP_UNIVERSE


GROWTH_LEADER_UNIVERSE = [
    "AAPL", "AMD", "AMZN", "ANET", "APP", "ARM", "AVGO", "AXON",
    "BKNG", "CAVA", "CELH", "CRWD", "DASH", "DDOG", "DECK", "DOCU",
    "DUOL", "FTNT", "GE", "GOOGL", "HUBS", "IOT", "ISRG", "LLY",
    "MELI", "META", "MNDY", "MDB", "MSFT", "NFLX", "NOW", "NVDA",
    "OKTA", "ORCL", "PANW", "PATH", "PLTR", "PODD", "RBLX", "RKLB",
    "SHOP", "SNOW", "SOFI", "TEAM", "TSLA", "TTD", "TXN", "UBER",
    "VRT", "VRTX", "ZS", "ADSK", "ADBE", "AMAT", "APPF", "ASML",
    "CDNS", "COIN", "CRM", "CYBR", "DE", "ELF", "ETSY", "FICO",
    "GDDY", "HIMS", "HOOD", "INTU", "KLAC", "LRCX", "MRVL", "NET",
    "NTNX", "NU", "ONON", "PINS", "PSTG", "QCOM", "RDDT", "ROKU",
    "SE", "SMCI", "SYM", "TEM", "TOST", "TTWO", "WIX", "XYZ",
]

MINERVINI_COMBINED_UNIVERSE = sorted(set(LARGE_CAP_UNIVERSE + GROWTH_LEADER_UNIVERSE))


def is_dynamic_universe(name: str) -> bool:
    key = (name or "").strip().lower()
    return key in {"broad", "broad_market", "broad-market", "market"}


def resolve_universe(name: str) -> list[str]:
    key = (name or "combined").strip().lower()
    if key in {"large", "large_cap", "large-cap"}:
        return LARGE_CAP_UNIVERSE
    if key in {"growth", "growth_leaders", "growth-leaders"}:
        return GROWTH_LEADER_UNIVERSE
    if key in {"combined", "all", "minervini"}:
        return MINERVINI_COMBINED_UNIVERSE
    raise ValueError(f"Unknown universe '{name}'")
