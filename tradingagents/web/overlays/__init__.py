"""Per-strategy overlay extractors.

Each extractor turns a (variant, trade) into a ``ChartPayload`` —
bars + structured overlays + reasoning panel. The frontend renders
the same vocabulary regardless of strategy.

Add a strategy by registering a ``BUILDERS`` entry. No frontend changes.

Imports are lazy (inside ``build_for``) to avoid a circular dependency
through ``..bars`` which itself imports ``overlays.base``.
"""

from __future__ import annotations

from typing import Optional

from tradingagents.storage.database import TradingDatabase

from .base import ChartPayload


def build_for(
    strategy_type: str,
    db: TradingDatabase,
    trade_id: int,
    variant_config: dict,
) -> Optional[ChartPayload]:
    """Dispatch to the appropriate extractor. Returns None if unsupported.

    Postcondition: payload carries ``is_open: bool`` — True iff the
    trade has no SELL fill yet. Computed centrally so each extractor
    doesn't need to think about it.
    """
    payload: Optional[ChartPayload] = None
    if strategy_type in ("chan", "chan_v2"):
        from . import chan
        payload = chan.build_chart(db, trade_id, variant_config)
    elif strategy_type == "chan_daily":
        from . import chan_daily
        payload = chan_daily.build_chart(db, trade_id, variant_config)
    elif strategy_type == "intraday_mechanical":
        from . import intraday_mechanical
        payload = intraday_mechanical.build_chart(db, trade_id, variant_config)
    elif strategy_type in ("mechanical", "llm"):
        from . import mechanical
        payload = mechanical.build_chart(db, trade_id, variant_config)
    elif strategy_type in ("pead", "pead_llm"):
        from . import pead
        payload = pead.build_chart(db, trade_id, variant_config)

    if payload is not None:
        # Open iff there's a BUY fill but no SELL fill. A trade with no
        # entry fill at all (e.g. rejected order) reads as not-open.
        sides = {(f.get("side") or "").lower() for f in payload.get("fills", [])}
        payload["is_open"] = ("buy" in sides) and ("sell" not in sides)
    return payload


__all__ = ["build_for", "ChartPayload"]
