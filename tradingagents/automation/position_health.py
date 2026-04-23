"""Snapshot + LLM commentary for open (still-held) positions.

The daily trade review covers CLOSED trades. Open positions — the MRVLs
and DELLs held 5+ days — have no narrative between entry and exit. This
module closes that gap: one LLM call per held position per day producing
a `### Health: HEALTHY | WATCH | WARNING` line plus 2-3 sentences of
concrete commentary.

Kill-switched, read-only against broker + DB, and wrapped in try/except
at every boundary so it can never block an exit.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def collect_position_snapshot(
    *,
    db: Any,
    symbol: str,
    position: Any,
    variant: str,
    features_fn: Optional[Callable[[str], Dict]] = None,
) -> Dict:
    """Compute current health metrics for one held position.

    Combines `position_states` entry context (persisted at BUY time)
    with live broker fields (current_price) and optional per-symbol
    features (SMA-50, RS percentile from caller's preflight cache).
    Every field is optional — None-valued keys are preserved so the
    downstream prompt can flag "n/a".
    """
    snap: Dict[str, Any] = {"symbol": symbol, "variant": variant}

    try:
        pos_state = db.get_position_state(symbol) or {}
    except Exception as e:
        logger.warning("position_health[%s]: get_position_state failed: %s", symbol, e)
        pos_state = {}

    entry_price = pos_state.get("entry_price")
    entry_date = pos_state.get("entry_date")
    highest_close = pos_state.get("highest_close")
    current_stop = pos_state.get("current_stop")
    current_price = getattr(position, "current_price", None)

    snap["entry_price"] = entry_price
    snap["entry_date"] = entry_date
    snap["current_price"] = current_price
    snap["highest_close_since_entry"] = highest_close
    snap["current_stop"] = current_stop
    snap["base_pattern"] = pos_state.get("base_pattern")
    snap["regime_at_entry"] = pos_state.get("regime_at_entry")
    snap["rs_at_entry"] = pos_state.get("rs_at_entry")
    snap["stage_at_entry"] = pos_state.get("stage_at_entry")

    # Hold days — only meaningful when entry_date is populated.
    if entry_date:
        try:
            from dateutil.parser import parse as parse_date
            snap["hold_days"] = max(
                0, (date.today() - parse_date(entry_date).date()).days
            )
        except Exception:
            snap["hold_days"] = None
    else:
        snap["hold_days"] = None

    # Unrealized P&L % vs entry.
    if entry_price and current_price and entry_price > 0:
        snap["unrealized_pct"] = round((current_price - entry_price) / entry_price, 4)
    else:
        snap["unrealized_pct"] = None

    # MFE-so-far: peak close since entry, relative to entry. Not the intraday
    # high — just the highest CLOSE the exit manager has observed.
    if entry_price and highest_close and entry_price > 0:
        snap["mfe_so_far_pct"] = round((highest_close - entry_price) / entry_price, 4)
    else:
        snap["mfe_so_far_pct"] = None

    # Distance to stop (positive = room to breathe, negative = stopped already).
    if current_stop and current_price and current_price > 0:
        snap["distance_to_stop_pct"] = round(
            (current_price - current_stop) / current_price, 4
        )
    else:
        snap["distance_to_stop_pct"] = None

    # Unrealized dollar P&L (per-share × qty).
    qty = float(getattr(position, "qty", 0) or 0)
    if entry_price and current_price and qty:
        snap["unrealized_pnl"] = round((current_price - entry_price) * qty, 2)
    else:
        snap["unrealized_pnl"] = None

    # Optional features (SMA-50, RS percentile now, ADX, etc). The caller
    # usually has an Orchestrator-bound `_get_latest_features` we can reuse.
    features: Dict = {}
    if features_fn is not None:
        try:
            features = features_fn(symbol) or {}
        except Exception as e:
            logger.warning(
                "position_health[%s]: features_fn failed: %s", symbol, e
            )
    snap["features"] = features

    # Derived: distance to 50-DMA now (negative = price below 50-DMA).
    sma50 = features.get("sma_50")
    if sma50 and current_price and current_price > 0:
        snap["distance_to_50dma_pct"] = round((current_price - sma50) / sma50, 4)
    else:
        snap["distance_to_50dma_pct"] = None

    # RS delta — did relative strength decay since entry?
    rs_now = features.get("rs_percentile")
    rs_entry = snap["rs_at_entry"]
    if rs_now is not None and rs_entry is not None:
        snap["rs_delta"] = round(float(rs_now) - float(rs_entry), 2)
    else:
        snap["rs_delta"] = None

    return snap


_HEALTH_PROMPT = """You are a concise position-health reviewer. A human
trader wants a fast read on whether to keep holding, tighten the stop,
or start watching closely. Do not be vague — cite numbers.

Position: {symbol} ({variant})
- Entry: ${entry_price} on {entry_date} · hold {hold_days} days
- Current: ${current_price} · unrealized {unrealized_pct} ({unrealized_pnl})
- Stop: ${current_stop} · distance_to_stop {distance_to_stop_pct}
- MFE so far: {mfe_so_far_pct} · highest close ${highest_close_since_entry}
- Setup at entry: base={base_pattern}, stage={stage_at_entry}, rs_entry={rs_at_entry}, regime={regime_at_entry}
- Now: sma_50=${sma50}, distance_to_50dma={distance_to_50dma_pct}, rs_now={rs_now}, rs_delta={rs_delta}, adx={adx}

Output EXACTLY these four lines, in order, nothing else:

### Health: <HEALTHY | WATCH | WARNING>
**Read:** one sentence on state (trend intact / stalling / breaking down).
**Watch:** one concrete trigger — a price level OR indicator threshold.
**Action:** one concrete action — hold / tighten stop / trim / exit — with a level.
"""


def render_health_prompt(snapshot: Dict) -> str:
    """Interpolate snapshot into the compact health prompt.

    Missing fields render as "n/a" so the LLM can note partial data
    rather than crash the template.
    """
    def _fmt(v, pct=False, money=False):
        if v is None:
            return "n/a"
        if pct and isinstance(v, (int, float)):
            return f"{v:+.2%}"
        if money and isinstance(v, (int, float)):
            return f"{v:+,.2f}"
        return str(v)

    feats = snapshot.get("features") or {}
    return _HEALTH_PROMPT.format(
        symbol=snapshot.get("symbol", "?"),
        variant=snapshot.get("variant", "?"),
        entry_price=_fmt(snapshot.get("entry_price")),
        entry_date=snapshot.get("entry_date") or "n/a",
        hold_days=snapshot.get("hold_days") if snapshot.get("hold_days") is not None else "n/a",
        current_price=_fmt(snapshot.get("current_price")),
        unrealized_pct=_fmt(snapshot.get("unrealized_pct"), pct=True),
        unrealized_pnl=_fmt(snapshot.get("unrealized_pnl"), money=True),
        current_stop=_fmt(snapshot.get("current_stop")),
        distance_to_stop_pct=_fmt(snapshot.get("distance_to_stop_pct"), pct=True),
        mfe_so_far_pct=_fmt(snapshot.get("mfe_so_far_pct"), pct=True),
        highest_close_since_entry=_fmt(snapshot.get("highest_close_since_entry")),
        base_pattern=snapshot.get("base_pattern") or "n/a",
        stage_at_entry=snapshot.get("stage_at_entry") if snapshot.get("stage_at_entry") is not None else "n/a",
        rs_at_entry=snapshot.get("rs_at_entry") if snapshot.get("rs_at_entry") is not None else "n/a",
        regime_at_entry=snapshot.get("regime_at_entry") or "n/a",
        sma50=_fmt(feats.get("sma_50")),
        distance_to_50dma_pct=_fmt(snapshot.get("distance_to_50dma_pct"), pct=True),
        rs_now=_fmt(feats.get("rs_percentile")),
        rs_delta=_fmt(snapshot.get("rs_delta")),
        adx=_fmt(feats.get("adx_14")),
    )


def compose_health_markdown(snapshot: Dict, llm_body: str) -> str:
    """Assemble the per-position _HELD.md content."""
    sym = snapshot.get("symbol", "?")
    variant = snapshot.get("variant", "?")
    hold_days = snapshot.get("hold_days")
    unrealized = snapshot.get("unrealized_pct")
    entry = snapshot.get("entry_price")
    current = snapshot.get("current_price")
    stop = snapshot.get("current_stop")
    dist_stop = snapshot.get("distance_to_stop_pct")
    mfe = snapshot.get("mfe_so_far_pct")

    def _pct(v):
        return f"{v:+.2%}" if isinstance(v, (int, float)) else "n/a"

    def _money(v):
        return f"${v:.2f}" if isinstance(v, (int, float)) else "n/a"

    header = f"# {sym} ({variant}) — held {hold_days or '?'} days · {_pct(unrealized)}\n\n"
    stats = (
        "## Snapshot\n"
        f"- Entry **{_money(entry)}** · now **{_money(current)}**\n"
        f"- Stop **{_money(stop)}** · distance-to-stop **{_pct(dist_stop)}**\n"
        f"- MFE so far **{_pct(mfe)}** · highest close **{_money(snapshot.get('highest_close_since_entry'))}**\n"
        f"- Setup: base={snapshot.get('base_pattern') or 'n/a'} · "
        f"regime_at_entry={snapshot.get('regime_at_entry') or 'n/a'}\n\n"
    )
    return header + stats + (llm_body or "") + "\n"
