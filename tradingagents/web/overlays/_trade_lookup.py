"""Trade-cycle lookup shared by every per-strategy overlay extractor.

Every overlay's contract is "given a trade_id, build a ChartPayload for
that trade's cycle". Until now each extractor centered the chart and
pulled reasoning from the row whose id was clicked — fine for BUY rows,
broken for SELL rows because:

  1. SELL signals don't carry the strategy's entry metadata
     (signal_metadata, full_analysis), so the criteria panel rendered
     all-red checks even though the entry decision was sound.
  2. The chart pivot was the SELL's timestamp; the entry fill, which
     happened earlier, fell outside the bar window.
  3. ``_exit_fills`` looked for SELLs *after* the clicked row's
     timestamp — which is empty when the click WAS the SELL.

Fix: clicking any trade in a cycle redirects to the BUY (entry). The
chart pivots on the entry, the reasoning panel reads the entry signal,
and we render all SELLs that close the cycle as exit markers.
"""

from __future__ import annotations

from typing import List, Optional

from tradingagents.storage.database import TradingDatabase


def find_entry_id(db: TradingDatabase, trade_id: int) -> Optional[int]:
    """Return the trade_id of the BUY that opened the cycle this trade is part of.

    If the input trade_id is itself a BUY, returns it unchanged.
    If it's a SELL, returns the most recent BUY of the same symbol with
    timestamp strictly before the SELL's. Returns None if the trade
    can't be located or no entry exists.

    Timestamp comparison is lexicographic — both formats present in the
    DB ('YYYY-MM-DD HH:MM:SS' and 'YYYY-MM-DDTHH:MM:SS+00:00') compare
    correctly under string ordering.
    """
    row = db.conn.execute(
        "SELECT id, symbol, side, timestamp FROM trades WHERE id = ?",
        [trade_id],
    ).fetchone()
    if row is None:
        return None
    side = (row["side"] or "").lower()
    if side == "buy":
        return int(row["id"])
    entry = db.conn.execute(
        "SELECT id FROM trades "
        "WHERE symbol = ? AND LOWER(side) = 'buy' AND timestamp < ? "
        "ORDER BY timestamp DESC LIMIT 1",
        [row["symbol"], row["timestamp"]],
    ).fetchone()
    return int(entry["id"]) if entry else None


def list_cycle_exits(
    db: TradingDatabase, symbol: str, entry_ts: str,
) -> List[dict]:
    """All SELL fills that belong to the cycle started at ``entry_ts``.

    Walks chronologically forward from ``entry_ts``; collects SELLs
    until the next BUY of the same symbol (which would be the start of
    the next cycle) or the end of the table. Returns full row dicts for
    each SELL — the caller picks out timestamp/price/reasoning/etc.
    """
    rows = db.conn.execute(
        "SELECT id, timestamp, side, filled_qty, filled_price, "
        "       reasoning, status "
        "FROM trades WHERE symbol = ? AND timestamp > ? "
        "ORDER BY timestamp ASC",
        [symbol, entry_ts],
    ).fetchall()
    out: List[dict] = []
    for r in rows:
        d = dict(r)
        if (d.get("side") or "").lower() == "buy":
            break  # next cycle started — stop
        out.append(d)
    return out


def trade_summary(
    entry_row: dict, exits: List[dict],
) -> dict:
    """Compose closed-trade metrics for the reasoning panel.

    Returns a dict suitable for inclusion in the metrics list. None values
    are included so the renderer can show a dash; empty exits → all None.
    """
    if not exits or not entry_row.get("filled_price"):
        return {"is_closed": False, "exit_price": None, "return_pct": None,
                "exit_reason": None, "held_str": None}
    last_exit = exits[-1]
    entry_price = float(entry_row["filled_price"])
    exit_price = float(last_exit.get("filled_price") or 0)
    ret = (exit_price / entry_price - 1.0) if entry_price > 0 else None
    # Held duration — ISO/space-mixed timestamps; parse leniently.
    import pandas as pd
    try:
        t0 = pd.to_datetime(entry_row["trade_ts"]
                            if "trade_ts" in entry_row else entry_row["timestamp"],
                            format="mixed", utc=True)
        t1 = pd.to_datetime(last_exit["timestamp"], format="mixed", utc=True)
        secs = (t1 - t0).total_seconds()
        if secs < 86400:
            held_str = f"{secs/3600:.1f}h"
        else:
            held_str = f"{secs/86400:.1f}d"
    except Exception:
        held_str = None
    return {
        "is_closed": True,
        "exit_price": exit_price,
        "return_pct": ret,
        "exit_reason": (last_exit.get("reasoning") or "").split("\n")[0][:120] or None,
        "held_str": held_str,
    }
