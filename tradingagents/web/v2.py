"""V2 dashboard API — endpoints for the redesigned UI in static/v2/.

Read-only across the per-variant SQLite DBs that ``_discover_variants``
already finds. Cursor pagination is by trade.id descending: callers pass
``?before=<id>&limit=30`` and the response includes ``next_before`` (or
null when exhausted).

The variant_name shape returned from ``_discover_variants`` is the
storage-layer name (e.g. ``chan_v2``, ``mechanical``). Display names
shown in the UI live in ``DISPLAY_MAP`` below — the mocks reference
labels like ``CHAN-V2`` / ``MINERVINI`` that don't map 1:1 to the
storage names, so this module owns the projection.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import statistics
from collections import defaultdict, deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException

from tradingagents.storage.database import TradingDatabase

_log = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v2")


# ── Display map ─────────────────────────────────────────────────────
# Storage-name → (display_name, color, sleeve_label).
# Anything not in this map falls back to upper-cased storage name +
# a neutral grey color so a new variant doesn't crash the UI.

# Layer ids match the SVG ``data-layer`` attributes in floor.html.
# Anything not listed here renders with no chan/minervini overlays —
# safer than falling back to a chan default and showing zhongshu boxes
# on a Minervini chart.
_LAYERS_CHAN     = ["chan-zs", "chan-seg", "chan-pivots"]
_LAYERS_MINERVINI = ["min-stage", "min-rs", "min-pivot", "ma"]
_LAYERS_BREAKOUT  = ["bk-cup", "bk-pivot", "bk-vol", "ma"]
_LAYERS_INTRADAY  = ["vwap"]
_LAYERS_NONE      = []  # generic event-driven; no TA overlays apply

DISPLAY_MAP: Dict[str, Dict] = {
    "chan_v2":             {"display": "CHAN-V2",       "color": "#f5a524", "sleeve": "swing",    "layers": _LAYERS_CHAN},
    "chan":                {"display": "CHAN",          "color": "#f5a524", "sleeve": "swing",    "layers": _LAYERS_CHAN},
    "chan_daily":          {"display": "CHAN-DAILY",    "color": "#d68a1c", "sleeve": "swing",    "layers": _LAYERS_CHAN},
    "mechanical":          {"display": "MINERVINI",     "color": "#c4a8ff", "sleeve": "swing",    "layers": _LAYERS_MINERVINI},
    "mechanical_v2":       {"display": "MINERVINI-V2",  "color": "#a888e8", "sleeve": "swing",    "layers": _LAYERS_MINERVINI},
    # llm = LLM-judged swing on the Minervini screener (same TA primitives)
    "llm":                 {"display": "LLM",           "color": "#9bb4ff", "sleeve": "swing",    "layers": _LAYERS_MINERVINI},
    "intraday_mechanical": {"display": "INTRADAY",      "color": "#5ec5b7", "sleeve": "intraday", "layers": _LAYERS_INTRADAY},
    # PEAD is event-driven (earnings); no TA overlays — show price only
    "pead":                {"display": "PEAD",          "color": "#7dd87f", "sleeve": "event",    "layers": _LAYERS_NONE},
    "pead_llm":            {"display": "PEAD-LLM",      "color": "#5fb866", "sleeve": "event",    "layers": _LAYERS_NONE},
}


def display_for(storage_name: str) -> Dict:
    return DISPLAY_MAP.get(storage_name, {
        "display": storage_name.upper(),
        "color":   "#7a7a82",
        "sleeve":  "swing",
        "layers":  _LAYERS_NONE,
    })


# ── Helpers ─────────────────────────────────────────────────────────


def _discover():
    """Lazy import — avoid circular import with app.py at module load."""
    from tradingagents.web.app import _discover_variants
    return _discover_variants()


def _variant(name: str) -> Optional[Dict]:
    for v in _discover():
        if v["name"] == name:
            return v
    return None


def _db(variant_name: str) -> TradingDatabase:
    v = _variant(variant_name)
    if v is None:
        raise HTTPException(404, f"variant {variant_name!r} not found")
    return TradingDatabase(v["db_path"])


def _to_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


# ── Closed round-trip computation ───────────────────────────────────
#
# A "closed round-trip" pairs a BUY with a later SELL of the same symbol.
# Because the DB doesn't store this directly, we walk the trades table
# in chronological order per symbol and emit a row each time a SELL
# zeros (or partially zeros) the running quantity. We keep this simple
# — full FIFO accounting lives in the overlays/_trade_lookup pipeline,
# which is overkill for the versus/review tables.


def _round_trips_for(db: TradingDatabase) -> List[Dict]:
    """All closed BUY→SELL round-trips for a variant, oldest-first.

    Each trip carries: entry_id, exit_id, symbol, entry_ts, exit_ts,
    qty, entry_px, exit_px, pnl_pct, hold_days, exit_reason.
    """
    rows = db.conn.execute(
        "SELECT id, timestamp, symbol, side, filled_qty, filled_price, "
        "       reasoning, status "
        "FROM trades WHERE LOWER(side) IN ('buy','sell') "
        "  AND filled_qty IS NOT NULL AND filled_qty > 0 "
        "  AND filled_price IS NOT NULL "
        "ORDER BY timestamp ASC, id ASC",
    ).fetchall()

    open_lots: Dict[str, List[Dict]] = defaultdict(list)
    trips: List[Dict] = []
    for r in rows:
        sym = r["symbol"]
        side = (r["side"] or "").lower()
        qty = float(r["filled_qty"] or 0)
        px = float(r["filled_price"] or 0)
        if side == "buy":
            open_lots[sym].append({
                "id":    int(r["id"]),
                "ts":    r["timestamp"],
                "qty":   qty,
                "px":    px,
                "reasoning": r["reasoning"],
            })
            continue
        # SELL — match against open lots FIFO until qty exhausted.
        remaining = qty
        while remaining > 0 and open_lots[sym]:
            lot = open_lots[sym][0]
            take = min(lot["qty"], remaining)
            entry_px = lot["px"]
            pnl_pct = ((px - entry_px) / entry_px * 100.0) if entry_px else 0.0
            try:
                entry_dt = datetime.fromisoformat(lot["ts"].replace(" ", "T").replace("Z", "+00:00"))
                exit_dt  = datetime.fromisoformat(r["timestamp"].replace(" ", "T").replace("Z", "+00:00"))
                hold_days = max(0.0, (exit_dt - entry_dt).total_seconds() / 86400.0)
            except Exception:
                hold_days = 0.0
            trips.append({
                "entry_id":    lot["id"],
                "exit_id":     int(r["id"]),
                "symbol":      sym,
                "entry_ts":    lot["ts"],
                "exit_ts":     r["timestamp"],
                "qty":         take,
                "entry_px":    entry_px,
                "exit_px":     px,
                "pnl_pct":     pnl_pct,
                "hold_days":   hold_days,
                "exit_reason": (r["reasoning"] or "").split("\n")[0][:120],
                "side":        "buy",  # entry side
            })
            lot["qty"] -= take
            remaining  -= take
            if lot["qty"] <= 1e-9:
                open_lots[sym].pop(0)
    return trips


# ── State ───────────────────────────────────────────────────────────
# /api/v2/state — single hydration call for floor.html. Aggregates
# pulse + per-variant lane snapshots from the SQLite DBs only. Live
# Alpaca state (cash, exposure %) is NOT included here — the existing
# /api/today/{variant} endpoint covers that and is read per-page.


@router.get("/state")
def state() -> dict:
    variants_out = []
    pulse_realized = 0.0
    pulse_open = 0
    pulse_today = 0
    today = date.today().isoformat()

    for v in _discover():
        vname = v["name"]
        vdisp = display_for(vname)
        try:
            db = TradingDatabase(v["db_path"])
        except Exception:
            continue

        # Sparkline: last 30 daily snapshots, percent off start.
        snaps = list(reversed(db.get_snapshots(days=30) or []))
        spark: List[float] = []
        if snaps:
            base = float(snaps[0]["equity"]) or 1.0
            spark = [round((float(s["equity"]) - base) / base * 100.0, 4) for s in snaps]

        # Day P&L from latest snapshot. daily_pl_pct is stored as a
        # decimal fraction; the UI shows percent, so multiply by 100.
        latest = snaps[-1] if snaps else None
        day_pnl_pct = (float(latest["daily_pl_pct"]) * 100.0) if latest and latest.get("daily_pl_pct") is not None else None
        day_pnl_usd = float(latest["daily_pl"]) if latest and latest.get("daily_pl") is not None else None

        # Open positions from position_states.
        try:
            open_rows = db.conn.execute(
                "SELECT symbol, entry_price, entry_date, current_stop, bars_held "
                "FROM position_states ORDER BY entry_date DESC"
            ).fetchall()
        except Exception:
            open_rows = []
        open_positions = [dict(r) for r in open_rows]

        # Today's fills (any side).
        try:
            today_fills = db.conn.execute(
                "SELECT id, timestamp, symbol, side, filled_qty, filled_price, "
                "       reasoning, status, signal_id "
                "FROM trades WHERE substr(timestamp,1,10) = ? "
                "  AND filled_qty IS NOT NULL AND filled_qty > 0 "
                "ORDER BY timestamp DESC",
                [today],
            ).fetchall()
            today_fills = [dict(r) for r in today_fills]
        except Exception:
            today_fills = []

        # Build ticket cards: open positions first, then closed-today fills.
        tickets = []
        seen_symbols = set()
        for p in open_positions:
            sym = p["symbol"]
            seen_symbols.add(sym)
            tickets.append({
                "kind":    "open",
                "symbol":  sym,
                "side":    "buy",
                "qty":     None,  # qty is in trades, not position_states
                "px":      float(p["entry_price"]),
                "stop":    float(p["current_stop"]) if p["current_stop"] else None,
                "ts":      p["entry_date"],
                "state":   "live",
            })
        for f in today_fills:
            sym = f["symbol"]
            tickets.append({
                "kind":    "fill",
                "trade_id": int(f["id"]),
                "symbol":  sym,
                "side":    (f["side"] or "").lower(),
                "qty":     float(f["filled_qty"]),
                "px":      float(f["filled_price"]),
                "ts":      f["timestamp"],
                "state":   "closed" if (f["side"] or "").lower() == "sell" else "live",
                "reasoning": (f["reasoning"] or "").split("\n")[0][:160],
            })

        variants_out.append({
            "name":         vname,
            "display":      vdisp["display"],
            "color":        vdisp["color"],
            "sleeve":       vdisp["sleeve"],
            "layers":       vdisp.get("layers", []),
            "strategy_type": v["strategy_type"],
            "day_pnl_pct":  day_pnl_pct,
            "day_pnl_usd":  day_pnl_usd,
            "open_count":   len(open_positions),
            "today_count":  len(today_fills),
            "spark":        spark,
            "tickets":      tickets,
        })

        if day_pnl_usd is not None:
            pulse_realized += day_pnl_usd
        pulse_open  += len(open_positions)
        pulse_today += len(today_fills)

    return {
        "asof":     _to_iso(datetime.now(timezone.utc)),
        "variants": variants_out,
        "pulse": {
            "day_pnl_usd_realized": round(pulse_realized, 2),
            "open_total":           pulse_open,
            "today_fills_total":    pulse_today,
        },
    }


# ── Fills (cursor-paginated) ────────────────────────────────────────


@router.get("/fills")
def fills(
    variant: Optional[str] = None,
    day: Optional[str] = None,
    before: Optional[int] = None,
    limit: int = 30,
) -> dict:
    """Cursor-paginated fills.

    - ``variant``: storage name, or omitted to fan out across all variants
    - ``day``: ``YYYY-MM-DD`` to filter to a single day
    - ``before``: trade.id strict-less-than for cursor pagination
    - ``limit``: capped at 200
    """
    limit = max(1, min(int(limit), 200))
    variants = [_variant(variant)] if variant else _discover()
    if any(v is None for v in variants):
        raise HTTPException(404, f"variant {variant!r} not found")

    out: List[Dict] = []
    for v in variants:
        try:
            db = TradingDatabase(v["db_path"])
        except Exception:
            continue
        sql = [
            "SELECT id, timestamp, symbol, side, filled_qty, filled_price, "
            "       reasoning, status, signal_id "
            "FROM trades WHERE filled_qty IS NOT NULL AND filled_qty > 0 ",
        ]
        params: List = []
        if day:
            sql.append(" AND substr(timestamp,1,10) = ? ")
            params.append(day)
        if before is not None:
            sql.append(" AND id < ? ")
            params.append(int(before))
        sql.append(" ORDER BY id DESC LIMIT ? ")
        params.append(limit)
        rows = db.conn.execute("".join(sql), params).fetchall()

        # Build a per-symbol BUY-ts → entry_px map so we can attach
        # pnl_pct to SELL fills without scanning the full table per row.
        entry_map: Dict[Tuple[str, str], float] = {}
        for r in rows:
            if (r["side"] or "").lower() == "sell":
                sym = r["symbol"]
                entry = db.conn.execute(
                    "SELECT filled_price, timestamp FROM trades "
                    "WHERE symbol = ? AND LOWER(side)='buy' AND timestamp < ? "
                    "ORDER BY timestamp DESC LIMIT 1",
                    [sym, r["timestamp"]],
                ).fetchone()
                if entry and entry["filled_price"]:
                    entry_map[(sym, r["timestamp"])] = float(entry["filled_price"])

        vdisp = display_for(v["name"])
        for r in rows:
            d = dict(r)
            d["variant"] = v["name"]
            d["variant_display"] = vdisp["display"]
            d["color"] = vdisp["color"]
            side = (d["side"] or "").lower()
            entry_px = entry_map.get((d["symbol"], d["timestamp"]))
            if side == "sell" and entry_px and d["filled_price"]:
                d["pnl_pct"] = round((float(d["filled_price"]) - entry_px) / entry_px * 100.0, 4)
                d["entry_px"] = entry_px
            else:
                d["pnl_pct"] = None
                d["entry_px"] = None
            d["why"] = (d.pop("reasoning", "") or "").split("\n")[0][:160]
            out.append(d)

    # Sort merged stream desc by id, take top N (cursor stays scoped per
    # variant when caller passes ``variant=``; multi-variant calls don't
    # paginate cleanly because ids aren't comparable across DBs).
    out.sort(key=lambda r: r["id"], reverse=True)
    if variant:
        out = out[:limit]
    next_before = out[-1]["id"] if out and len(out) >= limit else None
    return {"fills": out, "next_before": next_before}


# ── Daily equity / calendar ─────────────────────────────────────────


@router.get("/equity/daily")
def equity_daily(variant: str, range: str = "30d") -> dict:
    """Daily snapshots for a variant; ``range`` ∈ {5d, 20d, 30d, 90d, mtd, ytd}."""
    db = _db(variant)
    days = {"5d": 5, "20d": 20, "30d": 30, "90d": 90, "180d": 180}.get(range)
    if days is None and range not in ("mtd", "ytd"):
        raise HTTPException(400, f"unknown range {range!r}")

    if range == "mtd":
        first = date.today().replace(day=1).isoformat()
        rows = db.conn.execute(
            "SELECT date, equity, daily_pl, daily_pl_pct FROM daily_snapshots "
            "WHERE date >= ? ORDER BY date ASC",
            [first],
        ).fetchall()
    elif range == "ytd":
        first = date.today().replace(month=1, day=1).isoformat()
        rows = db.conn.execute(
            "SELECT date, equity, daily_pl, daily_pl_pct FROM daily_snapshots "
            "WHERE date >= ? ORDER BY date ASC",
            [first],
        ).fetchall()
    else:
        snaps = db.get_snapshots(days=days) or []
        rows = list(reversed(snaps))

    points = [dict(r) for r in rows]
    base = float(points[0]["equity"]) if points else None
    for p in points:
        p["cum_pct"] = ((float(p["equity"]) - base) / base * 100.0) if base else 0.0
        # daily_pl_pct is stored as decimal; surface percent for the UI.
        if p.get("daily_pl_pct") is not None:
            p["daily_pl_pct"] = float(p["daily_pl_pct"]) * 100.0
    return {"variant": variant, "range": range, "points": points}


@router.get("/calendar")
def calendar(variant: str, month: str) -> dict:
    """Per-day snapshot rows for a YYYY-MM grid."""
    try:
        y, m = map(int, month.split("-"))
        first = date(y, m, 1)
        next_first = date(y + (1 if m == 12 else 0), 1 if m == 12 else m + 1, 1)
    except Exception:
        raise HTTPException(400, "month must be YYYY-MM")
    db = _db(variant)
    rows = db.conn.execute(
        "SELECT date, equity, daily_pl, daily_pl_pct FROM daily_snapshots "
        "WHERE date >= ? AND date < ? ORDER BY date ASC",
        [first.isoformat(), next_first.isoformat()],
    ).fetchall()

    by_date = {r["date"]: dict(r) for r in rows}

    # Count fills per day for the calendar pill.
    fill_rows = db.conn.execute(
        "SELECT substr(timestamp,1,10) AS d, COUNT(*) AS n FROM trades "
        "WHERE filled_qty IS NOT NULL AND filled_qty > 0 "
        "  AND substr(timestamp,1,10) >= ? AND substr(timestamp,1,10) < ? "
        "GROUP BY d",
        [first.isoformat(), next_first.isoformat()],
    ).fetchall()
    by_date_fills = {r["d"]: int(r["n"]) for r in fill_rows}

    days_out: List[Dict] = []
    cur = first
    while cur < next_first:
        if cur.weekday() < 5:  # weekdays only
            iso = cur.isoformat()
            snap = by_date.get(iso) or {}
            days_out.append({
                "date":         iso,
                "weekday":      cur.strftime("%a").upper(),
                # daily_pl_pct is stored as decimal; emit percent units.
                "pnl_pct":      (float(snap["daily_pl_pct"]) * 100.0) if snap.get("daily_pl_pct") is not None else None,
                "pnl_usd":      float(snap["daily_pl"]) if snap.get("daily_pl") is not None else None,
                "fills":        by_date_fills.get(iso, 0),
            })
        cur += timedelta(days=1)
    return {"variant": variant, "month": month, "days": days_out}


# ── Versus ──────────────────────────────────────────────────────────


def _stats_for_trips(trips: List[Dict]) -> Dict:
    if not trips:
        return {
            "trades": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "expectancy": 0.0, "profit_factor": 0.0, "avg_hold": 0.0,
            "max_dd": 0.0,
        }
    pnls = [t["pnl_pct"] for t in trips]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    holds = [t["hold_days"] for t in trips]
    avg_win = statistics.fmean(wins) if wins else 0.0
    avg_loss = statistics.fmean(losses) if losses else 0.0
    win_rate = (len(wins) / len(pnls) * 100.0) if pnls else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0)
    # Cumulative pnl in pct space, then max drawdown of that curve.
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
    return {
        "trades":        len(trips),
        "win_rate":      round(win_rate, 2),
        "avg_win":       round(avg_win, 4),
        "avg_loss":      round(avg_loss, 4),
        "expectancy":    round(statistics.fmean(pnls), 4),
        "profit_factor": (None if math.isinf(profit_factor) else round(profit_factor, 3)),
        "avg_hold":      round(statistics.fmean(holds), 2) if holds else 0.0,
        "max_dd":        round(max_dd, 4),
    }


def _range_window(range_str: str) -> Optional[Tuple[date, date]]:
    """Return (start, end_exclusive) or None for unbounded."""
    today = date.today()
    if range_str == "5d":
        return today - timedelta(days=7), today + timedelta(days=1)
    if range_str == "30d":
        return today - timedelta(days=30), today + timedelta(days=1)
    if range_str == "90d":
        return today - timedelta(days=90), today + timedelta(days=1)
    if range_str == "qtd":
        q = (today.month - 1) // 3 * 3 + 1
        return date(today.year, q, 1), today + timedelta(days=1)
    if range_str == "ytd":
        return date(today.year, 1, 1), today + timedelta(days=1)
    if range_str == "1y":
        return today - timedelta(days=365), today + timedelta(days=1)
    if range_str == "all":
        return None
    return None


@router.get("/versus")
def versus(a: str, b: str, range: str = "30d", limit: int = 30,
           before: Optional[int] = None) -> dict:
    """A vs B comparison.

    Returns:
      - per side: trips, stats, cumulative pnl points (one per closed trip)
      - merged trades: most recent ``limit`` closed trips across A and B,
        cursor-paginated by exit_id within whichever variant produced the
        oldest row in the page.
    """
    if a == b:
        raise HTTPException(400, "A and B must differ")
    db_a, db_b = _db(a), _db(b)
    window = _range_window(range)

    def _filter(trips: List[Dict]) -> List[Dict]:
        if window is None:
            return trips
        start, end = window
        out = []
        for t in trips:
            try:
                d = datetime.fromisoformat(
                    t["exit_ts"].replace(" ", "T").replace("Z", "+00:00")
                ).date()
            except Exception:
                continue
            if start <= d < end:
                out.append(t)
        return out

    trips_a = _filter(_round_trips_for(db_a))
    trips_b = _filter(_round_trips_for(db_b))

    def _curve(trips: List[Dict]) -> List[Dict]:
        cum = 0.0
        out = []
        for t in trips:
            cum += t["pnl_pct"]
            out.append({"ts": t["exit_ts"], "cum_pct": round(cum, 4)})
        return out

    da = display_for(a)
    db_disp = display_for(b)

    # Merge for the trade table — newest first.
    merged: List[Dict] = []
    for t in trips_a:
        merged.append({**t, "variant": a, "variant_display": da["display"], "color": da["color"]})
    for t in trips_b:
        merged.append({**t, "variant": b, "variant_display": db_disp["display"], "color": db_disp["color"]})
    merged.sort(key=lambda r: r["exit_ts"], reverse=True)
    if before is not None:
        merged = [t for t in merged if t["exit_id"] < int(before)]
    page = merged[: max(1, min(limit, 200))]
    next_before = page[-1]["exit_id"] if page and len(page) >= limit else None

    return {
        "a": {
            "name": a, "display": da["display"], "color": da["color"],
            "stats": _stats_for_trips(trips_a),
            "curve": _curve(trips_a),
        },
        "b": {
            "name": b, "display": db_disp["display"], "color": db_disp["color"],
            "stats": _stats_for_trips(trips_b),
            "curve": _curve(trips_b),
        },
        "trades":      page,
        "next_before": next_before,
        "range":       range,
    }


# ── Spotlight (per-ticket panel on floor.html) ──────────────────────


@router.get("/spotlight/{variant}/{symbol}")
def spotlight(variant: str, symbol: str) -> dict:
    """Latest signal + position metadata for a symbol on a variant.

    Used by the spotlight overlay to populate the thesis text + entry/
    stop/target pills. The chart payload itself comes from the existing
    /api/trade/{variant}/{trade_id}/chart route.
    """
    db = _db(variant)

    sig = db.conn.execute(
        "SELECT id, timestamp, action, confidence, reasoning, "
        "       stop_loss, take_profit, signal_metadata "
        "FROM signals WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1",
        [symbol],
    ).fetchone()
    pos = db.conn.execute(
        "SELECT entry_price, entry_date, current_stop, base_pattern, "
        "       rs_at_entry, stage_at_entry, bars_held "
        "FROM position_states WHERE symbol = ?",
        [symbol],
    ).fetchone()
    last_trade = db.conn.execute(
        "SELECT id, timestamp, side, filled_qty, filled_price "
        "FROM trades WHERE symbol = ? AND filled_qty > 0 "
        "ORDER BY timestamp DESC LIMIT 1",
        [symbol],
    ).fetchone()
    last_buy = db.conn.execute(
        "SELECT id, timestamp, filled_qty, filled_price "
        "FROM trades WHERE symbol = ? AND LOWER(side)='buy' AND filled_qty > 0 "
        "ORDER BY timestamp DESC LIMIT 1",
        [symbol],
    ).fetchone()

    vdisp = display_for(variant)
    return {
        "variant":      variant,
        "display":      vdisp["display"],
        "color":        vdisp["color"],
        "sleeve":       vdisp["sleeve"],
        "layers":       vdisp.get("layers", []),
        "symbol":       symbol,
        "signal":       dict(sig) if sig else None,
        "position":     dict(pos) if pos else None,
        "last_trade":   dict(last_trade) if last_trade else None,
        "entry_trade":  dict(last_buy) if last_buy else None,
    }


# ── Intraday equity ring buffer ─────────────────────────────────────
#
# Daily snapshots are written by the scheduler at end-of-day. Live
# minute-cadence equity is not persisted anywhere — it lives only in
# the broker (Alpaca) until the next snapshot. To surface it on the
# floor pulse strip, we run a per-variant async poller that hits
# ``Orchestrator.get_status()`` once a minute and keeps a 24h ring
# buffer in memory.
#
# Trade-offs:
# - Volatile across web-app restarts (acceptable v1: 24h is short).
# - Skips PEAD variants — they don't own a live Alpaca account from
#   this app's perspective; their authoritative state lives in the
#   PEAD wrapper's JSON files (see CLAUDE.md PEAD architecture note).
# - Refresh interval is fixed at 60s. If the user wants finer cadence
#   we'll add an env override.

_INTRADAY_INTERVAL_S = int(os.environ.get("BAIBOT_V2_INTRADAY_INTERVAL_S", "60"))
_INTRADAY_BUFFER_SIZE = 60 * 24  # 24h at 1-min cadence
_PEAD_PREFIXES = ("pead",)        # storage names that should be skipped

# variant_name → deque[{ts, equity, cash, day_pnl_pct, day_pnl_usd, exposure_pct}]
_INTRADAY_BUFFER: Dict[str, Deque[Dict]] = defaultdict(
    lambda: deque(maxlen=_INTRADAY_BUFFER_SIZE)
)
_INTRADAY_TASK: Optional[asyncio.Task] = None


def _build_orchestrator(variant_name: str):
    """Same orchestrator construction the dashboard uses, isolated to
    this module so we don't re-import on every poll cycle."""
    from tradingagents.automation.config import build_config
    from tradingagents.automation.orchestrator import Orchestrator
    from tradingagents.testing.ab_config import build_variant_config, load_experiment

    yaml_path = os.environ.get(
        "EXPERIMENT_CONFIG_PATH", "experiments/paper_launch_v2.yaml"
    )
    base_cfg = build_config()
    if Path(yaml_path).exists():
        exp = load_experiment(yaml_path)
        v_obj = next((x for x in exp.variants if x.name == variant_name), None)
        cfg = build_variant_config(base_cfg, v_obj) if v_obj else base_cfg
    else:
        cfg = base_cfg
    if not cfg.get("alpaca_api_key") or not cfg.get("alpaca_secret_key"):
        return None
    return Orchestrator(cfg)


_ORCH_CACHE: Dict[str, object] = {}


async def _intraday_tick():
    """One sweep across all live variants. Errors are swallowed per
    variant so a single bad orchestrator doesn't kill the loop."""
    variants = _discover()
    now_iso = _to_iso(datetime.now(timezone.utc))
    for v in variants:
        name = v["name"]
        if any(name.startswith(p) for p in _PEAD_PREFIXES):
            continue
        try:
            orch = _ORCH_CACHE.get(name)
            if orch is None:
                orch = _build_orchestrator(name)
                if orch is None:
                    continue
                _ORCH_CACHE[name] = orch
            status = orch.get_status()
            acct = status.get("account") or {}
            positions = status.get("positions") or []
            equity = float(acct.get("equity") or acct.get("portfolio_value") or 0)
            cash = float(acct.get("cash") or 0)
            buying_power = float(acct.get("buying_power") or 0)
            # Day P&L: prefer broker-provided, else compute from snapshot start.
            day_pnl_usd = float(acct.get("daytrade_count") and 0 or
                                 acct.get("equity_change") or 0)
            day_pnl_pct = None
            if equity:
                pos_value = sum(float(p.get("market_value", 0) or 0) for p in positions)
                exposure_pct = (pos_value / equity) * 100.0
            else:
                exposure_pct = None
            # Compute day_pnl_pct vs the variant's last daily snapshot.
            try:
                db = TradingDatabase(v["db_path"])
                base_row = db.conn.execute(
                    "SELECT equity FROM daily_snapshots ORDER BY date DESC LIMIT 1"
                ).fetchone()
                base = float(base_row["equity"]) if base_row else None
                if base and equity:
                    day_pnl_pct = (equity - base) / base * 100.0
                    day_pnl_usd = equity - base
            except Exception:
                pass
            _INTRADAY_BUFFER[name].append({
                "ts":             now_iso,
                "equity":         equity,
                "cash":           cash,
                "buying_power":   buying_power,
                "day_pnl_usd":    day_pnl_usd,
                "day_pnl_pct":    day_pnl_pct,
                "exposure_pct":   exposure_pct,
                "n_positions":    len(positions),
            })
        except Exception as exc:
            _log.warning("intraday tick failed for %s: %s", name, exc)


async def _intraday_loop():
    while True:
        try:
            await _intraday_tick()
        except Exception as exc:
            _log.warning("intraday loop error: %s", exc)
        await asyncio.sleep(_INTRADAY_INTERVAL_S)


def start_intraday_loop():
    """Idempotent — safe to call multiple times. Returns the running task."""
    global _INTRADAY_TASK
    if _INTRADAY_TASK is None or _INTRADAY_TASK.done():
        loop = asyncio.get_event_loop()
        _INTRADAY_TASK = loop.create_task(_intraday_loop())
    return _INTRADAY_TASK


@router.get("/equity/live")
def equity_live() -> dict:
    """Latest intraday tick for every variant in the buffer.

    Cheap and frequently-polled by floor.html for the pulse strip.
    Variants with no buffer entry yet (just-started, PEAD, or auth
    failure) are returned with ``tick: null``.
    """
    out = {}
    for v in _discover():
        name = v["name"]
        buf = _INTRADAY_BUFFER.get(name)
        out[name] = buf[-1] if buf else None
    return {
        "variants":   out,
        "interval_s": _INTRADAY_INTERVAL_S,
    }


@router.get("/equity/intraday")
def equity_intraday(variant: str, day: Optional[str] = None) -> dict:
    """Return the in-memory minute-cadence equity buffer for ``variant``.

    Optionally filter to a single ``day`` (YYYY-MM-DD). Returns:
      ``{ variant, points: [{ts, equity, cash, day_pnl_pct, ...}], asof }``

    Buffer is volatile and clears on web-app restart.
    """
    if _variant(variant) is None:
        raise HTTPException(404, f"variant {variant!r} not found")
    rows = list(_INTRADAY_BUFFER.get(variant, ()))
    if day:
        rows = [r for r in rows if (r.get("ts") or "").startswith(day)]
    return {
        "variant": variant,
        "points":  rows,
        "asof":    rows[-1]["ts"] if rows else None,
        "interval_s": _INTRADAY_INTERVAL_S,
        "size":    len(rows),
    }
