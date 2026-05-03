"""Intraday mechanical (NR4 / gap-reclaim / ORB) overlay extractor.

Reasoning lives in ``signals.signal_metadata`` (see
``intraday_orchestrator.py:395-422``). The metadata schema:

  setup_family             — "nr4_breakout" | "gap_reclaim_long" | "orb_breakout"
  bar_ts                   — entry bar timestamp
  session_date             — session that fired the signal
  entry_close              — bar's close at entry
  volume_ratio             — relative volume vs 20d average
  breakout_distance_pct    — % move past the trigger level
  candidate_score          — composite ranking score
  vwap                     — session VWAP at the entry bar
  distance_from_vwap_pct
  opening_range_high/low   — first-N-bars range (ORB family)
  prior_session_high       — D-1 high (gap_reclaim, NR4 trigger)
  prior_session_close      — D-1 close (gap-reclaim baseline)
  prior_session_is_nr      — bool, NR4 detection on prior bar

Overlay rendering:
  - VWAP / OR-high / OR-low / prior_session_high → horizontal levels
  - prior_session_is_nr=True → marker on the prior session's bar
  - 15:55 ET EOD-flatten → vertical level on the entry day
"""

from __future__ import annotations

import json
from datetime import datetime, time as dtime, timezone
from typing import List, Optional

import pandas as pd

from tradingagents.storage.database import TradingDatabase

from ..bars import _to_unix, fetch_15m
from ._trade_lookup import find_entry_id, list_cycle_exits, trade_summary
from .base import (
    Bar,
    ChartPayload,
    Criterion,
    Fill,
    LevelOverlay,
    MarkerOverlay,
    Metric,
    Reasoning,
)


_DEFAULT_15M_DB = "research_data/intraday_15m.duckdb"


def _trade_with_signal(db: TradingDatabase, trade_id: int) -> Optional[dict]:
    rows = db.conn.execute(
        """
        SELECT t.id AS trade_id, t.timestamp AS trade_ts, t.symbol, t.side,
               t.qty, t.filled_qty, t.filled_price, t.status,
               t.reasoning AS trade_reasoning, t.signal_id,
               s.action, s.confidence,
               s.reasoning AS signal_reasoning, s.signal_metadata,
               s.stop_loss, s.take_profit
        FROM trades t
        LEFT JOIN signals s ON s.id = t.signal_id
        WHERE t.id = ?
        """,
        [trade_id],
    ).fetchall()
    return dict(rows[0]) if rows else None


def _build_overlays(meta: dict, bars: List[Bar]) -> List[dict]:
    """Translate intraday signal_metadata → overlay primitives."""
    overlays: List[dict] = []
    if not bars:
        return overlays

    first_t, last_t = bars[0]["time"], bars[-1]["time"]
    bar_ts_unix = _to_unix(meta["bar_ts"]) if meta.get("bar_ts") else None

    # VWAP at entry — horizontal level (one bar wide visually, but the
    # frontend extends levels across the chart by default).
    if meta.get("vwap"):
        overlays.append(LevelOverlay(
            kind="level",
            price=float(meta["vwap"]),
            from_t=first_t, to_t=last_t,
            label="VWAP@entry",
            style="dashed",
            color="#a78bfa",
        ))

    # Opening-range high/low (ORB)
    if meta.get("opening_range_high"):
        overlays.append(LevelOverlay(
            kind="level",
            price=float(meta["opening_range_high"]),
            from_t=first_t, to_t=last_t,
            label="OR high",
            style="solid",
            color="#10b981",
        ))
    if meta.get("opening_range_low"):
        overlays.append(LevelOverlay(
            kind="level",
            price=float(meta["opening_range_low"]),
            from_t=first_t, to_t=last_t,
            label="OR low",
            style="solid",
            color="#ef4444",
        ))

    # Prior session high — NR4/gap_reclaim trigger
    if meta.get("prior_session_high"):
        overlays.append(LevelOverlay(
            kind="level",
            price=float(meta["prior_session_high"]),
            from_t=first_t, to_t=last_t,
            label="D-1 high",
            style="dashed",
            color="#f5a524",
        ))

    # Prior session close — gap_reclaim baseline (visual context for the gap)
    if meta.get("prior_session_close"):
        overlays.append(LevelOverlay(
            kind="level",
            price=float(meta["prior_session_close"]),
            from_t=first_t, to_t=last_t,
            label="D-1 close",
            style="dotted",
            color="#64748b",
        ))

    # NR4 marker on the prior bar (the signal's setup bar, not the entry bar)
    if meta.get("prior_session_is_nr") and bar_ts_unix:
        # Locate the prior session's last bar in our window for marker placement.
        # We don't have the exact prior-session bar time, so we render the
        # marker label on the entry bar with a "NR4 prev session" hint.
        overlays.append(MarkerOverlay(
            kind="marker",
            time=bar_ts_unix,
            price=float(meta.get("entry_close") or meta.get("vwap") or 0.0),
            label="NR4 prev",
            side="info",
            color="#f5a524",
        ))

    # 15:55 ET EOD-flatten level on the entry day
    if bar_ts_unix:
        entry_dt = datetime.fromtimestamp(bar_ts_unix, tz=timezone.utc)
        # Convert to NYT (ET) approximately — the bars come in UTC; we mark
        # the EOD on the same calendar day we entered. Frontend renders
        # this as a vertical-implied horizontal level; we use a colored
        # level pinned to the latest bar to highlight the EOD ceiling.
        eod = datetime.combine(entry_dt.date(), dtime(19, 55), tzinfo=timezone.utc)  # 15:55 ET ≈ 19:55 UTC (EDT)
        overlays.append(LevelOverlay(
            kind="level",
            price=float(meta.get("entry_close") or 0.0),
            from_t=int(eod.timestamp()), to_t=int(eod.timestamp()),
            label="15:55 EOD flatten",
            style="solid",
            color="#475569",
        ))

    return overlays


def _build_reasoning(trade_row: dict, meta: dict, summary: dict) -> Reasoning:
    family = meta.get("setup_family", "?")
    vol_ratio = meta.get("volume_ratio")
    breakout_pct = meta.get("breakout_distance_pct")

    if summary.get("is_closed") and summary.get("return_pct") is not None:
        headline = f"{family} · Closed {summary['return_pct']*100:+.2f}% on {trade_row['symbol']}"
    else:
        headline = f"{family} BUY on {trade_row['symbol']}"

    metrics: List[Metric] = [
        {"label": "Setup family",   "value": family},
        {"label": "Volume ratio",   "value": f"{vol_ratio:.2f}x" if isinstance(vol_ratio, (int, float)) else "—"},
        {"label": "Breakout %",     "value": f"{breakout_pct:.3%}" if isinstance(breakout_pct, (int, float)) else "—"},
        {"label": "Candidate score","value": f"{meta.get('candidate_score'):.2f}" if isinstance(meta.get("candidate_score"), (int, float)) else "—"},
        {"label": "VWAP@entry",     "value": f"{meta.get('vwap'):.2f}" if meta.get("vwap") else "—"},
        {"label": "Dist from VWAP", "value": f"{meta.get('distance_from_vwap_pct'):.3%}" if isinstance(meta.get("distance_from_vwap_pct"), (int, float)) else "—"},
        {"label": "D-1 high",       "value": f"{meta.get('prior_session_high'):.2f}" if meta.get("prior_session_high") else "—"},
        {"label": "Stop pct",       "value": f"{trade_row.get('stop_loss') or 0:.2%}"},
    ]
    if summary.get("is_closed"):
        metrics.extend([
            {"label": "Entry price",  "value": f"{float(trade_row['filled_price']):.2f}" if trade_row.get('filled_price') else "—"},
            {"label": "Exit price",   "value": f"{summary['exit_price']:.2f}" if summary.get('exit_price') else "—"},
            {"label": "Return %",     "value": f"{summary['return_pct']*100:+.2f}%" if summary.get("return_pct") is not None else "—"},
            {"label": "Held",         "value": summary.get("held_str") or "—"},
            {"label": "Exit reason",  "value": summary.get("exit_reason") or "—"},
        ])

    criteria: List[Criterion] = [
        {"name": "Min volume ratio (≥1.3 NR4 / ≥1.5 default)",
         "passed": isinstance(vol_ratio, (int, float)) and vol_ratio >= 1.3,
         "value": vol_ratio},
        {"name": "Above VWAP",
         "passed": isinstance(meta.get("distance_from_vwap_pct"), (int, float))
                   and meta["distance_from_vwap_pct"] >= 0,
         "value": meta.get("distance_from_vwap_pct")},
        {"name": "Prior session NR4 (NR4 family only)",
         "passed": bool(meta.get("prior_session_is_nr")),
         "value": meta.get("prior_session_is_nr")},
        {"name": "Above prior session high",
         "passed": isinstance(meta.get("entry_close"), (int, float))
                   and isinstance(meta.get("prior_session_high"), (int, float))
                   and meta["entry_close"] > meta["prior_session_high"],
         "value": meta.get("prior_session_high")},
    ]

    return Reasoning(
        headline=headline,
        criteria=criteria,
        metrics=metrics,
        narrative=trade_row.get("signal_reasoning") or trade_row.get("trade_reasoning") or None,
    )


def build_chart(
    db: TradingDatabase,
    trade_id: int,
    variant_config: dict,
) -> ChartPayload:
    entry_id = find_entry_id(db, trade_id) or trade_id
    row = _trade_with_signal(db, entry_id)
    if row is None:
        return ChartPayload(
            symbol="?", variant=variant_config.get("name", "?"),
            strategy_type="intraday_mechanical", timeframe="15m",
            bars=[], overlays=[], fills=[],
            reasoning=Reasoning(headline="trade not found", criteria=[], metrics=[], narrative=None),
            error=f"trade {trade_id} not found",
        )

    symbol = row["symbol"]
    db_path = variant_config.get("intraday_15m_db", _DEFAULT_15M_DB)

    meta: dict = {}
    if row.get("signal_metadata"):
        try:
            meta = json.loads(row["signal_metadata"])
        except Exception:
            meta = {}

    bars: List[Bar] = fetch_15m(
        symbol=symbol,
        db_path=db_path,
        bars_before=80,
        bars_after=30,
        pivot_ts=str(row["trade_ts"]),
    )

    exits = list_cycle_exits(db, symbol, str(row["trade_ts"]))
    fills: List[Fill] = []
    if row.get("filled_price"):
        fills.append(Fill(
            time=_to_unix(row["trade_ts"]),
            price=float(row["filled_price"]),
            side="buy",
            qty=float(row.get("filled_qty") or row.get("qty") or 0),
            reasoning=row.get("trade_reasoning"),
        ))
    for ex in exits:
        if ex.get("filled_price"):
            fills.append(Fill(
                time=_to_unix(ex["timestamp"]),
                price=float(ex["filled_price"]),
                side="sell",
                qty=float(ex.get("filled_qty") or 0),
                reasoning=ex.get("reasoning"),
            ))

    summary = trade_summary(row, exits)
    return ChartPayload(
        symbol=symbol,
        variant=variant_config.get("name", "?"),
        strategy_type="intraday_mechanical",
        timeframe="15m",
        bars=bars,
        overlays=_build_overlays(meta, bars),
        fills=fills,
        reasoning=_build_reasoning(row, meta, summary),
        error=None if bars else "no 15m bars in window — check intraday_15m.duckdb",
    )
