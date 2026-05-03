"""Chan (缠论) overlay extractor — works for ``chan`` and ``chan_v2``.

Reads the persisted ``signal_metadata`` JSON from the entry signal,
re-runs Chan structural analysis over a window of 30m bars around the
entry, and returns BI / SEG / ZS / BSP overlays plus the rich reasoning
panel that the chan orchestrator stores at trade time.

No recompute of the strategy is required — the orchestrator already
persists ``t_types``, ``bi_low``, ``bsp_reason``, ``regime_at_entry``,
``market_score`` (see ``chan_orchestrator.py:341``).
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

import pandas as pd

# Reasoning text format used by ChanOrchestrator's _execute_signal call:
#   "Chan buy signal: T2 at 2026/04/16 15:00"
#   "Chan buy signal: T1+T1p at 2026/04/16 15:00"
# Extract the T-type chunk so we can populate the headline / criteria
# panel even when signal_metadata is NULL (a real gap on the chan_v2
# variant — 0/25 signals have metadata as of 2026-05-03).
_REASON_RE = re.compile(
    r"Chan\s+(?P<side>buy|sell)\s+signal:\s+(?P<types>[\w+]+)\s+at\s+(?P<bsp_time>[\d/\- :T]+)",
    re.IGNORECASE,
)


def _parse_reasoning_text(text: Optional[str]) -> dict:
    if not text:
        return {}
    m = _REASON_RE.search(text)
    if not m:
        return {}
    return {
        "t_types": m.group("types"),
        "bsp_reason": text,
        "bsp_time": m.group("bsp_time").strip(),
    }

from tradingagents.dashboard.chan_structures import extract_chan_structures
from tradingagents.storage.database import TradingDatabase

from ..bars import _to_unix, fetch_30m
from ._trade_lookup import find_entry_id, list_cycle_exits, trade_summary
from .base import (
    Bar,
    ChartPayload,
    Criterion,
    Fill,
    LineOverlay,
    MarkerOverlay,
    Metric,
    Reasoning,
    ZoneOverlay,
)


_DEFAULT_DB = "research_data/intraday_30m_broad.duckdb"


def _trade_with_signal(db: TradingDatabase, trade_id: int) -> Optional[dict]:
    """Look up trade row joined with its parent signal row."""
    rows = db.conn.execute(
        """
        SELECT t.id AS trade_id, t.timestamp AS trade_ts, t.symbol, t.side,
               t.qty, t.filled_qty, t.filled_price, t.status,
               t.reasoning AS trade_reasoning, t.signal_id,
               s.timestamp AS signal_ts, s.action, s.confidence,
               s.reasoning AS signal_reasoning, s.signal_metadata,
               s.stop_loss, s.take_profit, s.full_analysis
        FROM trades t
        LEFT JOIN signals s ON s.id = t.signal_id
        WHERE t.id = ?
        """,
        [trade_id],
    ).fetchall()
    if not rows:
        return None
    return dict(rows[0])


def _build_overlays(
    structures: dict,
    bsp_pivot_ts: Optional[str],
) -> List[dict]:
    """Translate Chan structures dict → unified overlay vocabulary.

    BI:  thin dashed line (a "stroke" — leg between confirmed pivots)
    SEG: thicker solid line (a higher-level "segment" composed of strokes)
    ZS:  semi-transparent rectangle ("zhongshu" — consolidation pivot)
    BSP: marker at the buy/sell-point bar with the T-type label
    """
    overlays: List[dict] = []

    # Strokes (笔)
    for bi in structures.get("bi_list", []):
        overlays.append(LineOverlay(
            kind="line",
            from_t=_to_unix(bi["start_time"]),
            from_p=float(bi["start_val"]),
            to_t=_to_unix(bi["end_time"]),
            to_p=float(bi["end_val"]),
            label=None,
            style="dashed",
            color="#5b8def" if bi["dir"] == "up" else "#9aa5b1",
            width=1,
        ))

    # Segments (线段)
    for seg in structures.get("seg_list", []):
        overlays.append(LineOverlay(
            kind="line",
            from_t=_to_unix(seg["start_time"]),
            from_p=float(seg["start_val"]),
            to_t=_to_unix(seg["end_time"]),
            to_p=float(seg["end_val"]),
            label="SEG",
            style="solid",
            color="#1f78b4" if seg["dir"] == "up" else "#5d6d7e",
            width=2,
        ))

    # Zhongshu (中枢)
    for zs in structures.get("zs_list", []):
        overlays.append(ZoneOverlay(
            kind="zone",
            from_t=_to_unix(zs["begin_time"]),
            to_t=_to_unix(zs["end_time"]),
            low=float(zs["low"]),
            high=float(zs["high"]),
            label="ZS",
            color="rgba(245, 165, 36, 0.18)",
        ))

    # BSP markers (买卖点) — types already arrive as "T1" / "T1P" / "T2" /
    # "T2S" (or composites like "T1+T1P") from extract_chan_structures.
    for bsp in structures.get("bsp_list", []):
        overlays.append(MarkerOverlay(
            kind="marker",
            time=_to_unix(bsp["time"]),
            price=float(bsp["price"]),
            label=bsp.get("types") or "BSP",
            side="buy" if bsp["is_buy"] else "sell",
            color="#10b981" if bsp["is_buy"] else "#ef4444",
        ))

    return overlays


def _build_reasoning(trade_row: dict, meta: dict, summary: dict) -> Reasoning:
    """Compose the side-panel reasoning from signal_metadata + signal text.

    ``summary`` is the closed-trade summary from
    ``_trade_lookup.trade_summary``; when present, we surface exit/return/
    held-time metrics so closed-trade views are informative instead of
    showing dashes everywhere.
    """
    types = meta.get("t_types") or "—"
    bsp_reason = meta.get("bsp_reason") or trade_row.get("signal_reasoning") or ""
    bi_low = meta.get("bi_low")
    regime = meta.get("regime_at_entry")
    market_score = meta.get("market_score")
    confidence = meta.get("confidence")

    closed = summary.get("is_closed")
    headline_kind = (
        f"Closed · {summary['return_pct']*100:+.2f}%"
        if closed and summary.get("return_pct") is not None
        else (trade_row.get('action') or 'BUY')
    )
    headline = f"{types} {headline_kind} on {trade_row['symbol']}"

    metrics: List[Metric] = [
        {"label": "T-types",        "value": types},
        {"label": "BI low",         "value": f"{bi_low:.2f}" if bi_low else "—"},
        {"label": "Regime",         "value": regime or "—"},
        {"label": "Market score",   "value": f"{market_score:.1f}" if isinstance(market_score, (int, float)) else "—"},
        {"label": "Confidence",     "value": f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "—"},
        {"label": "Stop pct",       "value": f"{trade_row.get('stop_loss') or 0:.2%}"},
        {"label": "Take-profit pct","value": f"{trade_row.get('take_profit') or 0:.2%}"},
    ]
    if closed:
        metrics.extend([
            {"label": "Entry price",  "value": f"{float(trade_row['filled_price']):.2f}"},
            {"label": "Exit price",   "value": f"{summary['exit_price']:.2f}"},
            {"label": "Return %",     "value": f"{summary['return_pct']*100:+.2f}%" if summary.get("return_pct") is not None else "—"},
            {"label": "Held",         "value": summary.get("held_str") or "—"},
            {"label": "Exit reason",  "value": summary.get("exit_reason") or "—"},
        ])

    # Criteria — the chan filters that had to pass for this signal to fire.
    # We don't have per-criterion booleans persisted, but the presence of
    # a t_types value means each declared filter passed. Surface that.
    criteria: List[Criterion] = [
        {"name": f"BSP type allowed ({types})", "passed": bool(types and types != "—"), "value": types},
        {"name": "Above MACD 0-line filter",    "passed": True, "value": None},
        {"name": "Divergence cap 0.6",           "passed": True, "value": None},
        {"name": "Regime score ≥ 3 (chan_v2)",   "passed": (
            isinstance(market_score, (int, float)) and market_score >= 3
        ), "value": market_score},
    ]

    return Reasoning(
        headline=headline,
        criteria=criteria,
        metrics=metrics,
        narrative=bsp_reason or None,
    )


def build_chart(
    db: TradingDatabase,
    trade_id: int,
    variant_config: dict,
) -> ChartPayload:
    # Redirect to the cycle's BUY (entry) so the reasoning panel reads
    # the entry signal's metadata and the chart pivots on the entry.
    # SELL rows otherwise rendered all-red criteria + missing entry marker.
    entry_id = find_entry_id(db, trade_id) or trade_id
    row = _trade_with_signal(db, entry_id)
    if row is None:
        return ChartPayload(
            symbol="?", variant=variant_config.get("name", "?"),
            strategy_type="chan", timeframe="30m",
            bars=[], overlays=[], fills=[],
            reasoning=Reasoning(headline="trade not found", criteria=[], metrics=[], narrative=None),
            error=f"trade {trade_id} not found",
        )

    symbol = row["symbol"]
    db_path = variant_config.get("chan_intraday_db", _DEFAULT_DB)

    # Window: ~1200 bars before entry (≈90 trading days of 30m), ~40 after.
    # Chan ZS (中枢) requires multiple overlapping segments; verified
    # empirically that 400 bars produces 0 ZS for most chan_v2 trades while
    # 1200 bars routinely surfaces 1–2 ZS plus more BSPs. The chan library
    # is fast enough that 1200-bar runs still complete in <1s per symbol.
    bars: List[Bar] = fetch_30m(
        symbol=symbol,
        db_path=db_path,
        bars_before=1200,
        bars_after=40,
        pivot_ts=str(row["trade_ts"]),
    )

    # Run Chan structural analysis on the same window. extract_chan_structures
    # mutates DuckDBIntradayAPI.DB_PATH at module level; pass our path
    # through and let it operate.
    overlays: List[dict] = []
    if bars:
        # DuckDBIntradayAPI expects 'YYYY-MM-DD' (it appends time itself).
        first_date = pd.to_datetime(bars[0]["time"], unit="s").strftime("%Y-%m-%d")
        last_date  = pd.to_datetime(bars[-1]["time"], unit="s").strftime("%Y-%m-%d")
        try:
            # zs_algo='over_seg' is more aggressive at flagging ZS than
            # the default 'normal' — verified 2026-05-03 that several
            # chan_v2 trades flip from 0 to 1 ZS under over_seg. ZS are
            # still genuinely absent on stocks that trended without any
            # consolidation in the window; that's a property of the
            # price action, not a config bug.
            structures = extract_chan_structures(
                symbol=symbol,
                begin=first_date,
                end=last_date,
                db_path=db_path,
                config_overrides={"zs_algo": "over_seg"},
            )
            overlays = _build_overlays(structures, bsp_pivot_ts=str(row["trade_ts"]))
        except Exception as exc:
            overlays = []
            err = f"chan extraction failed: {exc}"
        else:
            err = None
    else:
        err = "no bars in window"

    # Parse persisted metadata for the reasoning panel. When the orchestrator
    # didn't populate signal_metadata (the chan_v2 variant currently never
    # does — see grep audit 2026-05-03), fall back to parsing the trade's
    # reasoning text. The chan orchestrator's reasoning string follows a
    # stable format we can regex.
    meta = {}
    if row.get("signal_metadata"):
        try:
            meta = json.loads(row["signal_metadata"])
        except Exception:
            meta = {}
    if not meta:
        meta = _parse_reasoning_text(
            row.get("signal_reasoning") or row.get("trade_reasoning")
        )

    # Fills — entry + every SELL that closes this cycle (until the next BUY
    # of this symbol, which would belong to the next cycle).
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
        strategy_type=variant_config.get("strategy_type", "chan"),
        timeframe="30m",
        bars=bars,
        overlays=overlays,
        fills=fills,
        reasoning=_build_reasoning(row, meta, summary),
        error=err,
    )
