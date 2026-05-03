"""PEAD (post-earnings-announcement drift) overlay extractor.

PEAD trades are daily-timeframe, no stop, exit by date. Each open
position is identified by ``(symbol, event_date, exit_target_date)``.
Reasoning lives in two places:

* ``trading_pead.db`` — per-variant DB written by the bridge in
  ``automation/pead_ingest.py``. Standard trades/signals schema.
  The bridge sets ``base_pattern='earnings_surprise'`` and
  ``current_stop=0.0`` (sentinel for "no stop").
* ``research_data/earnings_data.duckdb.earnings_events`` — original
  earnings catalyst (eps_estimate, reported_eps, surprise_pct,
  time_hint, source). Cross-DB read.

The pead_llm variant adds a third source:
* ``earnings_data.duckdb.earnings_llm_decisions`` — the 13-agent
  debate's final_signal_text + trader_plan_text + screener_context.
  Used to populate the narrative panel.

Visualization:
* Daily candles, ~80 bars before the earnings event, ~30 after.
* Vertical level at the earnings event date (amber).
* Entry-price level (teal solid) and stop-sentinel removed.
* Exit-target-date level (faint, dashed).
* No structural overlays — PEAD has no chart structures, just dates.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from tradingagents.storage.database import TradingDatabase

from ..bars import _to_unix, fetch_daily
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


_DEFAULT_DAILY_DB    = "research_data/market_data.duckdb"
_DEFAULT_EARNINGS_DB = "research_data/earnings_data.duckdb"


def _trade_with_signal(db: TradingDatabase, trade_id: int) -> Optional[dict]:
    rows = db.conn.execute(
        """
        SELECT t.id AS trade_id, t.timestamp AS trade_ts, t.symbol, t.side,
               t.qty, t.filled_qty, t.filled_price, t.status,
               t.reasoning AS trade_reasoning, t.signal_id,
               s.action, s.confidence, s.reasoning AS signal_reasoning,
               s.full_analysis, s.signal_metadata,
               s.stop_loss, s.take_profit
        FROM trades t
        LEFT JOIN signals s ON s.id = t.signal_id
        WHERE t.id = ?
        """,
        [trade_id],
    ).fetchall()
    return dict(rows[0]) if rows else None


def _exit_fills(db: TradingDatabase, symbol: str, entry_ts: str) -> List[dict]:
    rows = db.conn.execute(
        """
        SELECT timestamp, side, filled_qty, filled_price, reasoning
        FROM trades
        WHERE symbol = ? AND LOWER(side) = 'sell' AND timestamp > ?
        ORDER BY timestamp ASC
        LIMIT 1
        """,
        [symbol, entry_ts],
    ).fetchall()
    return [dict(r) for r in rows]


def _fetch_earnings_event(
    earnings_db: str, symbol: str, near_date: str,
) -> Tuple[Optional[dict], Optional[str]]:
    """Look up the earnings event most likely to have triggered this trade.

    Returns ``(event_dict_or_None, error_or_None)``. Errors are non-fatal —
    the chart still renders without the catalyst overlay.
    """
    if not Path(earnings_db).exists():
        return None, f"{earnings_db} missing"
    try:
        import duckdb
        con = duckdb.connect(earnings_db, read_only=True)
        try:
            row = con.execute(
                """
                SELECT symbol, event_datetime, eps_estimate, reported_eps,
                       surprise_pct, time_hint, source
                FROM earnings_events
                WHERE symbol = ?
                  AND event_datetime BETWEEN ?::TIMESTAMP - INTERVAL 5 DAY
                                         AND ?::TIMESTAMP + INTERVAL 1 DAY
                ORDER BY ABS(EPOCH(event_datetime) - EPOCH(?::TIMESTAMP)) ASC
                LIMIT 1
                """,
                [symbol, near_date, near_date, near_date],
            ).fetchone()
            if row is None:
                return None, "no earnings event near trade date"
            cols = ["symbol", "event_datetime", "eps_estimate", "reported_eps",
                    "surprise_pct", "time_hint", "source"]
            return dict(zip(cols, row)), None
        finally:
            con.close()
    except Exception as e:
        return None, f"earnings lookup failed: {e}"


def _fetch_llm_decision(earnings_db: str, symbol: str, event_date: str) -> Optional[dict]:
    if not Path(earnings_db).exists():
        return None
    try:
        import duckdb
        con = duckdb.connect(earnings_db, read_only=True)
        try:
            row = con.execute(
                """
                SELECT llm_decision, llm_confidence, final_signal_text,
                       trader_plan_text, screener_context, deep_think_model,
                       quick_think_model, duration_seconds, total_llm_calls,
                       cost_estimate_usd, error
                FROM earnings_llm_decisions
                WHERE symbol = ? AND event_date = ?::DATE
                """,
                [symbol, event_date],
            ).fetchone()
            if row is None:
                return None
            cols = ["llm_decision", "llm_confidence", "final_signal_text",
                    "trader_plan_text", "screener_context", "deep_think_model",
                    "quick_think_model", "duration_seconds", "total_llm_calls",
                    "cost_estimate_usd", "error"]
            return dict(zip(cols, row))
        finally:
            con.close()
    except Exception:
        return None


def _build_reasoning(
    trade_row: dict,
    event: Optional[dict],
    llm_decision: Optional[dict],
    holding_days: Optional[int],
) -> Reasoning:
    sym = trade_row["symbol"]
    surprise = event["surprise_pct"] if event else None
    eps_est  = event["eps_estimate"] if event else None
    eps_act  = event["reported_eps"] if event else None
    time_hint= event.get("time_hint") if event else None

    if surprise is not None and surprise > 0:
        headline_kind = "Positive surprise"
    elif surprise is not None and surprise < 0:
        headline_kind = "Negative surprise"
    else:
        headline_kind = "PEAD entry"
    headline = f"{headline_kind} · {sym}"
    if surprise is not None:
        headline += f"  +{surprise:.1f}%" if surprise >= 0 else f"  {surprise:.1f}%"

    criteria: List[Criterion] = [
        {"name": "Surprise ≥ 5% (config min)",
         "passed": isinstance(surprise, (int, float)) and surprise >= 5.0,
         "value": (f"{surprise:+.2f}%" if isinstance(surprise, (int, float)) else None)},
        {"name": "Surprise ≤ 50% (config max — data noise cap)",
         "passed": isinstance(surprise, (int, float)) and surprise <= 50.0,
         "value": None},
        {"name": "Earnings event found",
         "passed": event is not None,
         "value": (event["event_datetime"].strftime("%Y-%m-%d") if event else None)},
    ]
    if llm_decision is not None:
        criteria.append({
            "name": f"LLM gate · {llm_decision.get('llm_decision') or '—'}",
            "passed": llm_decision.get("llm_decision") == "BUY",
            "value": llm_decision.get("llm_confidence"),
        })

    metrics: List[Metric] = [
        {"label": "Symbol",       "value": sym},
        {"label": "EPS estimate", "value": f"${eps_est:.2f}"  if isinstance(eps_est, (int, float)) else "—"},
        {"label": "EPS reported", "value": f"${eps_act:.2f}"  if isinstance(eps_act, (int, float)) else "—"},
        {"label": "Surprise %",   "value": f"{surprise:+.2f}%" if isinstance(surprise, (int, float)) else "—"},
        {"label": "Time hint",    "value": time_hint or "—"},
        {"label": "Source",       "value": (event.get("source") if event else "—") or "—"},
        {"label": "Entry price",  "value": f"{trade_row.get('filled_price') or 0:.2f}"},
        {"label": "Holding days", "value": holding_days if holding_days is not None else "—"},
    ]
    if llm_decision is not None:
        metrics.extend([
            {"label": "LLM model",    "value": llm_decision.get("deep_think_model") or "—"},
            {"label": "LLM duration", "value": f"{llm_decision.get('duration_seconds') or 0:.0f}s"},
            {"label": "LLM cost",     "value": f"${llm_decision.get('cost_estimate_usd') or 0:.2f}"},
        ])

    # Narrative — the 13-agent debate transcript if pead_llm, else the
    # trade's reasoning text.
    narrative_chunks: List[str] = []
    if llm_decision is not None:
        if llm_decision.get("final_signal_text"):
            narrative_chunks.append(
                "── FINAL DECISION ──\n" + llm_decision["final_signal_text"].strip()
            )
        if llm_decision.get("trader_plan_text"):
            narrative_chunks.append(
                "── TRADER PLAN ──\n" + llm_decision["trader_plan_text"].strip()
            )
        if llm_decision.get("screener_context"):
            narrative_chunks.append(
                "── EARNINGS CATALYST ──\n" + llm_decision["screener_context"].strip()
            )
    if not narrative_chunks:
        narrative_chunks.append(
            trade_row.get("signal_reasoning") or trade_row.get("trade_reasoning") or ""
        )

    return Reasoning(
        headline=headline,
        criteria=criteria,
        metrics=metrics,
        narrative="\n\n".join(c for c in narrative_chunks if c) or None,
    )


def build_chart(
    db: TradingDatabase,
    trade_id: int,
    variant_config: dict,
) -> ChartPayload:
    is_llm = variant_config.get("strategy_type") == "pead_llm"
    row = _trade_with_signal(db, trade_id)
    if row is None:
        return ChartPayload(
            symbol="?", variant=variant_config.get("name", "?"),
            strategy_type=variant_config.get("strategy_type", "pead"),
            timeframe="1d",
            bars=[], overlays=[], fills=[],
            reasoning=Reasoning(headline="trade not found",
                                criteria=[], metrics=[], narrative=None),
            error=f"trade {trade_id} not found",
        )

    symbol      = row["symbol"]
    daily_db    = variant_config.get("daily_db_path",   _DEFAULT_DAILY_DB)
    earnings_db = variant_config.get("earnings_db_path", _DEFAULT_EARNINGS_DB)
    pivot_date  = str(pd.to_datetime(row["trade_ts"]).date())

    bars: List[Bar] = fetch_daily(
        symbol=symbol, db_path=daily_db,
        bars_before=80, bars_after=40, pivot_date=pivot_date,
    )

    event, event_err = _fetch_earnings_event(earnings_db, symbol, pivot_date)
    llm_decision = (
        _fetch_llm_decision(earnings_db, symbol, str(event["event_datetime"].date()))
        if (is_llm and event) else None
    )

    overlays: List[dict] = []
    first_t = bars[0]["time"] if bars else None
    last_t  = bars[-1]["time"] if bars else None

    # Earnings event marker — drop a vertical accent at the event date
    if event and first_t is not None:
        ev_unix = int(pd.Timestamp(event["event_datetime"]).tz_localize("UTC").timestamp()) \
                  if pd.Timestamp(event["event_datetime"]).tzinfo is None \
                  else int(pd.Timestamp(event["event_datetime"]).timestamp())
        # Find a price near the event for marker placement (use entry close
        # if event matches a bar, else close-of-prior-bar)
        anchor_price = float(row.get("filled_price") or bars[-1]["close"])
        overlays.append(MarkerOverlay(
            kind="marker",
            time=ev_unix,
            price=anchor_price,
            label=f"EARNINGS{(' · ' + event['time_hint'].upper()) if event.get('time_hint') else ''}",
            side="info",
            color="#f5a524",
        ))

    # Entry-price reference level
    entry_price = float(row.get("filled_price") or 0)
    if entry_price > 0 and first_t is not None:
        overlays.append(LevelOverlay(
            kind="level", price=entry_price,
            from_t=first_t, to_t=last_t,
            label="entry", style="solid", color="#5ec5b7",
        ))

    # PEAD has no stop — surface the "exit by date" target instead.
    # The bridge populates current_stop=0.0; we don't draw one.

    # Fills (entry + exit)
    fills: List[Fill] = []
    if entry_price > 0:
        fills.append(Fill(
            time=_to_unix(row["trade_ts"]),
            price=entry_price,
            side="buy" if str(row.get("side", "")).lower() == "buy" else "sell",
            qty=float(row.get("filled_qty") or row.get("qty") or 0),
            reasoning=row.get("trade_reasoning"),
        ))
    holding_days = None
    for ex in _exit_fills(db, symbol, str(row["trade_ts"])):
        if ex.get("filled_price"):
            fills.append(Fill(
                time=_to_unix(ex["timestamp"]),
                price=float(ex["filled_price"]),
                side="sell",
                qty=float(ex.get("filled_qty") or 0),
                reasoning=ex.get("reasoning"),
            ))
            try:
                holding_days = (pd.to_datetime(ex["timestamp"]).date()
                                - pd.to_datetime(row["trade_ts"]).date()).days
            except Exception:
                pass

    err_parts = []
    if not bars:           err_parts.append("no daily bars")
    if event_err:          err_parts.append(event_err)

    return ChartPayload(
        symbol=symbol,
        variant=variant_config.get("name", "?"),
        strategy_type=variant_config.get("strategy_type", "pead"),
        timeframe="1d",
        bars=bars,
        overlays=overlays,
        fills=fills,
        reasoning=_build_reasoning(row, event, llm_decision, holding_days),
        error=" · ".join(err_parts) or None,
    )
