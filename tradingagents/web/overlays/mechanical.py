"""Mechanical (Minervini) + LLM overlay extractor.

The mechanical orchestrator persists a rich JSON snapshot in
``signals.full_analysis`` at trade time — see
``orchestrator.py:_screen_with_minervini`` and friends. The snapshot
has all the values we need for overlays:

  Spot values:
    close, sma_50, sma_150, sma_200, 52w_high, 52w_low,
    avg_volume_50, rs_score, rs_percentile, return_on_equity
  Setup levels:
    pivot_price, buy_point, buy_limit_price, initial_stop_price,
    initial_stop_pct, base_depth_pct, handle_depth_pct
  Flags:
    passed_template, breakout_ready, vcp_candidate, base_candidate,
    leader_continuation, leader_continuation_watch, breakout_signal,
    market_confirmed_uptrend
  Context:
    stage_number, base_label, candidate_status, market_regime,
    sector, industry, earnings_days_away

The MA *time series* is computed on the fly from daily bars so we can
draw real lines, not just spot values.

The llm variant uses the same orchestrator, so its full_analysis has
the same Minervini-snapshot shape — we register this extractor for
both. (The 13-agent debate transcript is not reliably persisted to
``signals.full_analysis`` for live trades; if/when that changes, the
narrative panel can be enriched without affecting overlays.)
"""

from __future__ import annotations

import json
from typing import List, Optional

import pandas as pd

from tradingagents.storage.database import TradingDatabase

from ..bars import _to_unix_date, fetch_daily
from .base import (
    Bar,
    ChartPayload,
    Criterion,
    Fill,
    LevelOverlay,
    MAOverlay,
    Metric,
    Reasoning,
)


_DEFAULT_DAILY_DB = "research_data/market_data.duckdb"

_MA_COLORS = {
    50:  "#5b8def",   # blue
    150: "#a78bfa",   # violet
    200: "#5ec5b7",   # teal
}


# ── DB queries ───────────────────────────────────────────────────────


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


# ── MA time series ───────────────────────────────────────────────────


def _ma_series(bars: List[Bar], period: int) -> List[List[float]]:
    """Rolling-mean MA, expressed as ``[[unix_ts, ma_value], ...]``.

    Bars before ``period`` have no MA; we drop those entries instead of
    emitting NaNs (lightweight-charts' line series can't render NaN).
    """
    if len(bars) < period:
        return []
    closes = [b["close"] for b in bars]
    times  = [b["time"]  for b in bars]
    out: List[List[float]] = []
    s = 0.0
    for i, c in enumerate(closes):
        s += c
        if i >= period:
            s -= closes[i - period]
        if i >= period - 1:
            out.append([times[i], round(s / period, 4)])
    return out


# ── Reasoning ────────────────────────────────────────────────────────


def _fmt_pct(v) -> str:
    if not isinstance(v, (int, float)):
        return "—"
    return f"{v:+.2%}" if abs(v) > 0 else "0.00%"


def _fmt_num(v, places: int = 2) -> str:
    if not isinstance(v, (int, float)):
        return "—"
    return f"{v:.{places}f}"


def _build_reasoning(trade_row: dict, fa: dict) -> Reasoning:
    sym       = trade_row["symbol"]
    close     = fa.get("close")
    pivot     = fa.get("pivot_price") or fa.get("buy_point")
    rs_pct    = fa.get("rs_percentile")
    sma50     = fa.get("sma_50")
    sma150    = fa.get("sma_150")
    sma200    = fa.get("sma_200")
    hi52      = fa.get("52w_high")
    lo52      = fa.get("52w_low")
    candidate = fa.get("candidate_status") or "—"
    stage     = fa.get("stage_number")
    base      = fa.get("base_label") or "—"

    # Distance from 52w high (Minervini wants ≤ 25% off-high)
    dist_from_high = None
    if isinstance(close, (int, float)) and isinstance(hi52, (int, float)) and hi52 > 0:
        dist_from_high = (close - hi52) / hi52

    # Distance from 52w low (Minervini wants ≥ 30% above)
    dist_from_low = None
    if isinstance(close, (int, float)) and isinstance(lo52, (int, float)) and lo52 > 0:
        dist_from_low = (close - lo52) / lo52

    headline_kind = (
        "Leader continuation" if fa.get("leader_continuation_actionable")
        else "Watch · leader" if fa.get("leader_continuation_watch")
        else "Breakout"        if fa.get("breakout_signal")
        else "Setup"
    )
    headline = f"{headline_kind} · {sym} @ {_fmt_num(close)}"

    criteria: List[Criterion] = [
        {"name": "Trend template (8/8)",
         "passed": bool(fa.get("passed_template")),
         "value": fa.get("template_score")},
        {"name": "Close > MA50 > MA150 > MA200",
         "passed": all(isinstance(v, (int, float)) for v in [close, sma50, sma150, sma200])
                   and close > sma50 > sma150 > sma200,
         "value": None},
        {"name": "RS percentile ≥ 70",
         "passed": isinstance(rs_pct, (int, float)) and rs_pct >= 70,
         "value": _fmt_num(rs_pct, 1)},
        {"name": "≥ 30% above 52w low",
         "passed": isinstance(dist_from_low, (int, float)) and dist_from_low >= 0.30,
         "value": _fmt_pct(dist_from_low)},
        {"name": "≤ 25% off 52w high",
         "passed": isinstance(dist_from_high, (int, float)) and dist_from_high >= -0.25,
         "value": _fmt_pct(dist_from_high)},
        {"name": "Confirmed uptrend regime",
         "passed": bool(fa.get("market_confirmed_uptrend")),
         "value": fa.get("market_regime")},
    ]

    metrics: List[Metric] = [
        {"label": "Close",       "value": _fmt_num(close)},
        {"label": "RS %ile",     "value": _fmt_num(rs_pct, 1)},
        {"label": "MA 50",       "value": _fmt_num(sma50)},
        {"label": "MA 150",      "value": _fmt_num(sma150)},
        {"label": "MA 200",      "value": _fmt_num(sma200)},
        {"label": "52w high",    "value": _fmt_num(hi52)},
        {"label": "52w low",     "value": _fmt_num(lo52)},
        {"label": "Off-high",    "value": _fmt_pct(dist_from_high)},
        {"label": "Pivot",       "value": _fmt_num(pivot)},
        {"label": "Buy limit",   "value": _fmt_num(fa.get("buy_limit_price"))},
        {"label": "Initial stop","value": _fmt_num(fa.get("initial_stop_price"))},
        {"label": "Stop %",      "value": _fmt_pct(fa.get("initial_stop_pct"))},
        {"label": "Stage",       "value": stage if stage is not None else "—"},
        {"label": "Base",        "value": base},
        {"label": "Status",      "value": candidate},
        {"label": "Sector",      "value": fa.get("sector") or "—"},
    ]

    narrative_lines = [trade_row.get("signal_reasoning") or trade_row.get("trade_reasoning") or ""]
    earnings_days = fa.get("earnings_days_away")
    if isinstance(earnings_days, (int, float)) and earnings_days < 999:
        narrative_lines.append(f"Earnings in {earnings_days:.0f} days.")
    revg = fa.get("revenue_growth"); epsg = fa.get("eps_growth")
    if isinstance(revg, (int, float)) or isinstance(epsg, (int, float)):
        narrative_lines.append(
            f"Revenue YoY {_fmt_pct(revg)} · EPS YoY {_fmt_pct(epsg)} · "
            f"ROE {_fmt_pct(fa.get('return_on_equity'))}"
        )

    return Reasoning(
        headline=headline,
        criteria=criteria,
        metrics=metrics,
        narrative="\n".join(line for line in narrative_lines if line) or None,
    )


# ── Top-level builder ────────────────────────────────────────────────


def build_chart(
    db: TradingDatabase,
    trade_id: int,
    variant_config: dict,
) -> ChartPayload:
    row = _trade_with_signal(db, trade_id)
    if row is None:
        return ChartPayload(
            symbol="?", variant=variant_config.get("name", "?"),
            strategy_type=variant_config.get("strategy_type", "mechanical"),
            timeframe="1d",
            bars=[], overlays=[], fills=[],
            reasoning=Reasoning(headline="trade not found",
                                criteria=[], metrics=[], narrative=None),
            error=f"trade {trade_id} not found",
        )

    symbol = row["symbol"]
    db_path = variant_config.get("daily_db_path", _DEFAULT_DAILY_DB)

    # Parse the Minervini snapshot. If absent (pre-snapshot trades) we
    # still render bars + fills + a barebones reasoning panel.
    fa: dict = {}
    if row.get("full_analysis"):
        try:
            fa = json.loads(row["full_analysis"])
        except Exception:
            fa = {}

    pivot_date = str(pd.to_datetime(row["trade_ts"]).date())
    bars: List[Bar] = fetch_daily(
        symbol=symbol,
        db_path=db_path,
        bars_before=260,
        bars_after=80,
        pivot_date=pivot_date,
    )

    overlays: List[dict] = []

    # Moving-average lines
    if bars:
        for period in (50, 150, 200):
            values = _ma_series(bars, period)
            if not values:
                continue
            overlays.append(MAOverlay(
                kind="ma",
                period=period,
                values=values,
                color=_MA_COLORS[period],
                label=f"MA{period}",
            ))

    # 52w high / low — horizontal levels across the visible window
    first_t = bars[0]["time"] if bars else None
    last_t  = bars[-1]["time"] if bars else None
    def _level(price, label, style, color):
        if not isinstance(price, (int, float)) or price <= 0 or first_t is None:
            return None
        return LevelOverlay(
            kind="level", price=float(price),
            from_t=first_t, to_t=last_t,
            label=label, style=style, color=color,
        )

    for ov in [
        _level(fa.get("52w_high"),         "52w hi",    "dashed", "#9aa5b1"),
        _level(fa.get("52w_low"),          "52w lo",    "dashed", "#9aa5b1"),
        _level(fa.get("pivot_price"),      "pivot",     "solid",  "#f5a524"),
        _level(fa.get("buy_limit_price"),  "buy limit", "dotted", "#f5a524"),
        _level(fa.get("initial_stop_price"), "stop",    "solid",  "#e85a5a"),
    ]:
        if ov is not None:
            # Skip duplicate levels (pivot == buy_point common case)
            existing = [o for o in overlays if o.get("kind") == "level"]
            dup = any(abs(o["price"] - ov["price"]) < 0.005 for o in existing)
            if not dup:
                overlays.append(ov)

    # Fills
    fills: List[Fill] = []
    if row.get("filled_price"):
        fills.append(Fill(
            time=_to_unix_date(row["trade_ts"]),
            price=float(row["filled_price"]),
            side="buy" if str(row.get("side", "")).lower() == "buy" else "sell",
            qty=float(row.get("filled_qty") or row.get("qty") or 0),
            reasoning=row.get("trade_reasoning"),
        ))
    for ex in _exit_fills(db, symbol, str(row["trade_ts"])):
        if ex.get("filled_price"):
            fills.append(Fill(
                time=_to_unix_date(ex["timestamp"]),
                price=float(ex["filled_price"]),
                side="sell",
                qty=float(ex.get("filled_qty") or 0),
                reasoning=ex.get("reasoning"),
            ))

    return ChartPayload(
        symbol=symbol,
        variant=variant_config.get("name", "?"),
        strategy_type=variant_config.get("strategy_type", "mechanical"),
        timeframe="1d",
        bars=bars,
        overlays=overlays,
        fills=fills,
        reasoning=_build_reasoning(row, fa),
        error=None if bars else "no daily bars in window — check market_data.duckdb",
    )
