"""Per-trade review chart builder — dispatches to strategy-specific overlays.

Pure rendering logic: given a trade outcome, price bars, and structured
setup context, produce a Plotly figure. File writing and LLM calls live in
`trade_review.py`.

All overlays are additive — a trade can appear on the chart even if the
strategy's metadata is sparse. Missing metadata degrades gracefully to a
plain candle chart with trade markers.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd

from tradingagents.dashboard.chart_builder import TradeChartBuilder


def _is_minervini_variant(variant: str) -> bool:
    return variant in {"mechanical", "llm", "mechanical_v2"}


def _is_chan_variant(variant: str) -> bool:
    return variant in {"chan", "chan_v2"}


def _is_intraday_variant(variant: str) -> bool:
    return variant.startswith("intraday")


def _parse_signal_metadata(raw: Any) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) or {}
    except Exception:
        return {}


def build_trade_chart(
    *,
    symbol: str,
    variant: str,
    bars: pd.DataFrame,
    outcome: dict,
    trades_for_marker: list[dict],
    signal_metadata: Optional[str] = None,
    setup_row: Optional[dict] = None,
    chan_structures: Optional[dict] = None,
):
    """Build a per-trade review chart.

    Parameters
    ----------
    symbol, variant : str
    bars : pd.DataFrame
        OHLCV indexed by datetime.
    outcome : dict
        `trade_outcomes` row (symbol, entry_date, exit_date, prices, MFE/MAE).
    trades_for_marker : list of dicts
        Raw `trades` rows to annotate on the chart (typically the entry BUY
        and the exit SELL for this one pairing).
    signal_metadata : str or None
        JSON string from `signals.signal_metadata` (intraday ORB/NR4/VWAP).
    setup_row : dict or None
        `setup_candidates` row (Minervini pivot / buy_point / stop).
    chan_structures : dict or None
        Output of `extract_chan_structures(...)`, keys: bi_list, seg_list,
        zs_list, bsp_list.
    """
    if bars is None or bars.empty:
        return None

    builder = TradeChartBuilder(symbol, bars)
    builder.add_candlesticks().add_volume()

    if _is_minervini_variant(variant):
        # Daily chart: SMA stack + pivot/buy_point/stop from setup_row.
        builder.add_sma([50, 150, 200])
        if setup_row:
            levels = {
                "buy_point": setup_row.get("buy_point"),
                "buy_limit": setup_row.get("buy_limit_price"),
                "stop_loss": setup_row.get("initial_stop_price")
                or outcome.get("entry_price", 0) * 0.92,
            }
            builder.add_minervini_levels(levels)
        else:
            # No setup row — fall back to approximate stop from entry price.
            entry = outcome.get("entry_price") or 0
            if entry:
                builder.add_minervini_levels(
                    {"buy_point": entry, "stop_loss": entry * 0.92}
                )

    elif _is_chan_variant(variant):
        # Chan overlays are live-extracted at review time — see trade_review.
        if chan_structures:
            bi = chan_structures.get("bi_list") or []
            seg = chan_structures.get("seg_list") or []
            zs = chan_structures.get("zs_list") or []
            bsp = chan_structures.get("bsp_list") or []
            if bi:
                builder.add_chan_bi(bi)
            if seg:
                builder.add_chan_seg(seg)
            if zs:
                builder.add_chan_zs(zs)
            if bsp:
                builder.add_chan_bsp(bsp)

    elif _is_intraday_variant(variant):
        # VWAP always; ORB/NR4 levels from signal_metadata when present.
        builder.add_vwap_line()
        meta = _parse_signal_metadata(signal_metadata)
        orb_high = meta.get("opening_range_high")
        orb_low = meta.get("opening_range_low")
        if orb_high and orb_low:
            builder.add_orb_levels(orb_high, orb_low)
        builder.add_prior_session_levels(
            prior_high=meta.get("prior_session_high"),
            prior_close=meta.get("prior_session_close"),
        )

    # Common: trade entry/exit markers.
    if trades_for_marker:
        builder.add_trade_markers(trades_for_marker)

    fig = builder.build()
    # Title encodes the variant + direction so the HTML is self-describing.
    return_pct = outcome.get("return_pct") or 0
    win_loss = "WIN" if return_pct > 0 else ("LOSS" if return_pct < 0 else "FLAT")
    fig.update_layout(
        title=dict(
            text=(
                f"{symbol} ({variant}) — {outcome.get('entry_date')} → "
                f"{outcome.get('exit_date')}  [{win_loss}] "
                f"{return_pct:+.2%}"
            ),
            x=0.02, xanchor="left",
        )
    )
    return fig
