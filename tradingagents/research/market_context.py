"""Composite market-context helpers for regime-aware backtests."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def _trend_frame(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame = df.copy().sort_index()
    if frame.empty:
        return pd.DataFrame()

    close = frame["close"]
    out = pd.DataFrame(index=frame.index)
    out[f"{prefix}_close"] = close
    out[f"{prefix}_sma_20"] = close.rolling(20).mean()
    out[f"{prefix}_sma_50"] = close.rolling(50).mean()
    out[f"{prefix}_sma_200"] = close.rolling(200).mean()
    out[f"{prefix}_roc_20"] = close.pct_change(20)
    out[f"{prefix}_ema_21"] = close.ewm(span=21, adjust=False).mean()
    out[f"{prefix}_roc_5"] = close.pct_change(5)
    return out


def build_market_context(data_by_symbol: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a composite regime frame from major index and volatility proxies."""
    frames = []
    mapping = {
        "SPY": "spy",
        "QQQ": "qqq",
        "IWM": "iwm",
        "SMH": "smh",
        "^VIX": "vix",
    }
    for symbol, prefix in mapping.items():
        frame = data_by_symbol.get(symbol)
        if frame is None or frame.empty:
            continue
        prepared = _trend_frame(frame, prefix)
        if not prepared.empty:
            frames.append(prepared)

    if not frames:
        return pd.DataFrame()

    context = pd.concat(frames, axis=1).sort_index()

    score = pd.Series(0, index=context.index, dtype="int64")
    if {"spy_close", "spy_sma_50", "spy_sma_200", "spy_roc_20"}.issubset(context.columns):
        score += (context["spy_close"] > context["spy_sma_200"]).astype(int)
        score += (context["spy_sma_50"] > context["spy_sma_200"]).astype(int)
        score += (context["spy_close"] > context["spy_sma_50"]).astype(int)
        score += (context["spy_roc_20"] > 0).astype(int)
    if {"qqq_close", "qqq_sma_200"}.issubset(context.columns):
        score += (context["qqq_close"] > context["qqq_sma_200"]).astype(int)
    if {"iwm_close", "iwm_sma_200"}.issubset(context.columns):
        score += (context["iwm_close"] > context["iwm_sma_200"]).astype(int)
    if {"smh_close", "smh_sma_200"}.issubset(context.columns):
        score += (context["smh_close"] > context["smh_sma_200"]).astype(int)
    if {"vix_close", "vix_sma_20"}.issubset(context.columns):
        score += (context["vix_close"] < context["vix_sma_20"]).astype(int)

    context["market_score"] = score
    context["market_confirmed_uptrend"] = score >= 6
    context["market_regime"] = "market_correction"
    context.loc[score >= 4, "market_regime"] = "uptrend_under_pressure"
    context.loc[score >= 6, "market_regime"] = "confirmed_uptrend"

    output_columns = [
        "market_score",
        "market_confirmed_uptrend",
        "market_regime",
    ]
    # Expose QQQ extension metrics so consumers (e.g. the market-extension
    # entry filter) can gate new entries. Computed by _trend_frame when QQQ
    # is supplied; absent columns simply aren't emitted.
    if {"qqq_close", "qqq_ema_21"}.issubset(context.columns):
        context["qqq_above_ema21_pct"] = (
            context["qqq_close"] / context["qqq_ema_21"] - 1.0
        )
        output_columns.append("qqq_above_ema21_pct")
    if "qqq_roc_5" in context.columns:
        output_columns.append("qqq_roc_5")

    return context[output_columns]
