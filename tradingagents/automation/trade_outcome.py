"""Shared trade-outcome logging + MFE/MAE computation.

Used by all orchestrator exit paths so every closed trade produces a
`trade_outcomes` row with consistent fields. Previously only
`Orchestrator._log_trade_outcome` wrote to the table; now Chan and
intraday exits call the same helper.

Kill switches respected:
  - `trade_outcome_excursion_enabled` (existing) — gates Alpaca hourly-bar
    fetch for MFE/MAE. When disabled, MFE/MAE are written as NULL and no
    network call fires.
  - `trade_outcome_live_hook_chan_enabled`, `trade_outcome_live_hook_intraday_enabled`
    — gate the call-site entirely, for paranoid one-variant rollouts.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_excursion(
    data_client: Any,
    symbol: str,
    entry_date_str: str,
    exit_date_str: str,
    entry_price: float,
    *,
    enabled: bool = True,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute max-favorable / max-adverse excursion over a trade window.

    Uses hourly bars from Alpaca via `data_client`. Returns (mfe_pct, mae_pct)
    as decimals (0.05 == +5%). Fails soft: returns (None, None) on any
    error, missing data, or when `enabled=False`.

    `data_client` is expected to be an Alpaca `StockHistoricalDataClient`
    (the `broker.data_client` attribute on `AlpacaBroker`). Passing None
    returns (None, None).
    """
    if not enabled:
        return (None, None)
    if data_client is None or entry_price <= 0 or not entry_date_str:
        return (None, None)
    try:
        from dateutil.parser import parse as parse_date
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        start = parse_date(entry_date_str)
        end = parse_date(exit_date_str) + timedelta(days=1)
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Hour,
            start=start,
            end=end,
            # Paper-account credentials require IEX feed; SIP is gated.
            feed="iex",
        )
        bars = data_client.get_stock_bars(req)
        series = getattr(bars, "data", {}).get(symbol) or []
        if not series:
            return (None, None)
        max_high = max(float(b.high) for b in series)
        min_low = min(float(b.low) for b in series)
        mfe = (max_high - entry_price) / entry_price
        mae = (min_low - entry_price) / entry_price
        return (round(mfe, 4), round(mae, 4))
    except Exception as e:
        logger.warning("compute_excursion(%s) failed: %s", symbol, e)
        return (None, None)


def _fallback_entry_context(db: Any, symbol: str, entry_date_str: str) -> dict:
    """Look up the most recent setup_candidates row for `symbol` on or before
    `entry_date_str`, returning Minervini entry-time context fields.

    Used as a fallback when `position_states.regime_at_entry` etc. are NULL
    (e.g. when the in-memory preflight cache was empty at entry time, or the
    scheduler restarted between preflight and entry). For chan / intraday
    variants there will be no setup_candidates row, so all fields stay None.
    """
    if not entry_date_str:
        return {}
    try:
        row = db.conn.execute(
            """SELECT base_label, stage_number, rs_percentile, market_regime
               FROM setup_candidates
               WHERE symbol = ? AND screen_date <= ?
               ORDER BY screen_date DESC LIMIT 1""",
            (symbol, entry_date_str),
        ).fetchone()
    except Exception:
        return {}
    if row is None:
        return {}
    r = dict(row)
    out = {}
    if r.get("base_label") is not None:
        out["base_pattern"] = r["base_label"]
    if r.get("stage_number") is not None:
        try:
            out["stage_at_entry"] = float(r["stage_number"])
        except (TypeError, ValueError):
            pass
    if r.get("rs_percentile") is not None:
        try:
            out["rs_at_entry"] = float(r["rs_percentile"])
        except (TypeError, ValueError):
            pass
    if r.get("market_regime") is not None:
        out["regime_at_entry"] = r["market_regime"]
    return out


def log_closed_trade(
    *,
    db: Any,
    symbol: str,
    pos_state: dict,
    exit_price: float,
    exit_reason: str,
    broker: Any = None,
    excursion_enabled: bool = True,
) -> Optional[int]:
    """Write one `trade_outcomes` row for a just-closed trade.

    Call AFTER the SELL has filled and BEFORE `delete_position_state`.
    Reads entry-time context directly from `pos_state` (entry_price,
    entry_date, regime_at_entry, base_pattern, rs_at_entry, stage_at_entry).
    For Minervini variants, falls back to the `setup_candidates` table when
    pos_state lacks these fields (e.g. preflight cache was empty at entry,
    scheduler restart, or a code path that bypassed entry_context capture).

    Returns the newly-inserted outcome row id, or None on failure. Never
    raises — the exit path must not be blocked by outcome bookkeeping.
    """
    try:
        entry_price = float(pos_state.get("entry_price") or 0)
        if entry_price <= 0:
            return None
        entry_date_str = pos_state.get("entry_date") or ""
        exit_date_str = date.today().isoformat()

        try:
            from dateutil.parser import parse as parse_date
            hold_days = (date.today() - parse_date(entry_date_str).date()).days
        except Exception:
            hold_days = 0

        return_pct = (exit_price - entry_price) / entry_price

        data_client = getattr(broker, "data_client", None) if broker is not None else None
        mfe, mae = compute_excursion(
            data_client=data_client,
            symbol=symbol,
            entry_date_str=entry_date_str,
            exit_date_str=exit_date_str,
            entry_price=entry_price,
            enabled=excursion_enabled,
        )

        # Fall back to setup_candidates lookup for any Minervini context
        # field that pos_state doesn't have. Chan/intraday writes hit this
        # function too — they just won't have a setup_candidates row, so
        # the lookup returns {} and pos_state values (including None) win.
        fallback = _fallback_entry_context(db, symbol, entry_date_str)
        def _pick(field: str):
            v = pos_state.get(field)
            if v is None:
                return fallback.get(field)
            return v

        return db.log_trade_outcome({
            "symbol": symbol,
            "entry_date": entry_date_str,
            "exit_date": exit_date_str,
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "return_pct": round(return_pct, 4),
            "hold_days": hold_days,
            "exit_reason": exit_reason or "closed",
            "base_pattern": _pick("base_pattern"),
            "stage_at_entry": _pick("stage_at_entry"),
            "rs_at_entry": _pick("rs_at_entry"),
            "regime_at_entry": _pick("regime_at_entry"),
            "max_favorable_excursion": mfe,
            "max_adverse_excursion": mae,
        })
    except Exception as e:
        logger.warning("log_closed_trade(%s) failed: %s", symbol, e)
        return None
