"""Shared earnings-report blackout helper.

Used by backtesters and (later) live orchestrators to answer:
  "given a candidate bar `ts` for `symbol`, is an earnings report close
   enough that we should skip entry or flatten an existing position?"

Time-of-day handling:
  * time_hint='bmo' (before market open, yfinance): the print lands before
    the ER-day's open. The last safe bar to hold is the prior session's
    close. Equivalently, a BMO print on Friday blackout-blocks Thursday.
  * time_hint='amc' (after market close, yfinance): the print lands after
    the ER-day's close. The last safe bar to hold is the ER-day's close
    itself IS NOT safe — the print happens the same evening, so the
    blackout includes the ER day.
  * time_hint='unknown' (Alpha Vantage or midday yfinance): treat as
    whole-day blackout on both sides — conservative because we don't know
    pre vs post.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd


def _row_to_dict(row, session_date: pd.Timestamp) -> dict:
    return {
        "event_datetime": row["event_datetime"],
        "session_date": session_date,
        "time_hint": row.get("time_hint", "unknown"),
        "source": row.get("source", "unknown"),
    }


class EarningsBlackout:
    """Per-process cache keyed by symbol. Instantiate once per backtest run.

    Pass a ``warehouse`` (MarketDataWarehouse) and the helper queries
    ``earnings_events`` lazily per symbol. Alternatively, pass
    ``events_by_symbol`` directly (pre-queried DataFrames keyed by symbol)
    for tight-loop backtests.
    """

    def __init__(
        self,
        warehouse=None,
        events_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        self._warehouse = warehouse
        self._cache: Dict[str, pd.DataFrame] = dict(events_by_symbol or {})
        self._provided_directly = events_by_symbol is not None

    def _events_for(self, symbol: str) -> pd.DataFrame:
        if symbol in self._cache:
            return self._cache[symbol]
        if self._warehouse is None:
            self._cache[symbol] = pd.DataFrame()
            return self._cache[symbol]
        frame = self._warehouse.get_earnings_events(symbol)
        if frame is None or frame.empty:
            self._cache[symbol] = pd.DataFrame()
        else:
            frame = frame.copy()
            frame["event_datetime"] = pd.to_datetime(frame["event_datetime"])
            self._cache[symbol] = frame
        return self._cache[symbol]

    def is_blackout(
        self,
        symbol: str,
        ts: pd.Timestamp,
        days_before: int = 0,
        days_after: int = 0,
    ) -> Tuple[bool, Optional[dict]]:
        """Return (blocked?, matched_event_dict_or_None).

        Semantics are keyed by time_hint on each event. ``delta_days`` is
        defined as (ER session date - ts's session date):

        * AMC event on day X (print AFTER close of X): close of X is
          pre-print. Block ts's close if ER is today or up to days_before
          calendar days ahead, OR if ER was 1..days_after calendar days
          ago.
        * BMO event on day X (print BEFORE open of X): close of X is
          post-print. Block ts if ER is 1..days_before calendar days
          ahead, OR if ER was 0..days_after-1 calendar days ago
          (X itself counts as 0 days after).
        * Unknown time_hint (e.g. Alpha Vantage rows): conservative —
          symmetric block for delta in [-days_after, +days_before].

        days_before=0 and days_after=0 disables the filter for that side.
        """
        if days_before <= 0 and days_after <= 0:
            return False, None
        events = self._events_for(symbol)
        if events is None or events.empty:
            return False, None

        ts_date = pd.Timestamp(ts).normalize()

        for _, row in events.iterrows():
            ev = pd.Timestamp(row["event_datetime"])
            hint = (row.get("time_hint") or "unknown").lower()
            ev_session_date = ev.normalize()
            delta_days = (ev_session_date - ts_date).days

            if hint == "amc":
                before_hit = days_before >= 1 and 0 <= delta_days <= days_before
                after_hit = days_after >= 1 and -days_after <= delta_days <= -1
            elif hint == "bmo":
                before_hit = days_before >= 1 and 1 <= delta_days <= days_before
                after_hit = days_after >= 1 and -days_after < delta_days <= 0
            else:  # unknown / missing
                before_hit = days_before >= 1 and 0 <= delta_days <= days_before
                after_hit = days_after >= 1 and -days_after <= delta_days <= 0

            if before_hit or after_hit:
                return True, _row_to_dict(row, ev_session_date)

        return False, None

    def is_last_safe_bar_before_er(
        self,
        symbol: str,
        ts: pd.Timestamp,
    ) -> Tuple[bool, Optional[dict]]:
        """Return True iff ts's close is the LAST bar before an ER print.

        Tight just-in-time flatten semantics:
          * AMC on day X: last safe bar is X's close (pre-print). Fires when
            ts's normalized date == X.
          * BMO on day X: last safe bar is X-1's close. Fires when
            ts's normalized date == X - 1 calendar day.
          * Unknown time_hint: conservative — fires on both the ER day
            itself AND the day before (we don't know which is pre-print).

        Returns the matched event dict for telemetry, or (False, None).
        """
        events = self._events_for(symbol)
        if events is None or events.empty:
            return False, None

        ts_date = pd.Timestamp(ts).normalize()

        for _, row in events.iterrows():
            ev = pd.Timestamp(row["event_datetime"])
            hint = (row.get("time_hint") or "unknown").lower()
            ev_session_date = ev.normalize()
            delta_days = (ev_session_date - ts_date).days

            if hint == "amc" and delta_days == 0:
                return True, _row_to_dict(row, ev_session_date)
            if hint == "bmo" and delta_days == 1:
                return True, _row_to_dict(row, ev_session_date)
            if hint not in ("amc", "bmo") and delta_days in (0, 1):
                return True, _row_to_dict(row, ev_session_date)

        return False, None


