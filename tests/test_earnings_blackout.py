"""Boundary tests for the earnings-blackout helper.

BMO semantics: print lands before the open of session D, so the last safe
bar to *hold* is session D-1's close. Entry on D-1's close should be
blocked when days_before >= 1.

AMC semantics: print lands after the close of session D. Session D's close
is the last bar before the print, so holding through D's close is blackout.
Entry on D with days_before >= 1 should be blocked.

Unknown: conservative — treat as whole-day symmetric block.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.research.earnings_blackout import EarningsBlackout


def _mk(symbol: str, rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["event_datetime"] = pd.to_datetime(frame["event_datetime"])
    return frame


def test_disabled_returns_false():
    eb = EarningsBlackout(events_by_symbol={"AAPL": _mk("AAPL", [
        {"event_datetime": "2024-01-25 21:00:00", "time_hint": "amc", "source": "yfinance"},
    ])})
    blocked, _ = eb.is_blackout("AAPL", pd.Timestamp("2024-01-24"), days_before=0, days_after=0)
    assert blocked is False


def test_empty_events_returns_false():
    eb = EarningsBlackout(events_by_symbol={"AAPL": pd.DataFrame()})
    blocked, _ = eb.is_blackout("AAPL", pd.Timestamp("2024-01-24"), days_before=5)
    assert blocked is False


def test_missing_symbol_returns_false():
    eb = EarningsBlackout(events_by_symbol={})
    blocked, _ = eb.is_blackout("MISSING", pd.Timestamp("2024-01-24"), days_before=5)
    assert blocked is False


def test_bmo_blocks_prior_session_close_at_days_before_1():
    # BMO on 2024-01-25: print fires before the open of the 25th.
    # Day-before (24th) close is pre-print → must be blocked when days_before=1.
    # ER-day (25th) close is post-print → NOT blocked on days_before alone.
    eb = EarningsBlackout(events_by_symbol={"AAPL": _mk("AAPL", [
        {"event_datetime": "2024-01-25 12:00:00", "time_hint": "bmo", "source": "yfinance"},
    ])})
    blocked_24, ev_24 = eb.is_blackout("AAPL", pd.Timestamp("2024-01-24"), days_before=1)
    assert blocked_24 is True
    assert ev_24["time_hint"] == "bmo"
    blocked_25, _ = eb.is_blackout("AAPL", pd.Timestamp("2024-01-25"), days_before=1)
    assert blocked_25 is False
    # BMO day itself is "0 days after" — days_after=1 covers it
    blocked_25_after, _ = eb.is_blackout("AAPL", pd.Timestamp("2024-01-25"), days_before=0, days_after=1)
    assert blocked_25_after is True


def test_bmo_does_not_block_more_than_n_days_out():
    eb = EarningsBlackout(events_by_symbol={"AAPL": _mk("AAPL", [
        {"event_datetime": "2024-01-25 12:00:00", "time_hint": "bmo", "source": "yfinance"},
    ])})
    # 4 days before print, days_before=3 → delta=4, upper bound = 3-1 = 2 → NOT blocked
    blocked, _ = eb.is_blackout("AAPL", pd.Timestamp("2024-01-21"), days_before=3)
    assert blocked is False


def test_amc_blocks_day_of_print_at_days_before_1():
    # AMC on 2024-01-25: print after the 25th's close. Holding through 25th's close = blackout.
    eb = EarningsBlackout(events_by_symbol={"AAPL": _mk("AAPL", [
        {"event_datetime": "2024-01-25 21:00:00", "time_hint": "amc", "source": "yfinance"},
    ])})
    blocked_25, ev_25 = eb.is_blackout("AAPL", pd.Timestamp("2024-01-25"), days_before=1)
    assert blocked_25 is True
    assert ev_25["time_hint"] == "amc"


def test_amc_blocks_prior_day_when_window_covers_it():
    eb = EarningsBlackout(events_by_symbol={"AAPL": _mk("AAPL", [
        {"event_datetime": "2024-01-25 21:00:00", "time_hint": "amc", "source": "yfinance"},
    ])})
    # days_before=2: blocks [ts, ts+2] → on 24th, delta=1, within window
    blocked, _ = eb.is_blackout("AAPL", pd.Timestamp("2024-01-24"), days_before=2)
    assert blocked is True
    # days_before=0 (disabled): returns False
    blocked_disabled, _ = eb.is_blackout("AAPL", pd.Timestamp("2024-01-24"), days_before=0)
    assert blocked_disabled is False


def test_unknown_is_symmetric_whole_day_block():
    eb = EarningsBlackout(events_by_symbol={"AAPL": _mk("AAPL", [
        {"event_datetime": "2024-01-25 00:00:00", "time_hint": "unknown", "source": "alphavantage"},
    ])})
    # ER day itself is blocked for any days_before>=0 with days_after>=0
    blocked_same, _ = eb.is_blackout("AAPL", pd.Timestamp("2024-01-25"), days_before=1)
    assert blocked_same is True
    # After the print — days_after controls
    blocked_after_1, _ = eb.is_blackout("AAPL", pd.Timestamp("2024-01-26"), days_before=1, days_after=1)
    assert blocked_after_1 is True
    blocked_after_2, _ = eb.is_blackout("AAPL", pd.Timestamp("2024-01-27"), days_before=1, days_after=1)
    assert blocked_after_2 is False


def test_days_after_window():
    eb = EarningsBlackout(events_by_symbol={"AAPL": _mk("AAPL", [
        {"event_datetime": "2024-01-25 21:00:00", "time_hint": "amc", "source": "yfinance"},
    ])})
    # 2 days after AMC on 25th: 27th is 2 days after
    blocked, _ = eb.is_blackout("AAPL", pd.Timestamp("2024-01-27"), days_before=0, days_after=2)
    assert blocked is True
    blocked_3, _ = eb.is_blackout("AAPL", pd.Timestamp("2024-01-28"), days_before=0, days_after=2)
    assert blocked_3 is False


def test_last_safe_bar_amc_fires_on_print_day():
    # AMC print lands after close of X. Last safe bar = X's close.
    eb = EarningsBlackout(events_by_symbol={"AAPL": _mk("AAPL", [
        {"event_datetime": "2024-01-25 21:00:00", "time_hint": "amc", "source": "yfinance"},
    ])})
    assert eb.is_last_safe_bar_before_er("AAPL", pd.Timestamp("2024-01-25"))[0] is True
    assert eb.is_last_safe_bar_before_er("AAPL", pd.Timestamp("2024-01-24"))[0] is False
    assert eb.is_last_safe_bar_before_er("AAPL", pd.Timestamp("2024-01-26"))[0] is False


def test_last_safe_bar_bmo_fires_on_day_prior():
    # BMO print lands before open of X. Last safe bar = X-1's close.
    eb = EarningsBlackout(events_by_symbol={"AAPL": _mk("AAPL", [
        {"event_datetime": "2024-01-25 12:00:00", "time_hint": "bmo", "source": "yfinance"},
    ])})
    assert eb.is_last_safe_bar_before_er("AAPL", pd.Timestamp("2024-01-24"))[0] is True
    assert eb.is_last_safe_bar_before_er("AAPL", pd.Timestamp("2024-01-25"))[0] is False
    assert eb.is_last_safe_bar_before_er("AAPL", pd.Timestamp("2024-01-23"))[0] is False


def test_last_safe_bar_unknown_fires_on_both_candidates():
    # Unknown hint: conservative — either X or X-1 is treated as last safe.
    eb = EarningsBlackout(events_by_symbol={"AAPL": _mk("AAPL", [
        {"event_datetime": "2024-01-25 00:00:00", "time_hint": "unknown", "source": "alphavantage"},
    ])})
    assert eb.is_last_safe_bar_before_er("AAPL", pd.Timestamp("2024-01-25"))[0] is True
    assert eb.is_last_safe_bar_before_er("AAPL", pd.Timestamp("2024-01-24"))[0] is True
    assert eb.is_last_safe_bar_before_er("AAPL", pd.Timestamp("2024-01-23"))[0] is False


def test_last_safe_bar_no_match():
    eb = EarningsBlackout(events_by_symbol={"AAPL": pd.DataFrame()})
    assert eb.is_last_safe_bar_before_er("AAPL", pd.Timestamp("2024-01-25"))[0] is False


def test_picks_closest_event_when_multiple():
    eb = EarningsBlackout(events_by_symbol={"AAPL": _mk("AAPL", [
        {"event_datetime": "2023-10-25 21:00:00", "time_hint": "amc", "source": "yfinance"},
        {"event_datetime": "2024-01-25 21:00:00", "time_hint": "amc", "source": "yfinance"},
    ])})
    # Any event within the window triggers — just assert we don't crash and
    # get a valid hit for the near one.
    blocked, ev = eb.is_blackout("AAPL", pd.Timestamp("2024-01-24"), days_before=3)
    assert blocked is True
    assert pd.Timestamp(ev["event_datetime"]).year == 2024
