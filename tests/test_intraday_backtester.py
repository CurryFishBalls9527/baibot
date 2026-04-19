import pandas as pd

from tradingagents.research.intraday_backtester import (
    IntradayBacktestConfig,
    IntradayBreakoutBacktester,
)


def test_filter_regular_session_keeps_only_rth_bars():
    index = pd.to_datetime(
        [
            "2026-04-01 07:30:00",
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-01 15:00:00",
            "2026-04-01 15:30:00",
        ]
    )
    frame = pd.DataFrame({"close": [1, 2, 3, 4, 5]}, index=index)

    filtered = IntradayBreakoutBacktester._filter_regular_session(frame)

    assert list(filtered.index.strftime("%H:%M")) == ["08:30", "09:00", "15:00"]


def test_prepare_symbol_features_does_not_leak_next_bar_across_sessions():
    config = IntradayBacktestConfig(opening_range_bars=1, min_volume_ratio=0.0)
    backtester = IntradayBreakoutBacktester(config)
    index = pd.to_datetime(
        [
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-02 08:30:00",
            "2026-04-02 09:00:00",
        ]
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 102.0, 104.0, 106.0],
            "high": [101.0, 103.0, 105.0, 107.0],
            "low": [99.0, 101.0, 103.0, 105.0],
            "close": [100.5, 102.5, 104.5, 106.5],
            "volume": [1000, 1200, 1100, 1300],
        },
        index=index,
    )

    prepared = backtester._prepare_symbol_features(frame)

    assert pd.isna(prepared.loc[pd.Timestamp("2026-04-01 09:00:00"), "next_open"])
    assert pd.isna(prepared.loc[pd.Timestamp("2026-04-01 09:00:00"), "next_ts"])
    assert prepared.loc[pd.Timestamp("2026-04-02 08:30:00"), "bar_in_session"] == 0
    assert prepared.loc[pd.Timestamp("2026-04-02 08:30:00"), "opening_range_high"] == 105.0


def test_prepare_symbol_features_marks_last_bar_per_session():
    config = IntradayBacktestConfig(opening_range_bars=1, min_volume_ratio=0.0)
    backtester = IntradayBreakoutBacktester(config)
    index = pd.to_datetime(
        [
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-02 08:30:00",
            "2026-04-02 09:00:00",
        ]
    )
    frame = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [10.5, 11.5, 12.5, 13.5],
            "low": [9.5, 10.5, 11.5, 12.5],
            "close": [10.2, 11.2, 12.2, 13.2],
            "volume": [1000, 1000, 1000, 1000],
        },
        index=index,
    )

    prepared = backtester._prepare_symbol_features(frame)

    assert bool(prepared.loc[pd.Timestamp("2026-04-01 08:30:00"), "is_last_bar"]) is False
    assert bool(prepared.loc[pd.Timestamp("2026-04-01 09:00:00"), "is_last_bar"]) is True
    assert bool(prepared.loc[pd.Timestamp("2026-04-02 08:30:00"), "is_last_bar"]) is False
    assert bool(prepared.loc[pd.Timestamp("2026-04-02 09:00:00"), "is_last_bar"]) is True


def test_prepare_symbol_features_assigns_setup_family_labels():
    config = IntradayBacktestConfig(opening_range_bars=1, min_volume_ratio=0.0)
    backtester = IntradayBreakoutBacktester(config)
    index = pd.date_range("2026-04-01 08:30:00", periods=23, freq="30min")
    frame = pd.DataFrame(
        {
            "open": [100.0] * 23,
            "high": [100.0] * 20 + [101.0, 102.0, 103.0],
            "low": [99.0] * 23,
            "close": [100.0] * 20 + [100.6, 101.3, 102.0],
            "volume": [1000] * 23,
        },
        index=index,
    )

    prepared = backtester._prepare_symbol_features(frame)

    assert prepared.iloc[20]["setup_family"] == "opening_drive_continuation"
    assert prepared.iloc[21]["setup_family"] == "opening_drive_expansion"
    assert prepared.iloc[22]["setup_family"] == "opening_drive_overextended"
    assert prepared.iloc[20]["candidate_score"] > 0


def test_prepare_symbol_features_can_convert_expansion_to_confirmation_entry():
    config = IntradayBacktestConfig(
        opening_range_bars=1,
        min_volume_ratio=0.0,
        require_above_vwap=False,
        use_expansion_confirmation_entry=True,
        expansion_confirmation_max_pullback_pct=0.01,
        expansion_confirmation_reclaim_buffer_pct=0.0,
        expansion_confirmation_max_bars_after_signal=3,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.date_range("2026-04-01 08:30:00", periods=23, freq="30min")
    frame = pd.DataFrame(
        {
            "open": [100.0] * 18 + [100.0, 100.5, 100.8, 101.0, 101.5],
            "high": [100.0] * 18 + [100.0, 101.4, 101.1, 101.7, 101.8],
            "low": [99.0] * 18 + [99.0, 100.4, 100.7, 100.9, 101.4],
            "close": [100.0] * 18 + [100.0, 101.3, 100.9, 101.6, 101.7],
            "volume": [1000] * 23,
        },
        index=index,
    )

    prepared = backtester._prepare_symbol_features(frame)

    assert prepared.iloc[19]["setup_family"] == "waiting_expansion_confirmation"
    assert bool(prepared.iloc[19]["entry_signal"]) is False
    assert prepared.iloc[21]["setup_family"] == "opening_drive_expansion_confirmation"
    assert bool(prepared.iloc[21]["entry_signal"]) is True


def test_prepare_symbol_features_can_fail_expansion_confirmation_if_pullback_breaks_structure():
    config = IntradayBacktestConfig(
        opening_range_bars=1,
        min_volume_ratio=0.0,
        require_above_vwap=False,
        use_expansion_confirmation_entry=True,
        expansion_confirmation_max_pullback_pct=0.005,
        expansion_confirmation_reclaim_buffer_pct=0.0,
        expansion_confirmation_max_bars_after_signal=3,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.date_range("2026-04-01 08:30:00", periods=22, freq="30min")
    frame = pd.DataFrame(
        {
            "open": [100.0] * 18 + [100.0, 100.5, 100.7, 101.4],
            "high": [100.0] * 18 + [100.0, 101.4, 100.9, 101.8],
            "low": [99.0] * 18 + [99.0, 100.4, 100.3, 101.2],
            "close": [100.0] * 18 + [100.0, 101.3, 100.4, 101.7],
            "volume": [1000] * 22,
        },
        index=index,
    )

    prepared = backtester._prepare_symbol_features(frame)

    assert prepared.iloc[19]["setup_family"] == "waiting_expansion_confirmation"
    assert prepared.iloc[21]["setup_family"] != "opening_drive_expansion_confirmation"


def test_backtest_can_trade_expansion_confirmation_entry(monkeypatch):
    config = IntradayBacktestConfig(
        min_volume_ratio=0.0,
        require_above_vwap=False,
        max_trades_per_symbol_per_day=1,
        stop_loss_pct=0.0,
        trail_stop_pct=0.0,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.to_datetime(
        [
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-01 09:30:00",
            "2026-04-01 10:00:00",
            "2026-04-01 10:30:00",
        ]
    )
    frame = pd.DataFrame({"open": [100.0, 100.5, 100.8, 101.0, 101.5]}, index=index)
    prepared = pd.DataFrame(
        {
            "open": [100.0, 100.5, 100.8, 101.0, 101.5],
            "high": [100.0, 101.4, 101.1, 101.7, 101.8],
            "low": [99.0, 100.4, 100.7, 100.9, 101.4],
            "close": [100.0, 101.3, 100.9, 101.6, 101.7],
            "volume": [1000] * 5,
            "session_date": [index[0].date()] * 5,
            "bar_in_session": [0, 1, 2, 3, 4],
            "next_open": [100.5, 100.8, 101.0, 101.5, pd.NA],
            "next_ts": [index[1], index[2], index[3], index[4], pd.NaT],
            "is_last_bar": [False, False, False, False, True],
            "opening_range_high": [100.0] * 5,
            "opening_range_low": [99.0] * 5,
            "opening_range_pct": [0.01] * 5,
            "breakout_distance_pct": [0.0, 0.013, 0.009, 0.016, 0.017],
            "distance_from_vwap_pct": [0.0, 0.011, 0.004, 0.012, 0.012],
            "prior_bar_high": [pd.NA, 100.0, 101.4, 101.1, 101.7],
            "prior_bar_low": [pd.NA, 99.0, 100.4, 100.7, 100.9],
            "vwap": [100.0, 100.3, 100.8, 101.0, 101.3],
            "avg_volume_20": [1000.0] * 5,
            "volume_ratio": [1.0, 2.0, 1.2, 1.8, 1.3],
            "entry_signal": [False, False, False, True, False],
            "base_entry_signal": [False, True, False, False, False],
            "setup_family": [
                "none",
                "waiting_expansion_confirmation",
                "none",
                "opening_drive_expansion_confirmation",
                "none",
            ],
            "base_setup_family": [
                "none",
                "opening_drive_expansion",
                "none",
                "none",
                "none",
            ],
            "candidate_score": [0.0, 2.0, 0.0, 2.2, 0.0],
            "base_candidate_score": [0.0, 2.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )

    monkeypatch.setattr(backtester, "_load_intraday_bars", lambda db_path, symbols, begin, end: {"AAA": frame})
    monkeypatch.setattr(backtester, "_load_daily_trend_filter", lambda symbols, begin, end: {})
    monkeypatch.setattr(backtester, "_prepare_symbol_features", lambda df: prepared)

    result = backtester.backtest_portfolio(["AAA"], "2026-04-01", "2026-04-01", "dummy.duckdb")

    assert result.summary["candidates_seen"] == 1
    assert result.summary["candidates_selected"] == 1
    assert result.candidate_log.iloc[0]["setup_family"] == "opening_drive_expansion_confirmation"
    assert result.trades.iloc[0]["setup_family"] == "opening_drive_expansion_confirmation"


def test_backtest_rejects_candidates_after_latest_entry_bar(monkeypatch):
    config = IntradayBacktestConfig(
        min_volume_ratio=0.0,
        require_above_vwap=False,
        latest_entry_bar_in_session=2,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.to_datetime(
        [
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-01 09:30:00",
        ]
    )
    frame = pd.DataFrame({"open": [100.0, 100.0, 100.0]}, index=index)
    prepared = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 101.0, 102.0],
            "low": [99.0, 100.0, 100.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1000, 1000, 1000],
            "session_date": [index[0].date()] * 3,
            "bar_in_session": [0, 1, 2],
            "next_open": [100.0, 100.0, pd.NA],
            "next_ts": [index[1], index[2], pd.NaT],
            "is_last_bar": [False, False, True],
            "opening_range_high": [100.0, 100.0, 100.0],
            "opening_range_low": [99.0, 99.0, 99.0],
            "opening_range_pct": [0.01, 0.01, 0.01],
            "breakout_distance_pct": [0.0, 0.01, 0.02],
            "prior_bar_high": [pd.NA, 100.0, 101.0],
            "vwap": [100.0, 100.0, 100.0],
            "avg_volume_20": [1000.0, 1000.0, 1000.0],
            "volume_ratio": [1.0, 1.0, 1.0],
            "entry_signal": [False, False, True],
        },
        index=index,
    )

    monkeypatch.setattr(
        backtester,
        "_load_intraday_bars",
        lambda db_path, symbols, begin, end: {"AAA": frame},
    )
    monkeypatch.setattr(backtester, "_load_daily_trend_filter", lambda symbols, begin, end: {})
    monkeypatch.setattr(backtester, "_prepare_symbol_features", lambda df: prepared)

    result = backtester.backtest_portfolio(["AAA"], "2026-04-01", "2026-04-01", "dummy.duckdb")

    assert result.summary["candidates_seen"] == 1
    assert result.summary["candidates_selected"] == 0
    reasons = result.candidate_log["filter_reason"].value_counts().to_dict()
    assert reasons["latest_entry_bar"] == 1


def test_backtest_limits_trades_per_symbol_per_day(monkeypatch):
    config = IntradayBacktestConfig(
        min_volume_ratio=0.0,
        require_above_vwap=False,
        max_trades_per_symbol_per_day=1,
        stop_loss_pct=0.0,
        trail_stop_pct=0.0,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.to_datetime(
        [
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-01 09:30:00",
            "2026-04-01 10:00:00",
        ]
    )
    frame = pd.DataFrame({"open": [100.0, 101.0, 100.0, 101.0]}, index=index)
    prepared = pd.DataFrame(
        {
            "open": [100.0, 101.0, 100.0, 101.0],
            "high": [100.0, 102.0, 100.0, 102.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 101.0, 100.0, 101.0],
            "volume": [1000, 1000, 1000, 1000],
            "session_date": [index[0].date()] * 4,
            "bar_in_session": [0, 1, 2, 3],
            "next_open": [101.0, 100.0, 101.0, pd.NA],
            "next_ts": [index[1], index[2], index[3], pd.NaT],
            "is_last_bar": [False, False, False, True],
            "opening_range_high": [100.0, 100.0, 100.0, 100.0],
            "opening_range_low": [99.0, 99.0, 99.0, 99.0],
            "opening_range_pct": [0.01, 0.01, 0.01, 0.01],
            "breakout_distance_pct": [0.0, 0.01, 0.0, 0.01],
            "prior_bar_high": [pd.NA, 100.0, 102.0, 100.0],
            "vwap": [100.0, 100.0, 100.0, 100.0],
            "avg_volume_20": [1000.0, 1000.0, 1000.0, 1000.0],
            "volume_ratio": [1.0, 1.0, 1.0, 1.0],
            "entry_signal": [False, True, False, True],
        },
        index=index,
    )

    monkeypatch.setattr(
        backtester,
        "_load_intraday_bars",
        lambda db_path, symbols, begin, end: {"AAA": frame},
    )
    monkeypatch.setattr(backtester, "_load_daily_trend_filter", lambda symbols, begin, end: {})
    monkeypatch.setattr(backtester, "_prepare_symbol_features", lambda df: prepared)

    result = backtester.backtest_portfolio(["AAA"], "2026-04-01", "2026-04-01", "dummy.duckdb")

    assert result.summary["total_trades"] == 1
    assert result.summary["candidates_seen"] == 2
    assert result.summary["candidates_selected"] == 1
    reasons = result.candidate_log["filter_reason"].value_counts().to_dict()
    assert reasons["queued_next_bar_open"] == 1
    assert reasons["max_trades_per_symbol_day"] == 1


def test_backtest_candidate_log_includes_setup_family(monkeypatch):
    config = IntradayBacktestConfig(
        min_volume_ratio=0.0,
        require_above_vwap=False,
        max_trades_per_symbol_per_day=1,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.to_datetime(
        [
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-01 09:30:00",
        ]
    )
    frame = pd.DataFrame({"open": [100.0, 100.0, 100.0]}, index=index)
    prepared = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 101.0, 101.0],
            "low": [99.0, 100.0, 100.0],
            "close": [100.0, 100.8, 100.9],
            "volume": [1000, 1000, 1000],
            "session_date": [index[0].date()] * 3,
            "bar_in_session": [0, 1, 2],
            "next_open": [100.0, 100.0, pd.NA],
            "next_ts": [index[1], index[2], pd.NaT],
            "is_last_bar": [False, False, True],
            "opening_range_high": [100.0, 100.0, 100.0],
            "opening_range_low": [99.0, 99.0, 99.0],
            "opening_range_pct": [0.01, 0.01, 0.01],
            "breakout_distance_pct": [0.0, 0.008, 0.009],
            "distance_from_vwap_pct": [0.0, 0.006, 0.007],
            "prior_bar_high": [pd.NA, 100.0, 101.0],
            "vwap": [100.0, 100.2, 100.2],
            "avg_volume_20": [1000.0, 1000.0, 1000.0],
            "volume_ratio": [1.0, 1.0, 1.0],
            "entry_signal": [False, True, False],
            "setup_family": ["none", "opening_drive_continuation", "none"],
            "candidate_score": [0.0, 1.2, 0.0],
        },
        index=index,
    )

    monkeypatch.setattr(
        backtester,
        "_load_intraday_bars",
        lambda db_path, symbols, begin, end: {"AAA": frame},
    )
    monkeypatch.setattr(backtester, "_load_daily_trend_filter", lambda symbols, begin, end: {})
    monkeypatch.setattr(backtester, "_prepare_symbol_features", lambda df: prepared)

    result = backtester.backtest_portfolio(["AAA"], "2026-04-01", "2026-04-01", "dummy.duckdb")

    assert result.summary["setup_family_counts"]["opening_drive_continuation"] == 1
    assert result.summary["selected_setup_family_counts"]["opening_drive_continuation"] == 1
    row = result.candidate_log.iloc[0]
    assert row["setup_family"] == "opening_drive_continuation"
    assert row["candidate_score"] == 1.2


def test_backtest_trades_include_setup_family_attribution(monkeypatch):
    config = IntradayBacktestConfig(
        min_volume_ratio=0.0,
        require_above_vwap=False,
        max_trades_per_symbol_per_day=1,
        stop_loss_pct=0.0,
        trail_stop_pct=0.0,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.to_datetime(
        [
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-01 09:30:00",
        ]
    )
    frame = pd.DataFrame({"open": [100.0, 100.0, 101.0]}, index=index)
    prepared = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0],
            "high": [100.0, 101.0, 101.0],
            "low": [99.0, 100.0, 100.0],
            "close": [100.0, 100.8, 101.0],
            "volume": [1000, 1000, 1000],
            "session_date": [index[0].date()] * 3,
            "bar_in_session": [0, 1, 2],
            "next_open": [100.0, 101.0, pd.NA],
            "next_ts": [index[1], index[2], pd.NaT],
            "is_last_bar": [False, False, True],
            "opening_range_high": [100.0, 100.0, 100.0],
            "opening_range_low": [99.0, 99.0, 99.0],
            "opening_range_pct": [0.01, 0.01, 0.01],
            "breakout_distance_pct": [0.0, 0.008, 0.01],
            "distance_from_vwap_pct": [0.0, 0.006, 0.006],
            "prior_bar_high": [pd.NA, 100.0, 101.0],
            "vwap": [100.0, 100.2, 100.4],
            "avg_volume_20": [1000.0, 1000.0, 1000.0],
            "volume_ratio": [1.0, 1.0, 1.0],
            "entry_signal": [False, True, False],
            "setup_family": ["none", "opening_drive_continuation", "none"],
            "candidate_score": [0.0, 1.2, 0.0],
        },
        index=index,
    )

    monkeypatch.setattr(
        backtester,
        "_load_intraday_bars",
        lambda db_path, symbols, begin, end: {"AAA": frame},
    )
    monkeypatch.setattr(backtester, "_load_daily_trend_filter", lambda symbols, begin, end: {})
    monkeypatch.setattr(backtester, "_prepare_symbol_features", lambda df: prepared)

    result = backtester.backtest_portfolio(["AAA"], "2026-04-01", "2026-04-01", "dummy.duckdb")

    trade = result.trades.iloc[0]
    assert trade["setup_family"] == "opening_drive_continuation"
    assert trade["candidate_score"] == 1.2
    assert trade["signal_time"] == "2026-04-01T09:00:00"
    assert result.summary["trade_setup_family_counts"]["opening_drive_continuation"] == 1
    assert result.setup_summary.iloc[0]["setup_family"] == "opening_drive_continuation"


def test_backtest_filters_continuation_by_setup_specific_rules(monkeypatch):
    config = IntradayBacktestConfig(
        min_volume_ratio=0.0,
        require_above_vwap=False,
        continuation_min_volume_ratio=1.5,
        continuation_max_distance_from_vwap_pct=0.005,
        continuation_latest_entry_bar_in_session=3,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.to_datetime(
        [
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-01 09:30:00",
            "2026-04-01 10:00:00",
        ]
    )
    frame_a = pd.DataFrame({"open": [100.0, 100.0, 100.0, 100.0]}, index=index)
    prepared_a = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 101.0, 101.0, 101.0],
            "low": [99.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.8, 100.9, 100.7],
            "volume": [1000, 1000, 1000, 1000],
            "session_date": [index[0].date()] * 4,
            "bar_in_session": [0, 1, 2, 3],
            "next_open": [100.0, 100.0, 100.0, pd.NA],
            "next_ts": [index[1], index[2], index[3], pd.NaT],
            "is_last_bar": [False, False, False, True],
            "opening_range_high": [100.0, 100.0, 100.0, 100.0],
            "opening_range_low": [99.0, 99.0, 99.0, 99.0],
            "opening_range_pct": [0.01, 0.01, 0.01, 0.01],
            "breakout_distance_pct": [0.0, 0.008, 0.009, 0.007],
            "distance_from_vwap_pct": [0.0, 0.006, 0.004, 0.004],
            "prior_bar_high": [pd.NA, 100.0, 101.0, 101.0],
            "vwap": [100.0, 100.2, 100.4, 100.4],
            "avg_volume_20": [1000.0, 1000.0, 1000.0, 1000.0],
            "volume_ratio": [1.0, 1.4, 1.6, 1.6],
            "entry_signal": [False, True, True, True],
            "setup_family": [
                "none",
                "opening_drive_continuation",
                "opening_drive_continuation",
                "opening_drive_continuation",
            ],
            "candidate_score": [0.0, 1.0, 1.1, 1.2],
        },
        index=index,
    )
    frame_b = pd.DataFrame({"open": [50.0, 50.0, 50.0, 50.0]}, index=index)
    prepared_b = pd.DataFrame(
        {
            "open": [50.0, 50.0, 50.0, 50.0],
            "high": [50.0, 50.5, 50.5, 50.5],
            "low": [49.0, 50.0, 50.0, 50.0],
            "close": [50.0, 50.2, 50.2, 50.3],
            "volume": [1000, 1000, 1000, 1000],
            "session_date": [index[0].date()] * 4,
            "bar_in_session": [0, 1, 2, 3],
            "next_open": [50.0, 50.0, 50.0, pd.NA],
            "next_ts": [index[1], index[2], index[3], pd.NaT],
            "is_last_bar": [False, False, False, True],
            "opening_range_high": [50.0, 50.0, 50.0, 50.0],
            "opening_range_low": [49.0, 49.0, 49.0, 49.0],
            "opening_range_pct": [0.02, 0.02, 0.02, 0.02],
            "breakout_distance_pct": [0.0, 0.004, 0.004, 0.006],
            "distance_from_vwap_pct": [0.0, 0.003, 0.003, 0.004],
            "prior_bar_high": [pd.NA, 50.0, 50.5, 50.5],
            "vwap": [50.0, 50.1, 50.1, 50.1],
            "avg_volume_20": [1000.0, 1000.0, 1000.0, 1000.0],
            "volume_ratio": [1.0, 1.6, 1.6, 1.6],
            "entry_signal": [False, False, False, True],
            "setup_family": [
                "none",
                "opening_drive_continuation",
                "opening_drive_continuation",
                "opening_drive_continuation",
            ],
            "candidate_score": [0.0, 1.0, 1.1, 1.2],
        },
        index=index,
    )

    monkeypatch.setattr(
        backtester,
        "_load_intraday_bars",
        lambda db_path, symbols, begin, end: {"AAA": frame_a, "BBB": frame_b},
    )
    monkeypatch.setattr(backtester, "_load_daily_trend_filter", lambda symbols, begin, end: {})
    monkeypatch.setattr(
        backtester,
        "_prepare_symbol_features",
        lambda df: prepared_a if float(df.iloc[0]["open"]) == 100.0 else prepared_b,
    )

    result = backtester.backtest_portfolio(["AAA", "BBB"], "2026-04-01", "2026-04-01", "dummy.duckdb")

    reasons = result.candidate_log["filter_reason"].value_counts().to_dict()
    assert reasons["continuation_volume_ratio"] == 1
    assert reasons["queued_next_bar_open"] == 1
    assert reasons["continuation_latest_entry_bar"] == 1


def test_backtest_filters_expansion_by_setup_specific_rules(monkeypatch):
    config = IntradayBacktestConfig(
        min_volume_ratio=0.0,
        require_above_vwap=False,
        expansion_min_volume_ratio=1.8,
        expansion_min_breakout_distance_pct=0.015,
        expansion_min_distance_from_vwap_pct=0.01,
        expansion_latest_entry_bar_in_session=4,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.to_datetime(
        [
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-01 09:30:00",
            "2026-04-01 10:00:00",
            "2026-04-01 10:30:00",
        ]
    )
    frame_a = pd.DataFrame({"open": [100.0] * 5}, index=index)
    prepared_a = pd.DataFrame(
        {
            "open": [100.0] * 5,
            "high": [100.0, 101.0, 101.5, 102.0, 102.0],
            "low": [99.0, 100.0, 100.0, 101.0, 101.0],
            "close": [100.0, 101.2, 101.7, 101.8, 101.5],
            "volume": [1000] * 5,
            "session_date": [index[0].date()] * 5,
            "bar_in_session": [0, 1, 2, 3, 4],
            "next_open": [100.0, 100.0, 100.0, 100.0, pd.NA],
            "next_ts": [index[1], index[2], index[3], index[4], pd.NaT],
            "is_last_bar": [False, False, False, False, True],
            "opening_range_high": [100.0] * 5,
            "opening_range_low": [99.0] * 5,
            "opening_range_pct": [0.01] * 5,
            "breakout_distance_pct": [0.0, 0.016, 0.016, 0.018, 0.017],
            "distance_from_vwap_pct": [0.0, 0.009, 0.011, 0.012, 0.012],
            "prior_bar_high": [pd.NA, 100.0, 101.0, 101.5, 102.0],
            "vwap": [100.0, 100.3, 100.6, 100.8, 101.0],
            "avg_volume_20": [1000.0] * 5,
            "volume_ratio": [1.0, 1.9, 1.7, 2.0, 2.0],
            "entry_signal": [False, True, True, True, True],
            "setup_family": [
                "none",
                "opening_drive_expansion",
                "opening_drive_expansion",
                "opening_drive_expansion",
                "opening_drive_expansion",
            ],
            "candidate_score": [0.0, 1.0, 1.1, 1.2, 1.3],
        },
        index=index,
    )
    frame_b = pd.DataFrame({"open": [50.0] * 5}, index=index)
    prepared_b = pd.DataFrame(
        {
            "open": [50.0] * 5,
            "high": [50.0, 50.0, 50.0, 50.0, 51.0],
            "low": [49.0, 49.0, 49.0, 49.0, 50.0],
            "close": [50.0, 50.0, 50.0, 50.0, 50.9],
            "volume": [1000] * 5,
            "session_date": [index[0].date()] * 5,
            "bar_in_session": [0, 1, 2, 3, 4],
            "next_open": [50.0, 50.0, 50.0, 50.0, pd.NA],
            "next_ts": [index[1], index[2], index[3], index[4], pd.NaT],
            "is_last_bar": [False, False, False, False, True],
            "opening_range_high": [50.0] * 5,
            "opening_range_low": [49.0] * 5,
            "opening_range_pct": [0.02] * 5,
            "breakout_distance_pct": [0.0, 0.0, 0.0, 0.0, 0.018],
            "distance_from_vwap_pct": [0.0, 0.0, 0.0, 0.0, 0.012],
            "prior_bar_high": [pd.NA, 50.0, 50.0, 50.0, 50.0],
            "vwap": [50.0, 50.0, 50.0, 50.0, 50.4],
            "avg_volume_20": [1000.0] * 5,
            "volume_ratio": [1.0, 1.0, 1.0, 1.0, 2.0],
            "entry_signal": [False, False, False, False, True],
            "setup_family": ["none", "none", "none", "none", "opening_drive_expansion"],
            "candidate_score": [0.0, 0.0, 0.0, 0.0, 1.4],
        },
        index=index,
    )

    monkeypatch.setattr(
        backtester,
        "_load_intraday_bars",
        lambda db_path, symbols, begin, end: {"AAA": frame_a, "BBB": frame_b},
    )
    monkeypatch.setattr(backtester, "_load_daily_trend_filter", lambda symbols, begin, end: {})
    monkeypatch.setattr(
        backtester,
        "_prepare_symbol_features",
        lambda df: prepared_a if float(df.iloc[0]["open"]) == 100.0 else prepared_b,
    )

    result = backtester.backtest_portfolio(["AAA", "BBB"], "2026-04-01", "2026-04-01", "dummy.duckdb")

    reasons = result.candidate_log["filter_reason"].value_counts().to_dict()
    assert reasons["expansion_vwap_distance"] == 1
    assert reasons["expansion_volume_ratio"] == 1
    assert reasons["queued_next_bar_open"] == 1
    assert reasons["expansion_latest_entry_bar"] == 1


def test_prepare_symbol_features_can_disable_overextended_setup():
    config = IntradayBacktestConfig(
        opening_range_bars=1,
        min_volume_ratio=0.0,
        allow_overextended_setup=False,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.date_range("2026-04-01 08:30:00", periods=23, freq="30min")
    frame = pd.DataFrame(
        {
            "open": [100.0] * 23,
            "high": [100.0] * 20 + [101.0, 102.0, 103.0],
            "low": [99.0] * 23,
            "close": [100.0] * 20 + [100.6, 101.3, 102.0],
            "volume": [1000] * 23,
        },
        index=index,
    )

    prepared = backtester._prepare_symbol_features(frame)

    assert prepared.iloc[21]["setup_family"] == "opening_drive_expansion"
    assert bool(prepared.iloc[21]["entry_signal"]) is True
    assert prepared.iloc[22]["setup_family"] == "disabled_overextended"
    assert bool(prepared.iloc[22]["entry_signal"]) is False


def test_prepare_symbol_features_can_disable_continuation_setup():
    config = IntradayBacktestConfig(
        opening_range_bars=1,
        min_volume_ratio=0.0,
        allow_continuation_setup=False,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.date_range("2026-04-01 08:30:00", periods=23, freq="30min")
    frame = pd.DataFrame(
        {
            "open": [100.0] * 23,
            "high": [100.0] * 20 + [101.0, 102.0, 103.0],
            "low": [99.0] * 23,
            "close": [100.0] * 20 + [100.6, 101.3, 102.0],
            "volume": [1000] * 23,
        },
        index=index,
    )

    prepared = backtester._prepare_symbol_features(frame)

    assert prepared.iloc[20]["setup_family"] == "disabled_continuation"
    assert bool(prepared.iloc[20]["entry_signal"]) is False
    assert prepared.iloc[21]["setup_family"] == "opening_drive_expansion"


def test_backtest_prioritizes_higher_value_setup_when_slots_are_limited(monkeypatch):
    config = IntradayBacktestConfig(
        min_volume_ratio=0.0,
        require_above_vwap=False,
        max_positions=1,
        max_trades_per_symbol_per_day=1,
        stop_loss_pct=0.0,
        trail_stop_pct=0.0,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.to_datetime(
        [
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-01 09:30:00",
        ]
    )
    frame_a = pd.DataFrame({"open": [100.0, 100.0, 101.0]}, index=index)
    frame_b = pd.DataFrame({"open": [50.0, 50.0, 51.0]}, index=index)
    prepared_a = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0],
            "high": [100.0, 102.0, 101.0],
            "low": [99.0, 100.0, 100.0],
            "close": [100.0, 101.3, 101.0],
            "volume": [1000, 1000, 1000],
            "session_date": [index[0].date()] * 3,
            "bar_in_session": [0, 1, 2],
            "next_open": [100.0, 101.0, pd.NA],
            "next_ts": [index[1], index[2], pd.NaT],
            "is_last_bar": [False, False, True],
            "opening_range_high": [100.0, 100.0, 100.0],
            "opening_range_low": [99.0, 99.0, 99.0],
            "opening_range_pct": [0.01, 0.01, 0.01],
            "breakout_distance_pct": [0.0, 0.013, 0.01],
            "distance_from_vwap_pct": [0.0, 0.01, 0.006],
            "prior_bar_high": [pd.NA, 100.0, 102.0],
            "vwap": [100.0, 100.2, 100.5],
            "avg_volume_20": [1000.0, 1000.0, 1000.0],
            "volume_ratio": [1.0, 2.0, 1.0],
            "entry_signal": [False, True, False],
            "setup_family": ["none", "opening_drive_expansion", "none"],
            "candidate_score": [0.0, 2.0, 0.0],
        },
        index=index,
    )
    prepared_b = pd.DataFrame(
        {
            "open": [50.0, 50.0, 51.0],
            "high": [50.0, 50.5, 51.0],
            "low": [49.0, 50.0, 50.0],
            "close": [50.0, 50.4, 51.0],
            "volume": [1000, 1000, 1000],
            "session_date": [index[0].date()] * 3,
            "bar_in_session": [0, 1, 2],
            "next_open": [50.0, 51.0, pd.NA],
            "next_ts": [index[1], index[2], pd.NaT],
            "is_last_bar": [False, False, True],
            "opening_range_high": [50.0, 50.0, 50.0],
            "opening_range_low": [49.0, 49.0, 49.0],
            "opening_range_pct": [0.02, 0.02, 0.02],
            "breakout_distance_pct": [0.0, 0.008, 0.02],
            "distance_from_vwap_pct": [0.0, 0.004, 0.005],
            "prior_bar_high": [pd.NA, 50.0, 50.5],
            "vwap": [50.0, 50.1, 50.5],
            "avg_volume_20": [1000.0, 1000.0, 1000.0],
            "volume_ratio": [1.0, 1.5, 1.0],
            "entry_signal": [False, True, False],
            "setup_family": ["none", "opening_drive_continuation", "none"],
            "candidate_score": [0.0, 1.0, 0.0],
        },
        index=index,
    )

    monkeypatch.setattr(
        backtester,
        "_load_intraday_bars",
        lambda db_path, symbols, begin, end: {"AAA": frame_a, "BBB": frame_b},
    )
    monkeypatch.setattr(backtester, "_load_daily_trend_filter", lambda symbols, begin, end: {})
    monkeypatch.setattr(
        backtester,
        "_prepare_symbol_features",
        lambda df: prepared_a if float(df.iloc[0]["open"]) == 100.0 else prepared_b,
    )

    result = backtester.backtest_portfolio(["AAA", "BBB"], "2026-04-01", "2026-04-01", "dummy.duckdb")

    assert list(result.trades["symbol"]) == ["AAA"]
    assert result.summary["dropped_entry_reason_counts"]["max_positions"] == 1
    assert float(result.trades.iloc[0]["selection_score"]) > 0


def test_backtest_can_filter_on_relative_strength_vs_benchmark(monkeypatch):
    config = IntradayBacktestConfig(
        min_volume_ratio=0.0,
        require_above_vwap=False,
        benchmark_symbol="SPY",
        min_relative_strength_pct=0.005,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.to_datetime(
        [
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-01 09:30:00",
        ]
    )
    frame_spy = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.2, 100.3],
            "low": [99.8, 99.9, 99.9],
            "close": [100.0, 100.1, 100.2],
            "volume": [1000, 1000, 1000],
        },
        index=index,
    )
    frame_strong = pd.DataFrame(
        {
            "open": [50.0, 50.0, 50.5],
            "high": [50.0, 50.6, 50.8],
            "low": [49.8, 50.0, 50.2],
            "close": [50.0, 50.5, 50.7],
            "volume": [1000, 1000, 1000],
        },
        index=index,
    )
    frame_weak = pd.DataFrame(
        {
            "open": [30.0, 30.0, 30.2],
            "high": [30.0, 30.2, 30.4],
            "low": [29.8, 29.9, 30.0],
            "close": [30.0, 30.15, 30.3],
            "volume": [1000, 1000, 1000],
        },
        index=index,
    )
    prepared_spy = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.2, 100.3],
            "low": [99.8, 99.9, 99.9],
            "close": [100.0, 100.1, 100.2],
            "volume": [1000, 1000, 1000],
            "session_date": [index[0].date()] * 3,
            "bar_in_session": [0, 1, 2],
            "next_open": [100.0, 100.0, pd.NA],
            "next_ts": [index[1], index[2], pd.NaT],
            "is_last_bar": [False, False, True],
            "opening_range_high": [100.0, 100.0, 100.0],
            "opening_range_low": [99.8, 99.8, 99.8],
            "opening_range_pct": [0.002, 0.002, 0.002],
            "session_open": [100.0, 100.0, 100.0],
            "return_since_open_pct": [0.0, 0.001, 0.002],
            "breakout_distance_pct": [0.0, 0.001, 0.002],
            "distance_from_vwap_pct": [0.0, 0.0005, 0.001],
            "prior_bar_high": [pd.NA, 100.0, 100.2],
            "vwap": [100.0, 100.05, 100.1],
            "avg_volume_20": [1000.0, 1000.0, 1000.0],
            "volume_ratio": [1.0, 1.0, 1.0],
            "entry_signal": [False, False, False],
            "setup_family": ["none", "none", "none"],
            "candidate_score": [0.0, 0.0, 0.0],
        },
        index=index,
    )
    prepared_strong = pd.DataFrame(
        {
            "open": [50.0, 50.0, 50.5],
            "high": [50.0, 50.6, 50.8],
            "low": [49.8, 50.0, 50.2],
            "close": [50.0, 50.5, 50.7],
            "volume": [1000, 1000, 1000],
            "session_date": [index[0].date()] * 3,
            "bar_in_session": [0, 1, 2],
            "next_open": [50.0, 50.5, pd.NA],
            "next_ts": [index[1], index[2], pd.NaT],
            "is_last_bar": [False, False, True],
            "opening_range_high": [50.0, 50.0, 50.0],
            "opening_range_low": [49.8, 49.8, 49.8],
            "opening_range_pct": [0.004, 0.004, 0.004],
            "session_open": [50.0, 50.0, 50.0],
            "return_since_open_pct": [0.0, 0.01, 0.014],
            "breakout_distance_pct": [0.0, 0.01, 0.014],
            "distance_from_vwap_pct": [0.0, 0.006, 0.007],
            "prior_bar_high": [pd.NA, 50.0, 50.6],
            "vwap": [50.0, 50.2, 50.3],
            "avg_volume_20": [1000.0, 1000.0, 1000.0],
            "volume_ratio": [1.0, 2.0, 1.0],
            "entry_signal": [False, True, False],
            "setup_family": ["none", "opening_drive_expansion", "none"],
            "candidate_score": [0.0, 2.0, 0.0],
        },
        index=index,
    )
    prepared_weak = pd.DataFrame(
        {
            "open": [30.0, 30.0, 30.2],
            "high": [30.0, 30.2, 30.4],
            "low": [29.8, 29.9, 30.0],
            "close": [30.0, 30.15, 30.3],
            "volume": [1000, 1000, 1000],
            "session_date": [index[0].date()] * 3,
            "bar_in_session": [0, 1, 2],
            "next_open": [30.0, 30.2, pd.NA],
            "next_ts": [index[1], index[2], pd.NaT],
            "is_last_bar": [False, False, True],
            "opening_range_high": [30.0, 30.0, 30.0],
            "opening_range_low": [29.8, 29.8, 29.8],
            "opening_range_pct": [0.0067, 0.0067, 0.0067],
            "session_open": [30.0, 30.0, 30.0],
            "return_since_open_pct": [0.0, 0.005, 0.01],
            "breakout_distance_pct": [0.0, 0.005, 0.01],
            "distance_from_vwap_pct": [0.0, 0.003, 0.004],
            "prior_bar_high": [pd.NA, 30.0, 30.2],
            "vwap": [30.0, 30.05, 30.1],
            "avg_volume_20": [1000.0, 1000.0, 1000.0],
            "volume_ratio": [1.0, 1.5, 1.0],
            "entry_signal": [False, True, False],
            "setup_family": ["none", "opening_drive_continuation", "none"],
            "candidate_score": [0.0, 1.0, 0.0],
        },
        index=index,
    )

    monkeypatch.setattr(
        backtester,
        "_load_intraday_bars",
        lambda db_path, symbols, begin, end: {
            "SPY": frame_spy,
            "AAA": frame_strong,
            "BBB": frame_weak,
        },
    )
    monkeypatch.setattr(backtester, "_load_daily_trend_filter", lambda symbols, begin, end: {})
    monkeypatch.setattr(
        backtester,
        "_prepare_symbol_features",
        lambda df: (
            prepared_spy
            if float(df.iloc[0]["open"]) == 100.0
            else prepared_strong
            if float(df.iloc[0]["open"]) == 50.0
            else prepared_weak
        ),
    )

    result = backtester.backtest_portfolio(["SPY", "AAA", "BBB"], "2026-04-01", "2026-04-01", "dummy.duckdb")

    reasons = result.candidate_log.groupby("symbol")["filter_reason"].first().to_dict()
    assert reasons["AAA"] == "queued_next_bar_open"
    assert reasons["BBB"] == "relative_strength_filter"


def test_backtest_can_exit_expansion_on_failed_follow_through(monkeypatch):
    config = IntradayBacktestConfig(
        min_volume_ratio=0.0,
        require_above_vwap=False,
        max_trades_per_symbol_per_day=1,
        stop_loss_pct=0.10,
        trail_stop_pct=0.10,
        expansion_failed_follow_through_bars=3,
        expansion_failed_follow_through_min_return_pct=0.005,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.to_datetime(
        [
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-01 09:30:00",
            "2026-04-01 10:00:00",
            "2026-04-01 10:30:00",
        ]
    )
    frame = pd.DataFrame({"open": [100.0, 100.0, 100.2, 100.1, 100.1]}, index=index)
    prepared = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.2, 100.1, 100.1],
            "high": [100.0, 101.0, 100.4, 100.3, 100.2],
            "low": [99.0, 100.0, 100.0, 99.9, 100.0],
            "close": [100.0, 101.2, 100.2, 100.1, 100.1],
            "volume": [1000, 1000, 1000, 1000, 1000],
            "session_date": [index[0].date()] * 5,
            "bar_in_session": [0, 1, 2, 3, 4],
            "next_open": [100.0, 100.2, 100.1, 100.1, pd.NA],
            "next_ts": [index[1], index[2], index[3], index[4], pd.NaT],
            "is_last_bar": [False, False, False, False, True],
            "opening_range_high": [100.0] * 5,
            "opening_range_low": [99.0] * 5,
            "opening_range_pct": [0.01] * 5,
            "breakout_distance_pct": [0.0, 0.012, 0.002, 0.001, 0.001],
            "distance_from_vwap_pct": [0.0, 0.01, 0.002, 0.001, 0.001],
            "prior_bar_high": [pd.NA, 100.0, 101.0, 100.4, 100.3],
            "vwap": [100.0, 100.3, 100.25, 100.2, 100.18],
            "avg_volume_20": [1000.0] * 5,
            "volume_ratio": [1.0, 2.0, 1.0, 1.0, 1.0],
            "entry_signal": [False, True, False, False, False],
            "setup_family": ["none", "opening_drive_expansion", "none", "none", "none"],
            "candidate_score": [0.0, 2.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )

    monkeypatch.setattr(
        backtester,
        "_load_intraday_bars",
        lambda db_path, symbols, begin, end: {"AAA": frame},
    )
    monkeypatch.setattr(backtester, "_load_daily_trend_filter", lambda symbols, begin, end: {})
    monkeypatch.setattr(backtester, "_prepare_symbol_features", lambda df: prepared)

    result = backtester.backtest_portfolio(["AAA"], "2026-04-01", "2026-04-01", "dummy.duckdb")

    trade = result.trades.iloc[0]
    assert trade["setup_family"] == "opening_drive_expansion"
    assert trade["exit_reason"] == "failed_follow_through"
    assert trade["exit_time"] == "2026-04-01T10:30:00"


def test_backtest_does_not_exit_strong_expansion_on_failed_follow_through(monkeypatch):
    config = IntradayBacktestConfig(
        min_volume_ratio=0.0,
        require_above_vwap=False,
        max_trades_per_symbol_per_day=1,
        stop_loss_pct=0.10,
        trail_stop_pct=0.10,
        expansion_failed_follow_through_bars=3,
        expansion_failed_follow_through_min_return_pct=0.005,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.to_datetime(
        [
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-01 09:30:00",
            "2026-04-01 10:00:00",
            "2026-04-01 10:30:00",
        ]
    )
    frame = pd.DataFrame({"open": [100.0, 100.0, 100.8, 101.2, 101.4]}, index=index)
    prepared = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.8, 101.2, 101.4],
            "high": [100.0, 101.0, 101.1, 101.4, 101.6],
            "low": [99.0, 100.0, 100.7, 101.0, 101.3],
            "close": [100.0, 101.2, 101.0, 101.3, 101.5],
            "volume": [1000, 1000, 1000, 1000, 1000],
            "session_date": [index[0].date()] * 5,
            "bar_in_session": [0, 1, 2, 3, 4],
            "next_open": [100.0, 100.8, 101.2, 101.4, pd.NA],
            "next_ts": [index[1], index[2], index[3], index[4], pd.NaT],
            "is_last_bar": [False, False, False, False, True],
            "opening_range_high": [100.0] * 5,
            "opening_range_low": [99.0] * 5,
            "opening_range_pct": [0.01] * 5,
            "breakout_distance_pct": [0.0, 0.012, 0.01, 0.013, 0.015],
            "distance_from_vwap_pct": [0.0, 0.01, 0.008, 0.01, 0.011],
            "prior_bar_high": [pd.NA, 100.0, 101.0, 101.1, 101.4],
            "vwap": [100.0, 100.3, 100.6, 100.9, 101.1],
            "avg_volume_20": [1000.0] * 5,
            "volume_ratio": [1.0, 2.0, 1.0, 1.0, 1.0],
            "entry_signal": [False, True, False, False, False],
            "setup_family": ["none", "opening_drive_expansion", "none", "none", "none"],
            "candidate_score": [0.0, 2.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )

    monkeypatch.setattr(
        backtester,
        "_load_intraday_bars",
        lambda db_path, symbols, begin, end: {"AAA": frame},
    )
    monkeypatch.setattr(backtester, "_load_daily_trend_filter", lambda symbols, begin, end: {})
    monkeypatch.setattr(backtester, "_prepare_symbol_features", lambda df: prepared)

    result = backtester.backtest_portfolio(["AAA"], "2026-04-01", "2026-04-01", "dummy.duckdb")

    trade = result.trades.iloc[0]
    assert trade["setup_family"] == "opening_drive_expansion"
    assert trade["exit_reason"] == "eod_flatten"


def test_prepare_symbol_features_fires_pullback_vwap_on_touch_and_reclaim():
    config = IntradayBacktestConfig(
        opening_range_bars=1,
        min_volume_ratio=0.0,
        allow_pullback_vwap=True,
        pullback_vwap_touch_tolerance_pct=0.002,
        pullback_vwap_touch_lookback_bars=3,
        pullback_vwap_reclaim_min_pct=0.001,
        pullback_vwap_min_session_trend_pct=0.001,
        pullback_vwap_min_volume_ratio=0.0,
        pullback_vwap_earliest_entry_bar=3,
        pullback_vwap_latest_entry_bar=10,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.date_range("2026-04-01 08:30:00", periods=6, freq="30min")
    frame = pd.DataFrame(
        {
            "open":   [100.00, 100.50, 100.60, 100.70, 100.65, 100.50],
            "high":   [101.00, 100.70, 100.80, 100.75, 100.70, 100.85],
            "low":    [ 99.90, 100.40, 100.50, 100.60, 100.40, 100.50],
            "close":  [100.50, 100.60, 100.70, 100.65, 100.50, 100.80],
            "volume": [  1000,   1000,   1000,   1000,   1000,   1000],
        },
        index=index,
    )

    prepared = backtester._prepare_symbol_features(frame)

    assert prepared.iloc[5]["setup_family"] == "pullback_vwap"
    assert bool(prepared.iloc[5]["entry_signal"]) is True
    assert prepared.iloc[4]["setup_family"] != "pullback_vwap"


def test_prepare_symbol_features_fires_gap_reclaim_long_on_gap_and_reclaim():
    config = IntradayBacktestConfig(
        opening_range_bars=1,
        min_volume_ratio=0.0,
        allow_gap_reclaim_long=True,
        gap_reclaim_min_gap_down_pct=0.015,
        gap_reclaim_max_gap_down_pct=0.06,
        gap_reclaim_min_reclaim_fraction=0.5,
        gap_reclaim_min_volume_ratio=0.0,
        gap_reclaim_earliest_entry_bar=1,
        gap_reclaim_latest_entry_bar=4,
    )
    backtester = IntradayBreakoutBacktester(config)
    session_a = pd.date_range("2026-04-01 09:30:00", periods=2, freq="30min")
    session_b = pd.date_range("2026-04-02 09:30:00", periods=6, freq="30min")
    index = session_a.append(session_b)
    frame = pd.DataFrame(
        {
            "open":   [99.00, 100.00, 97.00, 97.20, 98.60, 98.65, 98.70, 98.75],
            "high":   [100.20, 100.50, 99.00, 98.80, 98.85, 98.80, 98.85, 98.90],
            "low":    [98.80, 99.80, 96.50, 97.20, 98.50, 98.60, 98.65, 98.70],
            "close":  [100.00, 100.00, 97.20, 98.60, 98.65, 98.70, 98.75, 98.80],
            "volume": [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
        },
        index=index,
    )

    prepared = backtester._prepare_symbol_features(frame)

    session_b_rows = prepared.loc[session_b]
    fired = session_b_rows[session_b_rows["setup_family"] == "gap_reclaim_long"]
    assert len(fired) == 1
    assert fired.iloc[0]["bar_in_session"] == 1
    assert bool(fired.iloc[0]["entry_signal"]) is True


def test_prepare_symbol_features_gap_reclaim_long_requires_reclaim_fraction():
    config = IntradayBacktestConfig(
        opening_range_bars=1,
        min_volume_ratio=0.0,
        allow_gap_reclaim_long=True,
        gap_reclaim_min_gap_down_pct=0.015,
        gap_reclaim_max_gap_down_pct=0.06,
        gap_reclaim_min_reclaim_fraction=0.5,
        gap_reclaim_min_volume_ratio=0.0,
        gap_reclaim_earliest_entry_bar=1,
        gap_reclaim_latest_entry_bar=4,
    )
    backtester = IntradayBreakoutBacktester(config)
    session_a = pd.date_range("2026-04-01 09:30:00", periods=2, freq="30min")
    session_b = pd.date_range("2026-04-02 09:30:00", periods=6, freq="30min")
    index = session_a.append(session_b)
    frame = pd.DataFrame(
        {
            "open":   [99.00, 100.00, 97.00, 97.20, 97.50, 97.60, 97.70, 97.80],
            "high":   [100.20, 100.50, 99.00, 97.60, 97.90, 97.80, 97.90, 98.00],
            "low":    [98.80, 99.80, 96.50, 97.00, 97.20, 97.30, 97.50, 97.60],
            "close":  [100.00, 100.00, 97.20, 97.50, 97.60, 97.70, 97.80, 97.90],
            "volume": [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
        },
        index=index,
    )

    prepared = backtester._prepare_symbol_features(frame)

    assert (prepared["setup_family"] == "gap_reclaim_long").sum() == 0


def test_prepare_symbol_features_fires_nr4_breakout_on_narrow_range_prior():
    config = IntradayBacktestConfig(
        opening_range_bars=1,
        min_volume_ratio=0.0,
        allow_continuation_setup=False,
        allow_expansion_setup=False,
        allow_overextended_setup=False,
        allow_nr4_breakout=True,
        nr4_lookback_days=4,
        nr4_earliest_entry_bar=1,
        nr4_latest_entry_bar=12,
        nr4_min_volume_ratio=0.0,
        nr4_min_breakout_distance_pct=0.0,
    )
    backtester = IntradayBreakoutBacktester(config)
    session_a = pd.date_range("2026-04-01 09:30:00", periods=1, freq="30min")
    session_b = pd.date_range("2026-04-02 09:30:00", periods=1, freq="30min")
    session_c = pd.date_range("2026-04-03 09:30:00", periods=1, freq="30min")
    session_d = pd.date_range("2026-04-06 09:30:00", periods=1, freq="30min")
    session_e = pd.date_range("2026-04-07 09:30:00", periods=3, freq="30min")
    index = session_a.append(session_b).append(session_c).append(session_d).append(session_e)
    frame = pd.DataFrame(
        {
            "open":   [95.0, 100.0, 102.0, 99.5, 100.0, 100.5, 102.0],
            "high":   [110.0, 108.0, 107.0, 101.5, 103.0, 102.5, 103.0],
            "low":    [95.0, 96.0, 95.0, 99.0, 99.5, 100.3, 101.8],
            "close":  [105.0, 104.0, 100.0, 100.0, 100.0, 102.0, 102.5],
            "volume": [1000, 1000, 1000, 1000, 1000, 1000, 1000],
        },
        index=index,
    )

    prepared = backtester._prepare_symbol_features(frame)

    session_e_rows = prepared.loc[session_e]
    fired = session_e_rows[session_e_rows["setup_family"] == "nr4_breakout"]
    assert len(fired) == 1
    assert fired.iloc[0]["bar_in_session"] == 1
    assert bool(fired.iloc[0]["entry_signal"]) is True


def test_prepare_symbol_features_nr4_breakout_requires_narrowest_prior_range():
    config = IntradayBacktestConfig(
        opening_range_bars=1,
        min_volume_ratio=0.0,
        allow_continuation_setup=False,
        allow_expansion_setup=False,
        allow_overextended_setup=False,
        allow_nr4_breakout=True,
        nr4_lookback_days=4,
        nr4_earliest_entry_bar=1,
        nr4_latest_entry_bar=12,
        nr4_min_volume_ratio=0.0,
    )
    backtester = IntradayBreakoutBacktester(config)
    session_a = pd.date_range("2026-04-01 09:30:00", periods=1, freq="30min")
    session_b = pd.date_range("2026-04-02 09:30:00", periods=1, freq="30min")
    session_c = pd.date_range("2026-04-03 09:30:00", periods=1, freq="30min")
    session_d = pd.date_range("2026-04-06 09:30:00", periods=1, freq="30min")
    session_e = pd.date_range("2026-04-07 09:30:00", periods=3, freq="30min")
    index = session_a.append(session_b).append(session_c).append(session_d).append(session_e)
    frame = pd.DataFrame(
        {
            "open":   [95.0, 100.0, 102.0, 90.0, 100.0, 100.5, 102.0],
            "high":   [110.0, 108.0, 107.0, 112.0, 103.0, 102.5, 103.0],
            "low":    [95.0, 96.0, 95.0, 89.0, 99.5, 100.3, 101.8],
            "close":  [105.0, 104.0, 100.0, 100.0, 100.0, 102.0, 102.5],
            "volume": [1000, 1000, 1000, 1000, 1000, 1000, 1000],
        },
        index=index,
    )

    prepared = backtester._prepare_symbol_features(frame)

    assert (prepared["setup_family"] == "nr4_breakout").sum() == 0


def test_prepare_symbol_features_pullback_vwap_requires_reclaim():
    config = IntradayBacktestConfig(
        opening_range_bars=1,
        min_volume_ratio=0.0,
        allow_pullback_vwap=True,
        pullback_vwap_touch_tolerance_pct=0.002,
        pullback_vwap_touch_lookback_bars=3,
        pullback_vwap_reclaim_min_pct=0.001,
        pullback_vwap_min_session_trend_pct=0.001,
        pullback_vwap_min_volume_ratio=0.0,
        pullback_vwap_earliest_entry_bar=3,
        pullback_vwap_latest_entry_bar=10,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.date_range("2026-04-01 08:30:00", periods=6, freq="30min")
    frame = pd.DataFrame(
        {
            "open":   [100.00, 100.50, 100.60, 100.70, 100.65, 100.50],
            "high":   [101.00, 100.70, 100.80, 100.75, 100.70, 100.55],
            "low":    [ 99.90, 100.40, 100.50, 100.60, 100.40, 100.45],
            "close":  [100.50, 100.60, 100.70, 100.65, 100.50, 100.50],
            "volume": [  1000,   1000,   1000,   1000,   1000,   1000],
        },
        index=index,
    )

    prepared = backtester._prepare_symbol_features(frame)

    assert (prepared["setup_family"] == "pullback_vwap").sum() == 0


def test_backtest_gap_reclaim_can_delay_trailing_stop_activation(monkeypatch):
    config = IntradayBacktestConfig(
        min_volume_ratio=0.0,
        require_above_vwap=False,
        max_trades_per_symbol_per_day=1,
        stop_loss_pct=0.03,
        trail_stop_pct=0.04,
        gap_reclaim_trail_activation_return_pct=0.01,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.to_datetime(
        [
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-01 09:30:00",
            "2026-04-01 10:00:00",
            "2026-04-01 10:30:00",
        ]
    )
    frame = pd.DataFrame({"open": [100.0, 100.0, 100.0, 98.0, 98.0]}, index=index)
    prepared = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 98.0, 98.0],
            "high": [100.0, 101.0, 100.5, 98.5, 98.5],
            "low": [99.0, 100.0, 97.5, 97.8, 97.8],
            "close": [100.0, 100.8, 98.0, 98.0, 98.0],
            "volume": [1000, 1000, 1000, 1000, 1000],
            "session_date": [index[0].date()] * 5,
            "bar_in_session": [0, 1, 2, 3, 4],
            "next_open": [100.0, 100.0, 98.0, 98.0, pd.NA],
            "next_ts": [index[1], index[2], index[3], index[4], pd.NaT],
            "is_last_bar": [False, False, False, False, True],
            "opening_range_high": [100.0] * 5,
            "opening_range_low": [99.0] * 5,
            "opening_range_pct": [0.01] * 5,
            "breakout_distance_pct": [0.0, 0.008, -0.02, -0.02, -0.02],
            "distance_from_vwap_pct": [0.0, 0.004, -0.02, -0.02, -0.02],
            "prior_bar_high": [pd.NA, 100.0, 101.0, 100.5, 98.5],
            "prior_bar_low": [pd.NA, 99.0, 100.0, 97.5, 97.8],
            "vwap": [100.0, 100.2, 99.5, 99.0, 98.8],
            "avg_volume_20": [1000.0] * 5,
            "volume_ratio": [1.0, 1.5, 1.0, 1.0, 1.0],
            "entry_signal": [False, True, False, False, False],
            "setup_family": ["none", "gap_reclaim_long", "none", "none", "none"],
            "candidate_score": [0.0, 2.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )

    monkeypatch.setattr(backtester, "_load_intraday_bars", lambda db_path, symbols, begin, end: {"AAA": frame})
    monkeypatch.setattr(backtester, "_load_daily_trend_filter", lambda symbols, begin, end: {})
    monkeypatch.setattr(backtester, "_prepare_symbol_features", lambda df: prepared)

    result = backtester.backtest_portfolio(["AAA"], "2026-04-01", "2026-04-01", "dummy.duckdb")

    trade = result.trades.iloc[0]
    assert trade["setup_family"] == "gap_reclaim_long"
    assert trade["exit_reason"] == "eod_flatten"
    assert trade["exit_price"] == 98.0


def test_backtest_can_filter_on_entry_strength(monkeypatch):
    config = IntradayBacktestConfig(
        min_volume_ratio=0.0,
        require_above_vwap=False,
        min_entry_strength_breakout_pct=0.01,
        min_entry_strength_vwap_distance_pct=0.005,
    )
    backtester = IntradayBreakoutBacktester(config)
    index = pd.to_datetime(
        [
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-01 09:30:00",
        ]
    )
    frame_a = pd.DataFrame({"open": [100.0, 100.0, 101.0]}, index=index)
    frame_b = pd.DataFrame({"open": [50.0, 50.0, 51.0]}, index=index)
    prepared_a = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0],
            "high": [100.0, 102.0, 101.0],
            "low": [99.0, 100.0, 100.0],
            "close": [100.0, 101.3, 101.0],
            "volume": [1000, 1000, 1000],
            "session_date": [index[0].date()] * 3,
            "bar_in_session": [0, 1, 2],
            "next_open": [100.0, 101.0, pd.NA],
            "next_ts": [index[1], index[2], pd.NaT],
            "is_last_bar": [False, False, True],
            "opening_range_high": [100.0, 100.0, 100.0],
            "opening_range_low": [99.0, 99.0, 99.0],
            "opening_range_pct": [0.01, 0.01, 0.01],
            "session_open": [100.0, 100.0, 100.0],
            "return_since_open_pct": [0.0, 0.013, 0.01],
            "breakout_distance_pct": [0.0, 0.013, 0.01],
            "distance_from_vwap_pct": [0.0, 0.01, 0.006],
            "prior_bar_high": [pd.NA, 100.0, 102.0],
            "vwap": [100.0, 100.2, 100.5],
            "avg_volume_20": [1000.0, 1000.0, 1000.0],
            "volume_ratio": [1.0, 2.0, 1.0],
            "entry_signal": [False, True, False],
            "setup_family": ["none", "opening_drive_expansion", "none"],
            "candidate_score": [0.0, 2.0, 0.0],
        },
        index=index,
    )
    prepared_b = pd.DataFrame(
        {
            "open": [50.0, 50.0, 51.0],
            "high": [50.0, 50.4, 51.0],
            "low": [49.0, 50.0, 50.0],
            "close": [50.0, 50.25, 51.0],
            "volume": [1000, 1000, 1000],
            "session_date": [index[0].date()] * 3,
            "bar_in_session": [0, 1, 2],
            "next_open": [50.0, 51.0, pd.NA],
            "next_ts": [index[1], index[2], pd.NaT],
            "is_last_bar": [False, False, True],
            "opening_range_high": [50.0, 50.0, 50.0],
            "opening_range_low": [49.0, 49.0, 49.0],
            "opening_range_pct": [0.02, 0.02, 0.02],
            "session_open": [50.0, 50.0, 50.0],
            "return_since_open_pct": [0.0, 0.005, 0.02],
            "breakout_distance_pct": [0.0, 0.005, 0.02],
            "distance_from_vwap_pct": [0.0, 0.003, 0.005],
            "prior_bar_high": [pd.NA, 50.0, 50.4],
            "vwap": [50.0, 50.1, 50.3],
            "avg_volume_20": [1000.0, 1000.0, 1000.0],
            "volume_ratio": [1.0, 1.5, 1.0],
            "entry_signal": [False, True, False],
            "setup_family": ["none", "opening_drive_continuation", "none"],
            "candidate_score": [0.0, 1.0, 0.0],
        },
        index=index,
    )

    monkeypatch.setattr(
        backtester,
        "_load_intraday_bars",
        lambda db_path, symbols, begin, end: {"AAA": frame_a, "BBB": frame_b},
    )
    monkeypatch.setattr(backtester, "_load_daily_trend_filter", lambda symbols, begin, end: {})
    monkeypatch.setattr(
        backtester,
        "_prepare_symbol_features",
        lambda df: prepared_a if float(df.iloc[0]["open"]) == 100.0 else prepared_b,
    )

    result = backtester.backtest_portfolio(["AAA", "BBB"], "2026-04-01", "2026-04-01", "dummy.duckdb")

    reasons = result.candidate_log.groupby("symbol")["filter_reason"].first().to_dict()
    assert reasons["AAA"] == "queued_next_bar_open"
    assert reasons["BBB"] == "entry_strength_breakout"


def test_build_relative_volume_universe_picks_topk_per_session():
    config = IntradayBacktestConfig(
        allow_relative_volume_filter=True,
        relative_volume_lookback_days=3,
        relative_volume_top_k=2,
    )
    backtester = IntradayBreakoutBacktester(config)

    def make_frame(opening_vols: list[int]) -> pd.DataFrame:
        n_sessions = len(opening_vols)
        bars = []
        for day_idx in range(n_sessions):
            day = f"2026-04-{day_idx+1:02d}"
            bars.append((pd.Timestamp(f"{day} 08:30:00"), opening_vols[day_idx], 0))
            bars.append((pd.Timestamp(f"{day} 09:00:00"), 500, 1))
        index = [b[0] for b in bars]
        return pd.DataFrame(
            {
                "volume": [b[1] for b in bars],
                "bar_in_session": [b[2] for b in bars],
                "session_date": [b[0].date() for b in bars],
            },
            index=index,
        )

    prepared = {
        "AAA": make_frame([1000, 1000, 1000, 5000]),
        "BBB": make_frame([1000, 1000, 1000, 4000]),
        "CCC": make_frame([1000, 1000, 1000, 1500]),
    }

    universe = backtester._build_relative_volume_universe(prepared)

    day4 = pd.Timestamp("2026-04-04").date()
    assert day4 in universe
    assert universe[day4] == {"AAA", "BBB"}
    assert len(universe[day4]) == 2


def test_build_relative_volume_universe_empty_when_disabled_or_insufficient_history():
    config = IntradayBacktestConfig(
        allow_relative_volume_filter=True,
        relative_volume_lookback_days=5,
        relative_volume_top_k=2,
    )
    backtester = IntradayBreakoutBacktester(config)

    def tiny_frame() -> pd.DataFrame:
        index = pd.to_datetime(["2026-04-01 08:30:00", "2026-04-01 09:00:00"])
        return pd.DataFrame(
            {"volume": [1000, 500], "bar_in_session": [0, 1], "session_date": [d.date() for d in index]},
            index=index,
        )

    universe = backtester._build_relative_volume_universe({"AAA": tiny_frame()})
    assert universe == {}


def test_apply_execution_cost_zero_defaults_returns_raw_price():
    backtester = IntradayBreakoutBacktester(IntradayBacktestConfig())
    assert backtester._apply_execution_cost(100.0, "buy", None, 100, 10_000) == 100.0
    assert backtester._apply_execution_cost(100.0, "sell", "stop", 100, 10_000) == 100.0


def test_apply_execution_cost_half_spread_increases_buy_decreases_sell():
    config = IntradayBacktestConfig(execution_half_spread_bps=10.0)
    backtester = IntradayBreakoutBacktester(config)
    # 10 bps = 0.001 fraction
    assert backtester._apply_execution_cost(100.0, "buy", None, 0, None) == 100.0 * 1.001
    assert backtester._apply_execution_cost(100.0, "sell", "eod_flatten", 0, None) == 100.0 * 0.999


def test_apply_execution_cost_stop_slippage_only_on_stop_exits():
    config = IntradayBacktestConfig(
        execution_half_spread_bps=10.0,
        execution_stop_slippage_bps=50.0,
    )
    backtester = IntradayBreakoutBacktester(config)
    # non-stop sell pays only half spread (10 bps)
    eod = backtester._apply_execution_cost(100.0, "sell", "eod_flatten", 0, None)
    # stop sell pays half spread + stop slippage (60 bps)
    stop = backtester._apply_execution_cost(100.0, "sell", "stop", 0, None)
    failed = backtester._apply_execution_cost(100.0, "sell", "failed_follow_through", 0, None)
    # buys never see stop slippage
    buy = backtester._apply_execution_cost(100.0, "buy", None, 0, None)
    assert eod == 100.0 * 0.999
    assert stop == 100.0 * (1.0 - 0.006)
    assert failed == 100.0 * (1.0 - 0.006)
    assert buy == 100.0 * 1.001


def test_apply_execution_cost_impact_scales_with_participation():
    config = IntradayBacktestConfig(execution_impact_coeff_bps=100.0)
    backtester = IntradayBreakoutBacktester(config)
    # shares=100 / bar_volume=1000 => participation 0.1 => 10 bps impact
    assert backtester._apply_execution_cost(100.0, "buy", None, 100, 1_000) == 100.0 * 1.001
    assert backtester._apply_execution_cost(100.0, "sell", "eod_flatten", 100, 1_000) == 100.0 * 0.999
    # coeff > 0 but volume 0 or shares 0 => no impact
    assert backtester._apply_execution_cost(100.0, "buy", None, 0, 1_000) == 100.0
    assert backtester._apply_execution_cost(100.0, "buy", None, 100, 0) == 100.0


def test_backtest_execution_cost_shrinks_pnl(monkeypatch):
    """End-to-end: identical trade with non-zero execution cost yields lower PnL than zero-cost."""
    index = pd.to_datetime(
        [
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-01 09:30:00",
            "2026-04-01 10:00:00",
        ]
    )
    frame = pd.DataFrame({"open": [100.0, 100.0, 102.0, 103.0]}, index=index)
    prepared = pd.DataFrame(
        {
            "open": [100.0, 100.0, 102.0, 103.0],
            "high": [100.5, 101.0, 102.5, 104.0],
            "low": [99.5, 100.0, 101.5, 102.5],
            "close": [100.0, 100.8, 103.0, 104.0],
            "volume": [1000, 1000, 1000, 1000],
            "session_date": [index[0].date()] * 4,
            "bar_in_session": [0, 1, 2, 3],
            "next_open": [100.0, 102.0, 103.0, pd.NA],
            "next_ts": [index[1], index[2], index[3], pd.NaT],
            "is_last_bar": [False, False, False, True],
            "opening_range_high": [100.5] * 4,
            "opening_range_low": [99.5] * 4,
            "opening_range_pct": [0.01] * 4,
            "breakout_distance_pct": [0.0, 0.005, 0.02, 0.03],
            "distance_from_vwap_pct": [0.0, 0.004, 0.02, 0.03],
            "prior_bar_high": [pd.NA, 100.5, 101.0, 102.5],
            "prior_bar_low": [pd.NA, 99.5, 100.0, 101.5],
            "vwap": [100.0, 100.3, 100.8, 101.5],
            "avg_volume_20": [1000.0] * 4,
            "volume_ratio": [1.0, 1.5, 1.2, 1.0],
            "entry_signal": [False, True, False, False],
            "setup_family": ["none", "opening_drive_expansion", "none", "none"],
            "candidate_score": [0.0, 2.0, 0.0, 0.0],
        },
        index=index,
    )

    def _run(config: IntradayBacktestConfig) -> float:
        bt = IntradayBreakoutBacktester(config)
        monkeypatch.setattr(bt, "_load_intraday_bars", lambda db_path, symbols, begin, end: {"AAA": frame})
        monkeypatch.setattr(bt, "_load_daily_trend_filter", lambda symbols, begin, end: {})
        monkeypatch.setattr(bt, "_prepare_symbol_features", lambda df: prepared)
        result = bt.backtest_portfolio(["AAA"], "2026-04-01", "2026-04-01", "dummy.duckdb")
        return float(result.trades.iloc[0]["pnl"])

    zero_cost = _run(IntradayBacktestConfig(min_volume_ratio=0.0, require_above_vwap=False))
    with_cost = _run(
        IntradayBacktestConfig(
            min_volume_ratio=0.0,
            require_above_vwap=False,
            execution_half_spread_bps=10.0,
            execution_stop_slippage_bps=50.0,
        )
    )
    assert zero_cost > 0
    assert with_cost < zero_cost
