from datetime import time as dtime

import pandas as pd

from tradingagents.research.intraday_xsection_backtester import (
    XSectionReversionBacktester,
    XSectionReversionConfig,
)


def _make_frames():
    index = pd.to_datetime(
        [
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-01 10:00:00",
            "2026-04-01 11:00:00",
        ]
    )
    return {
        "AAA": pd.DataFrame(
            {
                "open": [100.0, 90.0, 95.0, 101.0],
                "high": [101.0, 91.0, 96.0, 102.0],
                "low": [99.0, 89.0, 94.0, 100.0],
                "close": [100.0, 90.0, 95.0, 101.0],
                "volume": [1_000_000, 1_000_000, 1_000_000, 1_000_000],
            },
            index=index,
        ),
        "BBB": pd.DataFrame(
            {
                "open": [100.0, 110.0, 105.0, 99.0],
                "high": [101.0, 111.0, 106.0, 100.0],
                "low": [99.0, 109.0, 104.0, 98.0],
                "close": [100.0, 110.0, 105.0, 99.0],
                "volume": [1_000_000, 1_000_000, 1_000_000, 1_000_000],
            },
            index=index,
        ),
        "SPY": pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 100.0],
                "high": [101.0, 104.0, 104.0, 104.0],
                "low": [99.0, 99.0, 99.0, 99.0],
                "close": [100.0, 103.0, 103.0, 103.0],
                "volume": [10_000_000, 10_000_000, 10_000_000, 10_000_000],
            },
            index=index,
        ),
    }


def _make_tilt_refresh_frames():
    index = pd.to_datetime(
        [
            "2026-04-01 08:30:00",
            "2026-04-01 09:00:00",
            "2026-04-01 10:00:00",
            "2026-04-01 11:00:00",
            "2026-04-01 12:00:00",
        ]
    )
    return {
        "AAA": pd.DataFrame(
            {
                "open": [100.0, 90.0, 95.0, 96.0, 102.0],
                "high": [101.0, 91.0, 96.0, 97.0, 103.0],
                "low": [99.0, 89.0, 94.0, 95.0, 101.0],
                "close": [100.0, 90.0, 95.0, 96.0, 102.0],
                "volume": [1_000_000] * 5,
            },
            index=index,
        ),
        "BBB": pd.DataFrame(
            {
                "open": [100.0, 110.0, 105.0, 104.0, 98.0],
                "high": [101.0, 111.0, 106.0, 105.0, 99.0],
                "low": [99.0, 109.0, 104.0, 103.0, 97.0],
                "close": [100.0, 110.0, 105.0, 104.0, 98.0],
                "volume": [1_000_000] * 5,
            },
            index=index,
        ),
        "SPY": pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 100.0, 100.0],
                "high": [101.0, 101.0, 101.0, 104.0, 104.0],
                "low": [99.0, 99.0, 99.0, 99.0, 99.0],
                "close": [100.0, 100.0, 100.0, 103.0, 103.0],
                "volume": [10_000_000] * 5,
            },
            index=index,
        ),
    }


def test_xsection_dollar_neutral_uses_full_side_notional(monkeypatch):
    frames = _make_frames()
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_symbol_frames",
        lambda *args, **kwargs: frames,
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_sector_map",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._compute_dollar_volume",
        lambda frame, lookback_bars: pd.Series(10_000_000.0, index=frame.index),
    )

    cfg = XSectionReversionConfig(
        initial_cash=100_000.0,
        min_dollar_volume_avg=0.0,
        interval_minutes=30,
        formation_minutes=30,
        hold_minutes=60,
        n_long=1,
        n_short=1,
        dollar_neutral=True,
        target_gross_exposure=1.0,
        earliest_rebalance_time=dtime(9, 0),
        latest_rebalance_time=dtime(9, 0),
        flatten_at_close_time=dtime(11, 0),
        half_spread_bps=0.0,
        slippage_bps=0.0,
        short_borrow_bps_per_day=0.0,
    )

    result = XSectionReversionBacktester(cfg).backtest(
        symbols=["AAA", "BBB"],
        begin="2026-04-01",
        end="2026-04-01",
        intraday_db_path="dummy.duckdb",
    )

    assert len(result.trades) == 2
    trade_by_symbol = {trade.symbol: trade for trade in result.trades}
    assert trade_by_symbol["AAA"].side == "long"
    assert trade_by_symbol["AAA"].qty == 1052
    assert trade_by_symbol["AAA"].entry_time == pd.Timestamp("2026-04-01 10:00:00")
    assert trade_by_symbol["BBB"].side == "short"
    assert trade_by_symbol["BBB"].qty == 952
    assert float(result.equity_curve["gross_exposure"].max()) > 1.5


def test_xsection_non_neutral_splits_total_gross(monkeypatch):
    frames = _make_frames()
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_symbol_frames",
        lambda *args, **kwargs: frames,
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_sector_map",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._compute_dollar_volume",
        lambda frame, lookback_bars: pd.Series(10_000_000.0, index=frame.index),
    )

    cfg = XSectionReversionConfig(
        initial_cash=100_000.0,
        min_dollar_volume_avg=0.0,
        interval_minutes=30,
        formation_minutes=30,
        hold_minutes=60,
        n_long=1,
        n_short=1,
        dollar_neutral=False,
        target_gross_exposure=1.0,
        earliest_rebalance_time=dtime(9, 0),
        latest_rebalance_time=dtime(9, 0),
        flatten_at_close_time=dtime(11, 0),
        half_spread_bps=0.0,
        slippage_bps=0.0,
        short_borrow_bps_per_day=0.0,
    )

    result = XSectionReversionBacktester(cfg).backtest(
        symbols=["AAA", "BBB"],
        begin="2026-04-01",
        end="2026-04-01",
        intraday_db_path="dummy.duckdb",
    )

    assert len(result.trades) == 2
    trade_by_symbol = {trade.symbol: trade for trade in result.trades}
    assert trade_by_symbol["AAA"].qty == 526
    assert trade_by_symbol["BBB"].qty == 476
    assert float(result.equity_curve["gross_exposure"].max()) < 1.1


def test_xsection_regime_gate_blocks_rebalance(monkeypatch):
    frames = _make_frames()
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_symbol_frames",
        lambda *args, **kwargs: frames,
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_sector_map",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._compute_dollar_volume",
        lambda frame, lookback_bars: pd.Series(10_000_000.0, index=frame.index),
    )

    cfg = XSectionReversionConfig(
        initial_cash=100_000.0,
        min_dollar_volume_avg=0.0,
        market_context_max_score=3,
        interval_minutes=30,
        formation_minutes=30,
        hold_minutes=60,
        n_long=1,
        n_short=1,
        dollar_neutral=True,
        target_gross_exposure=1.0,
        earliest_rebalance_time=dtime(9, 0),
        latest_rebalance_time=dtime(9, 0),
        flatten_at_close_time=dtime(11, 0),
        half_spread_bps=0.0,
        slippage_bps=0.0,
        short_borrow_bps_per_day=0.0,
    )
    backtester = XSectionReversionBacktester(cfg)
    monkeypatch.setattr(
        backtester,
        "_load_market_context",
        lambda begin, end, daily_db_path: pd.DataFrame(
            {
                "market_score": [6],
                "market_regime": ["confirmed_uptrend"],
            },
            index=pd.to_datetime(["2026-03-31"]),
        ),
    )

    result = backtester.backtest(
        symbols=["AAA", "BBB"],
        begin="2026-04-01",
        end="2026-04-01",
        intraday_db_path="dummy.duckdb",
    )

    assert len(result.trades) == 0
    assert result.summary["rebalance_count"] == 1
    assert result.summary["regime_blocked_rebalances"] == 1


def test_xsection_aux_market_context_filter_blocks_rebalance(monkeypatch):
    frames = _make_frames()
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_symbol_frames",
        lambda *args, **kwargs: frames,
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_sector_map",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._compute_dollar_volume",
        lambda frame, lookback_bars: pd.Series(10_000_000.0, index=frame.index),
    )

    cfg = XSectionReversionConfig(
        initial_cash=100_000.0,
        min_dollar_volume_avg=0.0,
        market_context_qqq_above_ema21_pct_max=-0.02,
        interval_minutes=30,
        formation_minutes=30,
        hold_minutes=60,
        n_long=1,
        n_short=1,
        dollar_neutral=True,
        target_gross_exposure=1.0,
        earliest_rebalance_time=dtime(9, 0),
        latest_rebalance_time=dtime(9, 0),
        flatten_at_close_time=dtime(11, 0),
        half_spread_bps=0.0,
        slippage_bps=0.0,
        short_borrow_bps_per_day=0.0,
    )
    backtester = XSectionReversionBacktester(cfg)
    monkeypatch.setattr(
        backtester,
        "_load_market_context",
        lambda begin, end, daily_db_path: pd.DataFrame(
            {
                "market_score": [1],
                "market_regime": ["market_correction"],
                "qqq_above_ema21_pct": [-0.01],
            },
            index=pd.to_datetime(["2026-03-31"]),
        ),
    )

    result = backtester.backtest(
        symbols=["AAA", "BBB"],
        begin="2026-04-01",
        end="2026-04-01",
        intraday_db_path="dummy.duckdb",
    )

    assert len(result.trades) == 0
    assert result.summary["rebalance_count"] == 1
    assert result.summary["regime_blocked_rebalances"] == 1


def test_xsection_regime_override_applies_stricter_filter(monkeypatch):
    frames = _make_frames()
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_symbol_frames",
        lambda *args, **kwargs: frames,
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_sector_map",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._compute_dollar_volume",
        lambda frame, lookback_bars: pd.Series(10_000_000.0, index=frame.index),
    )

    cfg = XSectionReversionConfig(
        initial_cash=100_000.0,
        min_dollar_volume_avg=0.0,
        allowed_market_regimes=("market_correction", "uptrend_under_pressure"),
        regime_filter_overrides={
            "market_correction": {
                "market_context_max_score": 1,
                "market_context_qqq_above_ema21_pct_max": -0.02,
            }
        },
        interval_minutes=30,
        formation_minutes=30,
        hold_minutes=60,
        n_long=1,
        n_short=1,
        dollar_neutral=True,
        target_gross_exposure=1.0,
        earliest_rebalance_time=dtime(9, 0),
        latest_rebalance_time=dtime(9, 0),
        flatten_at_close_time=dtime(11, 0),
        half_spread_bps=0.0,
        slippage_bps=0.0,
        short_borrow_bps_per_day=0.0,
    )
    backtester = XSectionReversionBacktester(cfg)
    monkeypatch.setattr(
        backtester,
        "_load_market_context",
        lambda begin, end, daily_db_path: pd.DataFrame(
            {
                "market_score": [3],
                "market_regime": ["market_correction"],
                "qqq_above_ema21_pct": [-0.01],
            },
            index=pd.to_datetime(["2026-03-31"]),
        ),
    )

    result = backtester.backtest(
        symbols=["AAA", "BBB"],
        begin="2026-04-01",
        end="2026-04-01",
        intraday_db_path="dummy.duckdb",
    )

    assert len(result.trades) == 0
    assert result.summary["rebalance_count"] == 1
    assert result.summary["regime_blocked_rebalances"] == 1


def test_xsection_adv_cap_reduces_dollar_neutral_book(monkeypatch):
    frames = _make_frames()
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_symbol_frames",
        lambda *args, **kwargs: frames,
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_sector_map",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._compute_dollar_volume",
        lambda frame, lookback_bars: pd.Series(10_000_000.0, index=frame.index),
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_daily_adv",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "AAA": [500_000.0],
                "BBB": [300_000.0],
            },
            index=pd.to_datetime(["2026-04-01"]),
        ),
    )

    cfg = XSectionReversionConfig(
        initial_cash=100_000.0,
        min_dollar_volume_avg=0.0,
        max_position_pct_of_adv=0.10,
        interval_minutes=30,
        formation_minutes=30,
        hold_minutes=60,
        n_long=1,
        n_short=1,
        dollar_neutral=True,
        target_gross_exposure=1.0,
        earliest_rebalance_time=dtime(9, 0),
        latest_rebalance_time=dtime(9, 0),
        flatten_at_close_time=dtime(11, 0),
        half_spread_bps=0.0,
        slippage_bps=0.0,
        short_borrow_bps_per_day=0.0,
    )

    result = XSectionReversionBacktester(cfg).backtest(
        symbols=["AAA", "BBB"],
        begin="2026-04-01",
        end="2026-04-01",
        intraday_db_path="dummy.duckdb",
    )

    assert len(result.trades) == 2
    trade_by_symbol = {trade.symbol: trade for trade in result.trades}
    assert trade_by_symbol["AAA"].qty == 315
    assert trade_by_symbol["BBB"].qty == 285
    assert result.summary["liquidity_capped_rebalances"] == 1


def test_xsection_flattens_on_last_bar_when_flatten_time_is_later(monkeypatch):
    frames = _make_frames()
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_symbol_frames",
        lambda *args, **kwargs: frames,
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_sector_map",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._compute_dollar_volume",
        lambda frame, lookback_bars: pd.Series(10_000_000.0, index=frame.index),
    )

    cfg = XSectionReversionConfig(
        initial_cash=100_000.0,
        min_dollar_volume_avg=0.0,
        interval_minutes=30,
        formation_minutes=30,
        hold_minutes=60,
        n_long=1,
        n_short=1,
        dollar_neutral=True,
        target_gross_exposure=1.0,
        earliest_rebalance_time=dtime(9, 0),
        latest_rebalance_time=dtime(9, 0),
        flatten_at_close_time=dtime(15, 55),
        half_spread_bps=0.0,
        slippage_bps=0.0,
        short_borrow_bps_per_day=0.0,
    )

    result = XSectionReversionBacktester(cfg).backtest(
        symbols=["AAA", "BBB"],
        begin="2026-04-01",
        end="2026-04-01",
        intraday_db_path="dummy.duckdb",
    )

    assert len(result.trades) == 2
    assert result.summary["final_equity"] == 112024.0


def test_xsection_intraday_market_tilt_leans_long(monkeypatch):
    frames = _make_frames()
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_symbol_frames",
        lambda *args, **kwargs: frames,
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_sector_map",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._compute_dollar_volume",
        lambda frame, lookback_bars: pd.Series(10_000_000.0, index=frame.index),
    )

    cfg = XSectionReversionConfig(
        initial_cash=100_000.0,
        min_dollar_volume_avg=0.0,
        interval_minutes=30,
        formation_minutes=30,
        hold_minutes=60,
        n_long=1,
        n_short=1,
        dollar_neutral=True,
        target_gross_exposure=1.0,
        intraday_market_tilt_symbol="SPY",
        intraday_market_tilt_threshold=0.01,
        intraday_market_tilt_strength=0.25,
        earliest_rebalance_time=dtime(9, 0),
        latest_rebalance_time=dtime(9, 0),
        flatten_at_close_time=dtime(11, 0),
        half_spread_bps=0.0,
        slippage_bps=0.0,
        short_borrow_bps_per_day=0.0,
    )

    result = XSectionReversionBacktester(cfg).backtest(
        symbols=["AAA", "BBB", "SPY"],
        begin="2026-04-01",
        end="2026-04-01",
        intraday_db_path="dummy.duckdb",
    )

    trade_by_symbol = {trade.symbol: trade for trade in result.trades}
    assert trade_by_symbol["AAA"].qty == 1315
    assert trade_by_symbol["BBB"].qty == 714
    assert result.summary["avg_net_exposure"] > 0


def test_xsection_intraday_market_tilt_strong_state_uses_larger_bias(monkeypatch):
    frames = _make_frames()
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_symbol_frames",
        lambda *args, **kwargs: frames,
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_sector_map",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._compute_dollar_volume",
        lambda frame, lookback_bars: pd.Series(10_000_000.0, index=frame.index),
    )

    cfg = XSectionReversionConfig(
        initial_cash=100_000.0,
        min_dollar_volume_avg=0.0,
        interval_minutes=30,
        formation_minutes=30,
        hold_minutes=60,
        n_long=1,
        n_short=1,
        dollar_neutral=True,
        target_gross_exposure=1.0,
        intraday_market_tilt_symbol="SPY",
        intraday_market_tilt_threshold=0.01,
        intraday_market_tilt_strength=0.25,
        intraday_market_tilt_strong_threshold=0.025,
        intraday_market_tilt_strong_strength=0.50,
        earliest_rebalance_time=dtime(9, 0),
        latest_rebalance_time=dtime(9, 0),
        flatten_at_close_time=dtime(11, 0),
        half_spread_bps=0.0,
        slippage_bps=0.0,
        short_borrow_bps_per_day=0.0,
    )

    result = XSectionReversionBacktester(cfg).backtest(
        symbols=["AAA", "BBB", "SPY"],
        begin="2026-04-01",
        end="2026-04-01",
        intraday_db_path="dummy.duckdb",
    )

    trade_by_symbol = {trade.symbol: trade for trade in result.trades}
    assert trade_by_symbol["AAA"].qty == 1578
    assert trade_by_symbol["BBB"].qty == 476
    assert result.summary["avg_net_exposure"] > 0


def test_xsection_regime_override_can_supply_intraday_tilt(monkeypatch):
    frames = _make_frames()
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_symbol_frames",
        lambda *args, **kwargs: frames,
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_sector_map",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._compute_dollar_volume",
        lambda frame, lookback_bars: pd.Series(10_000_000.0, index=frame.index),
    )

    cfg = XSectionReversionConfig(
        initial_cash=100_000.0,
        min_dollar_volume_avg=0.0,
        allowed_market_regimes=("uptrend_under_pressure",),
        regime_filter_overrides={
            "uptrend_under_pressure": {
                "intraday_market_tilt_symbol": "SPY",
                "intraday_market_tilt_threshold": 0.01,
                "intraday_market_tilt_strength": 0.25,
            }
        },
        interval_minutes=30,
        formation_minutes=30,
        hold_minutes=60,
        n_long=1,
        n_short=1,
        dollar_neutral=True,
        target_gross_exposure=1.0,
        earliest_rebalance_time=dtime(9, 0),
        latest_rebalance_time=dtime(9, 0),
        flatten_at_close_time=dtime(11, 0),
        half_spread_bps=0.0,
        slippage_bps=0.0,
        short_borrow_bps_per_day=0.0,
    )
    backtester = XSectionReversionBacktester(cfg)
    monkeypatch.setattr(
        backtester,
        "_load_market_context",
        lambda begin, end, daily_db_path: pd.DataFrame(
            {
                "market_score": [4],
                "market_regime": ["uptrend_under_pressure"],
            },
            index=pd.to_datetime(["2026-03-31"]),
        ),
    )

    result = backtester.backtest(
        symbols=["AAA", "BBB", "SPY"],
        begin="2026-04-01",
        end="2026-04-01",
        intraday_db_path="dummy.duckdb",
    )

    trade_by_symbol = {trade.symbol: trade for trade in result.trades}
    assert trade_by_symbol["AAA"].qty == 1315
    assert trade_by_symbol["BBB"].qty == 714


def test_xsection_tilt_refresh_resizes_same_basket_midday(monkeypatch):
    frames = _make_tilt_refresh_frames()
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_symbol_frames",
        lambda *args, **kwargs: frames,
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._load_sector_map",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "tradingagents.research.intraday_xsection_backtester._compute_dollar_volume",
        lambda frame, lookback_bars: pd.Series(10_000_000.0, index=frame.index),
    )

    cfg = XSectionReversionConfig(
        initial_cash=100_000.0,
        min_dollar_volume_avg=0.0,
        interval_minutes=30,
        formation_minutes=30,
        hold_minutes=120,
        tilt_refresh_minutes=60,
        n_long=1,
        n_short=1,
        dollar_neutral=True,
        target_gross_exposure=1.0,
        intraday_market_tilt_symbol="SPY",
        intraday_market_tilt_threshold=0.01,
        intraday_market_tilt_strength=0.25,
        earliest_rebalance_time=dtime(9, 0),
        latest_rebalance_time=dtime(11, 0),
        flatten_at_close_time=dtime(15, 55),
        half_spread_bps=0.0,
        slippage_bps=0.0,
        short_borrow_bps_per_day=0.0,
    )

    result = XSectionReversionBacktester(cfg).backtest(
        symbols=["AAA", "BBB", "SPY"],
        begin="2026-04-01",
        end="2026-04-01",
        intraday_db_path="dummy.duckdb",
    )

    assert result.summary["rebalance_count"] == 1
    assert result.summary["tilt_refresh_count"] == 1
    aaa_trades = [trade for trade in result.trades if trade.symbol == "AAA"]
    bbb_trades = [trade for trade in result.trades if trade.symbol == "BBB"]
    assert len(aaa_trades) == 2
    assert len(bbb_trades) == 2
    assert aaa_trades[0].qty == 1052
    assert bbb_trades[0].qty == 952
    assert aaa_trades[1].entry_time == pd.Timestamp("2026-04-01 12:00:00")
    assert bbb_trades[1].entry_time == pd.Timestamp("2026-04-01 12:00:00")
    assert aaa_trades[1].qty == 1225
    assert bbb_trades[1].qty == 765
    assert aaa_trades[1].qty > aaa_trades[0].qty
    assert bbb_trades[1].qty < bbb_trades[0].qty
