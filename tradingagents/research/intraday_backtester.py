"""Research-only mechanical intraday breakout prototype.

This module is intentionally additive. It does not change the existing daily
Minervini backtester or any live-paper execution paths. The first goal is
basic end-to-end validation on intraday data, not optimization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import duckdb
import pandas as pd

from .warehouse import MarketDataWarehouse


@dataclass
class IntradayBacktestConfig:
    initial_cash: float = 100_000.0
    max_positions: int = 6
    max_position_pct: float = 0.10
    stop_loss_pct: float = 0.03
    trail_stop_pct: float = 0.04
    opening_range_bars: int = 1
    min_opening_range_pct: float = 0.0
    min_breakout_distance_pct: float = 0.0
    max_breakout_distance_pct: Optional[float] = None
    require_above_prior_high: bool = False
    latest_entry_bar_in_session: Optional[int] = None
    max_trades_per_symbol_per_day: int = 1
    min_volume_ratio: float = 1.5
    continuation_min_volume_ratio: Optional[float] = None
    continuation_max_distance_from_vwap_pct: Optional[float] = None
    continuation_latest_entry_bar_in_session: Optional[int] = None
    expansion_min_volume_ratio: Optional[float] = None
    expansion_min_breakout_distance_pct: Optional[float] = None
    expansion_min_distance_from_vwap_pct: Optional[float] = None
    expansion_latest_entry_bar_in_session: Optional[int] = None
    expansion_failed_follow_through_bars: Optional[int] = None
    expansion_failed_follow_through_min_return_pct: Optional[float] = None
    use_expansion_confirmation_entry: bool = False
    expansion_confirmation_max_pullback_pct: Optional[float] = None
    expansion_confirmation_reclaim_buffer_pct: float = 0.0
    expansion_confirmation_max_bars_after_signal: Optional[int] = None
    allow_continuation_setup: bool = True
    allow_overextended_setup: bool = True
    allow_expansion_setup: bool = True
    allow_pullback_vwap: bool = False
    pullback_vwap_touch_tolerance_pct: float = 0.002
    pullback_vwap_touch_lookback_bars: int = 4
    pullback_vwap_reclaim_min_pct: float = 0.001
    pullback_vwap_min_session_trend_pct: float = 0.005
    pullback_vwap_min_volume_ratio: float = 1.2
    pullback_vwap_earliest_entry_bar: int = 3
    pullback_vwap_latest_entry_bar: int = 10
    pullback_vwap_min_distance_from_or_high_pct: Optional[float] = None
    pullback_vwap_max_position_pct: Optional[float] = None
    allow_gap_reclaim_long: bool = False
    gap_reclaim_min_gap_down_pct: float = 0.015
    gap_reclaim_max_gap_down_pct: float = 0.06
    gap_reclaim_min_reclaim_fraction: float = 0.5
    gap_reclaim_min_volume_ratio: float = 1.3
    gap_reclaim_earliest_entry_bar: int = 1
    gap_reclaim_latest_entry_bar: int = 4
    gap_reclaim_require_above_session_open: bool = True
    gap_reclaim_max_position_pct: Optional[float] = None
    gap_reclaim_trail_stop_pct: Optional[float] = None
    gap_reclaim_trail_activation_return_pct: Optional[float] = None
    allow_nr4_breakout: bool = False
    nr4_lookback_days: int = 4
    nr4_earliest_entry_bar: int = 1
    nr4_latest_entry_bar: int = 12
    nr4_min_volume_ratio: float = 1.3
    nr4_min_breakout_distance_pct: float = 0.0
    nr4_max_position_pct: Optional[float] = None
    execution_half_spread_bps: float = 0.0
    execution_stop_slippage_bps: float = 0.0
    execution_impact_coeff_bps: float = 0.0
    allow_relative_volume_filter: bool = False
    relative_volume_lookback_days: int = 20
    relative_volume_top_k: int = 20
    benchmark_symbol: str = "SPY"
    min_relative_strength_pct: Optional[float] = None
    min_entry_strength_breakout_pct: Optional[float] = None
    min_entry_strength_vwap_distance_pct: Optional[float] = None
    require_above_vwap: bool = True
    flatten_at_close: bool = True
    interval_minutes: int = 30
    daily_trend_filter: bool = False
    daily_trend_sma: int = 20
    daily_db_path: str = "research_data/market_data.duckdb"


@dataclass
class IntradayPortfolioResult:
    summary: Dict
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    daily_state: pd.DataFrame
    symbol_summary: pd.DataFrame
    setup_summary: pd.DataFrame
    candidate_log: pd.DataFrame
    metadata: Dict


class IntradayBreakoutBacktester:
    """Simple opening-range breakout portfolio backtester on intraday bars."""

    def __init__(self, config: IntradayBacktestConfig | None = None):
        self.config = config or IntradayBacktestConfig()

    def prepare_signals(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Public wrapper for feature + signal computation on a single symbol.

        Input: OHLCV DataFrame indexed by timestamp (regular-session bars).
        Output: the same frame annotated with `entry_signal`, `setup_family`,
        and the supporting feature columns the main loop uses. Intended for
        live orchestrators that need the same signal logic as the backtester
        but do not want to depend on its DuckDB load path.
        """
        return self._prepare_symbol_features(frame)

    def _apply_execution_cost(
        self,
        price: float,
        side: str,
        exit_reason: Optional[str],
        shares: int,
        bar_volume: Optional[float],
    ) -> float:
        cost_bps = float(self.config.execution_half_spread_bps)
        if side == "sell" and exit_reason in {"stop", "failed_follow_through"}:
            cost_bps += float(self.config.execution_stop_slippage_bps)
        impact_coeff = float(self.config.execution_impact_coeff_bps)
        if impact_coeff > 0.0 and bar_volume is not None and bar_volume > 0 and shares > 0:
            cost_bps += impact_coeff * (float(shares) / float(bar_volume))
        if cost_bps == 0.0:
            return price
        cost_fraction = cost_bps / 10_000.0
        if side == "buy":
            return price * (1.0 + cost_fraction)
        return price * (1.0 - cost_fraction)

    @staticmethod
    def _setup_priority(setup_family: str) -> int:
        priorities = {
            "opening_drive_expansion": 6,
            "pullback_vwap": 5,
            "gap_reclaim_long": 4,
            "nr4_breakout": 3,
            "opening_drive_overextended": 2,
            "opening_drive_continuation": 1,
        }
        return priorities.get(str(setup_family), 0)

    def _selection_score(
        self,
        setup_family: str,
        candidate_score: float | None,
        volume_ratio: float | None,
        breakout_distance_pct: float | None,
        distance_from_vwap_pct: float | None,
    ) -> float:
        setup_bonus = self._setup_priority(setup_family) * 10.0
        base_score = float(candidate_score or 0.0)
        volume_bonus = min(max(float(volume_ratio or 0.0), 0.0), 10.0) * 0.5
        breakout_bonus = min(max(float(breakout_distance_pct or 0.0), 0.0), 0.03) * 100.0 * 0.4
        vwap_penalty = min(max(float(distance_from_vwap_pct or 0.0), 0.0), 0.03) * 100.0 * 0.3
        return round(setup_bonus + base_score + volume_bonus + breakout_bonus - vwap_penalty, 4)

    def _build_metadata(
        self,
        symbols: list[str],
        begin: str,
        end: str,
        db_path: str,
        data_by_symbol: dict[str, pd.DataFrame],
    ) -> dict:
        bars_per_symbol = {symbol: int(len(df)) for symbol, df in data_by_symbol.items()}
        sessions_per_symbol = {
            symbol: int(df.index.normalize().nunique()) for symbol, df in data_by_symbol.items()
        }
        return {
            "strategy_name": "intraday_mechanical_breakout",
            "strategy_version": "prototype_v2",
            "config": asdict(self.config),
            "db_path": db_path,
            "begin": begin,
            "end": end,
            "symbols_requested": list(symbols),
            "symbols_loaded": sorted(data_by_symbol.keys()),
            "bars_loaded_per_symbol": bars_per_symbol,
            "sessions_loaded_per_symbol": sessions_per_symbol,
        }

    @staticmethod
    def _table_name(interval_minutes: int) -> str:
        return f"bars_{int(interval_minutes)}m"

    @staticmethod
    def _filter_regular_session(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # Alpaca intraday bars in our DuckDB are stored as naive timestamps that
        # align with the local workstation timezone used during ingestion. In
        # this repo that is America/Chicago, so regular US equities hours are
        # 08:30-15:00 local time. Keep bars whose timestamp falls inside that
        # session window and drop pre/post-market bars for the prototype.
        minutes = df.index.hour * 60 + df.index.minute
        session_start = 8 * 60 + 30
        session_end = 15 * 60
        return df.loc[(minutes >= session_start) & (minutes <= session_end)].copy()

    def _load_intraday_bars(
        self,
        db_path: str,
        symbols: list[str],
        begin: str,
        end: str,
    ) -> dict[str, pd.DataFrame]:
        table_name = self._table_name(self.config.interval_minutes)
        conn = duckdb.connect(db_path, read_only=True)
        frames: dict[str, pd.DataFrame] = {}
        try:
            for symbol in symbols:
                df = conn.execute(
                    f"""
                    SELECT ts, open, high, low, close, volume
                    FROM {table_name}
                    WHERE symbol = ? AND ts >= ? AND ts <= ?
                    ORDER BY ts
                    """,
                    [symbol, f"{begin} 00:00:00", f"{end} 23:59:59"],
                ).fetchdf()
                if df.empty:
                    continue
                df["ts"] = pd.to_datetime(df["ts"], utc=False)
                df = df.set_index("ts").sort_index()
                df = self._filter_regular_session(df)
                if df.empty:
                    continue
                frames[symbol] = df
        finally:
            conn.close()
        return frames

    def _prepare_symbol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy().sort_index()
        frame["session_date"] = frame.index.date
        frame["bar_in_session"] = frame.groupby("session_date").cumcount()
        frame["next_open"] = frame.groupby("session_date")["open"].shift(-1)
        frame["next_ts"] = (
            pd.Series(frame.index, index=frame.index)
            .groupby(frame["session_date"])
            .shift(-1)
        )
        frame["is_last_bar"] = frame["session_date"] != frame["session_date"].shift(-1)

        # Opening range from the first N bars of the session.
        opening_mask = frame["bar_in_session"] < self.config.opening_range_bars
        opening_high = (
            frame["high"].where(opening_mask)
            .groupby(frame["session_date"])
            .transform("max")
        )
        opening_low = (
            frame["low"].where(opening_mask)
            .groupby(frame["session_date"])
            .transform("min")
        )
        frame["opening_range_high"] = opening_high
        frame["opening_range_low"] = opening_low
        frame["opening_range_pct"] = (
            (frame["opening_range_high"] - frame["opening_range_low"])
            / frame["opening_range_high"].replace(0, pd.NA)
        )
        frame["session_open"] = frame.groupby("session_date")["open"].transform("first")
        frame["return_since_open_pct"] = (
            (frame["close"] - frame["session_open"])
            / frame["session_open"].replace(0, pd.NA)
        )
        session_close_by_date = frame.groupby("session_date")["close"].last()
        prior_close_by_date = session_close_by_date.shift(1)
        frame["prior_session_close"] = frame["session_date"].map(prior_close_by_date)

        session_high_by_date = frame.groupby("session_date")["high"].max()
        session_low_by_date = frame.groupby("session_date")["low"].min()
        daily_range_by_date = session_high_by_date - session_low_by_date
        prior_high_by_date = session_high_by_date.shift(1)
        prior_range_by_date = daily_range_by_date.shift(1)
        frame["prior_session_high"] = frame["session_date"].map(prior_high_by_date)
        if self.config.allow_nr4_breakout:
            lookback = max(2, int(self.config.nr4_lookback_days))
            prev_window_min = (
                daily_range_by_date.shift(2)
                .rolling(lookback - 1, min_periods=lookback - 1)
                .min()
            )
            prior_is_nr = (prior_range_by_date < prev_window_min).fillna(False)
            frame["prior_session_is_nr"] = frame["session_date"].map(prior_is_nr).fillna(False)
        else:
            frame["prior_session_is_nr"] = False
        frame["breakout_distance_pct"] = (
            (frame["close"] - frame["opening_range_high"])
            / frame["opening_range_high"].replace(0, pd.NA)
        )
        frame["distance_from_vwap_pct"] = (
            (frame["close"] - frame["vwap"])
            / frame["vwap"].replace(0, pd.NA)
        ) if "vwap" in frame.columns else pd.NA
        frame["prior_bar_high"] = frame["high"].shift(1)
        frame["prior_bar_low"] = frame["low"].shift(1)

        pv = frame["close"] * frame["volume"]
        frame["vwap"] = pv.groupby(frame["session_date"]).cumsum() / frame["volume"].groupby(frame["session_date"]).cumsum().replace(0, pd.NA)
        frame["avg_volume_20"] = frame["volume"].rolling(20, min_periods=20).mean()
        frame["volume_ratio"] = frame["volume"] / frame["avg_volume_20"].replace(0, pd.NA)
        frame["distance_from_vwap_pct"] = (
            (frame["close"] - frame["vwap"])
            / frame["vwap"].replace(0, pd.NA)
        )
        frame["entry_signal"] = (
            (frame["bar_in_session"] >= self.config.opening_range_bars)
            & (frame["close"] > frame["opening_range_high"])
            & (frame["volume_ratio"] >= self.config.min_volume_ratio)
        )
        if self.config.min_opening_range_pct > 0:
            frame["entry_signal"] &= frame["opening_range_pct"] >= self.config.min_opening_range_pct
        if self.config.min_breakout_distance_pct > 0:
            frame["entry_signal"] &= frame["breakout_distance_pct"] >= self.config.min_breakout_distance_pct
        if self.config.max_breakout_distance_pct is not None:
            frame["entry_signal"] &= frame["breakout_distance_pct"] <= self.config.max_breakout_distance_pct
        if self.config.require_above_prior_high:
            frame["entry_signal"] &= frame["close"] > frame["prior_bar_high"]
        if self.config.require_above_vwap:
            frame["entry_signal"] &= frame["close"] > frame["vwap"]
        frame["setup_family"] = pd.Series("none", index=frame.index, dtype="object")
        constructive_continuation_mask = (
            frame["entry_signal"]
            & frame["breakout_distance_pct"].notna()
            & (frame["breakout_distance_pct"] <= 0.01)
        )
        constructive_expansion_mask = (
            frame["entry_signal"]
            & frame["breakout_distance_pct"].notna()
            & (frame["breakout_distance_pct"] > 0.01)
            & frame["distance_from_vwap_pct"].notna()
            & (frame["distance_from_vwap_pct"] <= 0.015)
        )
        overextended_chase_mask = (
            frame["entry_signal"]
            & frame["breakout_distance_pct"].notna()
            & (frame["breakout_distance_pct"] > 0.01)
            & (
                frame["distance_from_vwap_pct"].isna()
                | (frame["distance_from_vwap_pct"] > 0.015)
            )
        )
        frame.loc[constructive_continuation_mask, "setup_family"] = "opening_drive_continuation"
        frame.loc[constructive_expansion_mask, "setup_family"] = "opening_drive_expansion"
        frame.loc[overextended_chase_mask, "setup_family"] = "opening_drive_overextended"
        frame["base_entry_signal"] = frame["entry_signal"]
        frame["base_setup_family"] = frame["setup_family"]
        frame["base_candidate_score"] = pd.NA
        if not self.config.allow_continuation_setup:
            frame.loc[frame["setup_family"] == "opening_drive_continuation", "entry_signal"] = False
            frame.loc[frame["setup_family"] == "opening_drive_continuation", "setup_family"] = "disabled_continuation"
        if not self.config.allow_overextended_setup:
            frame.loc[frame["setup_family"] == "opening_drive_overextended", "entry_signal"] = False
            frame.loc[frame["setup_family"] == "opening_drive_overextended", "setup_family"] = "disabled_overextended"
        if not self.config.allow_expansion_setup:
            frame.loc[frame["setup_family"] == "opening_drive_expansion", "entry_signal"] = False
            frame.loc[frame["setup_family"] == "opening_drive_expansion", "setup_family"] = "disabled_expansion"
        frame["candidate_score"] = (
            frame["volume_ratio"].fillna(0).clip(lower=0, upper=10) * 0.5
            + frame["breakout_distance_pct"].fillna(0).clip(lower=0, upper=0.03) * 100.0 * 0.3
            + frame["opening_range_pct"].fillna(0).clip(lower=0, upper=0.05) * 100.0 * 0.2
        )
        frame["base_candidate_score"] = frame["candidate_score"]
        if self.config.allow_pullback_vwap:
            frame = self._apply_pullback_vwap(frame)
        if self.config.allow_gap_reclaim_long:
            frame = self._apply_gap_reclaim_long(frame)
        if self.config.allow_nr4_breakout:
            frame = self._apply_nr4_breakout(frame)
        if self.config.use_expansion_confirmation_entry:
            frame = self._apply_expansion_confirmation_entry(frame)
        return frame

    def _apply_pullback_vwap(self, frame: pd.DataFrame) -> pd.DataFrame:
        tolerance = float(self.config.pullback_vwap_touch_tolerance_pct)
        lookback = max(1, int(self.config.pullback_vwap_touch_lookback_bars))
        reclaim_min = float(self.config.pullback_vwap_reclaim_min_pct)
        min_trend = float(self.config.pullback_vwap_min_session_trend_pct)
        min_vol = float(self.config.pullback_vwap_min_volume_ratio)
        earliest = int(self.config.pullback_vwap_earliest_entry_bar)
        latest = int(self.config.pullback_vwap_latest_entry_bar)
        or_gap = self.config.pullback_vwap_min_distance_from_or_high_pct

        vwap = frame["vwap"]
        vwap_valid = vwap.notna() & (vwap > 0)
        touch_depth = (frame["low"] / vwap.where(vwap_valid)) - 1.0
        touch_bar = (touch_depth <= tolerance) & vwap_valid
        touch_int = touch_bar.fillna(False).astype(int)
        touched_in_window = (
            touch_int.groupby(frame["session_date"], group_keys=False)
            .apply(lambda s: s.shift(1).rolling(lookback, min_periods=1).max())
            .fillna(0.0)
            .astype(bool)
        )

        strong_threshold = vwap * (1.0 + reclaim_min)
        current_reclaim = (frame["close"] >= strong_threshold) & vwap_valid
        prior_close = frame["close"].shift(1)
        prior_vwap = vwap.shift(1)
        prior_below_strong = (
            (prior_close < prior_vwap * (1.0 + reclaim_min))
            & prior_close.notna()
            & prior_vwap.notna()
        )

        trend_ok = frame["return_since_open_pct"] >= min_trend
        volume_ok = frame["volume_ratio"].fillna(0) >= min_vol
        bar_ok = (frame["bar_in_session"] >= earliest) & (frame["bar_in_session"] <= latest)
        if or_gap is not None:
            or_gap_ok = frame["close"] <= frame["opening_range_high"] * (1.0 - float(or_gap))
        else:
            or_gap_ok = pd.Series(True, index=frame.index)

        already_assigned = frame["setup_family"].isin(
            {
                "opening_drive_continuation",
                "opening_drive_expansion",
                "opening_drive_overextended",
            }
        )

        pullback_mask = (
            touched_in_window
            & current_reclaim
            & prior_below_strong
            & trend_ok
            & volume_ok
            & bar_ok
            & or_gap_ok
            & (~already_assigned)
        ).fillna(False)

        frame.loc[pullback_mask, "entry_signal"] = True
        frame.loc[pullback_mask, "setup_family"] = "pullback_vwap"
        return frame

    def _apply_gap_reclaim_long(self, frame: pd.DataFrame) -> pd.DataFrame:
        min_gap = float(self.config.gap_reclaim_min_gap_down_pct)
        max_gap = float(self.config.gap_reclaim_max_gap_down_pct)
        reclaim_fraction = float(self.config.gap_reclaim_min_reclaim_fraction)
        min_vol = float(self.config.gap_reclaim_min_volume_ratio)
        earliest = int(self.config.gap_reclaim_earliest_entry_bar)
        latest = int(self.config.gap_reclaim_latest_entry_bar)
        require_above_open = bool(self.config.gap_reclaim_require_above_session_open)

        prior_close = frame["prior_session_close"]
        session_open = frame["session_open"]
        prior_valid = prior_close.notna() & (prior_close > 0) & session_open.notna()
        gap_pct = (session_open - prior_close) / prior_close.where(prior_valid)
        gap_ok = prior_valid & (gap_pct <= -min_gap) & (gap_pct >= -max_gap)

        reclaim_threshold = session_open + reclaim_fraction * (prior_close - session_open)
        reclaim_ok = prior_valid & (frame["close"] >= reclaim_threshold)
        prior_bar_close = frame["close"].shift(1)
        same_session_as_prior = frame["session_date"] == frame["session_date"].shift(1)
        prior_below_reclaim = (
            prior_bar_close.notna()
            & same_session_as_prior
            & (prior_bar_close < reclaim_threshold)
        )
        volume_ok = frame["volume_ratio"].fillna(0) >= min_vol
        bar_ok = (frame["bar_in_session"] >= earliest) & (frame["bar_in_session"] <= latest)
        above_open_ok = (
            (frame["close"] > session_open) if require_above_open else pd.Series(True, index=frame.index)
        )

        already_assigned = frame["setup_family"].isin(
            {
                "opening_drive_continuation",
                "opening_drive_expansion",
                "opening_drive_overextended",
                "pullback_vwap",
            }
        )

        gap_reclaim_mask = (
            gap_ok
            & reclaim_ok
            & prior_below_reclaim
            & volume_ok
            & bar_ok
            & above_open_ok
            & (~already_assigned)
        ).fillna(False)

        frame.loc[gap_reclaim_mask, "entry_signal"] = True
        frame.loc[gap_reclaim_mask, "setup_family"] = "gap_reclaim_long"
        return frame

    def _apply_nr4_breakout(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "prior_session_high" not in frame.columns or "prior_session_is_nr" not in frame.columns:
            return frame
        min_vol = float(self.config.nr4_min_volume_ratio)
        earliest = int(self.config.nr4_earliest_entry_bar)
        latest = int(self.config.nr4_latest_entry_bar)
        min_breakout = float(self.config.nr4_min_breakout_distance_pct)

        prior_high = frame["prior_session_high"]
        prior_is_nr = frame["prior_session_is_nr"].fillna(False).astype(bool)
        prior_valid = prior_high.notna() & (prior_high > 0)

        breakout_dist_pct = (frame["close"] - prior_high) / prior_high.where(prior_valid)
        breakout_ok = (
            prior_valid
            & prior_is_nr
            & (frame["close"] > prior_high)
            & (breakout_dist_pct >= min_breakout)
        )

        prior_bar_close = frame["close"].shift(1)
        same_session_as_prior = frame["session_date"] == frame["session_date"].shift(1)
        prior_below_break = (
            prior_bar_close.notna()
            & same_session_as_prior
            & (prior_bar_close <= prior_high)
        )

        volume_ok = frame["volume_ratio"].fillna(0) >= min_vol
        bar_ok = (frame["bar_in_session"] >= earliest) & (frame["bar_in_session"] <= latest)

        already_assigned = frame["setup_family"].isin(
            {
                "opening_drive_continuation",
                "opening_drive_expansion",
                "opening_drive_overextended",
                "pullback_vwap",
                "gap_reclaim_long",
            }
        )

        nr4_mask = (
            breakout_ok
            & prior_below_break
            & volume_ok
            & bar_ok
            & (~already_assigned)
        ).fillna(False)

        frame.loc[nr4_mask, "entry_signal"] = True
        frame.loc[nr4_mask, "setup_family"] = "nr4_breakout"
        return frame

    def _build_relative_volume_universe(
        self,
        prepared: dict[str, pd.DataFrame],
    ) -> dict:
        lookback = int(self.config.relative_volume_lookback_days)
        top_k = int(self.config.relative_volume_top_k)
        if lookback <= 0 or top_k <= 0:
            return {}

        rows: list[pd.DataFrame] = []
        for symbol, frame in prepared.items():
            if frame.empty or "bar_in_session" not in frame.columns:
                continue
            open_bars = frame.loc[frame["bar_in_session"] == 0, ["volume", "session_date"]].copy()
            if open_bars.empty:
                continue
            open_bars = (
                open_bars.drop_duplicates(subset=["session_date"], keep="first")
                .sort_values("session_date")
                .reset_index(drop=True)
            )
            open_bars["rolling_avg"] = (
                open_bars["volume"].shift(1).rolling(lookback, min_periods=lookback).mean()
            )
            open_bars["rel_vol"] = open_bars["volume"] / open_bars["rolling_avg"].replace(0, pd.NA)
            open_bars["symbol"] = symbol
            rows.append(open_bars[["session_date", "symbol", "rel_vol"]])

        if not rows:
            return {}
        merged = pd.concat(rows, ignore_index=True)
        merged = merged.dropna(subset=["rel_vol"])
        universe: dict = {}
        for session_date, group in merged.groupby("session_date"):
            ranked = group.sort_values("rel_vol", ascending=False).head(top_k)
            universe[session_date] = set(ranked["symbol"].tolist())
        return universe

    def _apply_expansion_confirmation_entry(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        expansion_mask = working["base_setup_family"] == "opening_drive_expansion"
        if not expansion_mask.any():
            return working

        working.loc[expansion_mask, "entry_signal"] = False
        working.loc[expansion_mask, "setup_family"] = "waiting_expansion_confirmation"

        for session_date, session in working.groupby("session_date", sort=False):
            expansion_candidates = session.loc[session["base_setup_family"] == "opening_drive_expansion"]
            if expansion_candidates.empty:
                continue

            session_index = session.index
            for signal_ts, signal_row in expansion_candidates.iterrows():
                signal_pos = session_index.get_loc(signal_ts)
                if not isinstance(signal_pos, int):
                    continue
                signal_close = float(signal_row["close"])
                signal_vwap = float(signal_row["vwap"]) if pd.notna(signal_row.get("vwap")) else None
                signal_high = float(signal_row["high"])
                max_bars = self.config.expansion_confirmation_max_bars_after_signal
                end_pos = len(session_index) - 1
                if max_bars is not None:
                    end_pos = min(end_pos, signal_pos + max_bars)
                armed = False

                for follow_pos in range(signal_pos + 1, end_pos + 1):
                    follow_ts = session_index[follow_pos]
                    follow_row = working.loc[follow_ts]

                    close_price = float(follow_row["close"])
                    low_price = float(follow_row["low"])
                    pullback_pct = (signal_close - low_price) / signal_close if signal_close > 0 else 0.0
                    above_vwap = signal_vwap is None or (
                        pd.notna(follow_row.get("vwap")) and close_price >= float(follow_row["vwap"])
                    )
                    above_opening_range = close_price >= float(follow_row["opening_range_high"])
                    within_pullback = (
                        self.config.expansion_confirmation_max_pullback_pct is None
                        or pullback_pct <= self.config.expansion_confirmation_max_pullback_pct
                    )

                    if not armed:
                        if close_price <= signal_close and above_vwap and above_opening_range and within_pullback:
                            armed = True
                            continue
                        if follow_row["base_setup_family"] == "opening_drive_expansion":
                            break
                        continue

                    reclaim_level = signal_high * (1.0 + self.config.expansion_confirmation_reclaim_buffer_pct)
                    if close_price > reclaim_level:
                        working.loc[follow_ts, "entry_signal"] = True
                        working.loc[follow_ts, "setup_family"] = "opening_drive_expansion_confirmation"
                        breakout_distance = (
                            (close_price - float(follow_row["opening_range_high"]))
                            / float(follow_row["opening_range_high"])
                            if float(follow_row["opening_range_high"]) > 0
                            else 0.0
                        )
                        working.loc[follow_ts, "breakout_distance_pct"] = breakout_distance
                        working.loc[follow_ts, "candidate_score"] = (
                            float(follow_row["volume_ratio"] or 0.0) * 0.5
                            + min(max(breakout_distance, 0.0), 0.03) * 100.0 * 0.3
                            + min(max(float(follow_row["opening_range_pct"] or 0.0), 0.0), 0.05) * 100.0 * 0.2
                        )
                        break

        return working

    @staticmethod
    def _build_benchmark_relative_strength(
        prepared: dict[str, pd.DataFrame],
        benchmark_symbol: str,
    ) -> dict[str, pd.Series]:
        benchmark_frame = prepared.get(benchmark_symbol)
        if benchmark_frame is None or benchmark_frame.empty:
            return {}
        benchmark_series = benchmark_frame["return_since_open_pct"]
        relative_strength: dict[str, pd.Series] = {}
        for symbol, frame in prepared.items():
            if frame.empty:
                continue
            joined = frame[["return_since_open_pct"]].join(
                benchmark_series.rename("benchmark_return_since_open_pct"),
                how="left",
            )
            relative_strength[symbol] = (
                joined["return_since_open_pct"] - joined["benchmark_return_since_open_pct"]
            )
        return relative_strength

    def _load_daily_trend_filter(
        self,
        symbols: list[str],
        begin: str,
        end: str,
    ) -> dict[str, pd.Series]:
        if not self.config.daily_trend_filter:
            return {}
        warehouse = MarketDataWarehouse(self.config.daily_db_path, read_only=True)
        trend_map: dict[str, pd.Series] = {}
        try:
            for symbol in symbols:
                df = warehouse.get_daily_bars(symbol, begin, end)
                if df is None or df.empty:
                    continue
                daily = df.copy().sort_index()
                daily["sma_trend"] = daily["close"].rolling(
                    self.config.daily_trend_sma, min_periods=self.config.daily_trend_sma
                ).mean()
                trend_map[symbol] = (daily["close"] > daily["sma_trend"]).fillna(False)
        finally:
            warehouse.close()
        return trend_map

    @staticmethod
    def _symbol_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
        if trades_df.empty:
            return pd.DataFrame()
        grouped = trades_df.groupby("symbol").agg(
            trades=("symbol", "count"),
            total_pnl=("pnl", "sum"),
            total_return=("return_pct", "sum"),
            win_rate=("pnl", lambda s: float((s > 0).mean())),
        )
        return grouped.reset_index().sort_values("total_pnl", ascending=False)

    @staticmethod
    def _setup_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
        if trades_df.empty or "setup_family" not in trades_df.columns:
            return pd.DataFrame()
        grouped = trades_df.groupby("setup_family").agg(
            trades=("setup_family", "count"),
            total_pnl=("pnl", "sum"),
            total_return=("return_pct", "sum"),
            avg_return=("return_pct", "mean"),
            win_rate=("pnl", lambda s: float((s > 0).mean())),
            avg_bars_held=("bars_held", "mean"),
        )
        return grouped.reset_index().sort_values("total_pnl", ascending=False)

    def backtest_portfolio(
        self,
        symbols: list[str],
        begin: str,
        end: str,
        db_path: str,
    ) -> IntradayPortfolioResult:
        data_by_symbol = self._load_intraday_bars(db_path, symbols, begin, end)
        daily_trend = self._load_daily_trend_filter(symbols, begin, end)
        prepared = {
            symbol: self._prepare_symbol_features(df)
            for symbol, df in data_by_symbol.items()
            if not df.empty
        }
        relative_strength_map = self._build_benchmark_relative_strength(
            prepared,
            self.config.benchmark_symbol,
        )
        relative_volume_universe = (
            self._build_relative_volume_universe(prepared)
            if self.config.allow_relative_volume_filter
            else {}
        )
        if not prepared:
            empty = pd.DataFrame()
            return IntradayPortfolioResult(
                {},
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                {
                    "strategy_name": "intraday_mechanical_breakout",
                    "strategy_version": "prototype_v2",
                    "config": asdict(self.config),
                    "db_path": db_path,
                    "begin": begin,
                    "end": end,
                    "symbols_requested": list(symbols),
                    "symbols_loaded": [],
                },
            )

        all_ts = sorted({ts for frame in prepared.values() for ts in frame.index})
        cash = float(self.config.initial_cash)
        positions: dict[str, dict] = {}
        pending_entries: dict[pd.Timestamp, list[dict]] = {}
        trades: list[dict] = []
        equity_curve: list[dict] = []
        daily_state: list[dict] = []
        candidate_log: list[dict] = []
        dropped_entries: list[dict] = []
        running_peak = float(self.config.initial_cash)
        max_drawdown = 0.0
        last_day = None
        trades_per_symbol_day: dict[tuple[str, str], int] = {}

        for ts in all_ts:
            # Execute pending entries at this bar's open.
            queued_entries = sorted(
                pending_entries.pop(ts, []),
                key=lambda entry: (
                    float(entry.get("selection_score") or 0.0),
                    self._setup_priority(entry.get("setup_family", "unknown")),
                    float(entry.get("candidate_score") or 0.0),
                ),
                reverse=True,
            )
            for entry in queued_entries:
                symbol = entry["symbol"]
                if symbol in positions:
                    dropped_entries.append(
                        {
                            "ts": ts.isoformat(),
                            "symbol": symbol,
                            "setup_family": entry.get("setup_family", "unknown"),
                            "candidate_score": entry.get("candidate_score"),
                            "selection_score": entry.get("selection_score"),
                            "drop_reason": "already_in_position",
                        }
                    )
                    continue
                if len(positions) >= self.config.max_positions:
                    dropped_entries.append(
                        {
                            "ts": ts.isoformat(),
                            "symbol": symbol,
                            "setup_family": entry.get("setup_family", "unknown"),
                            "candidate_score": entry.get("candidate_score"),
                            "selection_score": entry.get("selection_score"),
                            "drop_reason": "max_positions",
                        }
                    )
                    continue
                frame = prepared.get(symbol)
                if frame is None or ts not in frame.index:
                    dropped_entries.append(
                        {
                            "ts": ts.isoformat(),
                            "symbol": symbol,
                            "setup_family": entry.get("setup_family", "unknown"),
                            "candidate_score": entry.get("candidate_score"),
                            "selection_score": entry.get("selection_score"),
                            "drop_reason": "missing_bar",
                        }
                    )
                    continue
                row = frame.loc[ts]
                price = float(row["open"])
                if price <= 0:
                    dropped_entries.append(
                        {
                            "ts": ts.isoformat(),
                            "symbol": symbol,
                            "setup_family": entry.get("setup_family", "unknown"),
                            "candidate_score": entry.get("candidate_score"),
                            "selection_score": entry.get("selection_score"),
                            "drop_reason": "invalid_open_price",
                        }
                    )
                    continue
                trade_day = str(ts.date())
                trade_key = (symbol, trade_day)
                if trades_per_symbol_day.get(trade_key, 0) >= self.config.max_trades_per_symbol_per_day:
                    dropped_entries.append(
                        {
                            "ts": ts.isoformat(),
                            "symbol": symbol,
                            "setup_family": entry.get("setup_family", "unknown"),
                            "candidate_score": entry.get("candidate_score"),
                            "selection_score": entry.get("selection_score"),
                            "drop_reason": "max_trades_per_symbol_day",
                        }
                    )
                    continue
                equity = cash + sum(
                    pos["shares"] * float(prepared[sym].loc[ts, "close"])
                    for sym, pos in positions.items()
                    if ts in prepared[sym].index
                )
                family_for_sizing = entry.get("setup_family", "unknown")
                position_pct = self.config.max_position_pct
                if (
                    family_for_sizing == "pullback_vwap"
                    and self.config.pullback_vwap_max_position_pct is not None
                ):
                    position_pct = self.config.pullback_vwap_max_position_pct
                elif (
                    family_for_sizing == "gap_reclaim_long"
                    and self.config.gap_reclaim_max_position_pct is not None
                ):
                    position_pct = self.config.gap_reclaim_max_position_pct
                elif (
                    family_for_sizing == "nr4_breakout"
                    and self.config.nr4_max_position_pct is not None
                ):
                    position_pct = self.config.nr4_max_position_pct
                budget = min(cash, equity * position_pct)
                shares = int(budget / price)
                if shares <= 0:
                    dropped_entries.append(
                        {
                            "ts": ts.isoformat(),
                            "symbol": symbol,
                            "setup_family": entry.get("setup_family", "unknown"),
                            "candidate_score": entry.get("candidate_score"),
                            "selection_score": entry.get("selection_score"),
                            "drop_reason": "insufficient_budget",
                        }
                    )
                    continue
                bar_volume = float(row["volume"]) if "volume" in row.index and pd.notna(row["volume"]) else None
                fill_price = self._apply_execution_cost(price, "buy", None, shares, bar_volume)
                cash -= shares * fill_price
                positions[symbol] = {
                    "entry_time": ts,
                    "entry_price": fill_price,
                    "shares": shares,
                    "high_since_entry": fill_price,
                    "stop_price": fill_price * (1.0 - self.config.stop_loss_pct),
                    "setup_family": entry.get("setup_family", "unknown"),
                    "signal_time": entry.get("signal_time"),
                    "candidate_score": entry.get("candidate_score"),
                    "selection_score": entry.get("selection_score"),
                }
                trades_per_symbol_day[trade_key] = trades_per_symbol_day.get(trade_key, 0) + 1

            # Process exits and new signals for symbols that have a bar at ts.
            for symbol, frame in prepared.items():
                if ts not in frame.index:
                    continue
                row = frame.loc[ts]
                relative_strength = None
                rs_series = relative_strength_map.get(symbol)
                if rs_series is not None and ts in rs_series.index and pd.notna(rs_series.loc[ts]):
                    relative_strength = float(rs_series.loc[ts])

                if symbol in positions:
                    pos = positions[symbol]
                    pos["high_since_entry"] = max(pos["high_since_entry"], float(row["high"]))
                    trail_stop_pct = self.config.trail_stop_pct
                    if (
                        pos["setup_family"] == "gap_reclaim_long"
                        and self.config.gap_reclaim_trail_stop_pct is not None
                    ):
                        trail_stop_pct = self.config.gap_reclaim_trail_stop_pct
                    activate_trail = True
                    if (
                        pos["setup_family"] == "gap_reclaim_long"
                        and self.config.gap_reclaim_trail_activation_return_pct is not None
                    ):
                        best_return = (pos["high_since_entry"] / pos["entry_price"]) - 1.0
                        activate_trail = best_return >= self.config.gap_reclaim_trail_activation_return_pct
                    if activate_trail:
                        trail_stop = pos["high_since_entry"] * (1.0 - trail_stop_pct)
                        pos["stop_price"] = max(pos["stop_price"], trail_stop)

                    exit_price = None
                    exit_reason = None
                    if float(row["low"]) <= pos["stop_price"]:
                        exit_price = min(float(row["open"]), pos["stop_price"])
                        exit_reason = "stop"
                    if (
                        exit_price is None
                        and (
                        pos["setup_family"] == "opening_drive_expansion"
                        and self.config.expansion_failed_follow_through_bars is not None
                        and self.config.expansion_failed_follow_through_min_return_pct is not None
                        )
                    ):
                        bars_held_now = int(
                            frame.loc[(frame.index >= pos["entry_time"]) & (frame.index <= ts)].shape[0]
                        )
                        current_return = (float(row["close"]) / pos["entry_price"]) - 1.0
                        if (
                            bars_held_now >= self.config.expansion_failed_follow_through_bars
                            and current_return < self.config.expansion_failed_follow_through_min_return_pct
                        ):
                            exit_price = float(row["close"])
                            exit_reason = "failed_follow_through"
                    if exit_price is None and self.config.flatten_at_close and bool(row["is_last_bar"]):
                        exit_price = float(row["close"])
                        exit_reason = "eod_flatten"

                    if exit_price is not None:
                        shares = pos["shares"]
                        bar_volume_exit = float(row["volume"]) if "volume" in row.index and pd.notna(row["volume"]) else None
                        exit_price = self._apply_execution_cost(
                            exit_price, "sell", exit_reason, shares, bar_volume_exit
                        )
                        proceeds = shares * exit_price
                        cost = shares * pos["entry_price"]
                        cash += proceeds
                        trades.append(
                            {
                                "symbol": symbol,
                                "entry_time": pos["entry_time"].isoformat(),
                                "exit_time": ts.isoformat(),
                                "signal_time": pos["signal_time"],
                                "entry_price": round(pos["entry_price"], 4),
                                "exit_price": round(exit_price, 4),
                                "shares": shares,
                                "pnl": round(proceeds - cost, 2),
                                "return_pct": round((exit_price / pos["entry_price"]) - 1.0, 4),
                                "bars_held": int(
                                    frame.loc[(frame.index >= pos["entry_time"]) & (frame.index <= ts)].shape[0]
                                ),
                                "setup_family": pos["setup_family"],
                                "candidate_score": pos["candidate_score"],
                                "selection_score": pos["selection_score"],
                                "exit_reason": exit_reason,
                            }
                        )
                        del positions[symbol]
                        continue

                # Queue entry for next bar open. One position per symbol, one signal per bar.
                if symbol not in positions and bool(row["entry_signal"]):
                    candidate = {
                        "ts": ts.isoformat(),
                        "symbol": symbol,
                        "session_date": str(ts.date()),
                        "bar_in_session": int(row["bar_in_session"]),
                        "close": round(float(row["close"]), 4),
                        "opening_range_high": round(float(row["opening_range_high"]), 4),
                        "opening_range_low": round(float(row["opening_range_low"]), 4),
                        "opening_range_pct": round(float(row["opening_range_pct"]), 6)
                        if pd.notna(row["opening_range_pct"])
                        else None,
                        "return_since_open_pct": round(float(row["return_since_open_pct"]), 6)
                        if pd.notna(row.get("return_since_open_pct"))
                        else None,
                        "breakout_distance_pct": round(float(row["breakout_distance_pct"]), 6)
                        if pd.notna(row["breakout_distance_pct"])
                        else None,
                        "distance_from_vwap_pct": round(float(row["distance_from_vwap_pct"]), 6)
                        if pd.notna(row.get("distance_from_vwap_pct"))
                        else None,
                        "volume_ratio": round(float(row["volume_ratio"]), 4)
                        if pd.notna(row["volume_ratio"])
                        else None,
                        "setup_family": row.get("setup_family", "unknown"),
                        "candidate_score": round(float(row["candidate_score"]), 4)
                        if pd.notna(row.get("candidate_score"))
                        else None,
                        "relative_strength_pct": round(relative_strength, 6)
                        if relative_strength is not None
                        else None,
                        "above_vwap": bool(pd.notna(row["vwap"]) and float(row["close"]) > float(row["vwap"])),
                        "above_prior_high": bool(
                            pd.notna(row["prior_bar_high"]) and float(row["close"]) > float(row["prior_bar_high"])
                        ),
                        "selected": False,
                        "filter_reason": None,
                    }
                    candidate["selection_score"] = self._selection_score(
                        candidate["setup_family"],
                        candidate["candidate_score"],
                        candidate["volume_ratio"],
                        candidate["breakout_distance_pct"],
                        candidate["distance_from_vwap_pct"],
                    )
                    if self.config.daily_trend_filter:
                        trend_series = daily_trend.get(symbol)
                        if trend_series is None:
                            candidate["filter_reason"] = "missing_daily_trend"
                            candidate_log.append(candidate)
                            continue
                        trend_day = pd.Timestamp(ts.date())
                        valid = trend_series.loc[trend_series.index <= trend_day]
                        if valid.empty or not bool(valid.iloc[-1]):
                            candidate["filter_reason"] = "daily_trend_filter"
                            candidate_log.append(candidate)
                            continue
                    if self.config.allow_relative_volume_filter:
                        session_date = ts.date()
                        top_k_symbols = relative_volume_universe.get(session_date)
                        if top_k_symbols is None or symbol not in top_k_symbols:
                            candidate["filter_reason"] = "relative_volume_universe"
                            candidate_log.append(candidate)
                            continue
                    if (
                        self.config.min_relative_strength_pct is not None
                        and symbol != self.config.benchmark_symbol
                    ):
                        if (
                            candidate["relative_strength_pct"] is None
                            or candidate["relative_strength_pct"] < self.config.min_relative_strength_pct
                        ):
                            candidate["filter_reason"] = "relative_strength_filter"
                            candidate_log.append(candidate)
                            continue
                    if (
                        self.config.min_entry_strength_breakout_pct is not None
                        and candidate["setup_family"] not in {"pullback_vwap", "gap_reclaim_long", "nr4_breakout"}
                        and (
                            candidate["breakout_distance_pct"] is None
                            or candidate["breakout_distance_pct"] < self.config.min_entry_strength_breakout_pct
                        )
                    ):
                        candidate["filter_reason"] = "entry_strength_breakout"
                        candidate_log.append(candidate)
                        continue
                    if (
                        self.config.min_entry_strength_vwap_distance_pct is not None
                        and candidate["setup_family"] not in {"pullback_vwap", "gap_reclaim_long", "nr4_breakout"}
                        and (
                            candidate["distance_from_vwap_pct"] is None
                            or candidate["distance_from_vwap_pct"] < self.config.min_entry_strength_vwap_distance_pct
                        )
                    ):
                        candidate["filter_reason"] = "entry_strength_vwap"
                        candidate_log.append(candidate)
                        continue
                    if (
                        self.config.latest_entry_bar_in_session is not None
                        and int(row["bar_in_session"]) >= self.config.latest_entry_bar_in_session
                    ):
                        candidate["filter_reason"] = "latest_entry_bar"
                        candidate_log.append(candidate)
                        continue
                    if candidate["setup_family"] == "opening_drive_continuation":
                        if (
                            self.config.continuation_latest_entry_bar_in_session is not None
                            and int(row["bar_in_session"]) >= self.config.continuation_latest_entry_bar_in_session
                        ):
                            candidate["filter_reason"] = "continuation_latest_entry_bar"
                            candidate_log.append(candidate)
                            continue
                        if (
                            self.config.continuation_min_volume_ratio is not None
                            and (
                                pd.isna(row["volume_ratio"])
                                or float(row["volume_ratio"]) < self.config.continuation_min_volume_ratio
                            )
                        ):
                            candidate["filter_reason"] = "continuation_volume_ratio"
                            candidate_log.append(candidate)
                            continue
                        if (
                            self.config.continuation_max_distance_from_vwap_pct is not None
                            and (
                                pd.isna(row["distance_from_vwap_pct"])
                                or float(row["distance_from_vwap_pct"])
                                > self.config.continuation_max_distance_from_vwap_pct
                            )
                        ):
                            candidate["filter_reason"] = "continuation_vwap_distance"
                            candidate_log.append(candidate)
                            continue
                    if candidate["setup_family"] == "opening_drive_expansion":
                        if (
                            self.config.expansion_latest_entry_bar_in_session is not None
                            and int(row["bar_in_session"]) >= self.config.expansion_latest_entry_bar_in_session
                        ):
                            candidate["filter_reason"] = "expansion_latest_entry_bar"
                            candidate_log.append(candidate)
                            continue
                        if (
                            self.config.expansion_min_volume_ratio is not None
                            and (
                                pd.isna(row["volume_ratio"])
                                or float(row["volume_ratio"]) < self.config.expansion_min_volume_ratio
                            )
                        ):
                            candidate["filter_reason"] = "expansion_volume_ratio"
                            candidate_log.append(candidate)
                            continue
                        if (
                            self.config.expansion_min_breakout_distance_pct is not None
                            and (
                                pd.isna(row["breakout_distance_pct"])
                                or float(row["breakout_distance_pct"])
                                < self.config.expansion_min_breakout_distance_pct
                            )
                        ):
                            candidate["filter_reason"] = "expansion_breakout_distance"
                            candidate_log.append(candidate)
                            continue
                        if (
                            self.config.expansion_min_distance_from_vwap_pct is not None
                            and (
                                pd.isna(row["distance_from_vwap_pct"])
                                or float(row["distance_from_vwap_pct"])
                                < self.config.expansion_min_distance_from_vwap_pct
                            )
                        ):
                            candidate["filter_reason"] = "expansion_vwap_distance"
                            candidate_log.append(candidate)
                            continue
                    signal_day = str(ts.date())
                    signal_key = (symbol, signal_day)
                    if trades_per_symbol_day.get(signal_key, 0) >= self.config.max_trades_per_symbol_per_day:
                        candidate["filter_reason"] = "max_trades_per_symbol_day"
                        candidate_log.append(candidate)
                        continue
                    next_ts = row.get("next_ts")
                    next_open = row.get("next_open")
                    if pd.notna(next_ts) and pd.notna(next_open):
                        candidate["selected"] = True
                        candidate["filter_reason"] = "queued_next_bar_open"
                        candidate["queued_entry_ts"] = pd.Timestamp(next_ts).isoformat()
                        candidate["queued_entry_price"] = round(float(next_open), 4)
                        candidate_log.append(candidate)
                        pending_entries.setdefault(pd.Timestamp(next_ts), []).append(
                            {
                                "symbol": symbol,
                                "setup_family": candidate["setup_family"],
                                "signal_time": candidate["ts"],
                                "candidate_score": candidate["candidate_score"],
                                "selection_score": candidate["selection_score"],
                            }
                        )
                    else:
                        candidate["filter_reason"] = "no_next_bar"
                        candidate_log.append(candidate)

            # Mark portfolio state.
            market_value = 0.0
            for symbol, pos in positions.items():
                frame = prepared[symbol]
                if ts in frame.index:
                    market_value += pos["shares"] * float(frame.loc[ts, "close"])
            equity = cash + market_value
            running_peak = max(running_peak, equity)
            if running_peak > 0:
                max_drawdown = max(max_drawdown, (running_peak - equity) / running_peak)
            exposure = market_value / equity if equity > 0 else 0.0
            equity_curve.append(
                {
                    "ts": ts.isoformat(),
                    "equity": round(equity, 2),
                    "cash": round(cash, 2),
                    "market_value": round(market_value, 2),
                    "exposure": round(exposure, 4),
                }
            )
            cur_day = str(ts.date())
            if cur_day != last_day:
                daily_state.append(
                    {
                        "date": cur_day,
                        "positions": len(positions),
                        "equity": round(equity, 2),
                        "cash": round(cash, 2),
                        "exposure": round(exposure, 4),
                    }
                )
                last_day = cur_day

        trades_df = pd.DataFrame(trades)
        equity_curve_df = pd.DataFrame(equity_curve)
        daily_state_df = pd.DataFrame(daily_state)
        symbol_summary_df = self._symbol_summary(trades_df)
        setup_summary_df = self._setup_summary(trades_df)
        candidate_log_df = pd.DataFrame(candidate_log)
        total_return = (equity_curve_df["equity"].iloc[-1] / self.config.initial_cash) - 1.0
        win_rate = float((trades_df["pnl"] > 0).mean()) if not trades_df.empty else 0.0
        n_candidates = int(len(candidate_log_df))
        n_candidates_selected = (
            int(candidate_log_df["selected"].sum()) if not candidate_log_df.empty else 0
        )
        filter_reason_counts = (
            candidate_log_df["filter_reason"].value_counts().to_dict()
            if not candidate_log_df.empty
            else {}
        )
        dropped_entry_reason_counts = (
            pd.DataFrame(dropped_entries)["drop_reason"].value_counts().to_dict()
            if dropped_entries
            else {}
        )
        dropped_entry_setup_counts = (
            pd.DataFrame(dropped_entries)["setup_family"].value_counts().to_dict()
            if dropped_entries
            else {}
        )
        setup_family_counts = (
            candidate_log_df["setup_family"].value_counts().to_dict()
            if not candidate_log_df.empty and "setup_family" in candidate_log_df
            else {}
        )
        selected_setup_family_counts = (
            candidate_log_df.loc[candidate_log_df["selected"], "setup_family"].value_counts().to_dict()
            if not candidate_log_df.empty and "setup_family" in candidate_log_df
            else {}
        )
        entry_hour_counts = (
            trades_df.assign(entry_hour=pd.to_datetime(trades_df["entry_time"]).dt.hour)["entry_hour"]
            .value_counts()
            .sort_index()
            .to_dict()
            if not trades_df.empty
            else {}
        )
        exit_reason_counts = (
            trades_df["exit_reason"].value_counts().to_dict() if not trades_df.empty else {}
        )
        trade_setup_family_counts = (
            trades_df["setup_family"].value_counts().to_dict()
            if not trades_df.empty and "setup_family" in trades_df
            else {}
        )
        trade_setup_pnl = (
            trades_df.groupby("setup_family")["pnl"].sum().round(2).to_dict()
            if not trades_df.empty and "setup_family" in trades_df
            else {}
        )
        trade_setup_win_rate = (
            trades_df.groupby("setup_family")["pnl"].apply(lambda s: float((s > 0).mean())).round(4).to_dict()
            if not trades_df.empty and "setup_family" in trades_df
            else {}
        )

        summary = {
            "initial_cash": round(self.config.initial_cash, 2),
            "final_equity": round(float(equity_curve_df["equity"].iloc[-1]), 2),
            "total_return_pct": round(total_return * 100.0, 2),
            "max_drawdown_pct": round(max_drawdown * 100.0, 2),
            "total_trades": int(len(trades_df)),
            "win_rate": round(win_rate, 4),
            "avg_return": round(float(trades_df["return_pct"].mean()), 4) if not trades_df.empty else 0.0,
            "avg_bars_held": round(float(trades_df["bars_held"].mean()), 1) if not trades_df.empty else 0.0,
            "interval_minutes": int(self.config.interval_minutes),
            "symbols_tested": len(prepared),
            "flatten_at_close": bool(self.config.flatten_at_close),
            "candidates_seen": n_candidates,
            "candidates_selected": n_candidates_selected,
            "candidate_select_rate": round(
                (n_candidates_selected / n_candidates), 4
            )
            if n_candidates > 0
            else 0.0,
            "setup_family_counts": setup_family_counts,
            "selected_setup_family_counts": selected_setup_family_counts,
            "trade_setup_family_counts": trade_setup_family_counts,
            "trade_setup_pnl": trade_setup_pnl,
            "trade_setup_win_rate": trade_setup_win_rate,
            "filter_reason_counts": filter_reason_counts,
            "dropped_entry_reason_counts": dropped_entry_reason_counts,
            "dropped_entry_setup_counts": dropped_entry_setup_counts,
            "exit_reason_counts": exit_reason_counts,
            "entry_hour_counts": entry_hour_counts,
        }
        metadata = self._build_metadata(symbols, begin, end, db_path, data_by_symbol)
        return IntradayPortfolioResult(
            summary=summary,
            trades=trades_df,
            equity_curve=equity_curve_df,
            daily_state=daily_state_df,
            symbol_summary=symbol_summary_df,
            setup_summary=setup_summary_df,
            candidate_log=candidate_log_df,
            metadata=metadata,
        )
