"""Minervini-style screening logic for swing-trading research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class MinerviniConfig:
    sma_short: int = 50
    sma_medium: int = 150
    sma_long: int = 200
    breakout_lookback: int = 20
    volume_window: int = 50
    min_avg_volume: float = 200_000
    min_avg_dollar_volume: float = 10_000_000
    min_rs_percentile: float = 70.0
    min_above_52w_low: float = 0.30
    max_below_52w_high: float = 0.25
    volume_surge_multiple: float = 1.5
    base_window: int = 65
    min_base_depth: float = 0.08
    max_base_depth: float = 0.35
    max_base_tightness: float = 0.12
    min_revenue_growth: float = 0.15
    min_eps_growth: float = 0.15
    min_return_on_equity: float = 0.15
    require_fundamentals: bool = True
    require_market_uptrend: bool = True
    require_acceleration: bool = False
    min_days_to_earnings: int = 5
    handle_window: int = 15
    max_handle_depth: float = 0.12
    breakout_ready_threshold: float = 0.05
    stage_lookback: int = 252
    stage_breakout_lookback: int = 120
    max_stage_number: int = 2
    pivot_buffer_pct: float = 0.001
    max_buy_zone_pct: float = 0.05
    min_initial_stop_pct: float = 0.03
    max_initial_stop_pct: float = 0.08
    leader_continuation_enabled: bool = True
    leader_continuation_min_rs_percentile: float = 75.0
    leader_continuation_min_close_range_pct: float = 0.15
    leader_continuation_min_adx_14: float = 12.0
    leader_continuation_min_roc_60: float = 0.0
    leader_continuation_min_roc_120: float = 0.0
    leader_continuation_max_extension_pct: float = 0.07
    leader_continuation_max_pullback_pct: float = 0.08


class MinerviniScreener:
    """Approximate the SEPA trend-template workflow with reproducible rules."""

    def __init__(self, config: MinerviniConfig | None = None):
        self.config = config or MinerviniConfig()

    @staticmethod
    def _safe_bool(value) -> bool:
        if pd.isna(value):
            return False
        return bool(value)

    @staticmethod
    def _safe_float(value, default: Optional[float] = None) -> Optional[float]:
        if value is None or pd.isna(value):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy().sort_index()
        if features.empty:
            return features

        close = features["close"]
        high = features["high"]
        low = features["low"]
        volume = features["volume"]
        prev_close = close.shift(1)
        daily_returns = close.pct_change()
        intraday_range = (high - low).replace(0, np.nan)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        atr_14 = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        plus_di_14 = (
            100.0
            * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
            / atr_14.replace(0, np.nan)
        )
        minus_di_14 = (
            100.0
            * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
            / atr_14.replace(0, np.nan)
        )
        dx = (
            100.0
            * (plus_di_14 - minus_di_14).abs()
            / (plus_di_14 + minus_di_14).replace(0, np.nan)
        )
        delta = close.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)

        features["ema_10"] = close.ewm(span=10, adjust=False).mean()
        features["ema_21"] = close.ewm(span=21, adjust=False).mean()
        features["sma_50"] = close.rolling(self.config.sma_short).mean()
        features["sma_150"] = close.rolling(self.config.sma_medium).mean()
        features["sma_200"] = close.rolling(self.config.sma_long).mean()
        features["sma_200_20d_ago"] = features["sma_200"].shift(20)
        features["52w_high"] = high.rolling(252).max()
        features["52w_low"] = low.rolling(252).min()
        features["avg_volume_10"] = volume.rolling(10).mean()
        features["avg_volume_50"] = volume.rolling(self.config.volume_window).mean()
        features["avg_dollar_volume_50"] = features["avg_volume_50"] * close
        features["close_range_pct"] = (close - low) / intraday_range
        features["atr_14"] = atr_14
        features["atr_pct_14"] = atr_14 / close.replace(0, np.nan)
        features["plus_di_14"] = plus_di_14
        features["minus_di_14"] = minus_di_14
        features["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        features["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
        features["roc_20"] = close.pct_change(20)
        features["roc_60"] = close.pct_change(60)
        features["roc_120"] = close.pct_change(120)
        features["volatility_20"] = daily_returns.rolling(20).std() * (252**0.5)
        features["handle_high"] = high.rolling(self.config.handle_window).max().shift(1)
        features["handle_low"] = low.rolling(self.config.handle_window).min().shift(1)
        features["handle_depth_pct"] = (
            (features["handle_high"] - features["handle_low"])
            / features["handle_high"].replace(0, np.nan)
        )
        features["pivot_price"] = high.rolling(self.config.breakout_lookback).max().shift(1)
        features["base_high"] = high.rolling(self.config.base_window).max().shift(1)
        features["base_low"] = low.rolling(self.config.base_window).min().shift(1)
        features["base_depth_pct"] = (
            (features["base_high"] - features["base_low"])
            / features["base_high"].replace(0, np.nan)
        )
        features["distance_from_base_high_pct"] = (
            (features["base_high"] - close) / features["base_high"].replace(0, np.nan)
        )
        features["drawdown_from_52w_high_pct"] = (
            (features["52w_high"] - close) / features["52w_high"].replace(0, np.nan)
        )
        features["prior_stage_high"] = (
            high.rolling(self.config.stage_breakout_lookback).max().shift(1)
        )

        range_pct = (high - low) / close.replace(0, np.nan)
        features["range_pct_10"] = range_pct.rolling(10).mean()
        features["range_pct_20"] = range_pct.rolling(20).mean()
        features["range_pct_40"] = range_pct.rolling(40).mean()
        features["tight_closes_10"] = (
            (close.rolling(10).max() - close.rolling(10).min())
            / close.replace(0, np.nan)
        )
        features["volume_contraction_ratio"] = (
            features["avg_volume_10"] / features["avg_volume_50"].replace(0, np.nan)
        )
        features["base_recovery_pct"] = (
            (close - features["base_low"])
            / (features["base_high"] - features["base_low"]).replace(0, np.nan)
        )
        features["vcp_candidate"] = (
            (features["range_pct_10"] < features["range_pct_20"])
            & (features["range_pct_20"] < features["range_pct_40"])
            & (features["avg_volume_10"] < features["avg_volume_50"])
            & (features["tight_closes_10"] < 0.10)
            & (close > features["sma_50"])
        )
        features["flat_base_candidate"] = (
            (features["base_depth_pct"] >= self.config.min_base_depth)
            & (features["base_depth_pct"] <= 0.15)
            & (features["tight_closes_10"] <= 0.08)
            & (features["volume_contraction_ratio"] <= 0.9)
            & (features["distance_from_base_high_pct"] <= 0.08)
            & (close > features["sma_50"])
        )
        features["cup_candidate"] = (
            (features["base_depth_pct"] >= 0.15)
            & (features["base_depth_pct"] <= self.config.max_base_depth)
            & (features["base_recovery_pct"] >= 0.70)
            & (close > features["sma_50"])
            & (close > features["sma_200"])
        )
        features["cup_handle_candidate"] = (
            features["cup_candidate"]
            & (features["handle_depth_pct"] <= self.config.max_handle_depth)
            & (features["volume_contraction_ratio"] <= 0.85)
            & (
                (features["handle_high"] - close)
                / features["handle_high"].replace(0, np.nan)
                <= 0.08
            )
        )
        features["base_candidate"] = (
            features["flat_base_candidate"]
            | features["cup_handle_candidate"]
            | features["vcp_candidate"]
        )
        features["base_label"] = "none"
        features.loc[features["base_candidate"], "base_label"] = "consolidation"
        features.loc[features["flat_base_candidate"], "base_label"] = "flat_base"
        features.loc[features["vcp_candidate"], "base_label"] = "vcp"
        features.loc[features["cup_handle_candidate"], "base_label"] = "cup_handle"
        features.loc[features["cup_handle_candidate"], "pivot_price"] = features["handle_high"]
        features.loc[features["flat_base_candidate"], "pivot_price"] = features["base_high"]
        features["stage_breakout_candidate"] = (
            (close > features["prior_stage_high"] * (1.0 + self.config.pivot_buffer_pct))
            & (close > features["sma_50"])
            & (features["avg_volume_10"] >= features["avg_volume_50"])
        )
        prior_stage_breakout = features["stage_breakout_candidate"].shift(
            1,
            fill_value=False,
        ).astype(bool)
        features["stage_breakout_event"] = (
            features["stage_breakout_candidate"] & ~prior_stage_breakout
        )
        features["stage_number"] = (
            features["stage_breakout_event"]
            .shift(1)
            .rolling(self.config.stage_lookback, min_periods=1)
            .sum()
            .fillna(0)
            + 1
        )
        features["preferred_stage"] = (
            features["stage_number"] <= self.config.max_stage_number
        )
        features["buy_point"] = (
            features["pivot_price"] * (1.0 + self.config.pivot_buffer_pct)
        )
        features["buy_limit_price"] = (
            features["buy_point"] * (1.0 + self.config.max_buy_zone_pct)
        )
        features["stop_reference"] = features["base_low"]
        features.loc[features["cup_handle_candidate"], "stop_reference"] = features["handle_low"]
        features["initial_stop_pct"] = (
            (features["buy_point"] - features["stop_reference"])
            / features["buy_point"].replace(0, np.nan)
        ).clip(
            lower=self.config.min_initial_stop_pct,
            upper=self.config.max_initial_stop_pct,
        )
        features["initial_stop_price"] = (
            features["buy_point"] * (1.0 - features["initial_stop_pct"])
        )
        features["distance_to_buy_point_pct"] = (
            (features["buy_point"] - close) / features["buy_point"].replace(0, np.nan)
        )
        features["buy_zone_pct"] = (
            (close - features["buy_point"]) / features["buy_point"].replace(0, np.nan)
        )
        features["breakout_volume_ratio"] = (
            volume / features["avg_volume_50"].replace(0, np.nan)
        )
        features["breakout_signal"] = (
            (close > features["buy_point"])
            & (features["breakout_volume_ratio"] > self.config.volume_surge_multiple)
        )
        features["breakout_ready"] = (
            features["base_candidate"]
            & (features["distance_from_base_high_pct"] <= self.config.breakout_ready_threshold)
        )
        features["candidate_status"] = "no_base"
        features.loc[features["base_candidate"], "candidate_status"] = "building_base"
        features.loc[
            features["base_candidate"] & ~features["preferred_stage"],
            "candidate_status",
        ] = "late_stage"
        features.loc[
            features["base_candidate"]
            & features["preferred_stage"]
            & features["breakout_ready"],
            "candidate_status",
        ] = "near_pivot"
        features.loc[
            features["base_candidate"]
            & features["preferred_stage"]
            & (close >= features["buy_point"])
            & (close <= features["buy_limit_price"]),
            "candidate_status",
        ] = "actionable"
        features.loc[
            features["base_candidate"]
            & features["preferred_stage"]
            & (close > features["buy_limit_price"]),
            "candidate_status",
        ] = "extended"

        return features

    @staticmethod
    def _weighted_return_score(df: pd.DataFrame) -> float:
        close = df["close"]
        score = 0.0
        windows = ((63, 0.40), (126, 0.20), (189, 0.20), (252, 0.20))
        for window, weight in windows:
            if len(close) <= window:
                continue
            base = close.iloc[-window - 1]
            if base <= 0:
                continue
            score += weight * ((close.iloc[-1] / base) - 1.0)
        return score

    def analyze_market_regime(self, benchmark_df: pd.DataFrame) -> Dict:
        prepared = self.prepare_features(benchmark_df)
        if prepared.empty:
            return {
                "regime": "unknown",
                "confirmed_uptrend": False,
                "benchmark_close": None,
                "benchmark_sma_50": None,
                "benchmark_sma_200": None,
                "benchmark_above_sma_50": False,
                "benchmark_above_sma_200": False,
                "benchmark_200dma_rising": False,
            }

        latest = prepared.iloc[-1]
        close = self._safe_float(latest["close"])
        sma_50 = self._safe_float(latest["sma_50"])
        sma_200 = self._safe_float(latest["sma_200"])
        sma_200_prev = self._safe_float(latest["sma_200_20d_ago"])

        above_50 = self._safe_bool(close is not None and sma_50 is not None and close > sma_50)
        above_200 = self._safe_bool(close is not None and sma_200 is not None and close > sma_200)
        rising_200 = self._safe_bool(
            sma_200 is not None and sma_200_prev is not None and sma_200 > sma_200_prev
        )

        if above_50 and above_200 and rising_200:
            regime = "confirmed_uptrend"
        elif above_200:
            regime = "uptrend_under_pressure"
        else:
            regime = "market_correction"

        return {
            "regime": regime,
            "confirmed_uptrend": regime == "confirmed_uptrend",
            "benchmark_close": close,
            "benchmark_sma_50": sma_50,
            "benchmark_sma_200": sma_200,
            "benchmark_above_sma_50": above_50,
            "benchmark_above_sma_200": above_200,
            "benchmark_200dma_rising": rising_200,
            "trade_date": prepared.index[-1].date().isoformat(),
        }

    def screen_universe(
        self,
        data_by_symbol: Dict[str, pd.DataFrame],
        benchmark_df: pd.DataFrame | None = None,
        fundamentals_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        market_regime = self.analyze_market_regime(benchmark_df) if benchmark_df is not None else {
            "regime": "unknown",
            "confirmed_uptrend": False,
        }
        benchmark_score = (
            self._weighted_return_score(benchmark_df)
            if benchmark_df is not None and not benchmark_df.empty
            else 0.0
        )
        fundamentals_lookup = {}
        if fundamentals_df is not None and not fundamentals_df.empty:
            fundamentals_lookup = fundamentals_df.set_index("symbol").to_dict("index")

        rs_scores = {}
        prepared_frames = {}
        for symbol, df in data_by_symbol.items():
            prepared = self.prepare_features(df)
            if prepared.empty or len(prepared) < self.config.sma_long + 20:
                continue
            prepared_frames[symbol] = prepared
            rs_scores[symbol] = self._weighted_return_score(prepared) - benchmark_score

        if not prepared_frames:
            return pd.DataFrame()

        rs_percentiles = (
            pd.Series(rs_scores).rank(method="average", pct=True).mul(100.0).to_dict()
        )

        rows = []
        for symbol, prepared in prepared_frames.items():
            latest = prepared.iloc[-1]
            close = latest["close"]
            sma_50 = latest["sma_50"]
            sma_150 = latest["sma_150"]
            sma_200 = latest["sma_200"]
            high_52w = latest["52w_high"]
            low_52w = latest["52w_low"]
            avg_volume_50 = latest["avg_volume_50"]
            avg_dollar_volume_50 = latest["avg_dollar_volume_50"]
            rs_percentile = rs_percentiles.get(symbol, 0.0)
            fundamentals = fundamentals_lookup.get(symbol, {})
            stage_number = self._safe_float(latest["stage_number"], 1.0)
            pivot_price = self._safe_float(latest["pivot_price"])
            buy_point = self._safe_float(latest["buy_point"])
            buy_limit_price = self._safe_float(latest["buy_limit_price"])
            initial_stop_price = self._safe_float(latest["initial_stop_price"])
            initial_stop_pct = self._safe_float(latest["initial_stop_pct"])
            candidate_status = latest.get("candidate_status", "no_base")

            revenue_growth = self._safe_float(
                fundamentals.get("revenue_growth_calc"),
                self._safe_float(fundamentals.get("revenue_growth"), 0.0),
            )
            eps_growth = self._safe_float(
                fundamentals.get("eps_growth_calc"),
                self._safe_float(fundamentals.get("earnings_quarterly_growth"), 0.0),
            )
            return_on_equity = self._safe_float(fundamentals.get("return_on_equity"), 0.0)
            revenue_acceleration = self._safe_float(fundamentals.get("revenue_acceleration"), 0.0)
            eps_acceleration = self._safe_float(fundamentals.get("eps_acceleration"), 0.0)
            next_earnings_datetime = fundamentals.get("next_earnings_datetime")
            earnings_days_away = None
            if next_earnings_datetime:
                earnings_ts = pd.to_datetime(next_earnings_datetime)
                current_ts = pd.Timestamp(prepared.index[-1])
                if earnings_ts.tzinfo is not None:
                    earnings_ts = earnings_ts.tz_localize(None)
                earnings_days_away = float((earnings_ts - current_ts).days)

            conditions = {
                "price_gt_sma_150": self._safe_bool(close > sma_150),
                "price_gt_sma_200": self._safe_bool(close > sma_200),
                "sma_150_gt_sma_200": self._safe_bool(sma_150 > sma_200),
                "sma_200_trending_up": self._safe_bool(sma_200 > latest["sma_200_20d_ago"]),
                "sma_stack_bullish": self._safe_bool(sma_50 > sma_150 > sma_200),
                "price_gt_sma_50": self._safe_bool(close > sma_50),
                "price_above_52w_low": self._safe_bool(
                    close >= low_52w * (1.0 + self.config.min_above_52w_low)
                ),
                "price_near_52w_high": self._safe_bool(
                    close >= high_52w * (1.0 - self.config.max_below_52w_high)
                ),
                "volume_ok": self._safe_bool(avg_volume_50 >= self.config.min_avg_volume),
                "dollar_volume_ok": self._safe_bool(
                    avg_dollar_volume_50 >= self.config.min_avg_dollar_volume
                ),
                "rs_ok": self._safe_bool(rs_percentile >= self.config.min_rs_percentile),
                "stage_ok": self._safe_bool(stage_number <= self.config.max_stage_number),
                "base_ok": (
                    self._safe_bool(latest["base_candidate"])
                    or self._safe_bool(latest["vcp_candidate"])
                ),
                "breakout_ready": self._safe_bool(latest["breakout_ready"]),
                "revenue_growth_ok": (
                    self._safe_bool(revenue_growth >= self.config.min_revenue_growth)
                    if self.config.require_fundamentals
                    else True
                ),
                "eps_growth_ok": (
                    self._safe_bool(eps_growth >= self.config.min_eps_growth)
                    if self.config.require_fundamentals
                    else True
                ),
                "roe_ok": (
                    self._safe_bool(return_on_equity >= self.config.min_return_on_equity)
                    if self.config.require_fundamentals
                    else True
                ),
                "revenue_acceleration_ok": (
                    self._safe_bool(revenue_acceleration >= 0)
                    if self.config.require_fundamentals and self.config.require_acceleration
                    else True
                ),
                "eps_acceleration_ok": (
                    self._safe_bool(eps_acceleration >= 0)
                    if self.config.require_fundamentals and self.config.require_acceleration
                    else True
                ),
                "earnings_window_ok": (
                    True
                    if earnings_days_away is None
                    else self._safe_bool(earnings_days_away >= self.config.min_days_to_earnings)
                ),
                "market_regime_ok": (
                    self._safe_bool(market_regime.get("confirmed_uptrend"))
                    if self.config.require_market_uptrend
                    else True
                ),
            }

            distance_to_pivot = (
                (pivot_price - close) / pivot_price if pivot_price and pivot_price > 0 else None
            )
            distance_to_buy_point = (
                (buy_point - close) / buy_point if buy_point and buy_point > 0 else None
            )
            buy_zone_pct = (
                (close - buy_point) / buy_point if buy_point and buy_point > 0 else None
            )

            base_template_passed = all(conditions.values())
            continuation_conditions = {
                "enabled": self.config.leader_continuation_enabled,
                "price_gt_sma_50": self._safe_bool(close > sma_50),
                "price_gt_sma_150": self._safe_bool(close > sma_150),
                "price_gt_sma_200": self._safe_bool(close > sma_200),
                "sma_stack_bullish": self._safe_bool(sma_50 > sma_150 > sma_200),
                "sma_200_trending_up": self._safe_bool(sma_200 > latest["sma_200_20d_ago"]),
                "price_near_52w_high": self._safe_bool(
                    close >= high_52w * (1.0 - self.config.max_below_52w_high)
                ),
                "dollar_volume_ok": self._safe_bool(
                    avg_dollar_volume_50 >= self.config.min_avg_dollar_volume
                ),
                "rs_ok": self._safe_bool(
                    rs_percentile >= self.config.leader_continuation_min_rs_percentile
                ),
                "adx_ok": self._safe_bool(
                    self._safe_float(latest.get("adx_14"), 0.0)
                    >= self.config.leader_continuation_min_adx_14
                ),
                "close_range_ok": self._safe_bool(
                    self._safe_float(latest.get("close_range_pct"), 0.0)
                    >= self.config.leader_continuation_min_close_range_pct
                ),
                "roc_60_ok": self._safe_bool(
                    self._safe_float(latest.get("roc_60"), -999.0)
                    >= self.config.leader_continuation_min_roc_60
                ),
                "roc_120_ok": self._safe_bool(
                    self._safe_float(latest.get("roc_120"), -999.0)
                    >= self.config.leader_continuation_min_roc_120
                ),
                "earnings_window_ok": (
                    True
                    if earnings_days_away is None
                    else self._safe_bool(earnings_days_away >= self.config.min_days_to_earnings)
                ),
            }
            leader_continuation = all(continuation_conditions.values())
            continuation_watch = bool(
                leader_continuation
                and distance_to_buy_point is not None
                and (
                    -self.config.leader_continuation_max_extension_pct
                    <= distance_to_buy_point
                    <= self.config.leader_continuation_max_pullback_pct
                )
            )
            continuation_actionable = bool(
                leader_continuation
                and buy_zone_pct is not None
                and 0.0 <= buy_zone_pct <= self.config.leader_continuation_max_extension_pct
            )

            if candidate_status == "no_base" and continuation_watch:
                candidate_status = "leader_continuation_watch"
            if continuation_actionable:
                candidate_status = "leader_continuation_actionable"

            template_score = max(
                sum(bool(v) for v in conditions.values()),
                sum(bool(v) for v in continuation_conditions.values()),
            )
            passed_template = base_template_passed or leader_continuation
            rule_watch_candidate = passed_template and candidate_status in {
                "near_pivot",
                "actionable",
                "leader_continuation_watch",
                "leader_continuation_actionable",
            }
            rule_entry_candidate = passed_template and candidate_status in {
                "actionable",
                "leader_continuation_actionable",
            }

            rows.append(
                {
                    "trade_date": prepared.index[-1].date().isoformat(),
                    "symbol": symbol,
                    "close": round(float(close), 2),
                    "sma_50": round(float(sma_50), 2),
                    "sma_150": round(float(sma_150), 2),
                    "sma_200": round(float(sma_200), 2),
                    "52w_high": round(float(high_52w), 2),
                    "52w_low": round(float(low_52w), 2),
                    "avg_volume_50": int(avg_volume_50) if pd.notna(avg_volume_50) else 0,
                    "avg_dollar_volume_50": round(float(avg_dollar_volume_50), 2),
                    "rs_score": round(float(rs_scores[symbol]), 4),
                    "rs_percentile": round(float(rs_percentile), 2),
                    "revenue_growth": round(revenue_growth, 4) if revenue_growth is not None else None,
                    "eps_growth": round(eps_growth, 4) if eps_growth is not None else None,
                    "revenue_acceleration": round(revenue_acceleration, 4)
                    if revenue_acceleration is not None
                    else None,
                    "eps_acceleration": round(eps_acceleration, 4)
                    if eps_acceleration is not None
                    else None,
                    "return_on_equity": round(return_on_equity, 4)
                    if return_on_equity is not None
                    else None,
                    "template_score": template_score,
                    "passed_template": passed_template,
                    "stage_number": int(stage_number) if stage_number is not None else None,
                    "base_label": latest["base_label"],
                    "base_depth_pct": round(float(latest["base_depth_pct"]), 4)
                    if pd.notna(latest["base_depth_pct"])
                    else None,
                    "handle_depth_pct": round(float(latest["handle_depth_pct"]), 4)
                    if pd.notna(latest["handle_depth_pct"])
                    else None,
                    "candidate_status": candidate_status,
                    "close_range_pct": round(float(latest["close_range_pct"]), 4)
                    if pd.notna(latest["close_range_pct"])
                    else None,
                    "breakout_ready": self._safe_bool(latest["breakout_ready"]),
                    "vcp_candidate": self._safe_bool(latest["vcp_candidate"]),
                    "base_candidate": self._safe_bool(latest["base_candidate"]),
                    "pivot_price": round(float(pivot_price), 2) if pd.notna(pivot_price) else None,
                    "buy_point": round(float(buy_point), 2) if pd.notna(buy_point) else None,
                    "buy_limit_price": round(float(buy_limit_price), 2)
                    if pd.notna(buy_limit_price)
                    else None,
                    "initial_stop_price": round(float(initial_stop_price), 2)
                    if pd.notna(initial_stop_price)
                    else None,
                    "initial_stop_pct": round(float(initial_stop_pct), 4)
                    if pd.notna(initial_stop_pct)
                    else None,
                    "distance_to_pivot_pct": round(float(distance_to_pivot), 4)
                    if distance_to_pivot is not None
                    else None,
                    "distance_to_buy_point_pct": round(float(distance_to_buy_point), 4)
                    if distance_to_buy_point is not None
                    else None,
                    "buy_zone_pct": round(float(buy_zone_pct), 4)
                    if buy_zone_pct is not None
                    else None,
                    "breakout_volume_ratio": round(float(latest["breakout_volume_ratio"]), 4)
                    if pd.notna(latest["breakout_volume_ratio"])
                    else None,
                    "rsi_14": round(float(latest["rsi_14"]), 2)
                    if pd.notna(latest["rsi_14"])
                    else None,
                    "adx_14": round(float(latest["adx_14"]), 2)
                    if pd.notna(latest["adx_14"])
                    else None,
                    "atr_pct_14": round(float(latest["atr_pct_14"]), 4)
                    if pd.notna(latest["atr_pct_14"])
                    else None,
                    "roc_20": round(float(latest["roc_20"]), 4)
                    if pd.notna(latest["roc_20"])
                    else None,
                    "roc_60": round(float(latest["roc_60"]), 4)
                    if pd.notna(latest["roc_60"])
                    else None,
                    "roc_120": round(float(latest["roc_120"]), 4)
                    if pd.notna(latest["roc_120"])
                    else None,
                    "breakout_signal": self._safe_bool(latest["breakout_signal"]),
                    "leader_continuation": leader_continuation,
                    "leader_continuation_watch": continuation_watch,
                    "leader_continuation_actionable": continuation_actionable,
                    "rule_watch_candidate": rule_watch_candidate,
                    "rule_entry_candidate": rule_entry_candidate,
                    "market_regime": market_regime.get("regime"),
                    "market_confirmed_uptrend": market_regime.get("confirmed_uptrend"),
                    "next_earnings_datetime": (
                        pd.to_datetime(next_earnings_datetime).isoformat()
                        if next_earnings_datetime is not None
                        else None
                    ),
                    "earnings_days_away": round(earnings_days_away, 1)
                    if earnings_days_away is not None
                    else None,
                    "sector": fundamentals.get("sector"),
                    "industry": fundamentals.get("industry"),
                }
            )

        results = pd.DataFrame(rows)
        if results.empty:
            return results

        status_rank = {
            "actionable": 4,
            "near_pivot": 3,
            "building_base": 2,
            "extended": 1,
            "late_stage": 0,
            "no_base": -1,
        }
        results["candidate_rank"] = results["candidate_status"].map(status_rank).fillna(-1)

        results = results.sort_values(
            [
                "rule_entry_candidate",
                "rule_watch_candidate",
                "passed_template",
                "market_confirmed_uptrend",
                "candidate_rank",
                "breakout_ready",
                "rs_percentile",
                "roc_60",
                "adx_14",
                "rsi_14",
                "template_score",
                "stage_number",
                "eps_acceleration",
                "eps_growth",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
            ],
        ).reset_index(drop=True)
        return results.drop(columns=["candidate_rank"])
