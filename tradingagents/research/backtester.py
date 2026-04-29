"""Rule-based Minervini-style breakout backtester."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .earnings_blackout import EarningsBlackout
from .minervini import MinerviniScreener


@dataclass
class BacktestConfig:
    initial_cash: float = 100_000.0
    max_position_pct: float = 0.10
    risk_per_trade: float = 0.01
    stop_loss_pct: float = 0.08
    trail_stop_pct: float = 0.10
    max_hold_days: int = 60
    min_template_score: int = 8
    require_volume_surge: bool = True
    require_base_pattern: bool = True
    require_breakout_ready: bool = True
    require_market_regime: bool = True
    exit_on_market_correction: bool = False
    progressive_entries: bool = False
    initial_entry_fraction: float = 1.0
    add_on_trigger_pct_1: float = 0.025
    add_on_trigger_pct_2: float = 0.05
    add_on_fraction_1: float = 0.30
    add_on_fraction_2: float = 0.20
    breakeven_trigger_pct: float = 0.05
    partial_profit_trigger_pct: float = 0.12
    partial_profit_fraction: float = 0.33
    # Future-blanked probe: when > 0, the partial-exit trigger is evaluated
    # against the close N bars ago instead of the current bar's close. Set to
    # 1 to simulate "signal known at yesterday's close, execute at today's
    # close" — if the edge disappears, the same-bar check was lookahead.
    partial_trigger_lag_bars: int = 0
    use_ema21_exit: bool = False
    use_50dma_exit: bool = True
    use_close_range_filter: bool = False
    min_close_range_pct: float = 0.60
    scale_exposure_in_weak_market: bool = False
    weak_market_position_scale: float = 0.50
    target_exposure_confirmed_uptrend: float = 1.00
    target_exposure_uptrend_under_pressure: float = 0.60
    target_exposure_market_correction: float = 0.00
    allow_new_entries_in_correction: bool = False
    max_positions: int = 6
    min_rsi_14: float = 0.0
    min_adx_14: float = 0.0
    max_atr_pct_14: float = 1.0
    min_roc_20: float = -1.0
    min_roc_60: float = -1.0
    min_roc_120: float = -1.0
    buy_point_tolerance: float = 1.0
    overlay_enabled: bool = False
    overlay_rebalance_threshold: float = 0.05
    allow_continuation_entry: bool = False
    continuation_min_template_score: int = 7
    continuation_min_roc_60: float = 0.10
    continuation_max_distance_from_high: float = 0.10
    continuation_max_atr_pct: float = 0.06
    vol_target_enabled: bool = False
    vol_target_annual: float = 0.15
    vol_target_halflife_days: int = 20
    vol_target_lookback_days: int = 60
    vol_target_min_scalar: float = 0.25
    vol_target_max_scalar: float = 1.50
    vol_target_warmup_days: int = 30
    min_breakout_volume_ratio: float = 0.0
    disable_breakouts_in_uptrend: bool = False
    # Chandelier exit (Change #1). When enabled, trail stop uses
    # highest_close - chandelier_atr_multiple * atr_14 instead of a flat %.
    # Defaults off so existing baselines/tests are byte-for-byte unchanged.
    use_chandelier_exit: bool = False
    chandelier_atr_multiple: float = 3.0
    # Dead-money time stop (Change #2). When enabled, exits if position has not
    # gained >= dead_money_min_gain_pct within dead_money_max_days trading days.
    # Defaults off.
    use_dead_money_stop: bool = False
    dead_money_min_gain_pct: float = 0.05
    dead_money_max_days: int = 15
    # Change #17: adaptive dead-money — when enabled, overrides dead_money_max_days
    # with a regime-specific value. Lets winners breathe in confirmed uptrends,
    # cuts stuck names faster in corrections.
    adaptive_dead_money: bool = False
    dead_money_max_days_uptrend: int = 15
    dead_money_max_days_pressure: int = 10
    dead_money_max_days_correction: int = 7
    # Change #3: breakeven lock offset. When stop is moved to breakeven,
    # lock it at average_cost * (1 + this). Default 0.0 = exact entry (old behavior).
    breakeven_lock_offset_pct: float = 0.0
    # B2L parity: port of live's `leader_continuation_*` entry path from
    # automation/config.py. The live orchestrator enters on this path in
    # addition to the VCP breakout path; B0/B1/B2 backtests do not model it.
    # Defaults mirror automation/config.py so enabling gives live parity.
    use_leader_continuation_entry: bool = False
    leader_cont_min_rs_percentile: float = 75.0
    leader_cont_min_adx_14: float = 12.0
    leader_cont_min_close_range_pct: float = 0.15
    leader_cont_min_roc_60: float = 0.0
    leader_cont_min_roc_120: float = 0.0
    leader_cont_max_below_52w_high: float = 0.30
    leader_cont_max_extension_pct: float = 0.07
    leader_cont_max_pullback_pct: float = 0.08
    # Change #16 (Wave 2): composite entry scoring. Replaces the raw-feature
    # ranking tuple with a composite score built from template, RS,
    # base depth, volume contraction, stage, RVOL, and 60d momentum. Intent:
    # pick the best-quality subset when candidates exceed max_positions.
    use_composite_scoring: bool = False
    # If "v2", composite drops stage/RS bands (already floor-filtered) and
    # double-weights template + ROC60 to avoid penalizing leader_continuation
    # entries (Stage 2 by definition).
    composite_variant: str = "v1"
    # Change #15 (Wave 2): RSI-2 mean-reversion sleeve on SPY. Activates only
    # when main strategy exposure is below the threshold (deploys idle cash
    # in flat/choppy years). Entry: SPY close > 200-DMA AND RSI(2) < entry_th.
    # Exit: close > 5-DMA OR bars_held >= max_hold_days.
    rsi2_sleeve_enabled: bool = False
    rsi2_sleeve_exposure_threshold: float = 0.50
    rsi2_sleeve_entry_threshold: float = 10.0
    rsi2_sleeve_exit_ma_days: int = 5
    rsi2_sleeve_max_hold_days: int = 5
    rsi2_sleeve_position_pct: float = 0.20
    rsi2_sleeve_long_term_ma_days: int = 200
    # Change #12 (Wave 2): regime-dependent trail stop. When enabled, the
    # flat-% trail uses regime-specific widths instead of trail_stop_pct.
    # Hypothesis: wider in uptrends (let winners breathe), tighter in
    # corrections (protect profits). Off by default.
    regime_aware_trail: bool = False
    trail_stop_pct_uptrend: float = 0.12
    trail_stop_pct_pressure: float = 0.10
    trail_stop_pct_correction: float = 0.07
    # Change #13 (Wave 2): cross-asset rotation. Deploy a fixed position in
    # a bond ETF (TLT default) when main strategy is light AND regime is
    # not confirmed_uptrend. Idle-cash sink for flat/down equity years.
    cross_asset_enabled: bool = False
    cross_asset_symbol: str = "TLT"
    cross_asset_exposure_threshold: float = 0.50
    cross_asset_position_pct: float = 0.50
    # Exit cross_asset position when regime flips back to confirmed_uptrend
    cross_asset_exit_on_uptrend: bool = True
    # Require cross_asset to be above its own long-term MA before buying.
    # Avoids buying into a bond bear market (e.g. TLT 2022-2024).
    cross_asset_require_trend: bool = True
    cross_asset_trend_ma_days: int = 200
    # Stop-loss on the cross_asset position itself (peak-to-close drop).
    cross_asset_stop_loss_pct: float = 0.10
    # Upstream-inspired "don't chase" gate: block new entries when QQQ is
    # stretched above its 21EMA or running too fast over the last 5 sessions.
    # Set *_enabled=True and tune the two thresholds to taste. When the
    # filter fires on a given day, NO new entries (breakout/continuation/
    # leader_continuation) are opened; existing positions are unaffected.
    market_extension_filter_enabled: bool = False
    market_extension_max_qqq_above_ema21_pct: float = 0.05
    market_extension_max_qqq_roc_5: float = 0.05
    # Future-blanked probe for bias audit: when > 0, evaluate the filter
    # against QQQ data from N bars ago instead of the current bar. If the
    # filter's effect disappears with lag=1, it was reading same-day info
    # that would not have been available at the open.
    market_extension_lag_bars: int = 0
    # Earnings-report blackout. When > 0, skip new entries if an ER is
    # within N days of the candidate bar; flatten existing positions if
    # an ER is within M days. Uses the earnings_events table populated by
    # warehouse.fetch_and_store_earnings_events. BMO/AMC time_hint is
    # respected; unknown-hint rows (AV fallback) are treated as whole-day
    # blackout. 0 = disabled (default; preserves baseline reproducibility).
    earnings_blackout_entry_days: int = 0
    earnings_flatten_days_before: int = 0
    # Just-in-time flatten: close positions ONLY at the last pre-print bar
    # (AMC → close of ER day; BMO → close of day prior). When True,
    # earnings_flatten_days_before is ignored for the flatten check.
    earnings_flatten_last_bar_only: bool = False


class MinerviniBacktester:
    """Backtest an approximate SEPA breakout workflow on daily data."""

    def __init__(
        self,
        screener: MinerviniScreener | None = None,
        config: BacktestConfig | None = None,
        earnings_blackout: EarningsBlackout | None = None,
    ):
        self.screener = screener or MinerviniScreener()
        self.config = config or BacktestConfig()
        # Only consulted when config.earnings_blackout_entry_days > 0 or
        # config.earnings_flatten_days_before > 0. If the caller sets the
        # config keys but does not provide a blackout, the filter is a
        # no-op (fails safe, emits a one-shot warning).
        self.earnings_blackout = earnings_blackout
        self._earnings_blackout_warned = False

    def _earnings_entry_blocked(self, symbol: str, ts) -> bool:
        days = int(self.config.earnings_blackout_entry_days)
        if days <= 0:
            return False
        if self.earnings_blackout is None:
            if not self._earnings_blackout_warned:
                import logging
                logging.getLogger(__name__).warning(
                    "earnings_blackout_entry_days=%d but no EarningsBlackout "
                    "instance wired — filter disabled",
                    days,
                )
                self._earnings_blackout_warned = True
            return False
        blocked, _ = self.earnings_blackout.is_blackout(
            symbol, pd.Timestamp(ts), days_before=days, days_after=0
        )
        return blocked

    def _earnings_flatten_triggered(self, symbol: str, ts) -> bool:
        if self.earnings_blackout is None:
            return False
        if self.config.earnings_flatten_last_bar_only:
            blocked, _ = self.earnings_blackout.is_last_safe_bar_before_er(
                symbol, pd.Timestamp(ts)
            )
            return blocked
        days = int(self.config.earnings_flatten_days_before)
        if days <= 0:
            return False
        blocked, _ = self.earnings_blackout.is_blackout(
            symbol, pd.Timestamp(ts), days_before=days, days_after=0
        )
        return blocked

    def _trail_stop_from_row(
        self,
        row,
        highest_close: float,
        regime_label: str | None = None,
    ) -> float:
        """Compute the trailing stop for a given bar.

        If chandelier is enabled and atr_14 is available, use
        highest_close - k * atr_14. Otherwise falls back to flat %.
        Change #12: if regime_aware_trail is on, branch width by regime.
        """
        if self.config.use_chandelier_exit:
            atr = row.get("atr_14") if hasattr(row, "get") else None
            if atr is not None and pd.notna(atr) and float(atr) > 0:
                return float(highest_close) - self.config.chandelier_atr_multiple * float(atr)
        if self.config.regime_aware_trail and regime_label is not None:
            if regime_label == "confirmed_uptrend":
                pct = self.config.trail_stop_pct_uptrend
            elif regime_label == "market_correction":
                pct = self.config.trail_stop_pct_correction
            else:
                pct = self.config.trail_stop_pct_pressure
        else:
            pct = self.config.trail_stop_pct
        return float(highest_close) * (1.0 - pct)

    def _build_regime_frame(
        self,
        benchmark_df: Optional[pd.DataFrame],
        market_context_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if (
            market_context_df is not None
            and not market_context_df.empty
            and {"market_confirmed_uptrend", "market_regime"}.issubset(market_context_df.columns)
        ):
            columns = ["market_confirmed_uptrend", "market_regime"]
            if "market_score" in market_context_df.columns:
                columns.append("market_score")
            return market_context_df[columns].copy().sort_index()

        if benchmark_df is None or benchmark_df.empty:
            return pd.DataFrame()

        prepared = self.screener.prepare_features(benchmark_df)
        if prepared.empty:
            return prepared

        prepared["market_confirmed_uptrend"] = (
            (prepared["close"] > prepared["sma_50"])
            & (prepared["close"] > prepared["sma_200"])
            & (prepared["sma_200"] > prepared["sma_200_20d_ago"])
        )
        prepared["market_regime"] = "market_correction"
        prepared.loc[
            prepared["close"] > prepared["sma_200"],
            "market_regime",
        ] = "uptrend_under_pressure"
        prepared.loc[
            prepared["market_confirmed_uptrend"],
            "market_regime",
        ] = "confirmed_uptrend"
        return prepared[["market_confirmed_uptrend", "market_regime"]]

    def backtest_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        market_context_df: Optional[pd.DataFrame] = None,
        trade_start_date: Optional[str] = None,
    ) -> Dict:
        if self.config.progressive_entries:
            return self._backtest_symbol_progressive(
                symbol,
                df,
                benchmark_df=benchmark_df,
                market_context_df=market_context_df,
                trade_start_date=trade_start_date,
            )

        features = self.screener.prepare_features(df)
        if features.empty:
            return {"symbol": symbol, "error": "No data available"}
        regime_frame = self._build_regime_frame(benchmark_df, market_context_df=market_context_df)
        trade_start_ts = pd.Timestamp(trade_start_date) if trade_start_date else None

        cash = self.config.initial_cash
        shares = 0
        entry_price = 0.0
        entry_date = None
        highest_close = 0.0
        stop_price = 0.0
        trades = []
        equity_curve = []

        running_peak = self.config.initial_cash
        max_drawdown = 0.0

        for trade_date, row in features.iterrows():
            price = float(row["close"])
            if price <= 0:
                continue
            regime_ok = True
            if not regime_frame.empty and trade_date in regime_frame.index:
                regime_ok = bool(regime_frame.loc[trade_date, "market_confirmed_uptrend"])

            position_value = shares * price
            equity = cash + position_value
            running_peak = max(running_peak, equity)
            max_drawdown = max(max_drawdown, (running_peak - equity) / running_peak)
            equity_curve.append({"trade_date": trade_date, "equity": equity})

            if trade_start_ts is not None and trade_date < trade_start_ts:
                continue

            if shares == 0:
                template_score = self._template_score(row)
                volume_ok = (
                    (not self.config.require_volume_surge)
                    or bool(row.get("breakout_signal"))
                )
                has_base = (
                    (pd.notna(row.get("base_candidate")) and bool(row.get("base_candidate")))
                    or (pd.notna(row.get("vcp_candidate")) and bool(row.get("vcp_candidate")))
                )
                base_ok = (
                    (not self.config.require_base_pattern)
                    or has_base
                )
                setup_ready = (
                    True
                    if pd.isna(row.get("breakout_ready"))
                    else bool(row.get("breakout_ready"))
                )
                stage_ok = (
                    pd.notna(row.get("stage_number"))
                    and float(row.get("stage_number")) <= self.screener.config.max_stage_number
                )
                buy_point = row.get("buy_point")
                buy_limit_price = row.get("buy_limit_price")
                stop_candidate = row.get("initial_stop_price")
                if (
                    template_score < self.config.min_template_score
                    or not volume_ok
                    or not base_ok
                    or not setup_ready
                    or not stage_ok
                    or (self.config.require_market_regime and not regime_ok)
                ):
                    continue

                if pd.isna(buy_point) or price < float(buy_point):
                    continue
                if pd.notna(buy_limit_price) and price > float(buy_limit_price):
                    continue
                if self._earnings_entry_blocked(symbol, trade_date):
                    continue

                if pd.notna(stop_candidate) and float(stop_candidate) < price:
                    risk_per_share = price - float(stop_candidate)
                else:
                    risk_per_share = price * self.config.stop_loss_pct
                if risk_per_share <= 0:
                    continue

                capital_cap = cash * self.config.max_position_pct
                qty_by_capital = int(capital_cap / price)
                qty_by_risk = int((cash * self.config.risk_per_trade) / risk_per_share)
                shares = min(qty_by_capital, qty_by_risk)
                if shares <= 0:
                    shares = 0
                    continue

                entry_price = price
                entry_date = trade_date
                cash -= shares * entry_price
                highest_close = entry_price
                stop_price = (
                    float(stop_candidate)
                    if pd.notna(stop_candidate) and float(stop_candidate) < entry_price
                    else entry_price * (1.0 - self.config.stop_loss_pct)
                )
                continue

            highest_close = max(highest_close, price)
            stop_price = max(stop_price, self._trail_stop_from_row(row, highest_close))
            hold_days = (trade_date - entry_date).days if entry_date is not None else 0

            exit_reason = None
            if self._earnings_flatten_triggered(symbol, trade_date):
                exit_reason = "earnings_flatten"
            elif price <= stop_price:
                exit_reason = "stop"
            elif self.config.exit_on_market_correction and not regime_ok:
                exit_reason = "market_regime"
            elif (
                self.config.use_dead_money_stop
                and hold_days >= self.config.dead_money_max_days
                and price < entry_price * (1.0 + self.config.dead_money_min_gain_pct)
            ):
                exit_reason = "dead_money"
            elif pd.notna(row.get("sma_50")) and price < float(row["sma_50"]):
                exit_reason = "lost_50dma"
            elif hold_days >= self.config.max_hold_days:
                exit_reason = "time"

            if exit_reason is None:
                continue

            cash += shares * price
            pnl = (price - entry_price) * shares
            return_pct = (price / entry_price) - 1.0 if entry_price > 0 else 0.0
            trades.append(
                {
                    "symbol": symbol,
                    "entry_date": entry_date.date().isoformat() if entry_date is not None else None,
                    "exit_date": trade_date.date().isoformat(),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(price, 2),
                    "shares": shares,
                    "pnl": round(pnl, 2),
                    "return_pct": round(return_pct, 4),
                    "hold_days": hold_days,
                    "exit_reason": exit_reason,
                }
            )
            shares = 0
            entry_price = 0.0
            entry_date = None
            highest_close = 0.0
            stop_price = 0.0

        if shares > 0 and not features.empty:
            last_date = features.index[-1]
            last_price = float(features["close"].iloc[-1])
            cash += shares * last_price
            pnl = (last_price - entry_price) * shares
            return_pct = (last_price / entry_price) - 1.0 if entry_price > 0 else 0.0
            hold_days = (last_date - entry_date).days if entry_date is not None else 0
            trades.append(
                {
                    "symbol": symbol,
                    "entry_date": entry_date.date().isoformat() if entry_date is not None else None,
                    "exit_date": last_date.date().isoformat(),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(last_price, 2),
                    "shares": shares,
                    "pnl": round(pnl, 2),
                    "return_pct": round(return_pct, 4),
                    "hold_days": hold_days,
                    "exit_reason": "final_bar",
                }
            )

        final_value = cash
        benchmark_return = 0.0
        if not features.empty and float(features["close"].iloc[0]) > 0:
            benchmark_return = (float(features["close"].iloc[-1]) / float(features["close"].iloc[0])) - 1.0

        return {
            "symbol": symbol,
            "start_date": features.index[0].date().isoformat(),
            "end_date": features.index[-1].date().isoformat(),
            "trade_start_date": trade_start_ts.date().isoformat() if trade_start_ts is not None else features.index[0].date().isoformat(),
            "start_value": self.config.initial_cash,
            "end_value": round(final_value, 2),
            "total_return": round((final_value / self.config.initial_cash) - 1.0, 4),
            "benchmark_return": round(benchmark_return, 4),
            "max_drawdown": round(max_drawdown, 4),
            "total_trades": len(trades),
            "win_rate": round(
                sum(1 for t in trades if t["pnl"] > 0) / len(trades),
                4,
            )
            if trades
            else 0.0,
            "profit_factor": self._profit_factor(trades),
            "avg_hold_days": round(
                sum(t["hold_days"] for t in trades) / len(trades),
                2,
            )
            if trades
            else 0.0,
            "regime_filter": self.config.require_market_regime,
            "trades": trades,
            "equity_curve": pd.DataFrame(equity_curve),
        }

    def _backtest_symbol_progressive(
        self,
        symbol: str,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        market_context_df: Optional[pd.DataFrame] = None,
        trade_start_date: Optional[str] = None,
    ) -> Dict:
        features = self.screener.prepare_features(df)
        if features.empty:
            return {"symbol": symbol, "error": "No data available"}
        regime_frame = self._build_regime_frame(benchmark_df, market_context_df=market_context_df)
        trade_start_ts = pd.Timestamp(trade_start_date) if trade_start_date else None

        cash = self.config.initial_cash
        lots: list[dict] = []
        trades = []
        equity_curve = []
        highest_close = 0.0
        stop_price = 0.0
        partial_taken = False
        add_on_1_done = False
        add_on_2_done = False
        running_peak = self.config.initial_cash
        max_drawdown = 0.0
        # Future-blanked probe: precompute close shifted by lag_bars so the
        # partial-exit trigger can be evaluated against strictly-past price.
        partial_lag = max(0, int(self.config.partial_trigger_lag_bars))
        lagged_close = (
            features["close"].shift(partial_lag) if partial_lag > 0 else None
        )

        for trade_date, row in features.iterrows():
            price = float(row["close"])
            if price <= 0:
                continue
            regime_ok = True
            if not regime_frame.empty and trade_date in regime_frame.index:
                regime_ok = bool(regime_frame.loc[trade_date, "market_confirmed_uptrend"])

            shares = self._total_shares(lots)
            equity = cash + (shares * price)
            running_peak = max(running_peak, equity)
            max_drawdown = max(max_drawdown, (running_peak - equity) / running_peak)
            equity_curve.append({"trade_date": trade_date, "equity": equity})

            if trade_start_ts is not None and trade_date < trade_start_ts:
                continue

            if shares <= 0:
                if not self._row_passes_entry(row, price, regime_ok):
                    continue
                if self._earnings_entry_blocked(symbol, trade_date):
                    continue

                stop_candidate = row.get("initial_stop_price")
                stop_reference = (
                    float(stop_candidate)
                    if pd.notna(stop_candidate) and float(stop_candidate) < price
                    else price * (1.0 - self.config.stop_loss_pct)
                )
                risk_per_share = price - stop_reference
                if risk_per_share <= 0:
                    continue

                regime_scale = self._regime_position_scale(regime_ok)
                position_cap_value = equity * self.config.max_position_pct * regime_scale
                initial_target_value = position_cap_value * self.config.initial_entry_fraction
                risk_budget = equity * self.config.risk_per_trade * self.config.initial_entry_fraction
                qty = min(
                    int(min(initial_target_value, cash) / price),
                    int(risk_budget / risk_per_share),
                )
                if qty <= 0:
                    continue

                cash -= qty * price
                lots.append(
                    {
                        "entry_price": price,
                        "shares": qty,
                        "entry_date": trade_date,
                        "leg_type": "initial",
                    }
                )
                highest_close = price
                stop_price = stop_reference
                partial_taken = False
                add_on_1_done = False
                add_on_2_done = False
                continue

            average_cost = self._average_cost(lots)
            shares = self._total_shares(lots)
            highest_close = max(highest_close, price)
            hold_days = (trade_date - lots[0]["entry_date"]).days if lots else 0

            if price >= average_cost * (1.0 + self.config.breakeven_trigger_pct):
                stop_price = max(
                    stop_price,
                    average_cost * (1.0 + self.config.breakeven_lock_offset_pct),
                )

            if partial_taken and pd.notna(row.get("ema_21")):
                stop_price = max(stop_price, float(row["ema_21"]))
            else:
                stop_price = max(stop_price, self._trail_stop_from_row(row, highest_close))

            if (
                not add_on_1_done
                and price >= average_cost * (1.0 + self.config.add_on_trigger_pct_1)
                and self._row_supports_pyramiding(row, price)
            ):
                added_qty = self._add_on_qty(
                    cash=cash,
                    equity=equity,
                    price=price,
                    stop_price=stop_price,
                    current_shares=shares,
                    regime_ok=regime_ok,
                    add_fraction=self.config.add_on_fraction_1,
                )
                if added_qty > 0:
                    cash -= added_qty * price
                    lots.append(
                        {
                            "entry_price": price,
                            "shares": added_qty,
                            "entry_date": trade_date,
                            "leg_type": "add_on_1",
                        }
                    )
                    add_on_1_done = True
                    average_cost = self._average_cost(lots)
                    shares = self._total_shares(lots)

            if (
                not add_on_2_done
                and price >= average_cost * (1.0 + self.config.add_on_trigger_pct_2)
                and self._row_supports_pyramiding(row, price)
            ):
                added_qty = self._add_on_qty(
                    cash=cash,
                    equity=equity,
                    price=price,
                    stop_price=stop_price,
                    current_shares=shares,
                    regime_ok=regime_ok,
                    add_fraction=self.config.add_on_fraction_2,
                )
                if added_qty > 0:
                    cash -= added_qty * price
                    lots.append(
                        {
                            "entry_price": price,
                            "shares": added_qty,
                            "entry_date": trade_date,
                            "leg_type": "add_on_2",
                        }
                    )
                    add_on_2_done = True
                    average_cost = self._average_cost(lots)
                    shares = self._total_shares(lots)

            shares = self._total_shares(lots)
            average_cost = self._average_cost(lots)
            if lagged_close is not None:
                lagged = lagged_close.loc[trade_date]
                trigger_price = float(lagged) if pd.notna(lagged) else None
            else:
                trigger_price = price
            if (
                not partial_taken
                and shares > 0
                and trigger_price is not None
                and trigger_price >= average_cost * (1.0 + self.config.partial_profit_trigger_pct)
            ):
                partial_qty = max(1, int(shares * self.config.partial_profit_fraction))
                partial_qty = min(partial_qty, shares)
                cash += partial_qty * price
                trades.extend(
                    self._close_lots(
                        lots=lots,
                        sell_qty=partial_qty,
                        exit_price=price,
                        exit_date=trade_date,
                        symbol=symbol,
                        exit_reason="partial_profit",
                    )
                )
                partial_taken = True
                if lots:
                    stop_price = max(stop_price, self._average_cost(lots))
                shares = self._total_shares(lots)

            exit_reason = None
            if shares <= 0:
                continue
            if self._earnings_flatten_triggered(symbol, trade_date):
                exit_reason = "earnings_flatten"
            elif price <= stop_price:
                exit_reason = "stop"
            elif self.config.exit_on_market_correction and not regime_ok:
                exit_reason = "market_regime"
            elif (
                self.config.use_ema21_exit
                and pd.notna(row.get("ema_21"))
                and hold_days >= 5
                and price < float(row["ema_21"])
            ):
                exit_reason = "lost_21ema"
            elif (
                not self.config.use_ema21_exit
                and pd.notna(row.get("sma_50"))
                and price < float(row["sma_50"])
            ):
                exit_reason = "lost_50dma"
            elif hold_days >= self.config.max_hold_days:
                exit_reason = "time"

            if exit_reason is None:
                continue

            remaining = self._total_shares(lots)
            cash += remaining * price
            trades.extend(
                self._close_lots(
                    lots=lots,
                    sell_qty=remaining,
                    exit_price=price,
                    exit_date=trade_date,
                    symbol=symbol,
                    exit_reason=exit_reason,
                )
            )
            highest_close = 0.0
            stop_price = 0.0
            partial_taken = False
            add_on_1_done = False
            add_on_2_done = False

        if lots and not features.empty:
            last_date = features.index[-1]
            last_price = float(features["close"].iloc[-1])
            remaining = self._total_shares(lots)
            cash += remaining * last_price
            trades.extend(
                self._close_lots(
                    lots=lots,
                    sell_qty=remaining,
                    exit_price=last_price,
                    exit_date=last_date,
                    symbol=symbol,
                    exit_reason="final_bar",
                )
            )

        final_value = cash
        benchmark_return = 0.0
        if not features.empty and float(features["close"].iloc[0]) > 0:
            benchmark_return = (float(features["close"].iloc[-1]) / float(features["close"].iloc[0])) - 1.0

        return {
            "symbol": symbol,
            "start_date": features.index[0].date().isoformat(),
            "end_date": features.index[-1].date().isoformat(),
            "trade_start_date": trade_start_ts.date().isoformat() if trade_start_ts is not None else features.index[0].date().isoformat(),
            "start_value": self.config.initial_cash,
            "end_value": round(final_value, 2),
            "total_return": round((final_value / self.config.initial_cash) - 1.0, 4),
            "benchmark_return": round(benchmark_return, 4),
            "max_drawdown": round(max_drawdown, 4),
            "total_trades": len(trades),
            "win_rate": round(
                sum(1 for t in trades if t["pnl"] > 0) / len(trades),
                4,
            )
            if trades
            else 0.0,
            "profit_factor": self._profit_factor(trades),
            "avg_hold_days": round(
                sum(t["hold_days"] for t in trades) / len(trades),
                2,
            )
            if trades
            else 0.0,
            "regime_filter": self.config.require_market_regime,
            "trades": trades,
            "equity_curve": pd.DataFrame(equity_curve),
        }

    def backtest_universe(
        self,
        data_by_symbol: Dict[str, pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame] = None,
        trade_start_date: Optional[str] = None,
    ) -> Dict:
        results = []
        all_trades = []
        for symbol, df in data_by_symbol.items():
            result = self.backtest_symbol(
                symbol,
                df,
                benchmark_df=benchmark_df,
                trade_start_date=trade_start_date,
            )
            results.append(
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"trades", "equity_curve"}
                }
            )
            all_trades.extend(result.get("trades", []))

        summary = pd.DataFrame(results)
        trades = pd.DataFrame(all_trades)
        portfolio_summary = {
            "symbols_tested": len(summary),
            "avg_return": round(float(summary["total_return"].mean()), 4)
            if not summary.empty
            else 0.0,
            "avg_benchmark_return": round(float(summary["benchmark_return"].mean()), 4)
            if not summary.empty
            else 0.0,
            "avg_win_rate": round(float(summary["win_rate"].mean()), 4)
            if not summary.empty
            else 0.0,
            "avg_max_drawdown": round(float(summary["max_drawdown"].mean()), 4)
            if not summary.empty
            else 0.0,
            "total_trades": int(summary["total_trades"].sum()) if not summary.empty else 0,
        }
        return {
            "summary": summary.sort_values("total_return", ascending=False)
            if not summary.empty
            else summary,
            "trades": trades,
            "portfolio_summary": portfolio_summary,
        }

    @staticmethod
    def _profit_factor(trades: list[dict]) -> float:
        gross_win = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
        if gross_loss == 0:
            return float("inf") if gross_win > 0 else 0.0
        return round(gross_win / gross_loss, 4)

    def _row_passes_entry(self, row: pd.Series, price: float, regime_ok: bool) -> bool:
        template_score = self._template_score(row)
        volume_ok = (not self.config.require_volume_surge) or bool(row.get("breakout_signal"))
        if self.config.min_breakout_volume_ratio > 0:
            rvol = row.get("breakout_volume_ratio")
            if pd.isna(rvol) or float(rvol) < self.config.min_breakout_volume_ratio:
                return False
        has_base = (
            (pd.notna(row.get("base_candidate")) and bool(row.get("base_candidate")))
            or (pd.notna(row.get("vcp_candidate")) and bool(row.get("vcp_candidate")))
        )
        base_ok = (not self.config.require_base_pattern) or has_base
        if self.config.require_breakout_ready:
            setup_ready = True if pd.isna(row.get("breakout_ready")) else bool(row.get("breakout_ready"))
        else:
            setup_ready = True
        stage_ok = (
            pd.notna(row.get("stage_number"))
            and float(row.get("stage_number")) <= self.screener.config.max_stage_number
        )
        buy_point = row.get("buy_point")
        buy_limit_price = row.get("buy_limit_price")
        close_range_ok = (
            (not self.config.use_close_range_filter)
            or (
                pd.notna(row.get("close_range_pct"))
                and float(row["close_range_pct"]) >= self.config.min_close_range_pct
            )
        )
        rsi_ok = (
            self.config.min_rsi_14 <= 0
            or (
                pd.notna(row.get("rsi_14"))
                and float(row["rsi_14"]) >= self.config.min_rsi_14
            )
        )
        adx_ok = (
            self.config.min_adx_14 <= 0
            or (
                pd.notna(row.get("adx_14"))
                and float(row["adx_14"]) >= self.config.min_adx_14
            )
        )
        atr_ok = (
            self.config.max_atr_pct_14 >= 1.0
            or (
                pd.notna(row.get("atr_pct_14"))
                and float(row["atr_pct_14"]) <= self.config.max_atr_pct_14
            )
        )
        roc_20_ok = (
            self.config.min_roc_20 <= -1.0
            or (
                pd.notna(row.get("roc_20"))
                and float(row["roc_20"]) >= self.config.min_roc_20
            )
        )
        roc_60_ok = (
            self.config.min_roc_60 <= -1.0
            or (
                pd.notna(row.get("roc_60"))
                and float(row["roc_60"]) >= self.config.min_roc_60
            )
        )
        roc_120_ok = (
            self.config.min_roc_120 <= -1.0
            or (
                pd.notna(row.get("roc_120"))
                and float(row["roc_120"]) >= self.config.min_roc_120
            )
        )
        if (
            template_score < self.config.min_template_score
            or not volume_ok
            or not base_ok
            or not setup_ready
            or not stage_ok
            or not close_range_ok
            or not rsi_ok
            or not adx_ok
            or not atr_ok
            or not roc_20_ok
            or not roc_60_ok
            or not roc_120_ok
            or (self.config.require_market_regime and not regime_ok)
        ):
            return False
        if pd.isna(buy_point) or price < float(buy_point) * self.config.buy_point_tolerance:
            return False
        if pd.notna(buy_limit_price) and price > float(buy_limit_price):
            return False
        return True

    def _row_passes_continuation_entry(self, row: pd.Series, price: float) -> bool:
        """Alt entry path for persistent leaders that never form a fresh base.

        The primary path requires a breakout from a 20-day pivot. Stocks in
        a sustained uptrend (AMZN 2015, NVDA 2023) rarely consolidate long
        enough to form that base — they just keep running. This path lets
        them enter on a mild pullback when the trend structure is intact.
        """
        if not self.config.allow_continuation_entry:
            return False

        template_score = self._template_score(row)
        if template_score < self.config.continuation_min_template_score:
            return False

        ema_21 = row.get("ema_21")
        sma_50 = row.get("sma_50")
        sma_150 = row.get("sma_150")
        sma_200 = row.get("sma_200")
        if any(pd.isna(x) for x in (ema_21, sma_50, sma_150, sma_200)):
            return False
        if price < float(ema_21) or price < float(sma_50):
            return False
        if not (float(sma_50) > float(sma_150) > float(sma_200)):
            return False

        roc_60 = row.get("roc_60")
        if pd.isna(roc_60) or float(roc_60) < self.config.continuation_min_roc_60:
            return False

        high_52w = row.get("52w_high")
        if pd.isna(high_52w) or high_52w <= 0:
            return False
        if price < float(high_52w) * (1.0 - self.config.continuation_max_distance_from_high):
            return False

        atr_pct = row.get("atr_pct_14")
        if pd.notna(atr_pct) and float(atr_pct) > self.config.continuation_max_atr_pct:
            return False

        return True

    def _row_passes_leader_continuation_entry(self, row: pd.Series, price: float) -> bool:
        """B2L parity: mirrors `leader_continuation_*` gate in minervini.py.

        Live orchestrator's screener (MinerviniScreener) emits
        `leader_continuation_actionable` signals on persistent leaders whose
        pullbacks sit inside a small buy zone around the current buy_point.
        The live orchestrator enters on those signals in addition to VCP
        breakouts. B0/B1/B2 backtests never modeled this path.

        Parity note: live's per-path RS >= 75 is not enforced here per-row
        because rs_percentile is computed cross-sectionally in walk_forward
        and isn't attached to prepared frames. The walk-forward RS floor
        (min_rs_percentile) applies to all candidates, so this path
        inherits that floor (typically 70 for live flavor, 60 for research).
        """
        if not self.config.use_leader_continuation_entry:
            return False

        sma_50 = row.get("sma_50")
        sma_150 = row.get("sma_150")
        sma_200 = row.get("sma_200")
        sma_200_20d_ago = row.get("sma_200_20d_ago")
        if any(pd.isna(x) for x in (sma_50, sma_150, sma_200, sma_200_20d_ago)):
            return False
        if price < float(sma_50) or price < float(sma_150) or price < float(sma_200):
            return False
        if not (float(sma_50) > float(sma_150) > float(sma_200)):
            return False
        if not (float(sma_200) > float(sma_200_20d_ago)):
            return False

        high_52w = row.get("52w_high")
        if pd.isna(high_52w) or high_52w <= 0:
            return False
        if price < float(high_52w) * (1.0 - self.config.leader_cont_max_below_52w_high):
            return False

        adx = row.get("adx_14")
        if pd.isna(adx) or float(adx) < self.config.leader_cont_min_adx_14:
            return False

        close_range = row.get("close_range_pct")
        if pd.isna(close_range) or float(close_range) < self.config.leader_cont_min_close_range_pct:
            return False

        roc_60 = row.get("roc_60")
        if pd.isna(roc_60) or float(roc_60) < self.config.leader_cont_min_roc_60:
            return False
        roc_120 = row.get("roc_120")
        if pd.isna(roc_120) or float(roc_120) < self.config.leader_cont_min_roc_120:
            return False

        buy_point = row.get("buy_point")
        if pd.isna(buy_point) or float(buy_point) <= 0:
            return False
        buy_zone_pct = (price - float(buy_point)) / float(buy_point)
        # Live actionable: 0 <= extension <= leader_cont_max_extension_pct
        # Live watch (also entered via same screener tag): pullback within
        # [-leader_cont_max_pullback_pct, 0]. Union covers both.
        if not (
            -self.config.leader_cont_max_pullback_pct
            <= buy_zone_pct
            <= self.config.leader_cont_max_extension_pct
        ):
            return False

        return True

    def _row_supports_pyramiding(self, row: pd.Series, price: float) -> bool:
        if pd.notna(row.get("ema_21")) and price < float(row["ema_21"]):
            return False
        if self.config.use_close_range_filter and pd.notna(row.get("close_range_pct")):
            if float(row["close_range_pct"]) < self.config.min_close_range_pct:
                return False
        return bool(row.get("breakout_ready")) or bool(row.get("breakout_signal"))

    def _regime_position_scale(self, regime_ok: bool) -> float:
        if self.config.scale_exposure_in_weak_market and not regime_ok:
            return self.config.weak_market_position_scale
        return 1.0

    def _add_on_qty(
        self,
        cash: float,
        equity: float,
        price: float,
        stop_price: float,
        current_shares: int,
        regime_ok: bool,
        add_fraction: float,
    ) -> int:
        risk_per_share = max(price - stop_price, 0.0)
        if risk_per_share <= 0:
            return 0
        regime_scale = self._regime_position_scale(regime_ok)
        max_position_value = equity * self.config.max_position_pct * regime_scale
        current_value = current_shares * price
        remaining_capacity = max(0.0, max_position_value - current_value)
        target_value = min(max_position_value * add_fraction, remaining_capacity, cash)
        risk_budget = equity * self.config.risk_per_trade * add_fraction
        return min(
            int(target_value / price),
            int(risk_budget / risk_per_share),
        )

    @staticmethod
    def _total_shares(lots: list[dict]) -> int:
        return int(sum(lot["shares"] for lot in lots))

    @staticmethod
    def _average_cost(lots: list[dict]) -> float:
        total_shares = sum(lot["shares"] for lot in lots)
        if total_shares <= 0:
            return 0.0
        total_cost = sum(lot["shares"] * lot["entry_price"] for lot in lots)
        return total_cost / total_shares

    @staticmethod
    def _close_lots(
        lots: list[dict],
        sell_qty: int,
        exit_price: float,
        exit_date,
        symbol: str,
        exit_reason: str,
    ) -> list[dict]:
        realized = []
        remaining = sell_qty
        while remaining > 0 and lots:
            lot = lots[0]
            lot_qty = min(remaining, int(lot["shares"]))
            pnl = (exit_price - lot["entry_price"]) * lot_qty
            return_pct = (exit_price / lot["entry_price"]) - 1.0 if lot["entry_price"] > 0 else 0.0
            hold_days = (exit_date - lot["entry_date"]).days
            realized.append(
                {
                    "symbol": symbol,
                    "entry_date": lot["entry_date"].date().isoformat(),
                    "exit_date": exit_date.date().isoformat(),
                    "entry_price": round(float(lot["entry_price"]), 2),
                    "exit_price": round(float(exit_price), 2),
                    "shares": lot_qty,
                    "pnl": round(float(pnl), 2),
                    "return_pct": round(float(return_pct), 4),
                    "hold_days": hold_days,
                    "exit_reason": exit_reason,
                    "leg_type": lot.get("leg_type", "initial"),
                }
            )
            lot["shares"] -= lot_qty
            remaining -= lot_qty
            if lot["shares"] <= 0:
                lots.pop(0)
        return realized

    @staticmethod
    def _template_score(row: pd.Series) -> int:
        score = 0
        score += int(row.get("close", 0) > row.get("sma_150", float("inf")))
        score += int(row.get("close", 0) > row.get("sma_200", float("inf")))
        score += int(row.get("sma_150", 0) > row.get("sma_200", float("inf")))
        score += int(row.get("sma_200", 0) > row.get("sma_200_20d_ago", float("inf")))
        score += int(
            row.get("sma_50", 0)
            > row.get("sma_150", float("inf"))
            > row.get("sma_200", float("inf"))
        )
        score += int(row.get("close", 0) > row.get("sma_50", float("inf")))
        score += int(
            row.get("close", 0)
            >= row.get("52w_low", float("inf")) * 1.30
        )
        score += int(
            row.get("close", 0)
            >= row.get("52w_high", float("inf")) * 0.75
        )
        score += int(row.get("avg_volume_50", 0) >= 200_000)
        score += int(
            (pd.notna(row.get("base_candidate")) and bool(row.get("base_candidate")))
            or (pd.notna(row.get("vcp_candidate")) and bool(row.get("vcp_candidate")))
        )
        score += int(
            pd.notna(row.get("stage_number")) and float(row.get("stage_number")) <= 2
        )
        score += int(
            pd.notna(row.get("breakout_ready")) and bool(row.get("breakout_ready"))
        )
        return score
