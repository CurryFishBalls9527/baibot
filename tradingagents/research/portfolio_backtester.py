"""Portfolio-level Minervini backtesting with shared capital and exposure scaling."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .backtester import BacktestConfig, MinerviniBacktester
from .minervini import MinerviniScreener

logger = logging.getLogger(__name__)


def _load_morning_volume_features() -> Optional[pd.DataFrame]:
    """Load the precomputed morning_volume_features table, indexed by
    (symbol, trade_date). Returns None if the table is unavailable. Tolerant
    so backtests that don't gate on this feature still run normally.
    """
    db_path = Path("research_data/market_data.duckdb")
    if not db_path.exists():
        return None
    try:
        import duckdb
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            df = con.execute("""
                SELECT symbol, trade_date, ratio AS morning_volume_ratio
                FROM morning_volume_features
            """).df()
        finally:
            con.close()
    except Exception as e:
        logger.warning("morning_volume_features unavailable: %s", e)
        return None
    if df.empty:
        return None
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index(["symbol", "trade_date"]).sort_index()
    return df


def _attach_morning_volume_ratio(
    prepared: pd.DataFrame,
    symbol: str,
    morning_volume_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge `morning_volume_ratio` column into a prepared per-symbol df by
    date index. NaN where the feature isn't available (no harm — gates check
    `pd.isna` before applying)."""
    if morning_volume_df is None or prepared.empty:
        prepared["morning_volume_ratio"] = pd.NA
        return prepared
    try:
        sym_slice = morning_volume_df.xs(symbol, level="symbol", drop_level=True)
    except KeyError:
        prepared["morning_volume_ratio"] = pd.NA
        return prepared
    # Index-align via reindex; df index is naive timestamps so coerce to
    # match. Lookup by date (truncate any time component on prepared index).
    target_dates = pd.to_datetime(prepared.index).normalize()
    aligned = sym_slice["morning_volume_ratio"].reindex(target_dates)
    prepared["morning_volume_ratio"] = aligned.values
    return prepared


def _load_max_bar_volume_features() -> Optional[pd.DataFrame]:
    """Load max_bar_volume_features for the pyramid add-on fake-breakout
    gate. Returns df indexed by (symbol, trade_date) with columns
    `max_bar_rvol_20d` and `max_bar_rvol_intraday`. Returns None if the
    table isn't present — backtest still runs with the feature absent."""
    db_path = Path("research_data/market_data.duckdb")
    if not db_path.exists():
        return None
    try:
        import duckdb
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            df = con.execute("""
                SELECT symbol, trade_date,
                       max_bar_rvol_20d, max_bar_rvol_intraday
                FROM max_bar_volume_features
            """).df()
        finally:
            con.close()
    except Exception as e:
        logger.warning("max_bar_volume_features unavailable: %s", e)
        return None
    if df.empty:
        return None
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index(["symbol", "trade_date"]).sort_index()
    return df


def _attach_max_bar_volume(
    prepared: pd.DataFrame,
    symbol: str,
    max_bar_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge max_bar_rvol_20d + max_bar_rvol_intraday into prepared df."""
    if max_bar_df is None or prepared.empty:
        prepared["max_bar_rvol_20d"] = pd.NA
        prepared["max_bar_rvol_intraday"] = pd.NA
        return prepared
    try:
        sym_slice = max_bar_df.xs(symbol, level="symbol", drop_level=True)
    except KeyError:
        prepared["max_bar_rvol_20d"] = pd.NA
        prepared["max_bar_rvol_intraday"] = pd.NA
        return prepared
    target_dates = pd.to_datetime(prepared.index).normalize()
    prepared["max_bar_rvol_20d"] = (
        sym_slice["max_bar_rvol_20d"].reindex(target_dates).values
    )
    prepared["max_bar_rvol_intraday"] = (
        sym_slice["max_bar_rvol_intraday"].reindex(target_dates).values
    )
    return prepared


@dataclass
class PortfolioBacktestResult:
    summary: Dict
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    daily_state: pd.DataFrame
    symbol_summary: pd.DataFrame


class PortfolioMinerviniBacktester(MinerviniBacktester):
    """Shared-capital SEPA-style backtester."""

    def __init__(
        self,
        screener: MinerviniScreener | None = None,
        config: BacktestConfig | None = None,
        earnings_blackout=None,
    ):
        super().__init__(
            screener=screener,
            config=config,
            earnings_blackout=earnings_blackout,
        )

    def backtest_portfolio(
        self,
        data_by_symbol: Dict[str, pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame] = None,
        market_context_df: Optional[pd.DataFrame] = None,
        trade_start_date: Optional[str] = None,
        candidate_schedule: Optional[Dict] = None,
        cross_asset_df: Optional[pd.DataFrame] = None,
        market_extension_frame: Optional[pd.DataFrame] = None,
    ) -> PortfolioBacktestResult:
        self._candidate_schedule = candidate_schedule
        self._market_extension_frame = market_extension_frame
        # Load the morning_volume_ratio feature once per backtest run if any
        # gate uses it. Cheap if absent (returns None).
        morning_volume_df = (
            _load_morning_volume_features()
            if getattr(self.config, "min_morning_volume_ratio", 0.0) > 0
            or getattr(self.config, "leader_cont_min_morning_volume_ratio", 0.0) > 0
            else None
        )
        # Same pattern for the bar-level fake-breakout gates (pyramid add-on).
        max_bar_df = (
            _load_max_bar_volume_features()
            if getattr(self.config, "min_add_on_max_bar_rvol_20d", 0.0) > 0
            or getattr(self.config, "min_add_on_max_bar_rvol_intraday", 0.0) > 0
            else None
        )
        prepared_frames = {}
        for symbol, df in data_by_symbol.items():
            prepared = self.screener.prepare_features(df)
            if prepared.empty:
                continue
            prepared = _attach_morning_volume_ratio(prepared, symbol, morning_volume_df)
            prepared = _attach_max_bar_volume(prepared, symbol, max_bar_df)
            prepared_frames[symbol] = prepared
        if not prepared_frames:
            empty = pd.DataFrame()
            return PortfolioBacktestResult(
                summary={},
                trades=empty,
                equity_curve=empty,
                daily_state=empty,
                symbol_summary=empty,
            )

        regime_frame = self._build_regime_frame(
            benchmark_df,
            market_context_df=market_context_df,
        )
        all_dates = sorted(
            {
                ts
                for frame in prepared_frames.values()
                for ts in frame.index
            }
        )
        trade_start_ts = pd.Timestamp(trade_start_date) if trade_start_date else all_dates[0]

        cash = float(self.config.initial_cash)
        positions: dict[str, dict] = {}
        trades: list[dict] = []
        equity_curve: list[dict] = []
        daily_state: list[dict] = []
        running_peak = float(self.config.initial_cash)
        max_drawdown = 0.0
        overlay_state = {"shares": 0, "cost_basis": 0.0}
        overlay_symbol = "SPY"
        rsi2_state = {
            "shares": 0,
            "entry_price": 0.0,
            "entry_date": None,
            "entry_bar_idx": -1,
        }
        rsi2_features = self._build_rsi2_features(benchmark_df)
        cross_asset_state = {
            "shares": 0,
            "entry_price": 0.0,
            "entry_date": None,
            "highest_close": 0.0,
        }
        cross_asset_trend = self._build_cross_asset_trend(cross_asset_df)

        for trade_date in all_dates:
            regime_label, regime_ok, target_exposure = self._regime_state(
                regime_frame, trade_date
            )
            cash_ref = {"cash": cash}

            stock_mv_before = self._portfolio_market_value(positions, prepared_frames, trade_date)
            overlay_mv_before = self._overlay_market_value(
                overlay_state, benchmark_df, trade_date
            )
            rsi2_mv_before = self._rsi2_market_value(
                rsi2_state, rsi2_features, trade_date
            )
            cross_mv_before = self._cross_asset_market_value(
                cross_asset_state, cross_asset_df, trade_date
            )
            equity_before = (
                cash_ref["cash"]
                + stock_mv_before
                + overlay_mv_before
                + rsi2_mv_before
                + cross_mv_before
            )
            running_peak = max(running_peak, equity_before)
            max_drawdown = max(max_drawdown, (running_peak - equity_before) / running_peak)

            self._process_existing_positions(
                trade_date=trade_date,
                positions=positions,
                prepared_frames=prepared_frames,
                regime_ok=regime_ok,
                regime_label=regime_label,
                trades=trades,
                cash_ref=cash_ref,
            )
            cash = float(cash_ref["cash"])

            vol_scalar = self._compute_vol_scalar(equity_curve)

            if trade_date >= trade_start_ts:
                self._process_new_entries(
                    trade_date=trade_date,
                    positions=positions,
                    prepared_frames=prepared_frames,
                    regime_ok=regime_ok,
                    regime_label=regime_label,
                    target_exposure=target_exposure,
                    cash_ref=cash_ref,
                    overlay_state=overlay_state,
                    overlay_symbol=overlay_symbol,
                    benchmark_df=benchmark_df,
                    trades=trades,
                    vol_scalar=vol_scalar,
                )
                cash = float(cash_ref["cash"])

                if self.config.overlay_enabled and benchmark_df is not None:
                    self._rebalance_overlay(
                        trade_date=trade_date,
                        overlay_symbol=overlay_symbol,
                        overlay_state=overlay_state,
                        benchmark_df=benchmark_df,
                        positions=positions,
                        prepared_frames=prepared_frames,
                        target_exposure=target_exposure,
                        regime_label=regime_label,
                        cash_ref=cash_ref,
                        trades=trades,
                    )
                    cash = float(cash_ref["cash"])

            if self.config.rsi2_sleeve_enabled and rsi2_features is not None:
                self._process_rsi2_sleeve(
                    trade_date=trade_date,
                    rsi2_state=rsi2_state,
                    rsi2_features=rsi2_features,
                    positions=positions,
                    prepared_frames=prepared_frames,
                    cash_ref=cash_ref,
                    trades=trades,
                )
                cash = float(cash_ref["cash"])

            if self.config.cross_asset_enabled and cross_asset_df is not None:
                self._process_cross_asset(
                    trade_date=trade_date,
                    cross_asset_state=cross_asset_state,
                    cross_asset_df=cross_asset_df,
                    cross_asset_trend=cross_asset_trend,
                    regime_label=regime_label,
                    positions=positions,
                    prepared_frames=prepared_frames,
                    cash_ref=cash_ref,
                    trades=trades,
                )
                cash = float(cash_ref["cash"])

            market_value = self._portfolio_market_value(positions, prepared_frames, trade_date)
            overlay_mv = self._overlay_market_value(overlay_state, benchmark_df, trade_date)
            rsi2_mv = self._rsi2_market_value(rsi2_state, rsi2_features, trade_date)
            cross_mv = self._cross_asset_market_value(cross_asset_state, cross_asset_df, trade_date)
            equity = cash + market_value + overlay_mv + rsi2_mv + cross_mv
            total_exposure_mv = market_value + overlay_mv + rsi2_mv + cross_mv
            exposure = total_exposure_mv / equity if equity > 0 else 0.0
            market_score = None
            if (
                not regime_frame.empty
                and trade_date in regime_frame.index
                and "market_score" in regime_frame.columns
            ):
                market_score = regime_frame.loc[trade_date, "market_score"]
            equity_curve.append(
                {
                    "trade_date": trade_date.date().isoformat(),
                    "equity": round(equity, 2),
                    "cash": round(cash, 2),
                    "market_value": round(market_value, 2),
                    "overlay_value": round(overlay_mv, 2),
                    "exposure": round(exposure, 4),
                }
            )
            daily_state.append(
                {
                    "trade_date": trade_date.date().isoformat(),
                    "regime": regime_label,
                    "regime_confirmed_uptrend": bool(regime_ok),
                    "target_exposure": target_exposure,
                    "vol_scalar": round(vol_scalar, 4),
                    "actual_exposure": round(exposure, 4),
                    "market_score": (
                        int(market_score) if pd.notna(market_score) else None
                    ),
                    "cash": round(cash, 2),
                    "equity": round(equity, 2),
                    "positions": len(positions),
                    "symbols": ",".join(sorted(positions)),
                }
            )

        if positions:
            final_date = all_dates[-1]
            for symbol in list(positions):
                frame = prepared_frames[symbol]
                if final_date in frame.index:
                    row = frame.loc[final_date]
                    exit_date = final_date
                else:
                    prior = frame.loc[:final_date]
                    if prior.empty:
                        continue
                    row = prior.iloc[-1]
                    exit_date = prior.index[-1]
                price = float(row["close"])
                cash += self._liquidate_position(
                    symbol=symbol,
                    position=positions.pop(symbol),
                    exit_price=price,
                    exit_date=exit_date,
                    trades=trades,
                    exit_reason="final_bar",
                )

        if cross_asset_state["shares"] > 0 and cross_asset_df is not None:
            final_date = all_dates[-1]
            final_price = self._cross_asset_price(cross_asset_df, final_date)
            if final_price is not None:
                shares = cross_asset_state["shares"]
                proceeds = shares * final_price
                cost = shares * cross_asset_state["entry_price"]
                cash += proceeds
                trades.append(
                    {
                        "symbol": self.config.cross_asset_symbol,
                        "entry_date": cross_asset_state["entry_date"],
                        "exit_date": final_date,
                        "entry_price": cross_asset_state["entry_price"],
                        "exit_price": final_price,
                        "shares": shares,
                        "pnl": round(proceeds - cost, 2),
                        "return_pct": round((proceeds - cost) / cost, 4) if cost else 0.0,
                        "hold_days": 0,
                        "exit_reason": "cross_asset_final",
                        "leg_type": "cross_asset",
                    }
                )
                cross_asset_state["shares"] = 0
                cross_asset_state["entry_price"] = 0.0
                cross_asset_state["entry_date"] = None

        if rsi2_state["shares"] > 0 and rsi2_features is not None:
            final_date = all_dates[-1]
            final_price = self._rsi2_price(rsi2_features, final_date)
            if final_price is not None:
                shares = rsi2_state["shares"]
                proceeds = shares * final_price
                cost = shares * rsi2_state["entry_price"]
                cash += proceeds
                trades.append(
                    {
                        "symbol": "SPY",
                        "entry_date": rsi2_state["entry_date"],
                        "exit_date": final_date,
                        "entry_price": rsi2_state["entry_price"],
                        "exit_price": final_price,
                        "shares": shares,
                        "pnl": round(proceeds - cost, 2),
                        "return_pct": round((proceeds - cost) / cost, 4) if cost else 0.0,
                        "hold_days": 0,
                        "exit_reason": "rsi2_sleeve_final",
                        "leg_type": "rsi2_sleeve",
                    }
                )
                rsi2_state["shares"] = 0
                rsi2_state["entry_price"] = 0.0
                rsi2_state["entry_date"] = None
                rsi2_state["entry_bar_idx"] = -1

        if overlay_state["shares"] > 0 and benchmark_df is not None:
            final_date = all_dates[-1]
            final_price = self._overlay_price(benchmark_df, final_date)
            if final_price is not None:
                shares = overlay_state["shares"]
                proceeds = shares * final_price
                cost = overlay_state["cost_basis"]
                cash += proceeds
                trades.append(
                    {
                        "symbol": overlay_symbol,
                        "entry_date": None,
                        "exit_date": final_date,
                        "entry_price": cost / shares if shares else 0.0,
                        "exit_price": final_price,
                        "shares": shares,
                        "pnl": round(proceeds - cost, 2),
                        "return_pct": round((proceeds - cost) / cost, 4) if cost else 0.0,
                        "hold_days": 0,
                        "exit_reason": "overlay_final",
                        "leg_type": "overlay",
                    }
                )
                overlay_state["shares"] = 0
                overlay_state["cost_basis"] = 0.0

        equity_curve_df = pd.DataFrame(equity_curve)
        daily_state_df = pd.DataFrame(daily_state)
        trades_df = pd.DataFrame(trades)
        symbol_summary_df = self._build_symbol_summary(trades_df)
        total_return = (cash / self.config.initial_cash) - 1.0 if self.config.initial_cash else 0.0
        benchmark_return = self._portfolio_benchmark_return(benchmark_df, trade_start_ts)

        avg_exposure_pct = 0.0
        sharpe_ratio = 0.0
        if not equity_curve_df.empty:
            avg_exposure_pct = float(equity_curve_df["exposure"].mean())
            daily_returns = equity_curve_df["equity"].pct_change().dropna()
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe_ratio = float(
                    (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
                )

        summary = {
            "start_value": round(float(self.config.initial_cash), 2),
            "end_value": round(float(cash), 2),
            "total_return": round(float(total_return), 4),
            "benchmark_return": round(float(benchmark_return), 4),
            "max_drawdown": round(float(max_drawdown), 4),
            "avg_exposure_pct": round(avg_exposure_pct, 4),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "symbols_tested": len(prepared_frames),
            "symbols_with_trades": int(trades_df["symbol"].nunique()) if not trades_df.empty else 0,
            "total_trades": int(len(trades_df)),
            "trade_win_rate": round(float((trades_df["pnl"] > 0).mean()), 4) if not trades_df.empty else 0.0,
            "avg_trade_return": round(float(trades_df["return_pct"].mean()), 4) if not trades_df.empty else 0.0,
            "median_trade_return": round(float(trades_df["return_pct"].median()), 4) if not trades_df.empty else 0.0,
            "avg_active_symbol_return": round(float(symbol_summary_df["total_return"].mean()), 4) if not symbol_summary_df.empty else 0.0,
            "median_active_symbol_return": round(float(symbol_summary_df["total_return"].median()), 4) if not symbol_summary_df.empty else 0.0,
            "positive_active_symbol_ratio": round(float((symbol_summary_df["total_return"] > 0).mean()), 4) if not symbol_summary_df.empty else 0.0,
            "realized_pnl": round(float(trades_df["pnl"].sum()), 2) if not trades_df.empty else 0.0,
            "trade_start_date": trade_start_ts.date().isoformat(),
            "end_date": all_dates[-1].date().isoformat(),
        }
        return PortfolioBacktestResult(
            summary=summary,
            trades=trades_df,
            equity_curve=equity_curve_df,
            daily_state=daily_state_df,
            symbol_summary=symbol_summary_df,
        )

    def _process_existing_positions(
        self,
        trade_date,
        positions: dict[str, dict],
        prepared_frames: dict[str, pd.DataFrame],
        regime_ok: bool,
        regime_label: str,
        trades: list[dict],
        cash_ref: dict[str, float],
    ) -> None:
        for symbol in list(positions):
            frame = prepared_frames.get(symbol)
            if frame is None or trade_date not in frame.index:
                continue
            row = frame.loc[trade_date]
            price = float(row["close"])
            position = positions[symbol]

            shares = self._total_shares(position["lots"])
            if shares <= 0:
                positions.pop(symbol, None)
                continue

            average_cost = self._average_cost(position["lots"])
            position["highest_close"] = max(position["highest_close"], price)
            hold_days = (trade_date - position["entry_date"]).days

            if price >= average_cost * (1.0 + self.config.breakeven_trigger_pct):
                position["stop_price"] = max(
                    position["stop_price"],
                    average_cost * (1.0 + self.config.breakeven_lock_offset_pct),
                )

            if position["partial_taken"] and pd.notna(row.get("ema_21")):
                position["stop_price"] = max(position["stop_price"], float(row["ema_21"]))
            else:
                position["stop_price"] = max(
                    position["stop_price"],
                    self._trail_stop_from_row(row, position["highest_close"], regime_label),
                )

            if (
                not position["add_on_1_done"]
                and regime_label != "market_correction"
                and price >= average_cost * (1.0 + self.config.add_on_trigger_pct_1)
                and self._row_supports_pyramiding(row, price)
            ):
                added_qty = self._add_on_qty(
                    cash=cash_ref["cash"],
                    equity=cash_ref["cash"] + self._portfolio_market_value(positions, prepared_frames, trade_date),
                    price=price,
                    stop_price=position["stop_price"],
                    current_shares=shares,
                    regime_ok=regime_ok,
                    add_fraction=self.config.add_on_fraction_1,
                )
                if added_qty > 0:
                    cash_ref["cash"] -= added_qty * price
                    position["lots"].append(
                        {
                            "entry_price": price,
                            "shares": added_qty,
                            "entry_date": trade_date,
                            "leg_type": "add_on_1",
                        }
                    )
                    position["add_on_1_done"] = True
                    shares = self._total_shares(position["lots"])

            if (
                not position["add_on_2_done"]
                and regime_label != "market_correction"
                and price >= average_cost * (1.0 + self.config.add_on_trigger_pct_2)
                and self._row_supports_pyramiding(row, price)
            ):
                added_qty = self._add_on_qty(
                    cash=cash_ref["cash"],
                    equity=cash_ref["cash"] + self._portfolio_market_value(positions, prepared_frames, trade_date),
                    price=price,
                    stop_price=position["stop_price"],
                    current_shares=shares,
                    regime_ok=regime_ok,
                    add_fraction=self.config.add_on_fraction_2,
                )
                if added_qty > 0:
                    cash_ref["cash"] -= added_qty * price
                    position["lots"].append(
                        {
                            "entry_price": price,
                            "shares": added_qty,
                            "entry_date": trade_date,
                            "leg_type": "add_on_2",
                        }
                    )
                    position["add_on_2_done"] = True

            shares = self._total_shares(position["lots"])
            average_cost = self._average_cost(position["lots"])
            if (
                not position["partial_taken"]
                and shares > 0
                and price >= average_cost * (1.0 + self.config.partial_profit_trigger_pct)
            ):
                partial_qty = max(1, int(shares * self.config.partial_profit_fraction))
                partial_qty = min(partial_qty, shares)
                cash_ref["cash"] += partial_qty * price
                trades.extend(
                    self._close_lots(
                        lots=position["lots"],
                        sell_qty=partial_qty,
                        exit_price=price,
                        exit_date=trade_date,
                        symbol=symbol,
                        exit_reason="partial_profit",
                    )
                )
                position["partial_taken"] = True
                if position["lots"]:
                    position["stop_price"] = max(
                        position["stop_price"],
                        self._average_cost(position["lots"]),
                    )

            shares = self._total_shares(position["lots"])
            if shares <= 0:
                positions.pop(symbol, None)
                continue

            exit_reason = None
            if self._earnings_flatten_triggered(symbol, trade_date):
                exit_reason = "earnings_flatten"
            elif price <= position["stop_price"]:
                exit_reason = "stop"
            elif self.config.exit_on_market_correction and regime_label == "market_correction":
                exit_reason = "market_regime"
            elif self.config.use_dead_money_stop:
                if self.config.adaptive_dead_money:
                    if regime_label == "confirmed_uptrend":
                        dm_days = self.config.dead_money_max_days_uptrend
                    elif regime_label == "market_correction":
                        dm_days = self.config.dead_money_max_days_correction
                    else:
                        dm_days = self.config.dead_money_max_days_pressure
                else:
                    dm_days = self.config.dead_money_max_days
                if (
                    hold_days >= dm_days
                    and price < average_cost * (1.0 + self.config.dead_money_min_gain_pct)
                ):
                    exit_reason = "dead_money"
            elif (
                self.config.use_ema21_exit
                and pd.notna(row.get("ema_21"))
                and hold_days >= 5
                and price < float(row["ema_21"])
            ):
                exit_reason = "lost_21ema"
            elif (
                not self.config.use_ema21_exit
                and self.config.use_50dma_exit
                and pd.notna(row.get("sma_50"))
                and price < float(row["sma_50"])
            ):
                exit_reason = "lost_50dma"
            elif hold_days >= self.config.max_hold_days:
                exit_reason = "time"

            if exit_reason is None:
                continue

            cash_ref["cash"] += self._liquidate_position(
                symbol=symbol,
                position=positions.pop(symbol),
                exit_price=price,
                exit_date=trade_date,
                trades=trades,
                exit_reason=exit_reason,
            )

    def _get_approved_for_date(self, trade_date) -> set:
        """Find the approved symbol set for a given trade date from the schedule."""
        if not self._candidate_schedule:
            return set()
        import bisect
        dates = sorted(self._candidate_schedule.keys())
        idx = bisect.bisect_right(dates, trade_date) - 1
        if idx < 0:
            return set()
        return self._candidate_schedule[dates[idx]]

    def _market_is_extended(self, trade_date) -> bool:
        if not self.config.market_extension_filter_enabled:
            return False
        frame = getattr(self, "_market_extension_frame", None)
        if frame is None or frame.empty:
            return False
        lag = int(self.config.market_extension_lag_bars or 0)
        if lag > 0:
            if trade_date not in frame.index:
                return False
            pos = frame.index.get_loc(trade_date)
            if pos - lag < 0:
                return False
            row = frame.iloc[pos - lag]
        else:
            if trade_date not in frame.index:
                return False
            row = frame.loc[trade_date]
        above = row.get("qqq_above_ema21_pct")
        roc_5 = row.get("qqq_roc_5")
        max_above = float(self.config.market_extension_max_qqq_above_ema21_pct)
        max_roc_5 = float(self.config.market_extension_max_qqq_roc_5)
        if pd.notna(above) and float(above) > max_above:
            return True
        if pd.notna(roc_5) and float(roc_5) > max_roc_5:
            return True
        return False

    def _compute_vol_scalar(self, equity_history: list) -> float:
        """Barroso & Santa-Clara style vol scalar computed from strategy equity.

        Scales intended exposure inversely with recent realized strategy vol.
        Returns 1.0 until enough equity history has accumulated.
        """
        if not self.config.vol_target_enabled:
            return 1.0
        warmup = self.config.vol_target_warmup_days
        if len(equity_history) < warmup + 2:
            return 1.0
        equities = pd.Series([row["equity"] for row in equity_history], dtype=float)
        rets = equities.pct_change().dropna()
        if len(rets) < warmup:
            return 1.0
        recent = rets.iloc[-self.config.vol_target_lookback_days:]
        ewm_var = recent.ewm(halflife=self.config.vol_target_halflife_days, adjust=False).var().iloc[-1]
        if pd.isna(ewm_var) or ewm_var <= 0:
            return 1.0
        realized_vol = float((ewm_var * 252) ** 0.5)
        scalar = self.config.vol_target_annual / max(realized_vol, 0.05)
        return float(
            max(
                self.config.vol_target_min_scalar,
                min(self.config.vol_target_max_scalar, scalar),
            )
        )

    def _process_new_entries(
        self,
        trade_date,
        positions: dict[str, dict],
        prepared_frames: dict[str, pd.DataFrame],
        regime_ok: bool,
        regime_label: str,
        target_exposure: float,
        cash_ref: dict[str, float],
        overlay_state: Optional[dict] = None,
        overlay_symbol: str = "SPY",
        benchmark_df: Optional[pd.DataFrame] = None,
        trades: Optional[list] = None,
        vol_scalar: float = 1.0,
    ) -> None:
        if regime_label == "market_correction" and not self.config.allow_new_entries_in_correction:
            return
        if self._market_is_extended(trade_date):
            return
        effective_target_exposure = min(1.0, target_exposure * vol_scalar)

        candidates = []
        approved = self._get_approved_for_date(trade_date) if self._candidate_schedule else None
        for symbol, frame in prepared_frames.items():
            if symbol in positions or trade_date not in frame.index:
                continue
            if approved is not None and symbol not in approved:
                continue
            row = frame.loc[trade_date]
            price = float(row["close"])
            breakouts_disabled = (
                self.config.disable_breakouts_in_uptrend
                and regime_label in ("confirmed_uptrend", "uptrend_under_pressure")
            )
            breakout_ok = (
                False
                if breakouts_disabled
                else self._row_passes_entry(row, price, regime_ok)
            )
            continuation_ok = (
                not breakout_ok
                and self._row_passes_continuation_entry(row, price)
            )
            leader_cont_ok = (
                not breakout_ok
                and not continuation_ok
                and self._row_passes_leader_continuation_entry(row, price)
            )
            if not (breakout_ok or continuation_ok or leader_cont_ok):
                continue
            if self._earnings_entry_blocked(symbol, trade_date):
                continue
            if breakout_ok:
                entry_type = "breakout"
            elif continuation_ok:
                entry_type = "continuation"
            else:
                entry_type = "leader_continuation"

            # Change #6: RVOL gate on breakout entries only (continuation is a
            # pullback setup with no breakout-day volume expectation).
            if entry_type == "breakout" and self.config.min_breakout_volume_ratio > 0:
                rvol = row.get("breakout_volume_ratio")
                if pd.isna(rvol) or float(rvol) < self.config.min_breakout_volume_ratio:
                    continue

            def _num(key: str, default: float = 0.0) -> float:
                val = row.get(key)
                return float(val) if pd.notna(val) else default

            if self.config.use_composite_scoring:
                template = _num("template_score")
                depth = _num("base_depth_pct", default=1.0)
                depth_band = (
                    2 if 0.08 <= depth <= 0.15
                    else 1 if 0.15 < depth <= 0.25
                    else 0
                )
                vcr = _num("volume_contraction_ratio", default=1.5)
                vcr_band = 2 if vcr <= 0.70 else 1 if vcr <= 0.85 else 0
                rvol = _num("breakout_volume_ratio")
                rvol_band = 2 if rvol >= 2.0 else 1 if rvol >= 1.5 else 0
                roc60 = _num("roc_60")
                roc_band = (
                    3 if roc60 >= 0.25
                    else 2 if roc60 >= 0.15
                    else 1 if roc60 >= 0.05
                    else 0
                )
                if self.config.composite_variant == "v2":
                    # v2: drop stage/RS bands (floor-filtered already); double
                    # template and roc to rank inside the leader-cont heavy pool
                    composite = 2.0 * template + depth_band + vcr_band + rvol_band + 2.0 * roc_band
                else:
                    rs = _num("rs_percentile")
                    rs_band = 3 if rs >= 90 else 2 if rs >= 80 else 1 if rs >= 70 else 0
                    stage = _num("stage_number", default=99.0)
                    stage_band = 3 if stage == 1 else 2 if stage == 2 else 0
                    composite = (
                        template + rs_band + depth_band + vcr_band
                        + stage_band + rvol_band + roc_band
                    )
                rank = (
                    int(bool(row.get("breakout_signal"))) if pd.notna(row.get("breakout_signal")) else 0,
                    composite,
                    _num("close_range_pct"),
                    -_num("atr_pct_14"),
                )
            else:
                rank = (
                    int(bool(row.get("breakout_signal"))) if pd.notna(row.get("breakout_signal")) else 0,
                    _num("template_score"),
                    _num("roc_60"),
                    _num("roc_120"),
                    _num("adx_14"),
                    _num("rsi_14"),
                    -_num("atr_pct_14"),
                    _num("close_range_pct"),
                    _num("breakout_volume_ratio"),
                    -_num("stage_number", 99.0),
                )

            candidates.append(
                {
                    "symbol": symbol,
                    "price": price,
                    "row": row,
                    "entry_type": entry_type,
                    "rank": rank,
                }
            )

        candidates.sort(key=lambda item: item["rank"], reverse=True)
        for candidate in candidates:
            if len(positions) >= self.config.max_positions:
                break

            market_value = self._portfolio_market_value(positions, prepared_frames, trade_date)
            overlay_mv = (
                self._overlay_market_value(overlay_state, benchmark_df, trade_date)
                if overlay_state is not None and benchmark_df is not None
                else 0.0
            )
            portfolio_value = cash_ref["cash"] + market_value + overlay_mv
            current_exposure = market_value / portfolio_value if portfolio_value > 0 else 0.0
            if current_exposure >= effective_target_exposure:
                break

            price = candidate["price"]
            row = candidate["row"]
            stop_candidate = row.get("initial_stop_price")
            stop_reference = (
                float(stop_candidate)
                if pd.notna(stop_candidate) and float(stop_candidate) < price
                else price * (1.0 - self.config.stop_loss_pct)
            )
            risk_per_share = price - stop_reference
            if risk_per_share <= 0:
                continue

            per_position_cap = portfolio_value * self.config.max_position_pct * vol_scalar
            remaining_to_target = max(0.0, (effective_target_exposure * portfolio_value) - market_value)
            desired_budget = min(per_position_cap, remaining_to_target)
            if desired_budget <= 0:
                continue

            # If cash is insufficient, liquidate SPY overlay to free the shortfall.
            if (
                self.config.overlay_enabled
                and overlay_state is not None
                and benchmark_df is not None
                and trades is not None
                and cash_ref["cash"] < desired_budget
                and overlay_state.get("shares", 0) > 0
            ):
                shortfall = desired_budget - cash_ref["cash"]
                self._liquidate_overlay(
                    shortfall=shortfall,
                    trade_date=trade_date,
                    overlay_symbol=overlay_symbol,
                    overlay_state=overlay_state,
                    benchmark_df=benchmark_df,
                    cash_ref=cash_ref,
                    trades=trades,
                )

            budget = min(desired_budget, cash_ref["cash"])
            if budget <= 0:
                continue
            risk_budget = (
                portfolio_value
                * self.config.risk_per_trade
                * self.config.initial_entry_fraction
                * vol_scalar
            )
            qty = min(
                int((budget * self.config.initial_entry_fraction) / price),
                int(risk_budget / risk_per_share),
            )
            if qty <= 0:
                continue

            cash_ref["cash"] -= qty * price
            positions[candidate["symbol"]] = {
                "lots": [
                    {
                        "entry_price": price,
                        "shares": qty,
                        "entry_date": trade_date,
                        "leg_type": "initial",
                    }
                ],
                "entry_date": trade_date,
                "highest_close": price,
                "stop_price": stop_reference,
                "partial_taken": False,
                "add_on_1_done": False,
                "add_on_2_done": False,
                "entry_type": candidate["entry_type"],
            }

    def _portfolio_market_value(
        self,
        positions: dict[str, dict],
        prepared_frames: dict[str, pd.DataFrame],
        trade_date,
    ) -> float:
        total = 0.0
        for symbol, position in positions.items():
            frame = prepared_frames.get(symbol)
            if frame is None or trade_date not in frame.index:
                continue
            total += self._total_shares(position["lots"]) * float(frame.loc[trade_date, "close"])
        return total

    @staticmethod
    def _overlay_price(
        benchmark_df: Optional[pd.DataFrame],
        trade_date,
    ) -> Optional[float]:
        if benchmark_df is None or benchmark_df.empty or trade_date not in benchmark_df.index:
            return None
        return float(benchmark_df.loc[trade_date, "close"])

    @staticmethod
    def _cross_asset_price(
        cross_asset_df: Optional[pd.DataFrame], trade_date
    ) -> Optional[float]:
        if (
            cross_asset_df is None
            or cross_asset_df.empty
            or trade_date not in cross_asset_df.index
        ):
            return None
        return float(cross_asset_df.loc[trade_date, "close"])

    def _build_cross_asset_trend(
        self,
        cross_asset_df: Optional[pd.DataFrame],
    ) -> Optional[pd.Series]:
        if cross_asset_df is None or cross_asset_df.empty:
            return None
        window = max(1, self.config.cross_asset_trend_ma_days)
        return cross_asset_df["close"].astype(float).rolling(window).mean()

    def _cross_asset_market_value(
        self,
        cross_asset_state: dict,
        cross_asset_df: Optional[pd.DataFrame],
        trade_date,
    ) -> float:
        if cross_asset_state["shares"] <= 0:
            return 0.0
        price = self._cross_asset_price(cross_asset_df, trade_date)
        if price is None:
            return cross_asset_state["shares"] * cross_asset_state["entry_price"]
        return cross_asset_state["shares"] * price

    def _process_cross_asset(
        self,
        trade_date,
        cross_asset_state: dict,
        cross_asset_df: pd.DataFrame,
        cross_asset_trend: Optional[pd.Series],
        regime_label: str,
        positions: dict[str, dict],
        prepared_frames: dict[str, pd.DataFrame],
        cash_ref: dict[str, float],
        trades: list[dict],
    ) -> None:
        price = self._cross_asset_price(cross_asset_df, trade_date)
        if price is None or price <= 0:
            return
        symbol = self.config.cross_asset_symbol
        trend_ma = (
            cross_asset_trend.get(trade_date)
            if cross_asset_trend is not None
            else None
        )

        if cross_asset_state["shares"] > 0:
            cross_asset_state["highest_close"] = max(
                cross_asset_state["highest_close"], price
            )
            exit_reason = None
            stop_level = cross_asset_state["highest_close"] * (
                1.0 - self.config.cross_asset_stop_loss_pct
            )
            if (
                self.config.cross_asset_exit_on_uptrend
                and regime_label == "confirmed_uptrend"
            ):
                exit_reason = "cross_asset_regime_uptrend"
            elif price <= stop_level:
                exit_reason = "cross_asset_stop"
            elif (
                self.config.cross_asset_require_trend
                and trend_ma is not None
                and pd.notna(trend_ma)
                and price < float(trend_ma)
            ):
                exit_reason = "cross_asset_trend_lost"
            if exit_reason is not None:
                shares = cross_asset_state["shares"]
                proceeds = shares * price
                cost = shares * cross_asset_state["entry_price"]
                cash_ref["cash"] += proceeds
                trades.append(
                    {
                        "symbol": symbol,
                        "entry_date": cross_asset_state["entry_date"],
                        "exit_date": trade_date,
                        "entry_price": cross_asset_state["entry_price"],
                        "exit_price": price,
                        "shares": shares,
                        "pnl": round(proceeds - cost, 2),
                        "return_pct": round((proceeds - cost) / cost, 4) if cost else 0.0,
                        "hold_days": 0,
                        "exit_reason": exit_reason,
                        "leg_type": "cross_asset",
                    }
                )
                cross_asset_state["shares"] = 0
                cross_asset_state["entry_price"] = 0.0
                cross_asset_state["entry_date"] = None
                cross_asset_state["highest_close"] = 0.0
            return

        if regime_label == "confirmed_uptrend":
            return
        if (
            self.config.cross_asset_require_trend
            and trend_ma is not None
            and pd.notna(trend_ma)
            and price < float(trend_ma)
        ):
            return

        stock_mv = self._portfolio_market_value(positions, prepared_frames, trade_date)
        equity = cash_ref["cash"] + stock_mv
        if equity <= 0:
            return
        main_exposure = stock_mv / equity
        if main_exposure >= self.config.cross_asset_exposure_threshold:
            return

        budget = equity * self.config.cross_asset_position_pct
        budget = min(budget, cash_ref["cash"])
        if budget <= 0:
            return
        qty = int(budget / price)
        if qty <= 0:
            return
        cost = qty * price
        cash_ref["cash"] -= cost
        cross_asset_state["shares"] = qty
        cross_asset_state["entry_price"] = price
        cross_asset_state["entry_date"] = trade_date
        cross_asset_state["highest_close"] = price

    def _build_rsi2_features(
        self,
        benchmark_df: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        if benchmark_df is None or benchmark_df.empty or "close" not in benchmark_df.columns:
            return None
        close = benchmark_df["close"].astype(float)
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=0.5, adjust=False).mean()
        avg_loss = loss.ewm(alpha=0.5, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, pd.NA)
        rsi2 = 100.0 - 100.0 / (1.0 + rs)
        rsi2 = rsi2.fillna(100.0)
        exit_ma = close.rolling(self.config.rsi2_sleeve_exit_ma_days).mean()
        long_ma = close.rolling(self.config.rsi2_sleeve_long_term_ma_days).mean()
        return pd.DataFrame(
            {
                "close": close,
                "rsi2": rsi2,
                "exit_ma": exit_ma,
                "long_ma": long_ma,
            },
            index=benchmark_df.index,
        )

    @staticmethod
    def _rsi2_price(rsi2_features: Optional[pd.DataFrame], trade_date) -> Optional[float]:
        if rsi2_features is None or trade_date not in rsi2_features.index:
            return None
        close = rsi2_features.loc[trade_date, "close"]
        return float(close) if pd.notna(close) else None

    def _rsi2_market_value(
        self,
        rsi2_state: dict,
        rsi2_features: Optional[pd.DataFrame],
        trade_date,
    ) -> float:
        if rsi2_state["shares"] <= 0:
            return 0.0
        price = self._rsi2_price(rsi2_features, trade_date)
        if price is None:
            return rsi2_state["shares"] * rsi2_state["entry_price"]
        return rsi2_state["shares"] * price

    def _process_rsi2_sleeve(
        self,
        trade_date,
        rsi2_state: dict,
        rsi2_features: pd.DataFrame,
        positions: dict[str, dict],
        prepared_frames: dict[str, pd.DataFrame],
        cash_ref: dict[str, float],
        trades: list[dict],
    ) -> None:
        if trade_date not in rsi2_features.index:
            return
        row = rsi2_features.loc[trade_date]
        price = float(row["close"]) if pd.notna(row["close"]) else None
        if price is None or price <= 0:
            return

        if rsi2_state["shares"] > 0:
            exit_ma = row.get("exit_ma")
            bars_held = rsi2_state["entry_bar_idx"]
            rsi2_state["entry_bar_idx"] = bars_held + 1
            exit_reason = None
            if pd.notna(exit_ma) and price > float(exit_ma):
                exit_reason = "rsi2_exit_ma"
            elif (bars_held + 1) >= self.config.rsi2_sleeve_max_hold_days:
                exit_reason = "rsi2_max_hold"
            if exit_reason is not None:
                shares = rsi2_state["shares"]
                proceeds = shares * price
                cost = shares * rsi2_state["entry_price"]
                cash_ref["cash"] += proceeds
                trades.append(
                    {
                        "symbol": "SPY",
                        "entry_date": rsi2_state["entry_date"],
                        "exit_date": trade_date,
                        "entry_price": rsi2_state["entry_price"],
                        "exit_price": price,
                        "shares": shares,
                        "pnl": round(proceeds - cost, 2),
                        "return_pct": round((proceeds - cost) / cost, 4) if cost else 0.0,
                        "hold_days": bars_held + 1,
                        "exit_reason": exit_reason,
                        "leg_type": "rsi2_sleeve",
                    }
                )
                rsi2_state["shares"] = 0
                rsi2_state["entry_price"] = 0.0
                rsi2_state["entry_date"] = None
                rsi2_state["entry_bar_idx"] = -1
            return

        long_ma = row.get("long_ma")
        rsi2 = row.get("rsi2")
        if pd.isna(long_ma) or pd.isna(rsi2):
            return
        if price <= float(long_ma):
            return
        if float(rsi2) >= self.config.rsi2_sleeve_entry_threshold:
            return

        stock_mv = self._portfolio_market_value(positions, prepared_frames, trade_date)
        equity = cash_ref["cash"] + stock_mv
        if equity <= 0:
            return
        main_exposure = stock_mv / equity
        if main_exposure >= self.config.rsi2_sleeve_exposure_threshold:
            return

        budget = equity * self.config.rsi2_sleeve_position_pct
        budget = min(budget, cash_ref["cash"])
        if budget <= 0:
            return
        qty = int(budget / price)
        if qty <= 0:
            return
        cost = qty * price
        cash_ref["cash"] -= cost
        rsi2_state["shares"] = qty
        rsi2_state["entry_price"] = price
        rsi2_state["entry_date"] = trade_date
        rsi2_state["entry_bar_idx"] = 0

    def _overlay_market_value(
        self,
        overlay_state: dict,
        benchmark_df: Optional[pd.DataFrame],
        trade_date,
    ) -> float:
        if overlay_state["shares"] <= 0:
            return 0.0
        price = self._overlay_price(benchmark_df, trade_date)
        if price is None:
            return overlay_state["cost_basis"]
        return overlay_state["shares"] * price

    def _liquidate_overlay(
        self,
        shortfall: float,
        trade_date,
        overlay_symbol: str,
        overlay_state: dict,
        benchmark_df: pd.DataFrame,
        cash_ref: dict,
        trades: list,
    ) -> None:
        """Sell just enough SPY overlay shares to free ``shortfall`` cash."""
        price = self._overlay_price(benchmark_df, trade_date)
        if price is None or price <= 0 or overlay_state["shares"] <= 0:
            return

        qty = min(overlay_state["shares"], int(shortfall / price) + 1)
        if qty <= 0:
            return
        proceeds = qty * price
        avg_cost = (
            overlay_state["cost_basis"] / overlay_state["shares"]
            if overlay_state["shares"] > 0
            else 0.0
        )
        realized_cost = avg_cost * qty
        cash_ref["cash"] += proceeds
        overlay_state["shares"] -= qty
        overlay_state["cost_basis"] = max(0.0, overlay_state["cost_basis"] - realized_cost)
        trades.append(
            {
                "symbol": overlay_symbol,
                "entry_date": None,
                "exit_date": trade_date,
                "entry_price": avg_cost,
                "exit_price": price,
                "shares": qty,
                "pnl": round(proceeds - realized_cost, 2),
                "return_pct": round((proceeds - realized_cost) / realized_cost, 4) if realized_cost else 0.0,
                "hold_days": 0,
                "exit_reason": "overlay_make_room",
                "leg_type": "overlay",
            }
        )

    def _rebalance_overlay(
        self,
        trade_date,
        overlay_symbol: str,
        overlay_state: dict,
        benchmark_df: pd.DataFrame,
        positions: dict[str, dict],
        prepared_frames: dict[str, pd.DataFrame],
        target_exposure: float,
        regime_label: str,
        cash_ref: dict,
        trades: list,
    ) -> None:
        price = self._overlay_price(benchmark_df, trade_date)
        if price is None or price <= 0:
            return

        effective_target = target_exposure

        stock_mv = self._portfolio_market_value(positions, prepared_frames, trade_date)
        overlay_mv = overlay_state["shares"] * price
        equity = cash_ref["cash"] + stock_mv + overlay_mv

        target_total_mv = equity * effective_target
        desired_overlay_mv = max(0.0, target_total_mv - stock_mv)
        # Cap overlay by available cash when buying.
        desired_overlay_mv = min(desired_overlay_mv, cash_ref["cash"] + overlay_mv)
        delta_mv = desired_overlay_mv - overlay_mv

        # Only act if the gap is material to avoid daily churn.
        if abs(delta_mv) < equity * self.config.overlay_rebalance_threshold:
            return

        if delta_mv > 0:
            qty = int(delta_mv / price)
            if qty <= 0:
                return
            cost = qty * price
            if cost > cash_ref["cash"]:
                return
            cash_ref["cash"] -= cost
            overlay_state["shares"] += qty
            overlay_state["cost_basis"] += cost
        else:
            qty = min(overlay_state["shares"], int((-delta_mv) / price) + 1)
            if qty <= 0:
                return
            proceeds = qty * price
            avg_cost = (
                overlay_state["cost_basis"] / overlay_state["shares"]
                if overlay_state["shares"] > 0
                else 0.0
            )
            realized_cost = avg_cost * qty
            cash_ref["cash"] += proceeds
            overlay_state["shares"] -= qty
            overlay_state["cost_basis"] = max(0.0, overlay_state["cost_basis"] - realized_cost)
            trades.append(
                {
                    "symbol": overlay_symbol,
                    "entry_date": None,
                    "exit_date": trade_date,
                    "entry_price": avg_cost,
                    "exit_price": price,
                    "shares": qty,
                    "pnl": round(proceeds - realized_cost, 2),
                    "return_pct": round((proceeds - realized_cost) / realized_cost, 4) if realized_cost else 0.0,
                    "hold_days": 0,
                    "exit_reason": "overlay_rebalance",
                    "leg_type": "overlay",
                }
            )

    def _regime_state(self, regime_frame: pd.DataFrame, trade_date) -> tuple[str, bool, float]:
        if not regime_frame.empty and trade_date in regime_frame.index:
            regime_label = str(regime_frame.loc[trade_date, "market_regime"])
            regime_ok = bool(regime_frame.loc[trade_date, "market_confirmed_uptrend"])
        else:
            regime_label = "uptrend_under_pressure"
            regime_ok = False

        exposure_map = {
            "confirmed_uptrend": self.config.target_exposure_confirmed_uptrend,
            "uptrend_under_pressure": self.config.target_exposure_uptrend_under_pressure,
            "market_correction": self.config.target_exposure_market_correction,
        }
        return regime_label, regime_ok, exposure_map.get(
            regime_label,
            self.config.target_exposure_uptrend_under_pressure,
        )

    def _portfolio_benchmark_return(
        self,
        benchmark_df: Optional[pd.DataFrame],
        trade_start_ts: pd.Timestamp,
    ) -> float:
        if benchmark_df is None or benchmark_df.empty:
            return 0.0
        frame = benchmark_df.loc[benchmark_df.index >= trade_start_ts]
        if frame.empty:
            return 0.0
        start_price = float(frame["close"].iloc[0])
        end_price = float(frame["close"].iloc[-1])
        if start_price <= 0:
            return 0.0
        return (end_price / start_price) - 1.0

    def _liquidate_position(
        self,
        symbol: str,
        position: dict,
        exit_price: float,
        exit_date,
        trades: list[dict],
        exit_reason: str,
    ) -> float:
        remaining = self._total_shares(position["lots"])
        trades.extend(
            self._close_lots(
                lots=position["lots"],
                sell_qty=remaining,
                exit_price=exit_price,
                exit_date=exit_date,
                symbol=symbol,
                exit_reason=exit_reason,
            )
        )
        return remaining * exit_price

    def _build_symbol_summary(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        if trades_df.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "total_trades",
                    "win_rate",
                    "total_pnl",
                    "total_return",
                    "avg_trade_return",
                    "avg_hold_days",
                ]
            )
        grouped = trades_df.groupby("symbol").agg(
            total_trades=("symbol", "size"),
            win_rate=("pnl", lambda s: float((s > 0).mean())),
            total_pnl=("pnl", "sum"),
            avg_trade_return=("return_pct", "mean"),
            avg_hold_days=("hold_days", "mean"),
        )
        grouped["total_return"] = grouped["total_pnl"] / float(self.config.initial_cash)
        return grouped.reset_index().sort_values("total_return", ascending=False)
