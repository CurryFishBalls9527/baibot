"""Portfolio-level Minervini backtesting with shared capital and exposure scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .backtester import BacktestConfig, MinerviniBacktester
from .minervini import MinerviniScreener


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
    ):
        super().__init__(screener=screener, config=config)

    def backtest_portfolio(
        self,
        data_by_symbol: Dict[str, pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame] = None,
        market_context_df: Optional[pd.DataFrame] = None,
        trade_start_date: Optional[str] = None,
        candidate_schedule: Optional[Dict] = None,
    ) -> PortfolioBacktestResult:
        self._candidate_schedule = candidate_schedule
        prepared_frames = {}
        for symbol, df in data_by_symbol.items():
            prepared = self.screener.prepare_features(df)
            if prepared.empty:
                continue
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

        for trade_date in all_dates:
            regime_label, regime_ok, target_exposure = self._regime_state(
                regime_frame, trade_date
            )
            cash_ref = {"cash": cash}

            stock_mv_before = self._portfolio_market_value(positions, prepared_frames, trade_date)
            overlay_mv_before = self._overlay_market_value(
                overlay_state, benchmark_df, trade_date
            )
            equity_before = cash_ref["cash"] + stock_mv_before + overlay_mv_before
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

            market_value = self._portfolio_market_value(positions, prepared_frames, trade_date)
            overlay_mv = self._overlay_market_value(overlay_state, benchmark_df, trade_date)
            equity = cash + market_value + overlay_mv
            total_exposure_mv = market_value + overlay_mv
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
                row = prepared_frames[symbol].loc[final_date]
                price = float(row["close"])
                cash += self._liquidate_position(
                    symbol=symbol,
                    position=positions.pop(symbol),
                    exit_price=price,
                    exit_date=final_date,
                    trades=trades,
                    exit_reason="final_bar",
                )

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
                position["stop_price"] = max(position["stop_price"], average_cost)

            if position["partial_taken"] and pd.notna(row.get("ema_21")):
                position["stop_price"] = max(position["stop_price"], float(row["ema_21"]))
            else:
                position["stop_price"] = max(
                    position["stop_price"],
                    position["highest_close"] * (1.0 - self.config.trail_stop_pct),
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
            if price <= position["stop_price"]:
                exit_reason = "stop"
            elif self.config.exit_on_market_correction and regime_label == "market_correction":
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
            if not (breakout_ok or continuation_ok):
                continue
            entry_type = "breakout" if breakout_ok else "continuation"

            def _num(key: str, default: float = 0.0) -> float:
                val = row.get(key)
                return float(val) if pd.notna(val) else default

            candidates.append(
                {
                    "symbol": symbol,
                    "price": price,
                    "row": row,
                    "entry_type": entry_type,
                    "rank": (
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
                    ),
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
