"""Portfolio-level backtesting — multiple stocks held simultaneously.

Solves the three core problems:
1. Low capital utilization → allocate across multiple stocks (e.g. 10 stocks × 10%)
2. Too much time in cash → stay invested, use trailing stops instead of quick exits
3. Single-stock testing → true portfolio with rebalancing

Also benchmarks against SPY (buy & hold S&P 500).
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Type

import backtrader as bt
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Portfolio-aware strategies ─────────────────────────────────────

class PortfolioSwingStrategy(bt.Strategy):
    """Multi-stock swing strategy: trend following with trailing stops.

    For each stock in the portfolio:
    - BUY when EMA9 > EMA21 + RSI ok + MACD bullish + above EMA50
    - SELL only on trailing stop (2x ATR) or strong bearish reversal
    - Stays in position as long as trend holds (no premature exits)
    - Each stock gets equal allocation (total_cash / num_stocks)
    """

    params = dict(
        ema_fast=9,
        ema_slow=21,
        ema_trend=50,
        rsi_period=14,
        rsi_low=35,
        rsi_high=80,
        atr_period=14,
        atr_trail_mult=2.5,
    )

    def __init__(self):
        self.indicators = {}
        self.stops = {}
        self.orders = {}

        for d in self.datas:
            name = d._name
            self.indicators[name] = {
                "ema_fast": bt.indicators.EMA(d.close, period=self.p.ema_fast),
                "ema_slow": bt.indicators.EMA(d.close, period=self.p.ema_slow),
                "ema_trend": bt.indicators.EMA(d.close, period=self.p.ema_trend),
                "rsi": bt.indicators.RSI(d.close, period=self.p.rsi_period),
                "macd": bt.indicators.MACD(d.close),
                "atr": bt.indicators.ATR(d, period=self.p.atr_period),
            }
            self.stops[name] = 0
            self.orders[name] = None

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.orders[order.data._name] = None

    def next(self):
        n_stocks = len(self.datas)
        alloc_per_stock = self.broker.getvalue() / n_stocks

        for d in self.datas:
            name = d._name
            ind = self.indicators[name]

            if self.orders[name]:
                continue

            pos = self.getposition(d)

            if not pos.size:
                # ── Entry: trend following ──
                bullish = (
                    ind["ema_fast"][0] > ind["ema_slow"][0]
                    and self.p.rsi_low < ind["rsi"][0] < self.p.rsi_high
                    and ind["macd"].macd[0] > ind["macd"].signal[0]
                    and d.close[0] > ind["ema_trend"][0]
                    and ind["atr"][0] > 0
                )
                if bullish:
                    size = int(alloc_per_stock / d.close[0])
                    if size > 0:
                        self.stops[name] = d.close[0] - ind["atr"][0] * self.p.atr_trail_mult
                        self.orders[name] = self.buy(data=d, size=size)
            else:
                # ── Exit: trailing stop or strong reversal ──
                # Trail the stop up
                new_stop = d.close[0] - ind["atr"][0] * self.p.atr_trail_mult
                if new_stop > self.stops[name]:
                    self.stops[name] = new_stop

                # Only exit on stop hit or strong bearish signal
                strong_bearish = (
                    ind["ema_fast"][0] < ind["ema_slow"][0]
                    and ind["macd"].macd[0] < ind["macd"].signal[0]
                    and ind["rsi"][0] > self.p.rsi_high
                )

                if d.close[0] <= self.stops[name] or strong_bearish:
                    self.orders[name] = self.close(data=d)
                    self.stops[name] = 0


class PortfolioDayTradingStrategy(bt.Strategy):
    """Multi-stock mean reversion with quick entries and wider exits."""

    params = dict(
        rsi_period=14,
        rsi_buy=42,
        rsi_sell=72,
        bb_period=20,
        bb_dev=2.0,
        atr_period=14,
        atr_trail_mult=2.0,
        max_hold_days=15,
    )

    def __init__(self):
        self.indicators = {}
        self.stops = {}
        self.targets = {}
        self.entry_bars = {}
        self.orders = {}

        for d in self.datas:
            name = d._name
            self.indicators[name] = {
                "rsi": bt.indicators.RSI(d.close, period=self.p.rsi_period),
                "boll": bt.indicators.BollingerBands(d.close, period=self.p.bb_period, devfactor=self.p.bb_dev),
                "macd": bt.indicators.MACD(d.close),
                "atr": bt.indicators.ATR(d, period=self.p.atr_period),
                "vol_avg": bt.indicators.SMA(d.volume, period=20),
            }
            self.stops[name] = 0
            self.targets[name] = 0
            self.entry_bars[name] = 0
            self.orders[name] = None

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.orders[order.data._name] = None

    def next(self):
        n_stocks = len(self.datas)
        alloc = self.broker.getvalue() / n_stocks

        for d in self.datas:
            name = d._name
            ind = self.indicators[name]

            if self.orders[name]:
                continue

            pos = self.getposition(d)

            if not pos.size:
                oversold = ind["rsi"][0] < self.p.rsi_buy
                near_lower = d.close[0] <= ind["boll"].mid[0]
                turning = ind["macd"].macd[0] > ind["macd"].macd[-1]

                if oversold and near_lower and turning:
                    size = int(alloc / d.close[0])
                    if size > 0:
                        atr = ind["atr"][0]
                        self.stops[name] = d.close[0] - atr * self.p.atr_trail_mult
                        self.targets[name] = d.close[0] + atr * 3.0
                        self.entry_bars[name] = len(self)
                        self.orders[name] = self.buy(data=d, size=size)
            else:
                # Trail stop
                new_stop = d.close[0] - ind["atr"][0] * self.p.atr_trail_mult
                if new_stop > self.stops[name]:
                    self.stops[name] = new_stop

                bars_held = len(self) - self.entry_bars[name]

                if (
                    d.close[0] <= self.stops[name]
                    or d.close[0] >= self.targets[name]
                    or ind["rsi"][0] > self.p.rsi_sell
                    or bars_held >= self.p.max_hold_days
                ):
                    self.orders[name] = self.close(data=d)


class PortfolioCombinedStrategy(bt.Strategy):
    """Multi-stock scoring strategy: stay invested, trail stops, wide targets."""

    params = dict(
        ema_fast=9,
        ema_slow=21,
        ema_trend=50,
        rsi_period=14,
        bb_period=20,
        bb_dev=2.0,
        atr_period=14,
        vol_avg_period=20,
        buy_threshold=4,
        sell_threshold=4,
        atr_trail_mult=2.5,
    )

    def __init__(self):
        self.indicators = {}
        self.stops = {}
        self.orders = {}

        for d in self.datas:
            name = d._name
            self.indicators[name] = {
                "ema_fast": bt.indicators.EMA(d.close, period=self.p.ema_fast),
                "ema_slow": bt.indicators.EMA(d.close, period=self.p.ema_slow),
                "ema_trend": bt.indicators.EMA(d.close, period=self.p.ema_trend),
                "rsi": bt.indicators.RSI(d.close, period=self.p.rsi_period),
                "boll": bt.indicators.BollingerBands(d.close, period=self.p.bb_period, devfactor=self.p.bb_dev),
                "macd": bt.indicators.MACD(d.close),
                "atr": bt.indicators.ATR(d, period=self.p.atr_period),
                "vol_avg": bt.indicators.SMA(d.volume, period=self.p.vol_avg_period),
            }
            self.stops[name] = 0
            self.orders[name] = None

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.orders[order.data._name] = None

    def _buy_score(self, d, ind) -> int:
        score = 0
        if ind["ema_fast"][0] > ind["ema_slow"][0]:
            score += 1
        if d.close[0] > ind["ema_trend"][0]:
            score += 1
        if 35 < ind["rsi"][0] < 70:
            score += 1
        if ind["macd"].macd[0] > ind["macd"].signal[0]:
            score += 1
        if ind["boll"].bot[0] < d.close[0] < ind["boll"].top[0]:
            score += 1
        if d.volume[0] > ind["vol_avg"][0]:
            score += 1
        return score

    def _sell_score(self, d, ind) -> int:
        score = 0
        if ind["ema_fast"][0] < ind["ema_slow"][0]:
            score += 1
        if ind["rsi"][0] > 75:
            score += 1
        if ind["macd"].macd[0] < ind["macd"].signal[0]:
            score += 1
        if d.close[0] > ind["boll"].top[0]:
            score += 1
        if d.close[0] < ind["ema_trend"][0]:
            score += 1
        return score

    def next(self):
        n_stocks = len(self.datas)
        alloc = self.broker.getvalue() / n_stocks

        for d in self.datas:
            name = d._name
            ind = self.indicators[name]

            if self.orders[name]:
                continue

            pos = self.getposition(d)

            if not pos.size:
                if self._buy_score(d, ind) >= self.p.buy_threshold and ind["atr"][0] > 0:
                    size = int(alloc / d.close[0])
                    if size > 0:
                        self.stops[name] = d.close[0] - ind["atr"][0] * self.p.atr_trail_mult
                        self.orders[name] = self.buy(data=d, size=size)
            else:
                new_stop = d.close[0] - ind["atr"][0] * self.p.atr_trail_mult
                if new_stop > self.stops[name]:
                    self.stops[name] = new_stop

                if d.close[0] <= self.stops[name] or self._sell_score(d, ind) >= self.p.sell_threshold:
                    self.orders[name] = self.close(data=d)


# ─── Portfolio Backtest Engine ──────────────────────────────────────

class PortfolioBacktestEngine:
    """Run portfolio-level backtests with multiple stocks held simultaneously."""

    def __init__(self, initial_cash: float = 100_000, commission: float = 0.0):
        self.initial_cash = initial_cash
        self.commission = commission

    def _download_data(self, symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        data = {}
        for sym in symbols:
            df = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[sym] = df
        return data

    def _get_spy_return(self, start: str, end: str) -> float:
        df = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            return 0.0
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0])

    def run(
        self,
        symbols: List[str],
        strategy_class: Type[bt.Strategy],
        start_date: str = None,
        end_date: str = None,
        strategy_params: dict = None,
    ) -> Dict:
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Download all stock data
        all_data = self._download_data(symbols, start_date, end_date)
        if not all_data:
            return {"error": "No data downloaded"}

        # Setup cerebro
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)

        for sym, df in all_data.items():
            feed = bt.feeds.PandasData(dataname=df, datetime=None, name=sym)
            cerebro.adddata(feed)

        params = strategy_params or {}
        cerebro.addstrategy(strategy_class, **params)

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.05)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        start_value = cerebro.broker.getvalue()
        results = cerebro.run()
        end_value = cerebro.broker.getvalue()
        strat = results[0]

        total_return = (end_value - start_value) / start_value
        spy_return = self._get_spy_return(start_date, end_date)

        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        sharpe = sharpe_analysis.get("sharperatio")

        dd = strat.analyzers.drawdown.get_analysis()
        max_dd = dd.get("max", {}).get("drawdown", 0) / 100

        trades = strat.analyzers.trades.get_analysis()
        total_trades = trades.get("total", {}).get("total", 0)
        won = trades.get("won", {}).get("total", 0)
        lost = trades.get("lost", {}).get("total", 0)
        win_rate = won / total_trades if total_trades > 0 else 0

        avg_win = trades.get("won", {}).get("pnl", {}).get("average", 0)
        avg_loss = abs(trades.get("lost", {}).get("pnl", {}).get("average", 0))
        profit_factor = (won * avg_win) / (lost * avg_loss) if lost > 0 and avg_loss > 0 else float("inf")

        return {
            "strategy": strategy_class.__name__,
            "symbols": list(all_data.keys()),
            "num_stocks": len(all_data),
            "period": f"{start_date} to {end_date}",
            "start_value": start_value,
            "end_value": round(end_value, 2),
            "total_return": round(total_return, 4),
            "total_return_pct": f"{total_return:.2%}",
            "spy_return": round(spy_return, 4),
            "spy_return_pct": f"{spy_return:.2%}",
            "beats_spy": total_return > spy_return,
            "excess_return": f"{total_return - spy_return:.2%}",
            "sharpe_ratio": round(sharpe, 2) if sharpe else None,
            "max_drawdown": round(max_dd, 4),
            "max_drawdown_pct": f"{max_dd:.2%}",
            "total_trades": total_trades,
            "wins": won,
            "losses": lost,
            "win_rate": round(win_rate, 4),
            "win_rate_pct": f"{win_rate:.1%}",
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
        }

    def run_all_strategies(self, symbols: List[str], start_date: str = None, end_date: str = None) -> Dict:
        strategies = [
            ("Swing", PortfolioSwingStrategy),
            ("DayTrading", PortfolioDayTradingStrategy),
            ("Combined", PortfolioCombinedStrategy),
        ]

        results = []
        for name, strat_class in strategies:
            try:
                r = self.run(symbols, strat_class, start_date, end_date)
                results.append(r)
            except Exception as e:
                logger.error(f"{name} failed: {e}")
                results.append({"strategy": name, "error": str(e)})

        return results

    @staticmethod
    def print_result(r: Dict):
        if "error" in r:
            print(f"  {r.get('strategy', '?')}: ERROR - {r['error']}")
            return

        beat_label = "YES ✓" if r["beats_spy"] else "NO"

        print(f"\n{'=' * 70}")
        print(f"  {r['strategy']} — {r['num_stocks']} stocks — {r['period']}")
        print(f"{'=' * 70}")
        print(f"  Stocks:            {', '.join(r['symbols'])}")
        print(f"  Starting capital:  ${r['start_value']:>12,.2f}")
        print(f"  Ending capital:    ${r['end_value']:>12,.2f}")
        print(f"  ---")
        print(f"  Strategy return:   {r['total_return_pct']:>12}")
        print(f"  S&P 500 return:    {r['spy_return_pct']:>12}")
        print(f"  Beats S&P 500:     {beat_label:>12}   (excess: {r['excess_return']})")
        print(f"  ---")
        print(f"  Sharpe ratio:      {str(r['sharpe_ratio']):>12}")
        print(f"  Max drawdown:      {r['max_drawdown_pct']:>12}")
        print(f"  Total trades:      {r['total_trades']:>12}")
        print(f"  Win rate:          {r['win_rate_pct']:>12}")
        print(f"  Avg win:           ${r['avg_win']:>11,.2f}")
        print(f"  Avg loss:          ${r['avg_loss']:>11,.2f}")
        print(f"  Profit factor:     {str(r['profit_factor']):>12}")

    @staticmethod
    def print_comparison(results: List[Dict]):
        print(f"\n{'#' * 70}")
        print(f"  PORTFOLIO BACKTEST COMPARISON vs S&P 500")
        print(f"{'#' * 70}")

        valid = [r for r in results if "error" not in r]
        if not valid:
            print("  No valid results.")
            return

        spy_ret = valid[0]["spy_return_pct"]
        print(f"\n  S&P 500 benchmark: {spy_ret}")
        print(f"  {'─' * 60}")

        for r in valid:
            beat = ">" if r["beats_spy"] else "<"
            print(
                f"  {r['strategy']:<28} | "
                f"Return: {r['total_return_pct']:>8} {beat} SPY {r['spy_return_pct']:>8} | "
                f"Win: {r['win_rate_pct']:>5} | "
                f"DD: {r['max_drawdown_pct']:>7} | "
                f"Trades: {r['total_trades']:>4} | "
                f"PF: {str(r['profit_factor']):>6}"
            )

        best = max(valid, key=lambda x: x["total_return"])
        print(f"\n  Best: {best['strategy']} at {best['total_return_pct']}")
        if best["beats_spy"]:
            print(f"  Outperformed S&P 500 by {best['excess_return']}")
        else:
            print(f"  Underperformed S&P 500 by {best['excess_return']}")
