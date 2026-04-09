"""Backtesting engine — run strategies against historical data."""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Type, Optional

import backtrader as bt
import yfinance as yf
import pandas as pd

from .strategies import SwingStrategy, DayTradingStrategy, CombinedStrategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Run backtests against historical data and compare strategies."""

    def __init__(self, initial_cash: float = 100_000, commission: float = 0.0):
        self.initial_cash = initial_cash
        self.commission = commission

    def run_single(
        self,
        symbol: str,
        strategy_class: Type[bt.Strategy],
        start_date: str = None,
        end_date: str = None,
        strategy_params: dict = None,
    ) -> Dict:
        """Run a single backtest for one symbol and strategy.

        Returns dict with performance metrics.
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Backtesting {strategy_class.__name__} on {symbol} ({start_date} to {end_date})")

        # Download data
        df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if df.empty:
            return {"symbol": symbol, "error": "No data available"}

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Setup backtrader
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)

        data = bt.feeds.PandasData(dataname=df, datetime=None)
        cerebro.adddata(data)

        params = strategy_params or {}
        cerebro.addstrategy(strategy_class, **params)

        # Analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.05)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

        # Run
        start_value = cerebro.broker.getvalue()
        results = cerebro.run()
        end_value = cerebro.broker.getvalue()
        strat = results[0]

        # Extract metrics
        total_return = (end_value - start_value) / start_value

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

        # Buy & hold comparison
        buy_hold_return = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]

        return {
            "symbol": symbol,
            "strategy": strategy_class.__name__,
            "period": f"{start_date} to {end_date}",
            "start_value": start_value,
            "end_value": round(end_value, 2),
            "total_return": round(total_return, 4),
            "total_return_pct": f"{total_return:.2%}",
            "buy_hold_return": round(float(buy_hold_return), 4),
            "buy_hold_return_pct": f"{float(buy_hold_return):.2%}",
            "beats_buy_hold": total_return > float(buy_hold_return),
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

    def run_comparison(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None,
    ) -> Dict:
        """Run all three strategies on multiple symbols and compare."""
        strategies = [
            ("Swing", SwingStrategy),
            ("DayTrading", DayTradingStrategy),
            ("Combined", CombinedStrategy),
        ]

        all_results = []

        for symbol in symbols:
            for name, strat_class in strategies:
                try:
                    result = self.run_single(symbol, strat_class, start_date, end_date)
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Backtest failed for {symbol}/{name}: {e}")
                    all_results.append({
                        "symbol": symbol, "strategy": name, "error": str(e)
                    })

        # Aggregate
        valid = [r for r in all_results if "error" not in r]
        by_strategy = {}
        for r in valid:
            s = r["strategy"]
            if s not in by_strategy:
                by_strategy[s] = []
            by_strategy[s].append(r)

        summary = {}
        for strat_name, results in by_strategy.items():
            returns = [r["total_return"] for r in results]
            win_rates = [r["win_rate"] for r in results]
            drawdowns = [r["max_drawdown"] for r in results]
            beat_count = sum(1 for r in results if r.get("beats_buy_hold"))

            summary[strat_name] = {
                "avg_return": f"{sum(returns)/len(returns):.2%}" if returns else "N/A",
                "avg_win_rate": f"{sum(win_rates)/len(win_rates):.1%}" if win_rates else "N/A",
                "avg_max_drawdown": f"{sum(drawdowns)/len(drawdowns):.2%}" if drawdowns else "N/A",
                "beats_buy_hold": f"{beat_count}/{len(results)}",
                "total_trades": sum(r["total_trades"] for r in results),
            }

        return {
            "results": all_results,
            "summary": summary,
        }

    @staticmethod
    def print_results(results: Dict):
        """Pretty-print backtest results."""
        if "error" in results:
            print(f"  ERROR: {results['error']}")
            return

        print(f"\n{'=' * 70}")
        print(f"  {results['symbol']} — {results['strategy']} — {results['period']}")
        print(f"{'=' * 70}")
        print(f"  Starting capital:  ${results['start_value']:>12,.2f}")
        print(f"  Ending capital:    ${results['end_value']:>12,.2f}")
        print(f"  Strategy return:   {results['total_return_pct']:>12}")
        print(f"  Buy & hold return: {results['buy_hold_return_pct']:>12}")
        print(f"  Beats buy & hold:  {'YES' if results['beats_buy_hold'] else 'NO':>12}")
        print(f"  Sharpe ratio:      {str(results['sharpe_ratio']):>12}")
        print(f"  Max drawdown:      {results['max_drawdown_pct']:>12}")
        print(f"  Total trades:      {results['total_trades']:>12}")
        print(f"  Win rate:          {results['win_rate_pct']:>12}")
        print(f"  Avg win:           ${results['avg_win']:>11,.2f}")
        print(f"  Avg loss:          ${results['avg_loss']:>11,.2f}")
        print(f"  Profit factor:     {str(results['profit_factor']):>12}")

    @staticmethod
    def print_comparison(comparison: Dict):
        """Pretty-print comparison results."""
        print(f"\n{'#' * 70}")
        print(f"  STRATEGY COMPARISON SUMMARY")
        print(f"{'#' * 70}")

        for strat_name, stats in comparison["summary"].items():
            print(f"\n  {strat_name}:")
            print(f"    Avg return:       {stats['avg_return']}")
            print(f"    Avg win rate:     {stats['avg_win_rate']}")
            print(f"    Avg max drawdown: {stats['avg_max_drawdown']}")
            print(f"    Beats buy & hold: {stats['beats_buy_hold']}")
            print(f"    Total trades:     {stats['total_trades']}")

        # Per-symbol detail
        print(f"\n{'=' * 70}")
        print(f"  PER-SYMBOL RESULTS")
        print(f"{'=' * 70}")

        for r in comparison["results"]:
            if "error" in r:
                print(f"  {r['symbol']}/{r['strategy']}: ERROR - {r['error']}")
            else:
                beat = ">" if r["beats_buy_hold"] else "<"
                print(
                    f"  {r['symbol']:>6} | {r['strategy']:<16} | "
                    f"Return: {r['total_return_pct']:>8} {beat} B&H: {r['buy_hold_return_pct']:>8} | "
                    f"Win: {r['win_rate_pct']:>5} | DD: {r['max_drawdown_pct']:>7} | "
                    f"Trades: {r['total_trades']:>3}"
                )
