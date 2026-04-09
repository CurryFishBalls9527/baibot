"""Smart portfolio strategies that address the core problems.

Key improvements over naive strategies:
1. Momentum ranking — allocate more to strongest stocks, less/none to weakest
2. Market regime filter — reduce exposure when S&P 500 is below 200 SMA
3. Wider trailing stops — 3x ATR to avoid being shaken out on normal pullbacks
4. Monthly rebalancing — continuously shift from losers to winners
5. Volatility-adjusted sizing — less volatile stocks get more capital (risk parity lite)
"""

import backtrader as bt
import numpy as np


class MomentumRankingStrategy(bt.Strategy):
    """Momentum + Trend strategy: buy the strongest, avoid the weakest.

    Each month:
    1. Rank all stocks by momentum (past N days return)
    2. Buy the top half, sell the bottom half
    3. Weight by momentum strength (stronger momentum = bigger position)
    4. Use trailing ATR stop for each position
    5. If the stock's own trend is down (below EMA50), skip it even if ranked high

    This captures the well-documented momentum anomaly while controlling risk.
    """

    params = dict(
        momentum_period=63,       # ~3 months lookback for ranking
        rebalance_days=21,        # rebalance monthly
        ema_trend=50,             # trend filter per stock
        atr_period=14,
        atr_trail_mult=3.0,      # wider stops to stay in trends
        top_pct=0.6,             # buy top 60% of ranked stocks
        max_position_pct=0.20,   # cap any single stock at 20%
    )

    def __init__(self):
        self.indicators = {}
        self.stops = {}
        self.orders = {}
        self.rebal_counter = 0

        for d in self.datas:
            name = d._name
            self.indicators[name] = {
                "ema_trend": bt.indicators.EMA(d.close, period=self.p.ema_trend),
                "atr": bt.indicators.ATR(d, period=self.p.atr_period),
            }
            self.stops[name] = 0
            self.orders[name] = None

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.orders[order.data._name] = None

    def _get_momentum(self, d) -> float:
        if len(d) < self.p.momentum_period:
            return 0.0
        old_price = d.close[-self.p.momentum_period]
        if old_price <= 0:
            return 0.0
        return (d.close[0] - old_price) / old_price

    def next(self):
        self.rebal_counter += 1

        # Update trailing stops for held positions
        for d in self.datas:
            name = d._name
            pos = self.getposition(d)
            if pos.size > 0 and self.indicators[name]["atr"][0] > 0:
                new_stop = d.close[0] - self.indicators[name]["atr"][0] * self.p.atr_trail_mult
                if new_stop > self.stops[name]:
                    self.stops[name] = new_stop
                # Check stop
                if d.close[0] <= self.stops[name] and not self.orders[name]:
                    self.orders[name] = self.close(data=d)
                    self.stops[name] = 0

        # Rebalance periodically
        if self.rebal_counter % self.p.rebalance_days != 0:
            return

        # Any pending orders? skip
        if any(self.orders[d._name] for d in self.datas):
            return

        # 1. Rank stocks by momentum
        rankings = []
        for d in self.datas:
            name = d._name
            mom = self._get_momentum(d)
            above_trend = d.close[0] > self.indicators[name]["ema_trend"][0]
            rankings.append((name, d, mom, above_trend))

        rankings.sort(key=lambda x: x[2], reverse=True)

        # 2. Determine which to hold (top momentum + above trend)
        n_top = max(1, int(len(rankings) * self.p.top_pct))
        to_hold = set()
        mom_weights = {}

        for i, (name, d, mom, above_trend) in enumerate(rankings[:n_top]):
            if mom > 0 and above_trend:
                to_hold.add(name)
                mom_weights[name] = max(mom, 0.01)

        # 3. Sell stocks no longer in top
        for d in self.datas:
            name = d._name
            pos = self.getposition(d)
            if pos.size > 0 and name not in to_hold and not self.orders[name]:
                self.orders[name] = self.close(data=d)
                self.stops[name] = 0

        if not to_hold:
            return

        # 4. Allocate capital weighted by momentum
        total_mom = sum(mom_weights.values())
        portfolio_value = self.broker.getvalue()

        for name, d, mom, above_trend in rankings:
            if name not in to_hold:
                continue
            if self.orders[name]:
                continue

            weight = mom_weights[name] / total_mom
            weight = min(weight, self.p.max_position_pct)

            target_value = portfolio_value * weight
            current_pos = self.getposition(d)
            current_value = current_pos.size * d.close[0] if current_pos.size > 0 else 0

            diff = target_value - current_value
            if abs(diff) < portfolio_value * 0.02:
                continue

            if diff > 0:
                shares_to_buy = int(diff / d.close[0])
                if shares_to_buy > 0:
                    self.orders[name] = self.buy(data=d, size=shares_to_buy)
                    if self.stops[name] == 0:
                        atr = self.indicators[name]["atr"][0]
                        self.stops[name] = d.close[0] - atr * self.p.atr_trail_mult
            elif diff < 0:
                shares_to_sell = min(int(-diff / d.close[0]), current_pos.size)
                if shares_to_sell > 0:
                    self.orders[name] = self.sell(data=d, size=shares_to_sell)


class RegimeAwareStrategy(bt.Strategy):
    """Market regime detection + momentum allocation.

    Uses SPY (first data feed) as regime indicator:
    - BULL:  SPY > 200 SMA and SPY > 50 SMA → invest 90-100%
    - CAUTION: SPY > 200 SMA but < 50 SMA → invest 50-60%
    - BEAR:  SPY < 200 SMA → invest only 20-30%

    Within the invested portion, uses momentum ranking to pick which stocks
    to hold and how much.

    IMPORTANT: First data feed must be SPY.
    """

    params = dict(
        sma_fast=50,
        sma_slow=200,
        momentum_period=63,
        rebalance_days=21,
        ema_trend=50,
        atr_period=14,
        atr_trail_mult=3.0,
        bull_exposure=0.95,
        caution_exposure=0.55,
        bear_exposure=0.25,
        max_position_pct=0.25,
    )

    def __init__(self):
        # SPY is first data feed for regime detection
        self.spy = self.datas[0]
        self.spy_sma_fast = bt.indicators.SMA(self.spy.close, period=self.p.sma_fast)
        self.spy_sma_slow = bt.indicators.SMA(self.spy.close, period=self.p.sma_slow)

        self.indicators = {}
        self.stops = {}
        self.orders = {}
        self.rebal_counter = 0

        # Stock data feeds start from index 1
        for d in self.datas[1:]:
            name = d._name
            self.indicators[name] = {
                "ema_trend": bt.indicators.EMA(d.close, period=self.p.ema_trend),
                "atr": bt.indicators.ATR(d, period=self.p.atr_period),
            }
            self.stops[name] = 0
            self.orders[name] = None

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            name = order.data._name
            if name in self.orders:
                self.orders[name] = None

    def _get_regime(self) -> str:
        if self.spy.close[0] > self.spy_sma_slow[0] and self.spy.close[0] > self.spy_sma_fast[0]:
            return "BULL"
        elif self.spy.close[0] > self.spy_sma_slow[0]:
            return "CAUTION"
        else:
            return "BEAR"

    def _get_target_exposure(self) -> float:
        regime = self._get_regime()
        if regime == "BULL":
            return self.p.bull_exposure
        elif regime == "CAUTION":
            return self.p.caution_exposure
        else:
            return self.p.bear_exposure

    def _get_momentum(self, d) -> float:
        if len(d) < self.p.momentum_period:
            return 0.0
        old = d.close[-self.p.momentum_period]
        return (d.close[0] - old) / old if old > 0 else 0.0

    def next(self):
        self.rebal_counter += 1

        stock_datas = self.datas[1:]

        # Trail stops
        for d in stock_datas:
            name = d._name
            pos = self.getposition(d)
            if pos.size > 0 and self.indicators[name]["atr"][0] > 0:
                new_stop = d.close[0] - self.indicators[name]["atr"][0] * self.p.atr_trail_mult
                if new_stop > self.stops[name]:
                    self.stops[name] = new_stop
                if d.close[0] <= self.stops[name] and not self.orders.get(name):
                    self.orders[name] = self.close(data=d)
                    self.stops[name] = 0

        if self.rebal_counter % self.p.rebalance_days != 0:
            return

        if any(self.orders.get(d._name) for d in stock_datas):
            return

        target_exposure = self._get_target_exposure()
        regime = self._get_regime()

        # Rank by momentum
        rankings = []
        for d in stock_datas:
            name = d._name
            mom = self._get_momentum(d)
            above = d.close[0] > self.indicators[name]["ema_trend"][0]
            rankings.append((name, d, mom, above))

        rankings.sort(key=lambda x: x[2], reverse=True)

        # Select top stocks with positive momentum and above trend
        to_hold = {}
        for name, d, mom, above in rankings:
            if mom > 0 and above:
                to_hold[name] = max(mom, 0.01)

        # Sell anything not in to_hold
        for d in stock_datas:
            name = d._name
            pos = self.getposition(d)
            if pos.size > 0 and name not in to_hold and not self.orders.get(name):
                self.orders[name] = self.close(data=d)
                self.stops[name] = 0

        if not to_hold:
            return

        # Allocate with momentum weighting, capped by regime exposure
        total_mom = sum(to_hold.values())
        portfolio_value = self.broker.getvalue()
        investable = portfolio_value * target_exposure

        for name, weight_mom in to_hold.items():
            d = None
            for dd in stock_datas:
                if dd._name == name:
                    d = dd
                    break
            if d is None or self.orders.get(name):
                continue

            weight = weight_mom / total_mom
            weight = min(weight, self.p.max_position_pct)
            target_value = investable * weight

            pos = self.getposition(d)
            current_value = pos.size * d.close[0] if pos.size > 0 else 0
            diff = target_value - current_value

            if abs(diff) < portfolio_value * 0.02:
                continue

            if diff > 0:
                shares = int(diff / d.close[0])
                if shares > 0:
                    self.orders[name] = self.buy(data=d, size=shares)
                    if self.stops[name] == 0:
                        atr = self.indicators[name]["atr"][0]
                        self.stops[name] = d.close[0] - atr * self.p.atr_trail_mult
            elif diff < 0:
                shares = min(int(-diff / d.close[0]), pos.size)
                if shares > 0:
                    self.orders[name] = self.sell(data=d, size=shares)


class TrendPlusMomentumStrategy(bt.Strategy):
    """The most aggressive approach: pure trend following + momentum, stay fully invested.

    Philosophy: In the long run, equity markets go up. The goal is to be
    invested as much as possible in the strongest stocks, and only reduce
    exposure when the overall market shows clear weakness.

    Rules:
    - Rank stocks by 3-month momentum every 2 weeks
    - Buy top 60-70% of stocks (those with positive momentum + uptrend)
    - Weight by momentum: strongest stocks get 2-3x more capital
    - Very wide trailing stops (3.5x ATR) — don't get shaken out
    - Only go defensive when a stock breaks below its 200-day SMA
    """

    params = dict(
        momentum_period=63,
        rebalance_days=14,
        sma_long=200,
        ema_trend=50,
        atr_period=14,
        atr_trail_mult=3.5,
        top_pct=0.70,
        max_position_pct=0.25,
    )

    def __init__(self):
        self.indicators = {}
        self.stops = {}
        self.orders = {}
        self.rebal_counter = 0

        for d in self.datas:
            name = d._name
            self.indicators[name] = {
                "sma_long": bt.indicators.SMA(d.close, period=self.p.sma_long),
                "ema_trend": bt.indicators.EMA(d.close, period=self.p.ema_trend),
                "atr": bt.indicators.ATR(d, period=self.p.atr_period),
            }
            self.stops[name] = 0
            self.orders[name] = None

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.orders[order.data._name] = None

    def _momentum(self, d) -> float:
        if len(d) < self.p.momentum_period:
            return 0.0
        old = d.close[-self.p.momentum_period]
        return (d.close[0] - old) / old if old > 0 else 0.0

    def next(self):
        self.rebal_counter += 1

        for d in self.datas:
            name = d._name
            pos = self.getposition(d)
            if pos.size > 0 and self.indicators[name]["atr"][0] > 0:
                new_stop = d.close[0] - self.indicators[name]["atr"][0] * self.p.atr_trail_mult
                if new_stop > self.stops[name]:
                    self.stops[name] = new_stop
                if d.close[0] <= self.stops[name] and not self.orders[name]:
                    self.orders[name] = self.close(data=d)
                    self.stops[name] = 0

        if self.rebal_counter % self.p.rebalance_days != 0:
            return

        if any(self.orders[d._name] for d in self.datas):
            return

        rankings = []
        for d in self.datas:
            name = d._name
            mom = self._momentum(d)
            above_long = d.close[0] > self.indicators[name]["sma_long"][0]
            above_trend = d.close[0] > self.indicators[name]["ema_trend"][0]
            rankings.append((name, d, mom, above_long, above_trend))

        rankings.sort(key=lambda x: x[2], reverse=True)

        n_top = max(1, int(len(rankings) * self.p.top_pct))
        to_hold = {}
        for name, d, mom, above_long, above_trend in rankings[:n_top]:
            if mom > 0 and above_long:
                to_hold[name] = mom

        for d in self.datas:
            name = d._name
            pos = self.getposition(d)
            if pos.size > 0 and name not in to_hold and not self.orders[name]:
                self.orders[name] = self.close(data=d)
                self.stops[name] = 0

        if not to_hold:
            return

        total_mom = sum(to_hold.values())
        pv = self.broker.getvalue()

        for name, mom in to_hold.items():
            d = None
            for dd in self.datas:
                if dd._name == name:
                    d = dd
                    break
            if d is None or self.orders.get(name):
                continue

            weight = min(mom / total_mom, self.p.max_position_pct)
            target = pv * weight * 0.95  # leave 5% cash buffer

            pos = self.getposition(d)
            current = pos.size * d.close[0] if pos.size > 0 else 0
            diff = target - current

            if abs(diff) < pv * 0.02:
                continue

            if diff > 0:
                shares = int(diff / d.close[0])
                if shares > 0:
                    self.orders[name] = self.buy(data=d, size=shares)
                    if self.stops[name] == 0:
                        atr = self.indicators[name]["atr"][0]
                        self.stops[name] = d.close[0] - atr * self.p.atr_trail_mult
            elif diff < 0:
                shares = min(int(-diff / d.close[0]), pos.size)
                if shares > 0:
                    self.orders[name] = self.sell(data=d, size=shares)
