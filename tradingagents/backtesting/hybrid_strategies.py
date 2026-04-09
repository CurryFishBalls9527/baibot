"""Hybrid strategies: simulate the optimized approach where LLM runs periodically
and generates conditional signals, while daily execution is pure rule-based.

Two strategies to compare:

1. PreScreenStrategy — Multi-indicator scoring (like CombinedStrategy) but with
   a pre-screening layer that skips trading on low-signal days. Simulates
   "only call LLM when indicators suggest something interesting."

2. WeeklySignalDailyExecStrategy — Simulates the proposed optimization:
   - Every N days (simulating weekly LLM analysis): evaluate all indicators
     and set a "conditional order" (target price + direction)
   - Every day: check if the conditional order should be triggered based on
     real-time price action — NO LLM call needed.
   This represents the pattern: "LLM says BUY NVDA if < $150" and then
   Alpaca real-time data handles execution.
"""

import backtrader as bt


class PreScreenStrategy(bt.Strategy):
    """Only trade when technical score is strong enough (simulates pre-screening).

    On days where the score is weak (|score| <= skip_threshold), do nothing.
    This simulates skipping the LLM call on quiet days.

    The key insight: most days, indicators are ambiguous. Only act on clear signals.
    Tracks "llm_calls_saved" to measure API savings.
    """

    params = dict(
        ema_fast=9,
        ema_slow=21,
        ema_trend=50,
        rsi_period=14,
        bb_period=20,
        bb_dev=2.0,
        atr_period=14,
        vol_avg_period=20,
        buy_threshold=5,       # Score >= 5 to buy (out of 6)
        sell_threshold=3,      # Sell score >= 3 to sell (out of 5)
        skip_threshold=2,      # |score| <= 2: skip (no trade, no LLM)
        atr_sl_mult=1.5,
        atr_tp_mult=2.5,
        position_pct=0.15,
    )

    def __init__(self):
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow)
        self.ema_trend = bt.indicators.EMA(self.data.close, period=self.p.ema_trend)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.boll = bt.indicators.BollingerBands(
            self.data.close, period=self.p.bb_period, devfactor=self.p.bb_dev
        )
        self.macd = bt.indicators.MACD(self.data.close)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.vol_avg = bt.indicators.SMA(self.data.volume, period=self.p.vol_avg_period)
        self.order = None
        self.stop_price = 0
        self.target_price = 0
        # Track savings
        self.days_screened = 0
        self.days_skipped = 0

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def _buy_score(self) -> int:
        score = 0
        if self.ema_fast[0] > self.ema_slow[0]:
            score += 1
        if self.data.close[0] > self.ema_trend[0]:
            score += 1
        if 35 < self.rsi[0] < 65:
            score += 1
        if self.macd.macd[0] > self.macd.signal[0]:
            score += 1
        if self.boll.bot[0] < self.data.close[0] < self.boll.top[0]:
            score += 1
        if self.data.volume[0] > self.vol_avg[0]:
            score += 1
        return score

    def _sell_score(self) -> int:
        score = 0
        if self.ema_fast[0] < self.ema_slow[0]:
            score += 1
        if self.rsi[0] > 72:
            score += 1
        if self.macd.macd[0] < self.macd.signal[0]:
            score += 1
        if self.data.close[0] > self.boll.top[0]:
            score += 1
        if self.data.close[0] < self.ema_trend[0]:
            score += 1
        return score

    def next(self):
        if self.order:
            return

        self.days_screened += 1
        buy_s = self._buy_score()
        sell_s = self._sell_score()

        if not self.position:
            # Pre-screen: skip if signal not strong enough
            if buy_s < self.p.buy_threshold:
                if buy_s <= self.p.skip_threshold:
                    self.days_skipped += 1  # Would have skipped LLM call
                return

            if self.atr[0] > 0:
                size = int(self.broker.getvalue() * self.p.position_pct / self.data.close[0])
                if size > 0:
                    self.stop_price = self.data.close[0] - self.atr[0] * self.p.atr_sl_mult
                    self.target_price = self.data.close[0] + self.atr[0] * self.p.atr_tp_mult
                    self.order = self.buy(size=size)
        else:
            # Trail stop
            new_stop = self.data.close[0] - self.atr[0] * self.p.atr_sl_mult
            if new_stop > self.stop_price:
                self.stop_price = new_stop

            if (
                self.data.close[0] <= self.stop_price
                or self.data.close[0] >= self.target_price
                or sell_s >= self.p.sell_threshold
            ):
                self.order = self.close()

    def stop(self):
        """Called at end of backtest — log pre-screening stats."""
        if self.days_screened > 0:
            pct = self.days_skipped / self.days_screened * 100
            self.llm_calls_saved_pct = pct


class WeeklySignalDailyExecStrategy(bt.Strategy):
    """Simulates: LLM analyzes weekly, sets conditional orders; daily execution
    is pure rule-based using price data (no LLM needed).

    Every `signal_interval` days (simulating weekly LLM run):
    - Compute a composite signal score
    - If bullish: set a conditional buy at "current price - dip%"
      (the LLM might say "buy NVDA if it dips to $X")
    - If bearish on held position: mark for exit on next weakness

    Every day between signal refreshes:
    - Check if price hits the conditional order level → execute
    - Monitor trailing stops → execute
    - NO LLM call needed

    This models the real proposed architecture:
    "LLM runs once/week → produces target prices → Alpaca monitors real-time"
    """

    params = dict(
        signal_interval=5,     # Re-evaluate every 5 trading days (~ weekly)
        ema_fast=9,
        ema_slow=21,
        ema_trend=50,
        rsi_period=14,
        bb_period=20,
        bb_dev=2.0,
        atr_period=14,
        vol_avg_period=20,
        buy_threshold=4,       # Composite score to generate buy signal
        dip_pct=0.02,          # Buy on 2% dip from signal price
        atr_sl_mult=1.5,
        atr_tp_mult=3.0,      # Wider target since we hold longer
        position_pct=0.15,
    )

    def __init__(self):
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow)
        self.ema_trend = bt.indicators.EMA(self.data.close, period=self.p.ema_trend)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.boll = bt.indicators.BollingerBands(
            self.data.close, period=self.p.bb_period, devfactor=self.p.bb_dev
        )
        self.macd = bt.indicators.MACD(self.data.close)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.vol_avg = bt.indicators.SMA(self.data.volume, period=self.p.vol_avg_period)

        self.order = None
        self.bar_count = 0

        # Conditional order state (set by "LLM" on signal day)
        self.pending_buy_price = 0       # Buy if price drops to this level
        self.pending_buy_valid = False
        self.pending_sell = False         # Sell on next weakness

        # Active position management
        self.stop_price = 0
        self.target_price = 0

        # Stats
        self.signal_days = 0             # Days where "LLM" ran
        self.execution_days = 0          # Days where an order actually fired
        self.total_days = 0

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def _composite_score(self) -> int:
        """Same scoring as CombinedStrategy."""
        score = 0
        if self.ema_fast[0] > self.ema_slow[0]:
            score += 1
        if self.data.close[0] > self.ema_trend[0]:
            score += 1
        if 35 < self.rsi[0] < 70:
            score += 1
        if self.macd.macd[0] > self.macd.signal[0]:
            score += 1
        if self.boll.bot[0] < self.data.close[0] < self.boll.top[0]:
            score += 1
        if self.data.volume[0] > self.vol_avg[0]:
            score += 1
        return score

    def _bearish_score(self) -> int:
        score = 0
        if self.ema_fast[0] < self.ema_slow[0]:
            score += 1
        if self.rsi[0] > 72:
            score += 1
        if self.macd.macd[0] < self.macd.signal[0]:
            score += 1
        if self.data.close[0] > self.boll.top[0]:
            score += 1
        return score

    def next(self):
        if self.order:
            return

        self.bar_count += 1
        self.total_days += 1
        is_signal_day = (self.bar_count % self.p.signal_interval == 0)

        # ── Signal Day: "LLM" runs and sets conditional orders ──
        if is_signal_day:
            self.signal_days += 1

            if not self.position:
                score = self._composite_score()
                if score >= self.p.buy_threshold and self.atr[0] > 0:
                    # Set conditional buy: buy if price dips to this level
                    self.pending_buy_price = self.data.close[0] * (1 - self.p.dip_pct)
                    self.pending_buy_valid = True
                else:
                    self.pending_buy_valid = False
            else:
                # Check if we should prepare to sell
                bear_score = self._bearish_score()
                if bear_score >= 3:
                    self.pending_sell = True
                else:
                    self.pending_sell = False
                    # Update targets on signal day
                    if self.atr[0] > 0:
                        new_target = self.data.close[0] + self.atr[0] * self.p.atr_tp_mult
                        if new_target > self.target_price:
                            self.target_price = new_target

        # ── Every Day: Pure rule-based execution (no LLM) ──

        if not self.position:
            # Check if conditional buy triggers
            if self.pending_buy_valid and self.data.close[0] <= self.pending_buy_price:
                size = int(
                    self.broker.getvalue() * self.p.position_pct / self.data.close[0]
                )
                if size > 0 and self.atr[0] > 0:
                    self.stop_price = self.data.close[0] - self.atr[0] * self.p.atr_sl_mult
                    self.target_price = self.data.close[0] + self.atr[0] * self.p.atr_tp_mult
                    self.order = self.buy(size=size)
                    self.pending_buy_valid = False
                    self.execution_days += 1
        else:
            # Trail stop
            if self.atr[0] > 0:
                new_stop = self.data.close[0] - self.atr[0] * self.p.atr_sl_mult
                if new_stop > self.stop_price:
                    self.stop_price = new_stop

            # Exit conditions (all rule-based, no LLM)
            hit_stop = self.data.close[0] <= self.stop_price
            hit_target = self.data.close[0] >= self.target_price
            pending_sell_weakness = (
                self.pending_sell
                and self.ema_fast[0] < self.ema_slow[0]
            )

            if hit_stop or hit_target or pending_sell_weakness:
                self.order = self.close()
                self.pending_sell = False
                self.execution_days += 1

    def stop(self):
        """Log efficiency stats at end of backtest."""
        if self.total_days > 0:
            self.llm_call_ratio = self.signal_days / self.total_days * 100


# ── Portfolio versions for multi-stock backtesting ──────────────────

class PortfolioPreScreenStrategy(bt.Strategy):
    """Portfolio version of PreScreenStrategy — multi-stock with pre-screening."""

    params = dict(
        ema_fast=9,
        ema_slow=21,
        ema_trend=50,
        rsi_period=14,
        bb_period=20,
        bb_dev=2.0,
        atr_period=14,
        vol_avg_period=20,
        buy_threshold=5,
        sell_threshold=3,
        skip_threshold=2,
        atr_sl_mult=1.5,
        atr_tp_mult=2.5,
    )

    def __init__(self):
        self.indicators = {}
        self.stops = {}
        self.orders = {}
        self.days_screened = 0
        self.days_skipped = 0

        for d in self.datas:
            name = d._name
            self.indicators[name] = {
                "ema_fast": bt.indicators.EMA(d.close, period=self.p.ema_fast),
                "ema_slow": bt.indicators.EMA(d.close, period=self.p.ema_slow),
                "ema_trend": bt.indicators.EMA(d.close, period=self.p.ema_trend),
                "rsi": bt.indicators.RSI(d.close, period=self.p.rsi_period),
                "boll": bt.indicators.BollingerBands(
                    d.close, period=self.p.bb_period, devfactor=self.p.bb_dev
                ),
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
        if 35 < ind["rsi"][0] < 65:
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
        if ind["rsi"][0] > 72:
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
        self.days_screened += 1

        skipped_all = True
        for d in self.datas:
            name = d._name
            ind = self.indicators[name]

            if self.orders[name]:
                continue

            pos = self.getposition(d)

            if not pos.size:
                buy_s = self._buy_score(d, ind)
                if buy_s <= self.p.skip_threshold:
                    continue  # Pre-screened out
                skipped_all = False
                if buy_s >= self.p.buy_threshold and ind["atr"][0] > 0:
                    size = int(alloc / d.close[0])
                    if size > 0:
                        self.stops[name] = d.close[0] - ind["atr"][0] * self.p.atr_sl_mult
                        self.orders[name] = self.buy(data=d, size=size)
            else:
                skipped_all = False
                new_stop = d.close[0] - ind["atr"][0] * self.p.atr_sl_mult
                if new_stop > self.stops[name]:
                    self.stops[name] = new_stop

                if (
                    d.close[0] <= self.stops[name]
                    or self._sell_score(d, ind) >= self.p.sell_threshold
                ):
                    self.orders[name] = self.close(data=d)
                    self.stops[name] = 0

        if skipped_all:
            self.days_skipped += 1


class PortfolioWeeklySignalStrategy(bt.Strategy):
    """Portfolio version: weekly signal generation + daily rule-based execution."""

    params = dict(
        signal_interval=5,
        ema_fast=9,
        ema_slow=21,
        ema_trend=50,
        rsi_period=14,
        bb_period=20,
        bb_dev=2.0,
        atr_period=14,
        vol_avg_period=20,
        buy_threshold=4,
        dip_pct=0.02,
        atr_sl_mult=1.5,
        atr_tp_mult=3.0,
    )

    def __init__(self):
        self.indicators = {}
        self.stops = {}
        self.targets = {}
        self.orders = {}
        self.pending_buys = {}
        self.pending_sells = {}
        self.bar_count = 0
        self.signal_days = 0

        for d in self.datas:
            name = d._name
            self.indicators[name] = {
                "ema_fast": bt.indicators.EMA(d.close, period=self.p.ema_fast),
                "ema_slow": bt.indicators.EMA(d.close, period=self.p.ema_slow),
                "ema_trend": bt.indicators.EMA(d.close, period=self.p.ema_trend),
                "rsi": bt.indicators.RSI(d.close, period=self.p.rsi_period),
                "boll": bt.indicators.BollingerBands(
                    d.close, period=self.p.bb_period, devfactor=self.p.bb_dev
                ),
                "macd": bt.indicators.MACD(d.close),
                "atr": bt.indicators.ATR(d, period=self.p.atr_period),
                "vol_avg": bt.indicators.SMA(d.volume, period=self.p.vol_avg_period),
            }
            self.stops[name] = 0
            self.targets[name] = 0
            self.orders[name] = None
            self.pending_buys[name] = 0
            self.pending_sells[name] = False

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.orders[order.data._name] = None

    def _score(self, d, ind) -> int:
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

    def _bear_score(self, d, ind) -> int:
        score = 0
        if ind["ema_fast"][0] < ind["ema_slow"][0]:
            score += 1
        if ind["rsi"][0] > 72:
            score += 1
        if ind["macd"].macd[0] < ind["macd"].signal[0]:
            score += 1
        if d.close[0] > ind["boll"].top[0]:
            score += 1
        return score

    def next(self):
        self.bar_count += 1
        n_stocks = len(self.datas)
        alloc = self.broker.getvalue() / n_stocks
        is_signal_day = (self.bar_count % self.p.signal_interval == 0)

        if is_signal_day:
            self.signal_days += 1

        for d in self.datas:
            name = d._name
            ind = self.indicators[name]

            if self.orders[name]:
                continue

            pos = self.getposition(d)

            # Signal day: update conditional orders
            if is_signal_day:
                if not pos.size:
                    score = self._score(d, ind)
                    if score >= self.p.buy_threshold and ind["atr"][0] > 0:
                        self.pending_buys[name] = d.close[0] * (1 - self.p.dip_pct)
                    else:
                        self.pending_buys[name] = 0
                else:
                    if self._bear_score(d, ind) >= 3:
                        self.pending_sells[name] = True
                    else:
                        self.pending_sells[name] = False

            # Daily execution (pure rule-based)
            if not pos.size:
                if (
                    self.pending_buys[name] > 0
                    and d.close[0] <= self.pending_buys[name]
                    and ind["atr"][0] > 0
                ):
                    size = int(alloc / d.close[0])
                    if size > 0:
                        self.stops[name] = d.close[0] - ind["atr"][0] * self.p.atr_sl_mult
                        self.targets[name] = d.close[0] + ind["atr"][0] * self.p.atr_tp_mult
                        self.orders[name] = self.buy(data=d, size=size)
                        self.pending_buys[name] = 0
            else:
                # Trail stop
                if ind["atr"][0] > 0:
                    new_stop = d.close[0] - ind["atr"][0] * self.p.atr_sl_mult
                    if new_stop > self.stops[name]:
                        self.stops[name] = new_stop

                hit_stop = d.close[0] <= self.stops[name]
                hit_target = d.close[0] >= self.targets[name]
                pending_weakness = (
                    self.pending_sells[name]
                    and ind["ema_fast"][0] < ind["ema_slow"][0]
                )

                if hit_stop or hit_target or pending_weakness:
                    self.orders[name] = self.close(data=d)
                    self.stops[name] = 0
                    self.pending_sells[name] = False
