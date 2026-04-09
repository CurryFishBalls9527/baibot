"""Trading strategies for backtesting using best-practice technical indicators.

Three strategies:
1. SwingStrategy — multi-day holds, EMA crossover + trend + momentum
2. DayTradingStrategy — short holds, mean reversion + RSI + Bollinger
3. CombinedStrategy — scoring system, highest confirmation for best win rate
"""

import backtrader as bt


class SwingStrategy(bt.Strategy):
    """Swing trading: EMA crossover + RSI filter + MACD confirmation + ATR stops.

    Entry: EMA fast > EMA slow AND RSI in range AND MACD bullish AND above trend.
    Exit: ATR-based SL/TP or bearish crossover.
    Re-enters after each exit when conditions align again.
    """

    params = dict(
        ema_fast=9,
        ema_slow=21,
        ema_trend=50,
        rsi_period=14,
        rsi_low=35,
        rsi_high=75,
        atr_period=14,
        atr_sl_mult=1.5,
        atr_tp_mult=2.5,
        position_pct=0.15,
    )

    def __init__(self):
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow)
        self.ema_trend = bt.indicators.EMA(self.data.close, period=self.p.ema_trend)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.macd = bt.indicators.MACD(self.data.close)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.crossover = bt.indicators.CrossOver(self.ema_fast, self.ema_slow)
        self.order = None
        self.stop_price = 0
        self.target_price = 0

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            bullish = (
                self.ema_fast[0] > self.ema_slow[0]
                and self.p.rsi_low < self.rsi[0] < self.p.rsi_high
                and self.macd.macd[0] > self.macd.signal[0]
                and self.data.close[0] > self.ema_trend[0]
                and self.atr[0] > 0
            )
            if bullish:
                size = int(self.broker.getvalue() * self.p.position_pct / self.data.close[0])
                if size > 0:
                    self.stop_price = self.data.close[0] - self.atr[0] * self.p.atr_sl_mult
                    self.target_price = self.data.close[0] + self.atr[0] * self.p.atr_tp_mult
                    self.order = self.buy(size=size)
        else:
            # Trailing stop: move stop up as price rises
            new_stop = self.data.close[0] - self.atr[0] * self.p.atr_sl_mult
            if new_stop > self.stop_price:
                self.stop_price = new_stop

            if (
                self.data.close[0] <= self.stop_price
                or self.data.close[0] >= self.target_price
                or self.crossover < 0
            ):
                self.order = self.close()


class DayTradingStrategy(bt.Strategy):
    """Mean reversion: RSI oversold/overbought + Bollinger Band touch + volume surge.

    Relaxed thresholds for daily bars — RSI 40 / 65 instead of 30 / 70.
    Short hold period targeting quick 2-4% moves.
    """

    params = dict(
        rsi_period=14,
        rsi_buy=40,
        rsi_sell=65,
        bb_period=20,
        bb_dev=2.0,
        vol_avg_period=20,
        vol_multiplier=1.2,
        stop_loss_pct=0.03,
        take_profit_pct=0.05,
        max_hold_days=10,
        position_pct=0.15,
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.boll = bt.indicators.BollingerBands(
            self.data.close, period=self.p.bb_period, devfactor=self.p.bb_dev
        )
        self.vol_avg = bt.indicators.SMA(self.data.volume, period=self.p.vol_avg_period)
        self.macd = bt.indicators.MACD(self.data.close)
        self.order = None
        self.entry_price = 0
        self.entry_bar = 0
        self.stop_price = 0
        self.target_price = 0

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # Buy on oversold + near lower band + volume spike
            near_lower = self.data.close[0] <= self.boll.mid[0]
            oversold = self.rsi[0] < self.p.rsi_buy
            vol_ok = self.data.volume[0] > self.vol_avg[0] * self.p.vol_multiplier
            turning_up = self.macd.macd[0] > self.macd.macd[-1]

            if oversold and near_lower and (vol_ok or turning_up):
                size = int(self.broker.getvalue() * self.p.position_pct / self.data.close[0])
                if size > 0:
                    self.entry_price = self.data.close[0]
                    self.entry_bar = len(self)
                    self.stop_price = self.entry_price * (1 - self.p.stop_loss_pct)
                    self.target_price = self.entry_price * (1 + self.p.take_profit_pct)
                    self.order = self.buy(size=size)
        else:
            bars_held = len(self) - self.entry_bar
            if (
                self.data.close[0] <= self.stop_price
                or self.data.close[0] >= self.target_price
                or self.rsi[0] > self.p.rsi_sell
                or bars_held >= self.p.max_hold_days
            ):
                self.order = self.close()


class CombinedStrategy(bt.Strategy):
    """Multi-indicator scoring: buy when >= threshold indicators agree.

    Scores:
    1. EMA 9 > EMA 21
    2. Price > EMA 50
    3. RSI 40-70 (trending, not extreme)
    4. MACD bullish crossover
    5. Price in middle of Bollinger Bands (not overbought)
    6. Volume above average

    Sell when >= sell_threshold bearish signals, or ATR stop/target hit.
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
        buy_threshold=4,
        sell_threshold=3,
        atr_sl_mult=1.5,
        atr_tp_mult=3.0,
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

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def _buy_score(self) -> int:
        score = 0
        if self.ema_fast[0] > self.ema_slow[0]:
            score += 1
        if self.data.close[0] > self.ema_trend[0]:
            score += 1
        if 40 < self.rsi[0] < 70:
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
        return score

    def next(self):
        if self.order:
            return

        if not self.position:
            if self._buy_score() >= self.p.buy_threshold and self.atr[0] > 0:
                size = int(self.broker.getvalue() * self.p.position_pct / self.data.close[0])
                if size > 0:
                    self.stop_price = self.data.close[0] - self.atr[0] * self.p.atr_sl_mult
                    self.target_price = self.data.close[0] + self.atr[0] * self.p.atr_tp_mult
                    self.order = self.buy(size=size)
        else:
            new_stop = self.data.close[0] - self.atr[0] * self.p.atr_sl_mult
            if new_stop > self.stop_price:
                self.stop_price = new_stop

            if (
                self.data.close[0] <= self.stop_price
                or self.data.close[0] >= self.target_price
                or self._sell_score() >= self.p.sell_threshold
            ):
                self.order = self.close()
