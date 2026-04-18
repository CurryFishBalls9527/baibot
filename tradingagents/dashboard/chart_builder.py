"""Plotly-based interactive chart builder for trade review.

Fluent API for composing candlestick charts with overlays:
    TradeChartBuilder("NVDA", df)
        .add_candlesticks()
        .add_volume()
        .add_sma([50, 150, 200])
        .add_trade_markers(trades)
        .add_macd()
        .build()
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_SMA_COLORS = {
    10: "#ff9800", 20: "#2196f3", 50: "#e91e63",
    150: "#9c27b0", 200: "#4caf50",
}


class TradeChartBuilder:
    def __init__(self, symbol: str, df: pd.DataFrame):
        self.symbol = symbol
        self.df = df.copy()
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if "date" in self.df.columns:
                self.df.index = pd.to_datetime(self.df["date"])
            elif "ts" in self.df.columns:
                self.df.index = pd.to_datetime(self.df["ts"])
        self._traces_main: list[go.BaseTraceType] = []
        self._traces_volume: list[go.BaseTraceType] = []
        self._subplots: list[tuple[str, list[go.BaseTraceType]]] = []
        self._shapes: list[dict] = []
        self._annotations: list[dict] = []
        self._has_volume = False

    def add_candlesticks(self) -> Self:
        self._traces_main.append(go.Candlestick(
            x=self.df.index,
            open=self.df["open"], high=self.df["high"],
            low=self.df["low"], close=self.df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
        ))
        return self

    def add_volume(self) -> Self:
        if "volume" not in self.df.columns:
            return self
        colors = [
            "#26a69a" if c >= o else "#ef5350"
            for c, o in zip(self.df["close"], self.df["open"])
        ]
        self._traces_volume.append(go.Bar(
            x=self.df.index, y=self.df["volume"],
            marker_color=colors, opacity=0.5,
            name="Volume", showlegend=False,
        ))
        self._has_volume = True
        return self

    def add_sma(self, periods: list[int]) -> Self:
        for p in periods:
            col = self.df["close"].rolling(p).mean()
            color = _SMA_COLORS.get(p, "#888")
            self._traces_main.append(go.Scatter(
                x=self.df.index, y=col,
                mode="lines", name=f"SMA {p}",
                line=dict(width=1.2, color=color),
            ))
        return self

    def add_bollinger(self, period: int = 20, std: float = 2.0) -> Self:
        mid = self.df["close"].rolling(period).mean()
        s = self.df["close"].rolling(period).std()
        upper = mid + std * s
        lower = mid - std * s
        self._traces_main.append(go.Scatter(
            x=self.df.index, y=upper, mode="lines",
            line=dict(width=0.8, color="rgba(100,100,255,0.4)"),
            name="BB Upper", showlegend=False,
        ))
        self._traces_main.append(go.Scatter(
            x=self.df.index, y=lower, mode="lines",
            line=dict(width=0.8, color="rgba(100,100,255,0.4)"),
            fill="tonexty", fillcolor="rgba(100,100,255,0.06)",
            name="Bollinger", showlegend=True,
        ))
        return self

    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Self:
        c = self.df["close"]
        ema_fast = c.ewm(span=fast).mean()
        ema_slow = c.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in histogram]
        traces = [
            go.Bar(x=self.df.index, y=histogram, marker_color=colors,
                   name="MACD Hist", showlegend=False),
            go.Scatter(x=self.df.index, y=macd_line, mode="lines",
                       line=dict(width=1.2, color="#2196f3"), name="MACD"),
            go.Scatter(x=self.df.index, y=signal_line, mode="lines",
                       line=dict(width=1, color="#ff9800"), name="Signal"),
        ]
        self._subplots.append(("MACD", traces))
        return self

    def add_rsi(self, period: int = 14) -> Self:
        delta = self.df["close"].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        traces = [
            go.Scatter(x=self.df.index, y=rsi, mode="lines",
                       line=dict(width=1.2, color="#9c27b0"), name="RSI"),
        ]
        self._subplots.append(("RSI", traces))
        return self

    def add_atr(self, period: int = 14) -> Self:
        h, l, c = self.df["high"], self.df["low"], self.df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([
            h - l, (h - prev_c).abs(), (l - prev_c).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        traces = [
            go.Scatter(x=self.df.index, y=atr, mode="lines",
                       line=dict(width=1.2, color="#ff5722"), name="ATR"),
        ]
        self._subplots.append(("ATR", traces))
        return self

    def add_trade_markers(self, trades: list[dict]) -> Self:
        for t in trades:
            is_buy = t.get("side", "").lower() == "buy"
            ts = pd.Timestamp(t.get("timestamp") or t.get("time"))
            price = t.get("filled_price") or t.get("price", 0)
            symbol_marker = "triangle-up" if is_buy else "triangle-down"
            color = "#2196f3" if is_buy else "#ff5722"
            label = t.get("side", "").upper()
            pl = t.get("pl")
            qty = t.get("qty") or t.get("filled_qty", "")
            reasoning = t.get("reasoning", "")

            hover = f"<b>{label} {self.symbol}</b><br>"
            hover += f"Price: ${price:.2f}<br>"
            if qty:
                hover += f"Qty: {qty}<br>"
            if pl is not None:
                hover += f"P&L: ${pl:+,.2f}<br>"
            if reasoning:
                hover += f"Reason: {reasoning[:80]}"

            self._traces_main.append(go.Scatter(
                x=[ts], y=[price],
                mode="markers+text",
                marker=dict(symbol=symbol_marker, size=14, color=color,
                            line=dict(width=1, color="white")),
                text=[label], textposition="top center" if is_buy else "bottom center",
                textfont=dict(size=10, color=color),
                hovertext=[hover], hoverinfo="text",
                name=f"{label} @ ${price:.2f}",
                showlegend=False,
            ))
        return self

    def add_minervini_levels(self, levels: dict) -> Self:
        for name, val, color, dash in [
            ("Buy Point", levels.get("buy_point"), "#4caf50", "dash"),
            ("Buy Limit", levels.get("buy_limit"), "#4caf50", "dot"),
            ("Stop Loss", levels.get("stop_loss"), "#f44336", "dash"),
        ]:
            if val:
                self._shapes.append(dict(
                    type="line", y0=val, y1=val,
                    x0=self.df.index[0], x1=self.df.index[-1],
                    line=dict(color=color, width=1, dash=dash),
                ))
                self._annotations.append(dict(
                    x=self.df.index[-1], y=val,
                    text=f"{name}: ${val:.2f}",
                    showarrow=False, xanchor="left",
                    font=dict(size=9, color=color),
                ))
        return self

    def add_chan_bi(self, bi_list: list[dict]) -> Self:
        for bi in bi_list:
            color = "#e91e63" if bi.get("dir") == "up" else "#4caf50"
            self._traces_main.append(go.Scatter(
                x=[bi["start_time"], bi["end_time"]],
                y=[bi["start_val"], bi["end_val"]],
                mode="lines+markers",
                line=dict(width=1.5, color=color),
                marker=dict(size=4, color=color),
                showlegend=False, hoverinfo="skip",
            ))
        return self

    def add_chan_seg(self, seg_list: list[dict]) -> Self:
        for seg in seg_list:
            color = "#d32f2f" if seg.get("dir") == "up" else "#1b5e20"
            self._traces_main.append(go.Scatter(
                x=[seg["start_time"], seg["end_time"]],
                y=[seg["start_val"], seg["end_val"]],
                mode="lines",
                line=dict(width=3, color=color, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))
        return self

    def add_chan_zs(self, zs_list: list[dict]) -> Self:
        for zs in zs_list:
            self._shapes.append(dict(
                type="rect",
                x0=zs["begin_time"], x1=zs["end_time"],
                y0=zs["low"], y1=zs["high"],
                fillcolor="rgba(255, 193, 7, 0.15)",
                line=dict(color="rgba(255, 193, 7, 0.6)", width=1),
            ))
        return self

    def add_chan_bsp(self, bsp_list: list[dict]) -> Self:
        for bsp in bsp_list:
            is_buy = bsp.get("is_buy", True)
            color = "#2196f3" if is_buy else "#ff5722"
            symbol = "triangle-up" if is_buy else "triangle-down"
            types_str = bsp.get("types", "")
            self._traces_main.append(go.Scatter(
                x=[bsp["time"]], y=[bsp["price"]],
                mode="markers",
                marker=dict(symbol=symbol, size=12, color=color,
                            line=dict(width=1.5, color="white")),
                hovertext=[f"{'Buy' if is_buy else 'Sell'} BSP: {types_str}"],
                hoverinfo="text",
                name=f"BSP {types_str}",
                showlegend=False,
            ))
        return self

    def build(self) -> go.Figure:
        n_subplots = len(self._subplots)
        has_vol = self._has_volume
        total_rows = 1 + (1 if has_vol else 0) + n_subplots

        row_heights = [0.5]
        subplot_titles = [self.symbol]
        if has_vol:
            row_heights.append(0.1)
            subplot_titles.append("Volume")
        for name, _ in self._subplots:
            row_heights.append(0.15)
            subplot_titles.append(name)

        total = sum(row_heights)
        row_heights = [h / total for h in row_heights]

        fig = make_subplots(
            rows=total_rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=row_heights,
            subplot_titles=subplot_titles,
        )

        for trace in self._traces_main:
            fig.add_trace(trace, row=1, col=1)

        current_row = 2
        if has_vol:
            for trace in self._traces_volume:
                fig.add_trace(trace, row=current_row, col=1)
            current_row += 1

        for _, traces in self._subplots:
            for trace in traces:
                fig.add_trace(trace, row=current_row, col=1)
            current_row += 1

        for shape in self._shapes:
            fig.add_shape(shape, row=1, col=1)
        for ann in self._annotations:
            fig.add_annotation(ann, row=1, col=1)

        fig.update_layout(
            height=250 + 150 * total_rows,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=60, r=20, t=60, b=30),
            hovermode="x unified",
        )

        fig.update_xaxes(type="date")
        fig.update_yaxes(title_text="Price", row=1, col=1)

        return fig
