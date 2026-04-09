"""Dynamic broad-market coarse scanner for Minervini-style screening."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.requests import StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from .minervini import MinerviniConfig, MinerviniScreener

logger = logging.getLogger(__name__)

_COMMON_FUND_KEYWORDS = (
    " ETF",
    " ETN",
    " FUND",
    " TRUST",
    "ISHARES",
    "SPDR",
    "PROSHARES",
    "INVESCO",
    "VANGUARD",
    "DIREXION",
    "GLOBAL X",
)


@dataclass
class BroadMarketConfig:
    min_price: float = 10.0
    max_price: Optional[float] = None
    min_prev_volume: float = 200_000
    min_prev_dollar_volume: float = 25_000_000
    min_avg_dollar_volume: float = 20_000_000
    max_seed_symbols: int = 600
    max_candidates: int = 160
    snapshot_batch_size: int = 200
    history_batch_size: int = 100
    history_period: str = "1y"
    exclude_funds: bool = True
    max_below_52w_high: float = 0.30
    min_above_52w_low: float = 0.25


class BroadMarketScreener:
    """Build a dynamic candidate pool before full Minervini screening."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        config: BroadMarketConfig | None = None,
        paper: bool = True,
    ):
        self.config = config or BroadMarketConfig()
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        self.minervini = MinerviniScreener(
            MinerviniConfig(
                require_fundamentals=False,
                require_market_uptrend=False,
                min_avg_dollar_volume=self.config.min_avg_dollar_volume,
                min_above_52w_low=self.config.min_above_52w_low,
                max_below_52w_high=self.config.max_below_52w_high,
                max_stage_number=3,
            )
        )

    def build_candidates(self, benchmark: str = "SPY") -> pd.DataFrame:
        assets = self._load_assets()
        if not assets:
            return pd.DataFrame()

        snapshot_frame = self._fetch_snapshot_frame(assets)
        if snapshot_frame.empty:
            return snapshot_frame

        seed_frame = snapshot_frame.sort_values(
            ["prev_dollar_volume", "prev_volume"],
            ascending=[False, False],
        ).head(self.config.max_seed_symbols)
        benchmark_df = self._download_symbol_history(benchmark, period=self.config.history_period)
        benchmark_score = 0.0
        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_prepared = self.minervini.prepare_features(benchmark_df)
            benchmark_score = self.minervini._weighted_return_score(benchmark_prepared)

        rows: list[dict] = []
        for chunk in self._chunked(seed_frame.index.tolist(), self.config.history_batch_size):
            history = self._download_batch_history(chunk, period=self.config.history_period)
            if history is None or history.empty:
                continue
            for symbol in chunk:
                normalized = self._extract_symbol_frame(history, symbol)
                if normalized.empty:
                    continue
                row = self._build_candidate_row(
                    symbol,
                    normalized,
                    benchmark_score,
                    snapshot_frame.loc[symbol].to_dict(),
                )
                if row is not None:
                    rows.append(row)

        candidates = pd.DataFrame(rows)
        if candidates.empty:
            return candidates

        candidates["rs_percentile"] = (
            candidates["rs_score"].rank(method="average", pct=True).mul(100.0).round(2)
        )
        candidates = candidates.sort_values(
            ["rs_score", "avg_dollar_volume_50", "prev_dollar_volume"],
            ascending=[False, False, False],
        ).head(self.config.max_candidates)
        return candidates.reset_index(drop=True)

    def _load_assets(self) -> List[dict]:
        assets = self.trading_client.get_all_assets(
            GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
        )
        rows: list[dict] = []
        for asset in assets:
            symbol = str(getattr(asset, "symbol", "") or "").upper()
            if not self._is_supported_symbol(symbol):
                continue
            if not bool(getattr(asset, "tradable", False)):
                continue
            exchange = str(getattr(asset, "exchange", "")).split(".")[-1]
            if exchange == "OTC":
                continue
            name = str(getattr(asset, "name", "") or "")
            if self.config.exclude_funds and self._looks_like_fund(name):
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "exchange": exchange,
                }
            )
        logger.info("Broad scanner loaded %s tradable common-stock assets", len(rows))
        return rows

    def _fetch_snapshot_frame(self, assets: Iterable[dict]) -> pd.DataFrame:
        rows: list[dict] = []
        asset_lookup = {row["symbol"]: row for row in assets}
        symbols = list(asset_lookup.keys())

        for chunk in self._chunked(symbols, self.config.snapshot_batch_size):
            try:
                snapshots = self.data_client.get_stock_snapshot(
                    StockSnapshotRequest(symbol_or_symbols=chunk)
                )
            except Exception as exc:
                logger.warning("Broad snapshot request failed for %s symbols: %s", len(chunk), exc)
                continue

            for symbol, snapshot in snapshots.items():
                previous_bar = getattr(snapshot, "previous_daily_bar", None)
                daily_bar = getattr(snapshot, "daily_bar", None)
                latest_trade = getattr(snapshot, "latest_trade", None)
                reference_bar = previous_bar or daily_bar
                last_price = None
                if latest_trade is not None and getattr(latest_trade, "price", None) is not None:
                    last_price = float(latest_trade.price)
                elif reference_bar is not None and getattr(reference_bar, "close", None) is not None:
                    last_price = float(reference_bar.close)

                if last_price is None or last_price < self.config.min_price:
                    continue
                if self.config.max_price is not None and last_price > self.config.max_price:
                    continue
                if reference_bar is None:
                    continue

                prev_close = float(getattr(reference_bar, "close", 0) or 0)
                prev_volume = float(getattr(reference_bar, "volume", 0) or 0)
                prev_dollar_volume = prev_close * prev_volume
                if prev_volume < self.config.min_prev_volume:
                    continue
                if prev_dollar_volume < self.config.min_prev_dollar_volume:
                    continue

                row = asset_lookup.get(symbol, {}).copy()
                row.update(
                    {
                        "symbol": symbol,
                        "last_price": round(last_price, 2),
                        "prev_close": round(prev_close, 2),
                        "prev_volume": prev_volume,
                        "prev_dollar_volume": prev_dollar_volume,
                    }
                )
                rows.append(row)

        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows).drop_duplicates(subset=["symbol"]).set_index("symbol")
        logger.info("Broad snapshot filter retained %s liquid symbols", len(frame))
        return frame

    def _build_candidate_row(
        self,
        symbol: str,
        frame: pd.DataFrame,
        benchmark_score: float,
        snapshot_meta: Dict,
    ) -> Optional[dict]:
        prepared = self.minervini.prepare_features(frame)
        if prepared.empty or len(prepared) < self.minervini.config.sma_long + 20:
            return None

        latest = prepared.iloc[-1]
        close = self.minervini._safe_float(latest.get("close"))
        sma_50 = self.minervini._safe_float(latest.get("sma_50"))
        sma_150 = self.minervini._safe_float(latest.get("sma_150"))
        sma_200 = self.minervini._safe_float(latest.get("sma_200"))
        sma_200_prev = self.minervini._safe_float(latest.get("sma_200_20d_ago"))
        high_52w = self.minervini._safe_float(latest.get("52w_high"))
        low_52w = self.minervini._safe_float(latest.get("52w_low"))
        avg_dollar_volume_50 = self.minervini._safe_float(latest.get("avg_dollar_volume_50"))
        avg_volume_50 = self.minervini._safe_float(latest.get("avg_volume_50"))
        if None in (close, sma_50, sma_150, sma_200, sma_200_prev, high_52w, low_52w):
            return None
        if avg_dollar_volume_50 is None or avg_dollar_volume_50 < self.config.min_avg_dollar_volume:
            return None

        if close <= sma_50 or close <= sma_150 or close <= sma_200:
            return None
        if sma_150 <= sma_200:
            return None
        if sma_200 <= sma_200_prev:
            return None
        if close < high_52w * (1.0 - self.config.max_below_52w_high):
            return None
        if close < low_52w * (1.0 + self.config.min_above_52w_low):
            return None

        rs_score = self.minervini._weighted_return_score(prepared) - benchmark_score
        return {
            "symbol": symbol,
            "name": snapshot_meta.get("name"),
            "exchange": snapshot_meta.get("exchange"),
            "last_price": round(close, 2),
            "prev_dollar_volume": round(float(snapshot_meta.get("prev_dollar_volume", 0.0)), 2),
            "prev_volume": round(float(snapshot_meta.get("prev_volume", 0.0)), 0),
            "avg_volume_50": round(avg_volume_50 or 0.0, 0),
            "avg_dollar_volume_50": round(avg_dollar_volume_50, 2),
            "distance_from_52w_high_pct": round((high_52w - close) / high_52w, 4),
            "distance_above_52w_low_pct": round((close - low_52w) / low_52w, 4),
            "rs_score": round(rs_score, 6),
        }

    @staticmethod
    def _period_to_dates(period: str) -> tuple[datetime, datetime]:
        end_dt = datetime.now(timezone.utc)
        normalized = (period or "1y").strip().lower()
        padding_days = 45
        if normalized.endswith("y"):
            start_dt = end_dt - timedelta(
                days=(365 * max(int(normalized[:-1] or "1"), 1)) + padding_days
            )
        elif normalized.endswith("mo"):
            start_dt = end_dt - timedelta(
                days=(30 * max(int(normalized[:-2] or "1"), 1)) + padding_days
            )
        elif normalized.endswith("d"):
            start_dt = end_dt - timedelta(
                days=max(int(normalized[:-1] or "1"), 1) + padding_days
            )
        else:
            start_dt = end_dt - timedelta(days=365 + padding_days)
        return start_dt, end_dt

    def _download_batch_history(self, symbols: list[str], period: str) -> pd.DataFrame:
        start_dt, end_dt = self._period_to_dates(period)
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
                feed=DataFeed.IEX,
            )
            response = self.data_client.get_stock_bars(request)
            frame = getattr(response, "df", pd.DataFrame())
            if frame is not None and not frame.empty:
                return frame
        except Exception as exc:
            logger.warning(
                "Broad Alpaca history download failed for batch of %s: %s",
                len(symbols),
                exc,
            )
        try:
            return yf.download(
                " ".join(symbols),
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:
            logger.warning("Broad history download failed for batch of %s: %s", len(symbols), exc)
            return pd.DataFrame()

    def _download_symbol_history(self, symbol: str, period: str) -> pd.DataFrame:
        history = self._download_batch_history([symbol], period=period)
        return self._extract_symbol_frame(history, symbol)

    @staticmethod
    def _extract_symbol_frame(history: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if history is None or history.empty:
            return pd.DataFrame()

        frame = history
        if isinstance(history.index, pd.MultiIndex) and "symbol" in history.index.names:
            try:
                frame = history.xs(symbol, level="symbol").copy()
            except KeyError:
                return pd.DataFrame()
        elif isinstance(history.columns, pd.MultiIndex):
            if symbol not in history.columns.get_level_values(0):
                return pd.DataFrame()
            frame = history[symbol].copy()
        else:
            frame = history.copy()

        normalized = frame.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        if "close" not in normalized.columns or "volume" not in normalized.columns:
            return pd.DataFrame()
        normalized = normalized[[column for column in ["open", "high", "low", "close", "volume"] if column in normalized.columns]]
        normalized = normalized.dropna().sort_index()
        normalized.index = pd.to_datetime(normalized.index)
        if getattr(normalized.index, "tz", None) is not None:
            normalized.index = normalized.index.tz_convert(None)
        return normalized

    @staticmethod
    def _chunked(values: list[str], size: int):
        for index in range(0, len(values), size):
            yield values[index:index + size]

    @staticmethod
    def _looks_like_fund(name: str) -> bool:
        upper = name.upper()
        return any(keyword in upper for keyword in _COMMON_FUND_KEYWORDS)

    @staticmethod
    def _is_supported_symbol(symbol: str) -> bool:
        return bool(re.fullmatch(r"[A-Z]{1,5}", symbol))
