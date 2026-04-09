"""Local market-data warehouse for research, screening, and backtesting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import duckdb
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketDataWarehouse:
    """Persist daily market data locally so research is reproducible."""

    def __init__(
        self,
        db_path: str = "research_data/market_data.duckdb",
        read_only: bool = False,
    ):
        self.db_path = Path(db_path)
        self.read_only = read_only
        if not self.read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path), read_only=self.read_only)
        self._create_tables()

    def _create_tables(self):
        if self.read_only:
            return
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_bars (
                symbol VARCHAR,
                trade_date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume DOUBLE,
                source VARCHAR,
                updated_at TIMESTAMP,
                PRIMARY KEY (symbol, trade_date)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fundamentals_snapshots (
                symbol VARCHAR,
                snapshot_date DATE,
                market_cap DOUBLE,
                average_volume DOUBLE,
                current_price DOUBLE,
                revenue_growth DOUBLE,
                earnings_quarterly_growth DOUBLE,
                return_on_equity DOUBLE,
                gross_margins DOUBLE,
                operating_margins DOUBLE,
                trailing_pe DOUBLE,
                forward_pe DOUBLE,
                total_revenue_latest DOUBLE,
                total_revenue_prev_year DOUBLE,
                revenue_growth_calc DOUBLE,
                diluted_eps_latest DOUBLE,
                diluted_eps_prev_year DOUBLE,
                eps_growth_calc DOUBLE,
                sector VARCHAR,
                industry VARCHAR,
                source VARCHAR,
                updated_at TIMESTAMP,
                PRIMARY KEY (symbol, snapshot_date)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quarterly_fundamentals (
                symbol VARCHAR,
                fiscal_date DATE,
                total_revenue DOUBLE,
                diluted_eps DOUBLE,
                revenue_yoy_growth DOUBLE,
                eps_yoy_growth DOUBLE,
                revenue_acceleration DOUBLE,
                eps_acceleration DOUBLE,
                source VARCHAR,
                updated_at TIMESTAMP,
                PRIMARY KEY (symbol, fiscal_date)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS earnings_events (
                symbol VARCHAR,
                event_datetime TIMESTAMP,
                eps_estimate DOUBLE,
                reported_eps DOUBLE,
                surprise_pct DOUBLE,
                revenue_average DOUBLE,
                is_future BOOLEAN,
                source VARCHAR,
                updated_at TIMESTAMP,
                PRIMARY KEY (symbol, event_datetime)
            )
            """
        )

    @staticmethod
    def _normalize_yfinance_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if df.empty:
            return df

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        normalized = df.reset_index().rename(
            columns={
                "Date": "trade_date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        if "adj_close" not in normalized.columns:
            normalized["adj_close"] = normalized["close"]

        normalized["symbol"] = symbol
        normalized["trade_date"] = pd.to_datetime(normalized["trade_date"]).dt.date
        normalized["source"] = "yfinance"
        normalized["updated_at"] = pd.Timestamp.utcnow()

        return normalized[
            [
                "symbol",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "source",
                "updated_at",
            ]
        ]

    def fetch_and_store_daily_bars(
        self,
        symbols: Iterable[str],
        start_date: str,
        end_date: str,
        auto_adjust: bool = True,
    ) -> Dict[str, int]:
        """Fetch daily bars and persist them to DuckDB."""
        row_counts: Dict[str, int] = {}
        for symbol in symbols:
            logger.info("Downloading %s daily bars (%s to %s)", symbol, start_date, end_date)
            try:
                df = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    auto_adjust=auto_adjust,
                    progress=False,
                )
            except Exception as exc:
                logger.warning("Daily bar download failed for %s: %s", symbol, exc)
                row_counts[symbol] = 0
                continue
            normalized = self._normalize_yfinance_frame(df, symbol)
            if normalized.empty:
                row_counts[symbol] = 0
                continue

            self.conn.execute(
                "DELETE FROM daily_bars WHERE symbol = ? AND trade_date BETWEEN ? AND ?",
                [symbol, start_date, end_date],
            )
            self.conn.register("bars_df", normalized)
            self.conn.execute("INSERT INTO daily_bars SELECT * FROM bars_df")
            self.conn.unregister("bars_df")
            row_counts[symbol] = len(normalized)

        return row_counts

    @staticmethod
    def _safe_numeric(value) -> Optional[float]:
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _extract_quarterly_metric_growth(
        cls, frame: pd.DataFrame, metric_name: str
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if frame is None or frame.empty or metric_name not in frame.index:
            return None, None, None

        series = frame.loc[metric_name]
        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            return None, None, None

        latest = cls._safe_numeric(series.iloc[0])
        prev_year = cls._safe_numeric(series.iloc[4]) if len(series) > 4 else None
        growth = None
        if latest is not None and prev_year not in (None, 0):
            growth = (latest / prev_year) - 1.0
        return latest, prev_year, growth

    def fetch_and_store_fundamentals(
        self,
        symbols: Iterable[str],
        snapshot_date: Optional[str] = None,
    ) -> Dict[str, bool]:
        snapshot_day = snapshot_date or pd.Timestamp.utcnow().date().isoformat()
        statuses: Dict[str, bool] = {}

        for symbol in symbols:
            logger.info("Downloading %s fundamentals snapshot", symbol)
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info or {}
                quarterly_income_stmt = ticker.quarterly_income_stmt
            except Exception as exc:
                logger.warning("Fundamentals download failed for %s: %s", symbol, exc)
                statuses[symbol] = False
                continue

            total_revenue_latest, total_revenue_prev_year, revenue_growth_calc = (
                self._extract_quarterly_metric_growth(quarterly_income_stmt, "Total Revenue")
            )
            diluted_eps_latest, diluted_eps_prev_year, eps_growth_calc = (
                self._extract_quarterly_metric_growth(quarterly_income_stmt, "Diluted EPS")
            )

            payload = pd.DataFrame(
                [
                    {
                        "symbol": symbol,
                        "snapshot_date": snapshot_day,
                        "market_cap": self._safe_numeric(info.get("marketCap")),
                        "average_volume": self._safe_numeric(info.get("averageVolume")),
                        "current_price": self._safe_numeric(
                            info.get("currentPrice") or info.get("regularMarketPrice")
                        ),
                        "revenue_growth": self._safe_numeric(info.get("revenueGrowth")),
                        "earnings_quarterly_growth": self._safe_numeric(
                            info.get("earningsQuarterlyGrowth")
                        ),
                        "return_on_equity": self._safe_numeric(info.get("returnOnEquity")),
                        "gross_margins": self._safe_numeric(info.get("grossMargins")),
                        "operating_margins": self._safe_numeric(info.get("operatingMargins")),
                        "trailing_pe": self._safe_numeric(info.get("trailingPE")),
                        "forward_pe": self._safe_numeric(info.get("forwardPE")),
                        "total_revenue_latest": total_revenue_latest,
                        "total_revenue_prev_year": total_revenue_prev_year,
                        "revenue_growth_calc": revenue_growth_calc,
                        "diluted_eps_latest": diluted_eps_latest,
                        "diluted_eps_prev_year": diluted_eps_prev_year,
                        "eps_growth_calc": eps_growth_calc,
                        "sector": info.get("sector"),
                        "industry": info.get("industry"),
                        "source": "yfinance",
                        "updated_at": pd.Timestamp.utcnow(),
                    }
                ]
            )

            self.conn.execute(
                "DELETE FROM fundamentals_snapshots WHERE symbol = ? AND snapshot_date = ?",
                [symbol, snapshot_day],
            )
            self.conn.register("fundamentals_df", payload)
            self.conn.execute("INSERT INTO fundamentals_snapshots SELECT * FROM fundamentals_df")
            self.conn.unregister("fundamentals_df")
            statuses[symbol] = True

        return statuses

    def fetch_and_store_quarterly_fundamentals(
        self,
        symbols: Iterable[str],
    ) -> Dict[str, int]:
        row_counts: Dict[str, int] = {}

        for symbol in symbols:
            logger.info("Downloading %s quarterly fundamentals", symbol)
            try:
                quarterly_income_stmt = yf.Ticker(symbol).quarterly_income_stmt
            except Exception as exc:
                logger.warning("Quarterly fundamentals download failed for %s: %s", symbol, exc)
                row_counts[symbol] = 0
                continue
            if quarterly_income_stmt is None or quarterly_income_stmt.empty:
                row_counts[symbol] = 0
                continue

            frame = quarterly_income_stmt.transpose().copy()
            frame.index = pd.to_datetime(frame.index).date
            frame = frame.rename_axis("fiscal_date").reset_index()
            frame["symbol"] = symbol
            frame["total_revenue"] = pd.to_numeric(frame.get("Total Revenue"), errors="coerce")
            frame["diluted_eps"] = pd.to_numeric(frame.get("Diluted EPS"), errors="coerce")
            frame = frame[["symbol", "fiscal_date", "total_revenue", "diluted_eps"]]
            frame = frame.sort_values("fiscal_date").reset_index(drop=True)

            frame["revenue_yoy_growth"] = pd.NA
            frame["eps_yoy_growth"] = pd.NA
            for idx in range(len(frame)):
                if idx >= 4:
                    prev_revenue = frame.loc[idx - 4, "total_revenue"]
                    prev_eps = frame.loc[idx - 4, "diluted_eps"]
                    curr_revenue = frame.loc[idx, "total_revenue"]
                    curr_eps = frame.loc[idx, "diluted_eps"]

                    if pd.notna(curr_revenue) and pd.notna(prev_revenue) and prev_revenue != 0:
                        frame.loc[idx, "revenue_yoy_growth"] = (curr_revenue / prev_revenue) - 1.0
                    if pd.notna(curr_eps) and pd.notna(prev_eps) and prev_eps != 0:
                        frame.loc[idx, "eps_yoy_growth"] = (curr_eps / prev_eps) - 1.0

            frame["revenue_acceleration"] = frame["revenue_yoy_growth"].diff()
            frame["eps_acceleration"] = frame["eps_yoy_growth"].diff()
            frame["source"] = "yfinance"
            frame["updated_at"] = pd.Timestamp.utcnow()

            self.conn.execute(
                "DELETE FROM quarterly_fundamentals WHERE symbol = ?",
                [symbol],
            )
            self.conn.register("quarterly_fundamentals_df", frame)
            self.conn.execute(
                "INSERT INTO quarterly_fundamentals SELECT * FROM quarterly_fundamentals_df"
            )
            self.conn.unregister("quarterly_fundamentals_df")
            row_counts[symbol] = len(frame)

        return row_counts

    def fetch_and_store_earnings_events(
        self,
        symbols: Iterable[str],
        limit: int = 8,
    ) -> Dict[str, int]:
        row_counts: Dict[str, int] = {}

        for symbol in symbols:
            logger.info("Downloading %s earnings events", symbol)
            try:
                ticker = yf.Ticker(symbol)
                events = ticker.get_earnings_dates(limit=limit)
                calendar = ticker.calendar if isinstance(ticker.calendar, dict) else {}
            except Exception as exc:
                logger.warning("Earnings events download failed for %s: %s", symbol, exc)
                row_counts[symbol] = 0
                continue
            if events is None or events.empty:
                row_counts[symbol] = 0
                continue

            frame = events.reset_index().rename(columns={"Earnings Date": "event_datetime"})
            frame["symbol"] = symbol
            frame["event_datetime"] = pd.to_datetime(frame["event_datetime"], utc=True).dt.tz_localize(None)
            frame["eps_estimate"] = pd.to_numeric(frame.get("EPS Estimate"), errors="coerce")
            frame["reported_eps"] = pd.to_numeric(frame.get("Reported EPS"), errors="coerce")
            frame["surprise_pct"] = pd.to_numeric(frame.get("Surprise(%)"), errors="coerce")
            revenue_average = calendar.get("Revenue Average")
            frame["revenue_average"] = self._safe_numeric(revenue_average)
            now_ts = pd.Timestamp.utcnow()
            frame["is_future"] = frame["event_datetime"] > now_ts.tz_localize(None)
            frame["source"] = "yfinance"
            frame["updated_at"] = now_ts
            frame = frame[
                [
                    "symbol",
                    "event_datetime",
                    "eps_estimate",
                    "reported_eps",
                    "surprise_pct",
                    "revenue_average",
                    "is_future",
                    "source",
                    "updated_at",
                ]
            ]
            frame = frame.dropna(subset=["event_datetime"])
            frame = frame.sort_values("event_datetime").drop_duplicates(
                subset=["symbol", "event_datetime"],
                keep="last",
            )

            self.conn.execute("DELETE FROM earnings_events WHERE symbol = ?", [symbol])
            self.conn.register("earnings_events_df", frame)
            self.conn.execute("INSERT INTO earnings_events SELECT * FROM earnings_events_df")
            self.conn.unregister("earnings_events_df")
            row_counts[symbol] = len(frame)

        return row_counts

    def get_daily_bars(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        query = """
            SELECT trade_date, open, high, low, close, adj_close, volume
            FROM daily_bars
            WHERE symbol = ?
        """
        params = [symbol]
        if start_date:
            query += " AND trade_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND trade_date <= ?"
            params.append(end_date)

        query += " ORDER BY trade_date"
        df = self.conn.execute(query, params).fetchdf()
        if df.empty:
            return df

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date")
        return df

    def available_symbols(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT symbol FROM daily_bars ORDER BY symbol"
        ).fetchall()
        return [row[0] for row in rows]

    def get_latest_fundamentals(self, symbols: Optional[Iterable[str]] = None) -> pd.DataFrame:
        query = """
            WITH ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY symbol
                           ORDER BY snapshot_date DESC, updated_at DESC
                       ) AS row_num
                FROM fundamentals_snapshots
            ),
            latest_quarterly AS (
                SELECT symbol, fiscal_date, revenue_yoy_growth, eps_yoy_growth,
                       revenue_acceleration, eps_acceleration
                FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY symbol
                               ORDER BY fiscal_date DESC
                           ) AS row_num
                    FROM quarterly_fundamentals
                )
                WHERE row_num = 1
            ),
            next_earnings AS (
                SELECT symbol, event_datetime, eps_estimate, revenue_average
                FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY symbol
                               ORDER BY event_datetime ASC
                           ) AS row_num
                    FROM earnings_events
                    WHERE is_future = TRUE
                )
                WHERE row_num = 1
            )
            SELECT symbol, snapshot_date, market_cap, average_volume, current_price,
                   revenue_growth, earnings_quarterly_growth, return_on_equity,
                   gross_margins, operating_margins, trailing_pe, forward_pe,
                   total_revenue_latest, total_revenue_prev_year, revenue_growth_calc,
                   diluted_eps_latest, diluted_eps_prev_year, eps_growth_calc,
                   latest_quarterly.fiscal_date AS latest_quarter_date,
                   latest_quarterly.revenue_yoy_growth,
                   latest_quarterly.eps_yoy_growth,
                   latest_quarterly.revenue_acceleration,
                   latest_quarterly.eps_acceleration,
                   next_earnings.event_datetime AS next_earnings_datetime,
                   next_earnings.eps_estimate AS next_earnings_eps_estimate,
                   next_earnings.revenue_average AS next_earnings_revenue_average,
                   sector, industry, source, updated_at
            FROM ranked
            LEFT JOIN latest_quarterly USING (symbol)
            LEFT JOIN next_earnings USING (symbol)
            WHERE row_num = 1
        """
        params = []
        if symbols:
            symbol_list = list(symbols)
            placeholders = ", ".join(["?"] * len(symbol_list))
            query += f" AND symbol IN ({placeholders})"
            params.extend(symbol_list)

        return self.conn.execute(query, params).fetchdf()

    def get_quarterly_fundamentals(
        self, symbols: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:
        query = """
            SELECT symbol, fiscal_date, total_revenue, diluted_eps,
                   revenue_yoy_growth, eps_yoy_growth,
                   revenue_acceleration, eps_acceleration
            FROM quarterly_fundamentals
        """
        params = []
        if symbols:
            symbol_list = list(symbols)
            placeholders = ", ".join(["?"] * len(symbol_list))
            query += f" WHERE symbol IN ({placeholders})"
            params.extend(symbol_list)
        query += " ORDER BY symbol, fiscal_date"
        return self.conn.execute(query, params).fetchdf()

    def close(self):
        self.conn.close()
