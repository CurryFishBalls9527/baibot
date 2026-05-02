"""Local market-data warehouse for research, screening, and backtesting."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, TypeVar

import duckdb
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# Earnings data lives in a separate DuckDB file so the standalone earnings
# ingest crons (AV / yfinance / calendar) and the live PEAD reader never
# contend with the scheduler's market_data.duckdb writer lock. Cross-DB
# queries use ATTACH ... (READ_ONLY) which does not take a writer lock.
# See plan: /Users/myu/.claude/plans/ethereal-strolling-rocket.md.
EARNINGS_DB_PATH = "research_data/earnings_data.duckdb"

T = TypeVar("T")


def _with_lock_retry(
    fn: Callable[[], T],
    max_tries: int = 3,
    backoff: tuple[float, ...] = (0.5, 2.0, 8.0),
) -> T:
    """Run ``fn`` and retry on DuckDB ``Could not set lock`` errors.

    Used by live readers (e.g., pead_trader) that may collide with brief
    writer windows held by the scheduler. Returns ``fn()`` on success;
    re-raises the final exception after exhausting retries.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_tries):
        try:
            return fn()
        except duckdb.IOException as exc:
            if "Could not set lock" not in str(exc):
                raise
            last_exc = exc
            if attempt + 1 >= max_tries:
                break
            sleep_s = backoff[min(attempt, len(backoff) - 1)]
            logger.warning(
                "DuckDB lock contention (attempt %d/%d), retrying in %.1fs",
                attempt + 1, max_tries, sleep_s,
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


class MarketDataWarehouse:
    """Persist daily market data locally so research is reproducible.

    Earnings tables (``earnings_events``, ``earnings_calendar``) live in a
    separate DuckDB file at :data:`EARNINGS_DB_PATH`. This warehouse
    attaches that file READ_ONLY for cross-DB JOINs (see
    :meth:`get_latest_fundamentals`) and opens dedicated write connections
    inside :meth:`fetch_and_store_earnings_events`.
    """

    def __init__(
        self,
        db_path: str = "research_data/market_data.duckdb",
        read_only: bool = False,
        earnings_db_path: str = EARNINGS_DB_PATH,
    ):
        self.db_path = Path(db_path)
        self.read_only = read_only
        self.earnings_db_path = Path(earnings_db_path)
        if not self.read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path), read_only=self.read_only)
        self._earnings_attached = False
        self._attach_earnings_readonly()
        self._create_tables()

    # Context-manager support for `with MarketDataWarehouse(...) as wh:` —
    # ensures close() runs even on exception. Backward compatible with
    # existing try/finally close() callers.
    def __enter__(self) -> "MarketDataWarehouse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _attach_earnings_readonly(self) -> None:
        """Attach earnings_data.duckdb READ_ONLY for cross-DB JOINs.

        Idempotency: DuckDB attaches are PROCESS-WIDE — once any
        connection in the process attaches a file, all other connections
        can query ``ed.*`` without re-attaching. A 2nd ATTACH from
        another warehouse instance fails with "already exists", which we
        treat as success (the alias IS visible to this connection via
        process-shared catalog state — verified empirically).

        Safety vs. concurrent writers: keep this attach READ_ONLY so it
        does not take a writer lock on the file. Ingest crons writing to
        earnings_data.duckdb from a SEPARATE process are unaffected.
        Within THIS process: writers (e.g. fetch_and_store_earnings_events
        via _write_earnings_events) cannot open the file directly while
        any attach is active — see _write_earnings_events for the
        detach-write-reattach dance.

        Skipped silently if the earnings DB file doesn't exist yet
        (one-shot migration via scripts/migrate_split_earnings_db.py
        creates it).
        """
        if not self.earnings_db_path.exists():
            logger.warning(
                "earnings DB %s missing; cross-DB JOINs disabled. "
                "Run scripts/migrate_split_earnings_db.py.",
                self.earnings_db_path,
            )
            return
        try:
            self.conn.execute(
                f"ATTACH '{self.earnings_db_path}' AS ed (READ_ONLY)"
            )
            self._earnings_attached = True
        except duckdb.Error as exc:
            msg = str(exc)
            if "already exists" in msg or "already attached" in msg:
                # Another warehouse in this process already attached. The
                # `ed` alias resolves through process-shared catalog —
                # mark as attached so queries via `ed.*` work.
                self._earnings_attached = True
            else:
                logger.warning("failed to attach earnings DB: %s", exc)

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
        # earnings_events lives in earnings_data.duckdb (attached READ_ONLY
        # at __init__). We no longer define it here. The legacy table in
        # market_data.duckdb is kept around as fallback until the rollout
        # is proven, but is no longer the source of truth.
        self._migrate_earnings_events_add_time_hint()

    def _migrate_earnings_events_add_time_hint(self) -> None:
        """Legacy migration: add time_hint to local earnings_events if it
        still exists. After the DB split (earnings_data.duckdb), this
        method is a no-op when the legacy table is gone. Idempotent.
        """
        if self.read_only:
            return
        # Check whether earnings_events still exists in this local DB
        # (it may have been dropped during the post-split cleanup).
        cols = self.conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'earnings_events' AND table_catalog = current_database()"
        ).fetchall()
        if not cols:
            return  # legacy table already cleaned up — nothing to do
        col_names = {row[0] for row in cols}
        if "time_hint" not in col_names:
            self.conn.execute(
                "ALTER TABLE earnings_events ADD COLUMN time_hint VARCHAR"
            )
        self.conn.execute(
            """
            UPDATE earnings_events
            SET time_hint = CASE
                WHEN EXTRACT(hour FROM event_datetime) <= 13 THEN 'bmo'
                WHEN EXTRACT(hour FROM event_datetime) >= 20 THEN 'amc'
                ELSE 'unknown'
            END
            WHERE time_hint IS NULL
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
        limit: int = 40,
        min_events: int = 20,
        use_alpha_vantage_fallback: bool = True,
    ) -> Dict[str, int]:
        """Fetch historical earnings events per symbol.

        yfinance is the primary source (preserves BMO/AMC via event hour). If
        the returned event count is below ``min_events`` and the Alpha Vantage
        API key is configured, the AV EARNINGS endpoint fills the historical
        gap (date-only, so those rows get time_hint='unknown').
        """
        row_counts: Dict[str, int] = {}

        for symbol in symbols:
            logger.info("Downloading %s earnings events", symbol)
            yf_frame = self._fetch_yfinance_earnings(symbol, limit=limit)
            frame = yf_frame

            if (
                use_alpha_vantage_fallback
                and (frame is None or len(frame) < min_events)
            ):
                av_frame = self._fetch_alpha_vantage_earnings(symbol)
                if av_frame is not None and not av_frame.empty:
                    frame = self._merge_earnings_sources(yf_frame, av_frame)

            if frame is None or frame.empty:
                row_counts[symbol] = 0
                continue

            frame = frame.sort_values("event_datetime").drop_duplicates(
                subset=["symbol", "event_datetime"],
                keep="last",
            )

            # Write to earnings_data.duckdb (split DB). We open a fresh
            # write connection per symbol-batch so we hold the lock for
            # only a few ms — long enough for the standalone PEAD reader
            # to never effectively contend.
            self._write_earnings_events(symbol, frame)
            row_counts[symbol] = len(frame)

        return row_counts

    def _write_earnings_events(self, symbol: str, frame: pd.DataFrame) -> None:
        """Open a short-lived write connection to earnings_data.duckdb,
        delete-then-insert all rows for this symbol, close. Used by
        fetch_and_store_earnings_events to avoid holding the earnings
        DB writer lock across the slow yfinance/AV network calls.
        """
        self.earnings_db_path.parent.mkdir(parents=True, exist_ok=True)
        write_conn = duckdb.connect(str(self.earnings_db_path))
        try:
            # Ensure the target table exists (matches migrate_split_earnings_db.py)
            write_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS earnings_events (
                    symbol VARCHAR NOT NULL,
                    event_datetime TIMESTAMP NOT NULL,
                    eps_estimate DOUBLE,
                    reported_eps DOUBLE,
                    surprise_pct DOUBLE,
                    revenue_average DOUBLE,
                    is_future BOOLEAN,
                    source VARCHAR,
                    updated_at TIMESTAMP,
                    time_hint VARCHAR,
                    PRIMARY KEY (symbol, event_datetime)
                )
                """
            )
            write_conn.execute(
                "DELETE FROM earnings_events WHERE symbol = ?", [symbol]
            )
            write_conn.register("earnings_events_df", frame)
            write_conn.execute(
                "INSERT INTO earnings_events SELECT * FROM earnings_events_df"
            )
            write_conn.unregister("earnings_events_df")
        finally:
            write_conn.close()

    def _fetch_yfinance_earnings(
        self,
        symbol: str,
        limit: int = 40,
    ) -> Optional[pd.DataFrame]:
        try:
            ticker = yf.Ticker(symbol)
            events = ticker.get_earnings_dates(limit=limit)
            calendar = ticker.calendar if isinstance(ticker.calendar, dict) else {}
        except Exception as exc:
            logger.warning("yfinance earnings download failed for %s: %s", symbol, exc)
            return None
        if events is None or events.empty:
            return None

        frame = events.reset_index().rename(columns={"Earnings Date": "event_datetime"})
        frame["symbol"] = symbol
        frame["event_datetime"] = pd.to_datetime(
            frame["event_datetime"], utc=True
        ).dt.tz_localize(None)
        frame["eps_estimate"] = pd.to_numeric(frame.get("EPS Estimate"), errors="coerce")
        frame["reported_eps"] = pd.to_numeric(frame.get("Reported EPS"), errors="coerce")
        frame["surprise_pct"] = pd.to_numeric(frame.get("Surprise(%)"), errors="coerce")
        revenue_average = calendar.get("Revenue Average")
        frame["revenue_average"] = self._safe_numeric(revenue_average)
        now_ts = pd.Timestamp.utcnow()
        frame["is_future"] = frame["event_datetime"] > now_ts.tz_localize(None)
        frame["source"] = "yfinance"
        frame["updated_at"] = now_ts
        # BMO = before market open (hour <= 13 UTC ~ <= 09:00 ET)
        # AMC = after market close (hour >= 20 UTC ~ >= 16:00 ET)
        hours = frame["event_datetime"].dt.hour
        frame["time_hint"] = hours.apply(
            lambda h: "bmo" if h <= 13 else ("amc" if h >= 20 else "unknown")
        )
        frame = frame.dropna(subset=["event_datetime"])
        return frame[
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
                "time_hint",
            ]
        ]

    def _fetch_alpha_vantage_earnings(
        self,
        symbol: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch historical earnings from Alpha Vantage EARNINGS endpoint.

        AV provides reportedDate back to IPO but no BMO/AMC flag, so rows
        get time_hint='unknown' and the blackout module treats them as
        whole-day conservatively.
        """
        api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
        if not api_key:
            logger.warning(
                "ALPHAVANTAGE_API_KEY not set; skipping AV earnings fallback for %s",
                symbol,
            )
            return None
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "EARNINGS",
            "symbol": symbol,
            "apikey": api_key,
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning("AV earnings download failed for %s: %s", symbol, exc)
            return None

        quarterly = payload.get("quarterlyEarnings") or []
        if not quarterly:
            return None

        rows = []
        now_ts = pd.Timestamp.utcnow().tz_localize(None)
        for item in quarterly:
            reported_date = item.get("reportedDate")
            if not reported_date:
                continue
            try:
                event_dt = pd.Timestamp(reported_date)
            except Exception:
                continue

            def _num(key: str):
                raw = item.get(key)
                if raw in (None, "None", "", "NaN"):
                    return None
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    return None

            rows.append(
                {
                    "symbol": symbol,
                    "event_datetime": event_dt,
                    "eps_estimate": _num("estimatedEPS"),
                    "reported_eps": _num("reportedEPS"),
                    "surprise_pct": _num("surprisePercentage"),
                    "revenue_average": None,
                    "is_future": event_dt > now_ts,
                    "source": "alphavantage",
                    "updated_at": pd.Timestamp.utcnow(),
                    "time_hint": "unknown",
                }
            )

        if not rows:
            return None
        return pd.DataFrame(rows)

    @staticmethod
    def _merge_earnings_sources(
        yf_frame: Optional[pd.DataFrame],
        av_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combine yfinance and AV frames; yfinance wins when dates match
        within one day (yfinance has BMO/AMC precision)."""
        if yf_frame is None or yf_frame.empty:
            return av_frame
        yf_dates = set(yf_frame["event_datetime"].dt.normalize())
        av_mask = ~av_frame["event_datetime"].dt.normalize().apply(
            lambda d: any(abs((d - yd).days) <= 1 for yd in yf_dates)
        )
        av_filtered = av_frame[av_mask]
        if av_filtered.empty:
            return yf_frame
        return pd.concat([yf_frame, av_filtered], ignore_index=True)

    def get_earnings_events(
        self,
        symbol: str,
    ) -> pd.DataFrame:
        """Return all earnings events for a symbol, sorted by date.

        Reads from the attached read-only earnings DB (ed.earnings_events).
        Returns an empty frame if the attach failed.
        """
        if not self._earnings_attached:
            return pd.DataFrame(columns=[
                "symbol", "event_datetime", "eps_estimate", "reported_eps",
                "surprise_pct", "revenue_average", "is_future", "source",
                "updated_at", "time_hint",
            ])
        query = """
            SELECT symbol, event_datetime, eps_estimate, reported_eps,
                   surprise_pct, revenue_average, is_future, source,
                   updated_at, time_hint
            FROM ed.earnings_events
            WHERE symbol = ?
            ORDER BY event_datetime
        """
        return self.conn.execute(query, [symbol]).fetchdf()

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
        # Earnings tables live in the attached read-only DB. If attach
        # failed (file missing), fall back to a SELECT that yields no
        # rows so the LEFT JOIN populates NULLs and the rest of the query
        # still works.
        next_earnings_source = (
            "ed.earnings_events"
            if self._earnings_attached
            else "(SELECT NULL::VARCHAR AS symbol, NULL::TIMESTAMP AS event_datetime, "
                 "NULL::DOUBLE AS eps_estimate, NULL::DOUBLE AS revenue_average, "
                 "FALSE AS is_future WHERE FALSE)"
        )
        query = f"""
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
                    FROM {next_earnings_source}
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

    def get_daily_bars_bulk(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch daily bars for multiple symbols in one query, split by symbol.

        Much faster than calling get_daily_bars() in a loop (single SQL query).
        """
        if not symbols:
            return {}
        placeholders = ", ".join(["?"] * len(symbols))
        query = f"""
            SELECT symbol, trade_date, open, high, low, close, adj_close, volume
            FROM daily_bars
            WHERE symbol IN ({placeholders})
        """
        params: list = list(symbols)
        if start_date:
            query += " AND trade_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND trade_date <= ?"
            params.append(end_date)
        query += " ORDER BY symbol, trade_date"

        df = self.conn.execute(query, params).fetchdf()
        if df.empty:
            return {}

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        result = {}
        for symbol, group in df.groupby("symbol"):
            frame = group.drop(columns=["symbol"]).set_index("trade_date")
            result[str(symbol)] = frame
        return result

    def symbols_with_data_on(self, trade_date: str, min_bars: int = 220) -> list[str]:
        """Return symbols that have at least min_bars of data ending on or before trade_date.

        Used by walk-forward backtester to filter to symbols with sufficient history.
        """
        query = """
            SELECT symbol
            FROM daily_bars
            WHERE trade_date <= ?
            GROUP BY symbol
            HAVING COUNT(*) >= ?
            ORDER BY symbol
        """
        rows = self.conn.execute(query, [trade_date, min_bars]).fetchall()
        return [row[0] for row in rows]

    def close(self):
        self.conn.close()
