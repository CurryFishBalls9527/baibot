"""chan.py custom DataAPI backed by our DuckDB intraday warehouse.

chan.py expects data via a class that inherits CCommonStockApi and yields
CKLine_Unit objects from get_kl_data(). This adapter:

1. Queries a DuckDB intraday warehouse for a single symbol
2. Filters to regular trading hours (14:30 - 21:00 UTC = 9:30-16:00 EST)
3. Yields bars in ascending time order as CKLine_Unit

Usage requires chan.py's root on sys.path — see chan_spike.py for the
entry point that wires everything together.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

# chan.py uses top-level imports (from Common.CEnum import ...), so its root
# must be on sys.path before this module is imported.
CHAN_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "chan.py"
if CHAN_ROOT.exists() and str(CHAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CHAN_ROOT))

import duckdb  # noqa: E402

from Common.CEnum import DATA_FIELD, KL_TYPE  # noqa: E402
from Common.CTime import CTime  # noqa: E402
from DataAPI.CommonStockAPI import CCommonStockApi  # noqa: E402
from KLine.KLine_Unit import CKLine_Unit  # noqa: E402


DEFAULT_DB_PATH = "research_data/intraday_30m.duckdb"

INTRADAY_TABLES = {
    KL_TYPE.K_5M: "bars_5m",
    KL_TYPE.K_15M: "bars_15m",
    KL_TYPE.K_30M: "bars_30m",
}


def _hours_filter_for_level(k_type: KL_TYPE) -> tuple[int, int] | None:
    """Return (start_hour_utc, end_hour_utc) window or None for daily+."""
    if k_type in (KL_TYPE.K_DAY, KL_TYPE.K_WEEK, KL_TYPE.K_MON):
        return None
    # Regular US session: 9:30-16:00 EST = 14:30-21:00 UTC (standard time)
    # We use [14, 21) inclusive — DST shifts ±1 hour, worth revisiting
    return (14, 21)


class DuckDBIntradayAPI(CCommonStockApi):
    """Feed chan.py from our intraday DuckDB warehouse.

    `code` is the stock ticker (e.g. 'AAPL').
    `k_type` must match the granularity you're reading.
    `begin_date` / `end_date` are 'YYYY-MM-DD' strings (chan.py convention).
    """

    DB_PATH: str = DEFAULT_DB_PATH  # class-level override hook

    def __init__(self, code, k_type=KL_TYPE.K_30M, begin_date=None, end_date=None, autype=None):
        super().__init__(code, k_type, begin_date, end_date, autype)

    def get_kl_data(self) -> Iterator[CKLine_Unit]:
        table_name = INTRADAY_TABLES.get(self.k_type)
        if table_name is None:
            raise ValueError(
                "DuckDBIntradayAPI only supports intraday levels "
                f"{tuple(INTRADAY_TABLES.keys())} (got {self.k_type})"
            )
        if not Path(self.DB_PATH).exists():
            raise FileNotFoundError(
                f"DuckDB warehouse not found: {self.DB_PATH}. "
                "Run scripts/download_alpaca_30m.py first."
            )

        conn = duckdb.connect(self.DB_PATH, read_only=True)
        try:
            sql = f"""
                SELECT ts, open, high, low, close, volume
                FROM {table_name}
                WHERE symbol = ?
            """
            params: list = [self.code]
            if self.begin_date:
                sql += " AND ts >= ?"
                params.append(f"{self.begin_date} 00:00:00")
            if self.end_date:
                sql += " AND ts <= ?"
                params.append(f"{self.end_date} 23:59:59")
            sql += " ORDER BY ts"
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        hours_window = _hours_filter_for_level(self.k_type)

        for ts, o, h, l, c, v in rows:
            if hours_window is not None:
                hr = ts.hour
                if hr < hours_window[0] or hr >= hours_window[1]:
                    continue
            item = {
                DATA_FIELD.FIELD_TIME: CTime(ts.year, ts.month, ts.day, ts.hour, ts.minute),
                DATA_FIELD.FIELD_OPEN: float(o),
                DATA_FIELD.FIELD_HIGH: float(h),
                DATA_FIELD.FIELD_LOW: float(l),
                DATA_FIELD.FIELD_CLOSE: float(c),
                DATA_FIELD.FIELD_VOLUME: float(v),
            }
            yield CKLine_Unit(item)

    def SetBasciInfo(self):
        pass

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass


DEFAULT_DAILY_DB_PATH = "research_data/market_data.duckdb"


class DuckDBDailyAPI(CCommonStockApi):
    """Feed chan.py from daily bars in market_data.duckdb."""

    DB_PATH: str = DEFAULT_DAILY_DB_PATH

    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=None):
        super().__init__(code, k_type, begin_date, end_date, autype)

    def get_kl_data(self) -> Iterator[CKLine_Unit]:
        if self.k_type != KL_TYPE.K_DAY:
            raise ValueError(f"DuckDBDailyAPI only supports K_DAY (got {self.k_type})")
        if not Path(self.DB_PATH).exists():
            raise FileNotFoundError(f"Daily warehouse not found: {self.DB_PATH}")

        conn = duckdb.connect(self.DB_PATH, read_only=True)
        try:
            sql = """
                SELECT trade_date, open, high, low, close, volume
                FROM daily_bars
                WHERE symbol = ?
            """
            params: list = [self.code]
            if self.begin_date:
                sql += " AND trade_date >= ?"
                params.append(self.begin_date)
            if self.end_date:
                sql += " AND trade_date <= ?"
                params.append(self.end_date)
            sql += " ORDER BY trade_date"
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        for dt, o, h, l, c, v in rows:
            item = {
                DATA_FIELD.FIELD_TIME: CTime(dt.year, dt.month, dt.day, 0, 0),
                DATA_FIELD.FIELD_OPEN: float(o),
                DATA_FIELD.FIELD_HIGH: float(h),
                DATA_FIELD.FIELD_LOW: float(l),
                DATA_FIELD.FIELD_CLOSE: float(c),
                DATA_FIELD.FIELD_VOLUME: float(v) if v else 0.0,
            }
            yield CKLine_Unit(item)

    def SetBasciInfo(self):
        pass

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass
