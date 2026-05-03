"""Chan daily overlay extractor (``chan_daily`` variant).

Symmetrical to ``chan.py`` but operates on daily bars from
``market_data.duckdb.daily_bars``. Uses ``DuckDBDailyAPI`` from
``chan_adapter`` to feed the chan library.

The chan_daily orchestrator persists an enriched signal_metadata blob
(see ``chan_daily_orchestrator.py:427``) — same shape as ``chan`` plus a
``signal_type`` field (e.g. "donchian_30" / "seg_bsp").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from tradingagents.storage.database import TradingDatabase

from ..bars import _to_unix_date, fetch_daily
from .base import (
    Bar,
    ChartPayload,
    Criterion,
    Fill,
    LineOverlay,
    MarkerOverlay,
    Metric,
    Reasoning,
    ZoneOverlay,
)


_DEFAULT_DB = "research_data/market_data.duckdb"


def _trade_with_signal(db: TradingDatabase, trade_id: int) -> Optional[dict]:
    rows = db.conn.execute(
        """
        SELECT t.id AS trade_id, t.timestamp AS trade_ts, t.symbol, t.side,
               t.qty, t.filled_qty, t.filled_price, t.status,
               t.reasoning AS trade_reasoning, t.signal_id,
               s.timestamp AS signal_ts, s.action, s.confidence,
               s.reasoning AS signal_reasoning, s.signal_metadata,
               s.stop_loss, s.take_profit
        FROM trades t
        LEFT JOIN signals s ON s.id = t.signal_id
        WHERE t.id = ?
        """,
        [trade_id],
    ).fetchall()
    return dict(rows[0]) if rows else None


def _exit_fills(db: TradingDatabase, symbol: str, entry_ts: str) -> List[dict]:
    rows = db.conn.execute(
        """
        SELECT timestamp, symbol, side, filled_qty, filled_price, reasoning
        FROM trades
        WHERE symbol = ? AND LOWER(side) = 'sell' AND timestamp > ?
        ORDER BY timestamp ASC
        LIMIT 1
        """,
        [symbol, entry_ts],
    ).fetchall()
    return [dict(r) for r in rows]


def _extract_daily_structures(symbol: str, begin: str, end: str, db_path: str) -> dict:
    """Run Chan analysis on daily bars.

    Mirrors ``dashboard.chan_structures.extract_chan_structures`` but
    forces the daily timeframe + ``DuckDBDailyAPI``.
    """
    chan_root = Path(__file__).resolve().parents[3] / "third_party" / "chan.py"
    if str(chan_root) not in sys.path:
        sys.path.insert(0, str(chan_root))

    from Chan import CChan
    from ChanConfig import CChanConfig
    from Common.CEnum import AUTYPE, KL_TYPE

    from tradingagents.research.chan_adapter import DuckDBDailyAPI

    DuckDBDailyAPI.DB_PATH = db_path

    chan_cfg = CChanConfig({
        "trigger_step": True,
        "bi_strict": True,
        "divergence_rate": 0.8,
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 1,
        "bs1_peak": False,
        "macd_algo": "area",
        "bs_type": "1,1p,2,2s",
        "max_bs2_rate": 0.618,
        "print_warning": False,
        "zs_algo": "normal",
    })

    chan = CChan(
        code=symbol,
        begin_time=begin,
        end_time=end,
        data_src="custom:DuckDBDailyAPI.DuckDBDailyAPI",
        lv_list=[KL_TYPE.K_DAY],
        config=chan_cfg,
        autype=AUTYPE.QFQ,
    )
    for _ in chan.step_load():
        pass

    lvl = chan[0]
    return {
        "bi_list":  [{
            "start_time": bi.get_begin_klu().time.to_str(),
            "start_val":  float(bi.get_begin_val()),
            "end_time":   bi.get_end_klu().time.to_str(),
            "end_val":    float(bi.get_end_val()),
            "dir":        "up" if bi.dir.value == 1 else "down",
        } for bi in lvl.bi_list],
        "seg_list": [{
            "start_time": seg.get_begin_klu().time.to_str(),
            "start_val":  float(seg.get_begin_val()),
            "end_time":   seg.get_end_klu().time.to_str(),
            "end_val":    float(seg.get_end_val()),
            "dir":        "up" if seg.dir.value == 1 else "down",
        } for seg in lvl.seg_list],
        "zs_list":  [{
            "begin_time": zs.begin.time.to_str(),
            "end_time":   zs.end.time.to_str(),
            "low":        float(zs.low),
            "high":       float(zs.high),
        } for zs in lvl.zs_list],
        "bsp_list": [{
            "time":   bsp.klu.time.to_str(),
            "price":  float(bsp.klu.close),
            "is_buy": bool(bsp.is_buy),
            "types":  "+".join(t.name.split("_")[-1] for t in bsp.type),
        } for bsp in lvl.bs_point_lst.bsp_store_flat_dict.values()],
    }


def _structures_to_overlays(structures: dict) -> List[dict]:
    overlays: List[dict] = []
    for bi in structures.get("bi_list", []):
        overlays.append(LineOverlay(
            kind="line",
            from_t=_to_unix_date(bi["start_time"]),
            from_p=float(bi["start_val"]),
            to_t=_to_unix_date(bi["end_time"]),
            to_p=float(bi["end_val"]),
            label=None, style="dashed",
            color="#5b8def" if bi["dir"] == "up" else "#9aa5b1",
            width=1,
        ))
    for seg in structures.get("seg_list", []):
        overlays.append(LineOverlay(
            kind="line",
            from_t=_to_unix_date(seg["start_time"]),
            from_p=float(seg["start_val"]),
            to_t=_to_unix_date(seg["end_time"]),
            to_p=float(seg["end_val"]),
            label="SEG", style="solid",
            color="#1f78b4" if seg["dir"] == "up" else "#5d6d7e",
            width=2,
        ))
    for zs in structures.get("zs_list", []):
        overlays.append(ZoneOverlay(
            kind="zone",
            from_t=_to_unix_date(zs["begin_time"]),
            to_t=_to_unix_date(zs["end_time"]),
            low=float(zs["low"]), high=float(zs["high"]),
            label="ZS", color="rgba(245, 165, 36, 0.18)",
        ))
    for bsp in structures.get("bsp_list", []):
        overlays.append(MarkerOverlay(
            kind="marker",
            time=_to_unix_date(bsp["time"]),
            price=float(bsp["price"]),
            label=bsp.get("types") or "BSP",
            side="buy" if bsp["is_buy"] else "sell",
            color="#10b981" if bsp["is_buy"] else "#ef4444",
        ))
    return overlays


def _build_reasoning(trade_row: dict, meta: dict) -> Reasoning:
    signal_type = meta.get("signal_type") or "—"
    types = meta.get("t_types") or "—"
    bsp_reason = meta.get("bsp_reason") or trade_row.get("signal_reasoning") or ""
    bi_low = meta.get("bi_low")
    regime = meta.get("regime_at_entry")
    market_score = meta.get("market_score")

    return Reasoning(
        headline=f"chan_daily {signal_type}/{types} BUY on {trade_row['symbol']}",
        criteria=[
            {"name": f"Signal type ({signal_type})",       "passed": signal_type != "—", "value": signal_type},
            {"name": f"BSP types ({types})",                "passed": types != "—",       "value": types},
            {"name": "Donchian-30 / segment-BSP qualified", "passed": True,               "value": None},
        ],
        metrics=[
            {"label": "Signal type",  "value": signal_type},
            {"label": "T-types",       "value": types},
            {"label": "BI low",        "value": f"{bi_low:.2f}" if bi_low else "—"},
            {"label": "Regime",        "value": regime or "—"},
            {"label": "Market score",  "value": f"{market_score:.1f}" if isinstance(market_score, (int, float)) else "—"},
            {"label": "Stop pct",      "value": f"{trade_row.get('stop_loss') or 0:.2%}"},
            {"label": "TP pct",        "value": f"{trade_row.get('take_profit') or 0:.2%}"},
        ],
        narrative=bsp_reason or None,
    )


def build_chart(
    db: TradingDatabase,
    trade_id: int,
    variant_config: dict,
) -> ChartPayload:
    row = _trade_with_signal(db, trade_id)
    if row is None:
        return ChartPayload(
            symbol="?", variant=variant_config.get("name", "?"),
            strategy_type="chan_daily", timeframe="1d",
            bars=[], overlays=[], fills=[],
            reasoning=Reasoning(headline="trade not found", criteria=[], metrics=[], narrative=None),
            error=f"trade {trade_id} not found",
        )

    symbol = row["symbol"]
    db_path = variant_config.get("daily_db_path", _DEFAULT_DB)

    bars: List[Bar] = fetch_daily(
        symbol=symbol,
        db_path=db_path,
        bars_before=200,
        bars_after=40,
        pivot_date=str(pd.to_datetime(row["trade_ts"]).date()),
    )

    overlays: List[dict] = []
    err: Optional[str] = None
    if bars:
        first = pd.to_datetime(bars[0]["time"], unit="s").strftime("%Y-%m-%d")
        last  = pd.to_datetime(bars[-1]["time"], unit="s").strftime("%Y-%m-%d")
        try:
            structures = _extract_daily_structures(symbol, first, last, db_path)
            overlays = _structures_to_overlays(structures)
        except Exception as exc:
            err = f"chan_daily extraction failed: {exc}"
    else:
        err = "no daily bars in window"

    meta = {}
    if row.get("signal_metadata"):
        try:
            meta = json.loads(row["signal_metadata"])
        except Exception:
            pass

    fills: List[Fill] = []
    if row.get("filled_price"):
        fills.append(Fill(
            time=_to_unix_date(row["trade_ts"]),
            price=float(row["filled_price"]),
            side="buy" if str(row.get("side", "")).lower() == "buy" else "sell",
            qty=float(row.get("filled_qty") or row.get("qty") or 0),
            reasoning=row.get("trade_reasoning"),
        ))
    for ex in _exit_fills(db, symbol, str(row["trade_ts"])):
        if ex.get("filled_price"):
            fills.append(Fill(
                time=_to_unix_date(ex["timestamp"]),
                price=float(ex["filled_price"]),
                side="sell",
                qty=float(ex.get("filled_qty") or 0),
                reasoning=ex.get("reasoning"),
            ))

    return ChartPayload(
        symbol=symbol,
        variant=variant_config.get("name", "?"),
        strategy_type="chan_daily",
        timeframe="1d",
        bars=bars,
        overlays=overlays,
        fills=fills,
        reasoning=_build_reasoning(row, meta),
        error=err,
    )
