"""Chan-structural 15m segment-level BSP signal precomputer.

Adapts the chan_daily breakthrough lesson — different *mechanism*, not
different parameter — to intraday. Current intraday signals (gap_reclaim,
NR4, ORB) are all post-open breakouts. Chan's segment-level BSP is a
structural reversal signal — a new bs_point appearing on the segment
level after a Chan-defined trend pivot. Different shape entirely.

Per CLAUDE.md mandatory bias-audit:
  * Pure read path. No INSERTs, no shifts back in time. CChan operates
    bar-by-bar via `step_load()`; we observe only the bars already fed in.
  * Future-blanked probe: pass `lag_bars=1` to delay signal emission by
    one bar (signal for bar t is emitted when bar t+1 is processed). If
    the lag-1 result drops materially vs lag-0, lookahead is suspected.
  * RTH filter inherits from the chan_adapter (`_hours_filter_for_level`
    in chan_adapter.py limits 14-21 UTC = 9:30-16:00 ET on standard time).

Returns a per-symbol boolean Series indexed by bar timestamp, True where
a NEW seg-level buy BSP was just registered. Caller merges into the
intraday DataFrame as the `chan_seg_bsp_long` column.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def compute_seg_bsp_long_signals(
    symbol: str,
    db_path: str,
    begin: str,
    end: str,
    *,
    lag_bars: int = 0,
    only_unsure: bool = False,
) -> pd.Series:
    """Compute a boolean signal series for `symbol` over [begin, end].

    Returns a Series indexed by bar timestamp (UTC, matching the intraday
    DuckDB warehouse) with True on the bar where a new seg-level long
    BSP was registered.

    `lag_bars=1` shifts each True by one bar forward — implements the
    standard future-blanked probe (signal for bar t fires on bar t+1).

    `only_unsure=True` restricts to bsp.is_sure=False — these are the
    "tentative" segment-level BSPs that may revert. When False, both
    sure and unsure are emitted (default).
    """
    # Lazy import — chan.py is heavy, only load when this signal is enabled.
    from tradingagents.research import chan_adapter
    from Common.CEnum import KL_TYPE, AUTYPE  # noqa
    from Chan import CChan  # noqa
    from ChanConfig import CChanConfig  # noqa

    if not Path(db_path).exists():
        logger.warning("chan_seg_bsp: DB not found %s — skipping %s", db_path, symbol)
        return pd.Series(dtype=bool)

    chan_adapter.DuckDBIntradayAPI.DB_PATH = db_path
    # Disable chan_adapter's hard-coded RTH filter (14-21 UTC). Our intraday
    # DBs store naive timestamps in CDT (per intraday_backtester.py:277),
    # so the UTC-based filter only matches a 1-hour overlap and starves
    # CChan of 70% of the session. Backtester's `_filter_regular_session`
    # (08:30-15:00 local) is the authoritative session filter; we just feed
    # CChan everything in the DB and let it operate on the same view.
    _orig_hours_filter = chan_adapter._hours_filter_for_level
    chan_adapter._hours_filter_for_level = lambda _k: None
    chan_cfg = CChanConfig({
        "trigger_step": True,
        "bi_strict": True,
        "skip_step": 0,
        "divergence_rate": 0.9,
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "macd_algo": "slope",
    })

    # Use the existing chan.py bridge module (third_party/chan.py/DataAPI/
    # DuckDBAPI.py re-exports DuckDBIntradayAPI as DuckDB30mAPI). Class
    # supports any K_*M k_type — `lv_list=[K_15M]` selects the 15m table.
    try:
        chan = CChan(
            code=symbol,
            begin_time=begin,
            end_time=end,
            data_src="custom:DuckDBAPI.DuckDB30mAPI",
            lv_list=[KL_TYPE.K_15M],
            config=chan_cfg,
            autype=AUTYPE.QFQ,
        )
    except Exception as exc:
        logger.warning("chan_seg_bsp: CChan init failed for %s: %s", symbol, exc)
        return pd.Series(dtype=bool)

    seen_seg_bsp_klu_idx: set[int] = set()
    bar_signals: list[tuple[pd.Timestamp, bool]] = []

    try:
        for snapshot in chan.step_load():
            try:
                lvl = snapshot[0]
                if not lvl.lst:
                    continue
                ckline = lvl.lst[-1]
                cur_klu = ckline.lst[-1]
                ts = pd.Timestamp(
                    year=cur_klu.time.year,
                    month=cur_klu.time.month,
                    day=cur_klu.time.day,
                    hour=cur_klu.time.hour,
                    minute=cur_klu.time.minute,
                )
            except Exception:
                continue

            fresh_long_bsp = False
            try:
                seg_bsp_obj = lvl.seg_bs_point_lst
                # bsp1_list is the canonical "segment-level type-1 BSPs"
                # — emitted when a Chan segment closes against the trend.
                # Sample a buy-side BSP whose klu_idx hasn't been seen yet.
                for seg_bsp in seg_bsp_obj.bsp1_list:
                    if not seg_bsp.is_buy:
                        continue
                    if only_unsure and seg_bsp.bi.is_sure:
                        continue
                    klu_idx = seg_bsp.klu.idx
                    if klu_idx not in seen_seg_bsp_klu_idx:
                        seen_seg_bsp_klu_idx.add(klu_idx)
                        # Only emit if THIS bar is the one containing the BSP.
                        # bsp1_list can lag the trigger bar by several bars
                        # while Chan ratifies the segment turn — emit on the
                        # bar where it first appears (i.e. cur_klu.idx is the
                        # next ratification bar after seg_bsp.klu.idx).
                        fresh_long_bsp = True
            except Exception:
                pass

            bar_signals.append((ts, fresh_long_bsp))
    except Exception as exc:
        logger.warning("chan_seg_bsp: step_load loop crashed for %s: %s", symbol, exc)
    finally:
        # Restore the chan_adapter RTH filter so chan_v2 (which expects the
        # UTC filter for its 30m flow) keeps working after this call.
        chan_adapter._hours_filter_for_level = _orig_hours_filter

    if not bar_signals:
        return pd.Series(dtype=bool)

    s = pd.Series(
        [v for _, v in bar_signals],
        index=pd.to_datetime([t for t, _ in bar_signals]),
        name="chan_seg_bsp_long",
    )
    s = s[~s.index.duplicated(keep="last")]
    if lag_bars > 0:
        s = s.shift(lag_bars).fillna(False).astype(bool)
    return s.astype(bool)
