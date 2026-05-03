"""Extract Chan 缠论 structural data for Plotly chart overlays."""

from __future__ import annotations

import sys
from pathlib import Path

CHAN_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "chan.py"
if str(CHAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CHAN_ROOT))

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, KL_TYPE

from tradingagents.research.chan_adapter import DuckDBIntradayAPI


_DEFAULT_CHAN_CFG = {
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
}


def extract_chan_structures(
    symbol: str,
    begin: str,
    end: str,
    db_path: str,
    config_overrides: dict | None = None,
) -> dict:
    """Run Chan analysis and return structures as plain dicts for Plotly.

    Returns dict with keys: bi_list, seg_list, zs_list, bsp_list.

    ``config_overrides`` lets callers tweak the chan config (e.g. switch
    to ``zs_algo='over_seg'`` for more aggressive ZS detection on the web
    view without changing the existing dashboard's defaults).
    """
    DuckDBIntradayAPI.DB_PATH = db_path
    # CChanConfig mutates (empties) the dict it's given in-place — verified
    # 2026-05-03 against chan.py vendored at third_party/chan.py. Without a
    # copy here, any process that calls this function twice gets an
    # AssertionError on the second call (`assert self.conf.trigger_step`
    # fires because trigger_step has been popped). Pass a deep copy.
    import copy as _copy
    cfg = _copy.deepcopy(_DEFAULT_CHAN_CFG)
    if config_overrides:
        cfg.update(config_overrides)
    chan_cfg = CChanConfig(cfg)

    chan = CChan(
        code=symbol,
        begin_time=begin,
        end_time=end,
        data_src="custom:DuckDBAPI.DuckDB30mAPI",
        lv_list=[KL_TYPE.K_30M],
        config=chan_cfg,
        autype=AUTYPE.QFQ,
    )

    for snapshot in chan.step_load():
        pass

    lvl = chan[0]

    bi_list = []
    for bi in lvl.bi_list:
        bi_list.append({
            "start_time": bi.get_begin_klu().time.to_str(),
            "start_val": float(bi.get_begin_val()),
            "end_time": bi.get_end_klu().time.to_str(),
            "end_val": float(bi.get_end_val()),
            "dir": "up" if bi.dir.value == 1 else "down",
        })

    seg_list = []
    for seg in lvl.seg_list:
        seg_list.append({
            "start_time": seg.get_begin_klu().time.to_str(),
            "start_val": float(seg.get_begin_val()),
            "end_time": seg.get_end_klu().time.to_str(),
            "end_val": float(seg.get_end_val()),
            "dir": "up" if seg.dir.value == 1 else "down",
        })

    zs_list = []
    for zs in lvl.zs_list:
        zs_list.append({
            "begin_time": zs.begin.time.to_str(),
            "end_time": zs.end.time.to_str(),
            "low": float(zs.low),
            "high": float(zs.high),
        })

    bsp_list = []
    for bsp in lvl.bs_point_lst.bsp_store_flat_dict.values():
        types = [t.name.split("_")[-1] for t in bsp.type]
        bsp_list.append({
            "time": bsp.klu.time.to_str(),
            "price": float(bsp.klu.close),
            "is_buy": bool(bsp.is_buy),
            "types": "+".join(types),
        })

    return {
        "bi_list": bi_list,
        "seg_list": seg_list,
        "zs_list": zs_list,
        "bsp_list": bsp_list,
    }
