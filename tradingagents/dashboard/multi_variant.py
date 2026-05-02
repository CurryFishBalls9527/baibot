"""Multi-variant data layer for cross-experiment dashboard views."""

import os
from pathlib import Path
from typing import Callable, Dict

import pandas as pd
import streamlit as st

from tradingagents.storage.database import TradingDatabase
from tradingagents.testing.ab_config import load_experiment

VARIANT_COLORS = {
    "mechanical": "#4caf50",
    "llm": "#2196f3",
    "chan": "#ff9800",
    "mechanical_v2": "#8bc34a",        # lighter green
    "chan_v2": "#ffb74d",              # lighter orange
    "intraday_mechanical": "#9c27b0",  # purple
    "chan_daily": "#e91e63",           # pink — distinct from chan/chan_v2 oranges
    "pead": "#00bcd4",                 # cyan — standalone PEAD strategy bridge
    "pead_llm": "#0097a7",             # darker cyan — PEAD LLM-gated A/B treatment arm
}

_DEFAULT_EXPERIMENT = "experiments/paper_launch_v2.yaml"


# Standalone strategies (separate launchd jobs, not in paper_launch_v2.yaml)
# whose data the dashboard should still surface. Each entry is
# (display_name, db_path). The corresponding launchd job mirrors its
# JSON state into this DB via a one-way bridge.
#
# 2026-05-01: PEAD added here. Do NOT add it back to paper_launch_v2.yaml —
# the scheduler will treat it as a tradeable variant and corrupt the
# strategy (proven incident: scheduler opened ~$50K of unwanted positions
# on PEAD's Alpaca account). See pead_ingest.py for the bridge details.
_STANDALONE_VIEW_DBS = (
    ("pead", "trading_pead.db"),
    # 2026-05-02: PEAD LLM-gated A/B treatment arm. Dry-run only — no live
    # broker submissions. Mirrored from results/pead/llm_dryrun/ via the
    # com.baibot.pead_llm launchd job. Compare against `pead` (control) on
    # the Performance + Reviews pages to gauge the LLM gate's edge.
    ("pead_llm", "trading_pead_llm.db"),
)


@st.cache_resource(show_spinner=False)
def get_variant_dbs() -> Dict[str, TradingDatabase]:
    yaml_path = os.getenv("EXPERIMENT_CONFIG_PATH", _DEFAULT_EXPERIMENT)
    dbs: Dict[str, TradingDatabase] = {}
    if Path(yaml_path).exists():
        experiment = load_experiment(yaml_path)
        for v in experiment.variants:
            if v.db_path and Path(v.db_path).exists():
                dbs[v.name] = TradingDatabase(v.db_path)
    # Standalone view-only DBs (mirrored by separate launchd bridges)
    for name, db_path in _STANDALONE_VIEW_DBS:
        if Path(db_path).exists() and name not in dbs:
            dbs[name] = TradingDatabase(db_path)
    return dbs


def query_all_variants(query_fn: Callable) -> pd.DataFrame:
    """Run query_fn(db) per variant, concatenate with 'variant' column."""
    dbs = get_variant_dbs()
    frames = []
    for name, db in dbs.items():
        rows = query_fn(db)
        if rows:
            df = pd.DataFrame(rows)
            df["variant"] = name
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["variant"])
    return pd.concat(frames, ignore_index=True)
