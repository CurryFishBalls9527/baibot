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
}

_DEFAULT_EXPERIMENT = "experiments/paper_launch_v2.yaml"


@st.cache_resource(show_spinner=False)
def get_variant_dbs() -> Dict[str, TradingDatabase]:
    yaml_path = os.getenv("EXPERIMENT_CONFIG_PATH", _DEFAULT_EXPERIMENT)
    if not Path(yaml_path).exists():
        return {}
    experiment = load_experiment(yaml_path)
    dbs = {}
    for v in experiment.variants:
        if v.db_path and Path(v.db_path).exists():
            dbs[v.name] = TradingDatabase(v.db_path)
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
