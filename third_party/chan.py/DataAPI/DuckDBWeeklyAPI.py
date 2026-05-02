"""Thin wrapper so chan.py's custom data_src mechanism can find our weekly adapter.

Usage: CChan(..., data_src="custom:DuckDBWeeklyAPI.DuckDBWeeklyAPI")
"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from tradingagents.research.chan_adapter import DuckDBWeeklyAPI  # noqa: F401, E402
