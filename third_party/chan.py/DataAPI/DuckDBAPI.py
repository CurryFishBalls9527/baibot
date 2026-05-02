"""Thin wrapper so chan.py's custom data_src mechanism can find our adapter.

Usage: CChan(..., data_src="custom:DuckDBAPI.DuckDB30mAPI")
"""
import sys
from pathlib import Path

# Ensure the baibot project root is on sys.path so we can import our adapter
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from tradingagents.research.chan_adapter import DuckDBIntradayAPI as DuckDB30mAPI  # noqa: F401, E402
