"""Streamlit dashboard for monitoring the automated trading system."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from tradingagents.automation.config import build_config
from tradingagents.automation.orchestrator import Orchestrator
from tradingagents.dashboard.multi_variant import get_variant_dbs
from tradingagents.storage.database import TradingDatabase
from tradingagents.testing.ab_config import build_variant_config, load_experiment

SERVICE_LABEL = "com.tradingagents.scheduler"
_DEFAULT_EXPERIMENT = "experiments/paper_launch_v2.yaml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _log_paths(repo_root: Path) -> tuple[Path, Path]:
    log_dir = repo_root / "results" / "service_logs"
    return (
        log_dir / "automation_service.out.log",
        log_dir / "automation_service.err.log",
    )


@st.cache_resource(show_spinner=False)
def _load_variant_orchestrators() -> Dict[str, Orchestrator]:
    """Build an Orchestrator per experiment variant that has valid Alpaca keys."""
    import os
    yaml_path = os.getenv("EXPERIMENT_CONFIG_PATH", _DEFAULT_EXPERIMENT)
    if not Path(yaml_path).exists():
        return {}
    experiment = load_experiment(yaml_path)
    base_config = build_config()
    orchestrators = {}
    for v in experiment.variants:
        vconfig = build_variant_config(base_config, v)
        if not vconfig.get("alpaca_api_key") or not vconfig.get("alpaca_secret_key"):
            continue
        try:
            orchestrators[v.name] = Orchestrator(vconfig)
        except Exception:
            continue
    return orchestrators


def _launchctl_status(label: str) -> dict:
    try:
        result = subprocess.run(
            ["launchctl", "print", f"gui/{_uid()}/{label}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return {"state": "unknown", "detail": str(exc)}

    text = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        return {"state": "not_loaded", "detail": text.strip()}

    state = "unknown"
    pid = None
    last_exit_code = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("state = "):
            state = stripped.split("=", 1)[1].strip()
        elif stripped.startswith("pid = "):
            pid = stripped.split("=", 1)[1].strip()
        elif stripped.startswith("last exit code = "):
            last_exit_code = stripped.split("=", 1)[1].strip()
    return {
        "state": state,
        "pid": pid,
        "last_exit_code": last_exit_code,
        "detail": text.strip(),
    }


def _uid() -> str:
    return subprocess.check_output(["id", "-u"], text=True).strip()


def _tail_text(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return "(log file not found)"
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        content = handle.readlines()
    return "".join(content[-lines:]).strip() or "(empty)"


def _snapshots_frame(db: TradingDatabase) -> pd.DataFrame:
    snapshots = db.get_snapshots(days=60)
    if not snapshots:
        return pd.DataFrame()
    frame = pd.DataFrame(list(reversed(snapshots)))
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _recent_trades_frame(db: TradingDatabase, limit: int = 50) -> pd.DataFrame:
    frame = pd.DataFrame(db.get_recent_trades(limit=limit))
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def _setups_frame(db: TradingDatabase) -> pd.DataFrame:
    frame = pd.DataFrame(db.get_latest_setup_candidates())
    if frame.empty:
        return frame
    ordered = [
        "screen_date",
        "symbol",
        "selected_for_analysis",
        "passed_template",
        "market_regime",
        "rs_percentile",
        "revenue_growth",
        "eps_growth",
        "stage_number",
        "base_label",
        "candidate_status",
        "pivot_price",
        "buy_point",
        "buy_limit_price",
        "initial_stop_price",
        "earnings_days_away",
    ]
    available = [column for column in ordered if column in frame.columns]
    return frame[available]


def _latest_report(repo_root: Path) -> tuple[dict | None, Path | None]:
    report_dir = repo_root / "results" / "daily_reports"
    reports = sorted(report_dir.glob("*.json"))
    if not reports:
        return None, None
    latest = reports[-1]
    with latest.open("r", encoding="utf-8") as handle:
        return json.load(handle), latest


def _render_variant(variant_name: str, orchestrator: Orchestrator, repo_root: Path):
    """Render the dashboard content for a single variant."""
    db = orchestrator.db
    stdout_log, stderr_log = _log_paths(repo_root)

    try:
        status = orchestrator.get_status()
    except Exception as exc:
        st.error(f"Failed to fetch status for {variant_name}: {exc}")
        status = None

    service = _launchctl_status(SERVICE_LABEL)
    snapshots = _snapshots_frame(db)
    recent_trades = _recent_trades_frame(db)
    latest_setups = _setups_frame(db)
    report, report_path = _latest_report(repo_root)

    if status is None:
        acct = {"equity": 0, "cash": 0, "daily_pl": 0, "daily_pl_pct": "N/A"}
        today = {"trade_summary": {"total_orders": 0}}
        screening = {}
        overlay = {}
        positions_list = []
    else:
        acct = status["account"]
        today = status["today"]
        screening = status.get("screening", {})
        overlay = status.get("overlay", {})
        positions_list = status["positions"]

    metric_cols = st.columns(6)
    metric_cols[0].metric("Equity", f"${acct['equity']:,.2f}")
    metric_cols[1].metric("Cash", f"${acct['cash']:,.2f}")
    metric_cols[2].metric("Daily P&L", f"${acct['daily_pl']:,.2f}", acct["daily_pl_pct"])
    metric_cols[3].metric("Open Positions", str(len(positions_list)))
    metric_cols[4].metric("Today Orders", str(today["trade_summary"]["total_orders"]))
    metric_cols[5].metric("Approved Setups", str(len(screening.get("approved_symbols", []))))

    left, right = st.columns([1.3, 1.0])

    with left:
        st.subheader("Equity Curve")
        if snapshots.empty:
            st.info("No daily snapshots saved yet.")
        else:
            chart = snapshots.set_index("date")[["equity", "cash"]]
            st.line_chart(chart)
            st.dataframe(
                snapshots[["date", "equity", "cash", "daily_pl", "daily_pl_pct"]].sort_values("date", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Open Positions")
        positions = pd.DataFrame(positions_list)
        if positions.empty:
            st.info("No open positions.")
        else:
            st.dataframe(positions, use_container_width=True, hide_index=True)

        st.subheader("Recent Trades")
        if recent_trades.empty:
            st.info("No trades logged yet.")
        else:
            st.dataframe(recent_trades, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Screening")
        st.write(
            {
                "screen_date": screening.get("screen_date"),
                "market_regime": screening.get("market_regime"),
                "confirmed_uptrend": screening.get("confirmed_uptrend"),
                "approved_symbols": screening.get("approved_symbols", []),
                "setup_count": screening.get("setup_count"),
            }
        )
        if latest_setups.empty:
            st.info("No setup rows saved yet.")
        else:
            st.dataframe(latest_setups.head(25), use_container_width=True, hide_index=True)

        st.subheader("Overlay")
        if not overlay.get("enabled"):
            st.info("Overlay is disabled.")
        else:
            st.write(overlay)

        st.subheader("Latest Daily Report")
        if report is None:
            st.info("No daily report saved yet.")
        else:
            st.write(
                {
                    "report_file": str(report_path),
                    "date": report.get("date"),
                    "gross_filled_notional": report.get("trade_summary", {}).get("gross_filled_notional"),
                    "open_positions": report.get("position_summary", {}).get("open_positions"),
                }
            )

        st.subheader("Service Logs")
        out_tab, err_tab = st.tabs(["stdout", "stderr"])
        with out_tab:
            st.code(_tail_text(stdout_log), language="text")
        with err_tab:
            st.code(_tail_text(stderr_log), language="text")


def main():
    st.set_page_config(
        page_title="TradingAgents Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("TradingAgents Dashboard")
    st.caption("Live monitor for Alpaca paper trading, Minervini screening, and automation service.")

    repo_root = _repo_root()
    orchestrators = _load_variant_orchestrators()

    if not orchestrators:
        st.warning(
            "No variant accounts found. Check that experiments/paper_launch_v2.yaml "
            "exists and ALPACA_*_API_KEY / ALPACA_*_SECRET_KEY env vars are set."
        )
        st.stop()

    with st.sidebar:
        st.subheader("System")
        st.write(f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
        st.write(f"Variants: `{', '.join(orchestrators.keys())}`")
        if st.button("Refresh"):
            st.rerun()

    variant_names = list(orchestrators.keys())
    tabs = st.tabs(variant_names)

    for tab, name in zip(tabs, variant_names):
        with tab:
            _render_variant(name, orchestrators[name], repo_root)


if __name__ == "__main__":
    main()
