"""Today — single-variant operational view.

User picks a variant from the top-bar dropdown; the page shows that
variant's live KPI, equity curve, open positions, and trade history.
Cross-variant overview moved to the Reviews page.

Live log lives on its own page — see pages/4_Live_Log.py.
"""

from __future__ import annotations

import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from tradingagents.automation.config import build_config
from tradingagents.automation.orchestrator import Orchestrator
from tradingagents.storage.database import TradingDatabase
from tradingagents.testing.ab_config import build_variant_config, load_experiment

_DEFAULT_EXPERIMENT = "experiments/paper_launch_v2.yaml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@st.cache_resource(show_spinner=False)
def _load_variant_orchestrators() -> Dict[str, Orchestrator]:
    import os
    yaml_path = os.getenv("EXPERIMENT_CONFIG_PATH", _DEFAULT_EXPERIMENT)
    if not Path(yaml_path).exists():
        return {}
    experiment = load_experiment(yaml_path)
    base_config = build_config()
    orchestrators: Dict[str, Orchestrator] = {}
    for v in experiment.variants:
        vconfig = build_variant_config(base_config, v)
        if not vconfig.get("alpaca_api_key") or not vconfig.get("alpaca_secret_key"):
            continue
        try:
            orchestrators[v.name] = Orchestrator(vconfig)
        except Exception:
            continue
    return orchestrators


@st.cache_data(ttl=20, show_spinner=False)
def _status_for(variant: str) -> dict:
    orch = _load_variant_orchestrators()[variant]
    return orch.get_status()


@st.cache_data(ttl=20, show_spinner=False)
def _snapshots_for(variant: str) -> pd.DataFrame:
    db: TradingDatabase = _load_variant_orchestrators()[variant].db
    rows = db.get_snapshots(days=180)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(list(reversed(rows)))
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date")


@st.cache_data(ttl=20, show_spinner=False)
def _trades_for(variant: str, limit: int = 50) -> pd.DataFrame:
    db: TradingDatabase = _load_variant_orchestrators()[variant].db
    rows = db.get_recent_trades(limit=limit)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "timestamp" in frame.columns:
        # Trades table holds mixed timestamp formats — orchestrator live
        # inserts use a space separator, reconciler bracket-leg inserts use
        # "T" + timezone. `format="mixed"` handles both.
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], format="mixed", errors="coerce", utc=True,
        )
        frame = frame.dropna(subset=["timestamp"])
        frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
        frame = frame.sort_values("timestamp", ascending=False)
    return frame


def _service_pill() -> str:
    try:
        uid = subprocess.check_output(["id", "-u"], text=True).strip()
        out = subprocess.run(
            ["launchctl", "print", f"gui/{uid}/com.tradingagents.scheduler"],
            capture_output=True, text=True, check=False,
        )
        running = "state = running" in (out.stdout or "")
    except Exception:
        running = False
    color = "#10b981" if running else "#ef4444"
    label = "RUNNING" if running else "STOPPED"
    return (
        f'<span style="display:inline-flex;align-items:center;gap:0.4rem;font-size:0.72rem;'
        f'letter-spacing:0.18em;color:#9ca3af;">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:{color};'
        f'box-shadow:0 0 6px {color};"></span>{label}</span>'
    )


_CSS = """
<style>
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }
/* Sidebar kept visible so multi-page nav (Performance / Trade Journal /
   Reviews) is reachable. The built-in header stays visible too — without
   it, Streamlit's sidebar-toggle chevron is invisible. */
/* Top padding must clear Streamlit's fixed header (~3rem), otherwise the
   variant dropdown at page top clips behind it. */
.block-container { padding: 4rem 1.25rem 1rem; max-width: 1500px; }
.stApp { background: #0a0e14; }
/* Apply the monospace font to text-bearing elements only. The previous
   `[class*="st-"]` selector swept up every Streamlit emotion class,
   including the spans Streamlit uses for Material icons — Material
   icons render via font ligatures, so forcing monospace produced
   literal `keyboard_double_arrow_left` text where the sidebar collapse
   chevron should be a glyph. */
html, body, .stMarkdown, .stText, .stDataFrame, .stTable, .stCaption,
.stCode, .stMetric, .stButton button, .stSelectbox, .stMultiSelect,
.stRadio, .stCheckbox, .stTextInput, .stTextArea, .stNumberInput {
  font-family: 'IBM Plex Mono', 'Menlo', 'Consolas', monospace;
}
[class*="material-icons"], [class*="material-symbols"],
span[class*="EmotionIcon"], span[data-testid*="Icon"] {
  font-family: 'Material Symbols Outlined', 'Material Symbols Rounded',
               'Material Icons' !important;
}

.tb-outer {
  border: 1px solid #1f2937; border-radius: 10px; background: #0d1117;
  padding: 1rem 1.25rem; margin-bottom: 1rem;
}
.tb-title { font-size: 1.45rem; letter-spacing: 0.12em; font-weight: 600; color: #e6edf3; }
.tb-title em { font-style: normal; color: #f5a524; margin: 0 0.35em; }
.tb-title small { color: #64748b; font-size: 0.72rem; letter-spacing: 0.25em; margin-left: 0.6em; }
.tb-meta { text-align: right; color: #64748b; font-size: 0.82rem; }

.kpi-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.75rem; margin-bottom: 1rem; }
.kpi {
  border: 1px solid #1f2937; border-left: 3px solid #f5a524;
  padding: 0.9rem 1rem; border-radius: 6px; background: #0d1117;
  min-height: 96px;
}
.kpi.pos { border-left-color: #10b981; }
.kpi.neg { border-left-color: #ef4444; }
.kpi-label { color: #64748b; font-size: 0.7rem; letter-spacing: 0.22em; font-weight: 600; }
.kpi-value { font-size: 1.55rem; font-weight: 500; margin-top: 0.25rem; color: #e6edf3; }
.kpi-sub { color: #64748b; font-size: 0.75rem; margin-top: 0.15rem; }
.pos { color: #10b981; }
.neg { color: #ef4444; }

.panel {
  border: 1px solid #1f2937; border-radius: 6px; padding: 0.75rem 1rem;
  background: #0d1117; margin-bottom: 1rem;
}
.panel-head {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 0.6rem;
}
.panel-title {
  color: #9ca3af; font-size: 0.72rem; letter-spacing: 0.22em; font-weight: 600;
}
.panel-meta { color: #64748b; font-size: 0.72rem; letter-spacing: 0.12em; }

table.tb { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
table.tb thead th {
  color: #64748b; font-weight: 600; font-size: 0.68rem; letter-spacing: 0.18em;
  text-align: left; padding: 0.4rem 0.5rem; border-bottom: 1px solid #1f2937;
}
table.tb td {
  padding: 0.55rem 0.5rem; border-bottom: 1px solid #161b22; color: #d1d5db;
  vertical-align: middle;
}
table.tb tr:last-child td { border-bottom: none; }

.tag {
  display: inline-block; padding: 0.08rem 0.55rem; font-size: 0.7rem;
  letter-spacing: 0.14em; font-weight: 600; border-radius: 3px;
}
.tag-buy   { background: rgba(16,185,129,0.12); color: #10b981; border: 1px solid #10b981; }
.tag-sell  { background: rgba(239,68,68,0.12);  color: #ef4444; border: 1px solid #ef4444; }

.footer {
  color: #4b5563; font-size: 0.68rem; text-align: center; margin-top: 1rem;
  letter-spacing: 0.28em;
}

.stSelectbox label, .stSelectbox div[data-baseweb="select"] > div { background: #161b22; color: #e6edf3; }
div[data-testid="stVerticalBlock"] > div { gap: 0; }
</style>
"""


def _kpi_card(label: str, value: str, sub: str = "", accent: str = "") -> str:
    sub_html = f'<div class="kpi-sub {accent}">{sub}</div>' if sub else ""
    cls = f"kpi {accent}" if accent in ("pos", "neg") else "kpi"
    return f'<div class="{cls}"><div class="kpi-label">{label}</div>' \
           f'<div class="kpi-value">{value}</div>{sub_html}</div>'


def _equity_curve_fig(snapshots: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=snapshots["date"], y=snapshots["equity"],
            mode="lines",
            line=dict(color="#f5a524", width=2),
            fill="tozeroy",
            fillcolor="rgba(245, 165, 36, 0.08)",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
        )
    )
    y_min = float(snapshots["equity"].min()) * 0.97
    y_max = float(snapshots["equity"].max()) * 1.02
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=320, margin=dict(t=10, b=30, l=55, r=15),
        showlegend=False,
        xaxis=dict(gridcolor="#1a2230", showgrid=True, color="#64748b", zeroline=False),
        yaxis=dict(
            gridcolor="#1a2230", showgrid=True, color="#64748b",
            tickformat="$,.0f", range=[y_min, y_max], zeroline=False,
        ),
    )
    return fig


def _positions_html(positions: List[dict]) -> str:
    if not positions:
        return '<div style="color:#64748b;padding:1.5rem 0;text-align:center;">No open positions.</div>'
    rows = []
    for p in positions:
        pnl = float(p.get("pl", 0) or 0)
        pnl_pct_str = p.get("pl_pct", "0.00%")
        color = "pos" if pnl >= 0 else "neg"
        rows.append(
            '<tr>'
            f'<td style="font-weight:600;">{p.get("symbol","")}</td>'
            f'<td>{int(float(p.get("qty",0)))}</td>'
            f'<td>${float(p.get("entry",0)):,.2f}</td>'
            f'<td>${float(p.get("current",0)):,.2f}</td>'
            f'<td class="{color}">${pnl:+,.2f}'
            f'<div style="font-size:0.7rem;">{pnl_pct_str}</div></td>'
            '</tr>'
        )
    return (
        '<table class="tb"><thead><tr>'
        '<th>SYMBOL</th><th>QTY</th><th>ENTRY</th><th>CURRENT</th><th>P&amp;L</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
    )


def _trades_html(trades: pd.DataFrame, limit: int = 14) -> str:
    if trades.empty:
        return '<div style="color:#64748b;padding:1.5rem 0;text-align:center;">No trades logged yet.</div>'

    rows = []
    for _, t in trades.head(limit).iterrows():
        side = str(t.get("side", "")).lower()
        tag_class = "tag-buy" if side == "buy" else "tag-sell"
        tag_label = "BUY" if side == "buy" else "SELL"

        ts = t.get("timestamp")
        ts_str = ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else ""

        qty_raw = t.get("filled_qty") or t.get("qty") or 0
        try:
            qty = f"{int(float(qty_raw))}"
        except (TypeError, ValueError):
            qty = "—"

        fill = t.get("filled_price")
        status = str(t.get("status", "") or "").lower()
        reasoning = str(t.get("reasoning", "") or "")[:60]
        if fill and float(fill) > 0:
            detail = f"@ ${float(fill):.2f} · {status}"
        else:
            detail = f"{status} · {reasoning}" if reasoning else status

        rows.append(
            '<tr>'
            f'<td style="color:#9ca3af;">{ts_str}</td>'
            f'<td><span class="tag {tag_class}">{tag_label}</span></td>'
            f'<td style="font-weight:600;">{t.get("symbol","")}</td>'
            f'<td>{qty}</td>'
            f'<td style="color:#9ca3af;font-size:0.75rem;">{detail}</td>'
            '</tr>'
        )
    return (
        '<table class="tb"><thead><tr>'
        '<th>TIME</th><th>ACTION</th><th>SYMBOL</th><th>QTY</th><th>DETAILS</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
    )


def main():
    st.set_page_config(
        page_title="Today",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    orchestrators = _load_variant_orchestrators()
    if not orchestrators:
        st.error("No variant accounts available. Check experiments/paper_launch_v2.yaml and .env.")
        st.stop()

    variant_names = list(orchestrators.keys())
    if "variant" not in st.session_state:
        st.session_state["variant"] = variant_names[0]

    # ── Top bar ────────────────────────────────────────────────────────
    with st.container():
        c1, c2, c3, c4 = st.columns([4, 2, 1.2, 0.8])
        with c1:
            st.markdown(
                '<div class="tb-title">TODAY<em>·</em>'
                f'<small>{date.today().isoformat()}</small></div>',
                unsafe_allow_html=True,
            )
        with c2:
            variant = st.selectbox(
                "variant", variant_names,
                index=variant_names.index(st.session_state["variant"]),
                label_visibility="collapsed",
                key="variant",
            )
        with c3:
            st.markdown(
                f'<div class="tb-meta">{datetime.now().strftime("%H:%M:%S")} ET &nbsp; {_service_pill()}</div>',
                unsafe_allow_html=True,
            )
        with c4:
            if st.button("↻ REFRESH", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

    variant = st.session_state["variant"]
    orch = orchestrators[variant]

    try:
        status = _status_for(variant)
    except Exception as exc:
        st.error(f"Status fetch failed for {variant}: {exc}")
        st.stop()

    acct = status["account"]
    positions = status["positions"]
    snapshots = _snapshots_for(variant)

    starting_equity = orch.db.get_starting_equity()
    if starting_equity is None and not snapshots.empty:
        starting_equity = float(snapshots["equity"].iloc[0])
    if starting_equity is None:
        starting_equity = float(acct["equity"])

    current_equity = float(acct["equity"])
    vs_start = current_equity - float(starting_equity)
    vs_start_pct = (vs_start / starting_equity * 100) if starting_equity else 0.0
    vs_cls = "pos" if vs_start >= 0 else "neg"

    daily_pl = float(acct.get("daily_pl", 0) or 0)
    daily_pl_pct = acct.get("daily_pl_pct", "0.00%")
    daily_cls = "pos" if daily_pl >= 0 else "neg"

    max_positions = int(orch.config.get("max_positions", 10))

    # ── KPI row ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="kpi-row">'
        + _kpi_card("EQUITY", f"${current_equity:,.2f}")
        + _kpi_card("CASH", f"${float(acct['cash']):,.2f}")
        + _kpi_card("DAY P&L", f"${daily_pl:+,.2f}", daily_pl_pct, daily_cls)
        + _kpi_card("VS. START", f"${vs_start:+,.2f}", f"{vs_start_pct:+.2f}%", vs_cls)
        + _kpi_card("POSITIONS", f"{len(positions)}", f"max {max_positions}")
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── Equity curve + positions ─────────────────────────────────────
    left, right = st.columns([1.6, 1.0])
    with left:
        total_ret = f"{vs_start_pct:+.2f}% total"
        st.markdown(
            f'<div class="panel"><div class="panel-head">'
            f'<div class="panel-title">EQUITY CURVE</div>'
            f'<div class="panel-meta {vs_cls}">{total_ret}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if snapshots.empty:
            st.markdown(
                '<div style="color:#64748b;padding:3rem 0;text-align:center;">'
                'No equity history yet. Snapshots appear after the first trading day.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.plotly_chart(
                _equity_curve_fig(snapshots),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown(
            f'<div class="panel"><div class="panel-head">'
            f'<div class="panel-title">OPEN POSITIONS</div>'
            f'<div class="panel-meta">{len(positions)} / {max_positions}</div>'
            f'</div>{_positions_html(positions)}</div>',
            unsafe_allow_html=True,
        )

    # ── Trade history ────────────────────────────────────────────────
    trades = _trades_for(variant)
    st.markdown(
        f'<div class="panel"><div class="panel-head">'
        f'<div class="panel-title">TRADE HISTORY</div>'
        f'<div class="panel-meta">last 50</div>'
        f'</div>{_trades_html(trades)}</div>',
        unsafe_allow_html=True,
    )

    # ── Footer ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="footer">PAPER TRADING // FOR TESTING ONLY // NOT FINANCIAL ADVICE '
        f'&nbsp;·&nbsp; LAST REFRESH: {datetime.now().strftime("%H:%M:%S")}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
