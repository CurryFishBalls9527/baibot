"""Performance Overview — cross-variant equity, returns, drawdown, and activity."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from tradingagents.dashboard.multi_variant import (
    VARIANT_COLORS,
    get_variant_dbs,
    query_all_variants,
)

st.set_page_config(page_title="Performance Overview", layout="wide")
st.title("Performance Overview")

# ── Sidebar controls ─────────────────────────────────────────────
dbs = get_variant_dbs()
if not dbs:
    st.warning("No variant databases found. Set EXPERIMENT_CONFIG_PATH or check experiments/paper_launch_v2.yaml.")
    st.stop()

all_variants = list(dbs.keys())
selected = st.sidebar.multiselect("Variants", all_variants, default=all_variants)
period = st.sidebar.radio("Period", ["Daily", "Weekly", "Monthly"], index=0)
freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
freq = freq_map[period]

if not selected:
    st.info("Select at least one variant.")
    st.stop()

# ── Load snapshots ───────────────────────────────────────────────
snapshots = query_all_variants(lambda db: db.get_all_snapshots())
if snapshots.empty or "date" not in snapshots.columns:
    st.info("No daily snapshots yet. Data will appear after the first trading day.")
    st.stop()

snapshots["date"] = pd.to_datetime(snapshots["date"])
snapshots = snapshots[snapshots["variant"].isin(selected)]

if snapshots.empty:
    st.info("No snapshots for selected variants.")
    st.stop()

# ── Metric cards ─────────────────────────────────────────────────
cols = st.columns(len(selected))
for i, variant in enumerate(selected):
    vdf = snapshots[snapshots["variant"] == variant].sort_values("date")
    if vdf.empty:
        cols[i].metric(variant, "No data")
        continue

    equities = vdf["equity"].values
    starting = equities[0]
    current = equities[-1]
    total_return = (current - starting) / starting if starting > 0 else 0

    # Max drawdown
    peak = 0.0
    max_dd = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Sharpe
    daily_returns = []
    for j in range(1, len(equities)):
        if equities[j - 1] > 0:
            daily_returns.append((equities[j] - equities[j - 1]) / equities[j - 1])
    avg_r = sum(daily_returns) / len(daily_returns) if daily_returns else 0
    std_r = (sum((r - avg_r) ** 2 for r in daily_returns) / len(daily_returns)) ** 0.5 if daily_returns else 0
    sharpe = (avg_r / std_r * 252**0.5) if std_r > 0 else 0

    cols[i].metric(f"{variant}", f"${current:,.0f}")
    cols[i].caption(
        f"Return: {total_return:+.2%} | DD: {max_dd:.2%} | Sharpe: {sharpe:.2f}"
    )

# ── Equity curves ────────────────────────────────────────────────
st.subheader("Equity Curves")
fig_eq = go.Figure()
for variant in selected:
    vdf = snapshots[snapshots["variant"] == variant].sort_values("date")
    if vdf.empty:
        continue
    fig_eq.add_trace(go.Scatter(
        x=vdf["date"],
        y=vdf["equity"],
        mode="lines",
        name=variant,
        line=dict(color=VARIANT_COLORS.get(variant)),
    ))
fig_eq.update_layout(
    xaxis_title="Date",
    yaxis_title="Equity ($)",
    hovermode="x unified",
    height=420,
    margin=dict(t=30, b=40),
)
st.plotly_chart(fig_eq, use_container_width=True)

# ── Period returns ───────────────────────────────────────────────
st.subheader(f"{period} Returns")
fig_ret = go.Figure()
for variant in selected:
    vdf = snapshots[snapshots["variant"] == variant].sort_values("date").set_index("date")
    if vdf.empty:
        continue
    resampled = vdf["equity"].resample(freq).last().dropna()
    period_returns = resampled.pct_change().dropna()
    if period_returns.empty:
        continue
    fig_ret.add_trace(go.Bar(
        x=period_returns.index,
        y=period_returns.values * 100,
        name=variant,
        marker_color=VARIANT_COLORS.get(variant),
    ))
fig_ret.update_layout(
    barmode="group",
    xaxis_title="Date",
    yaxis_title="Return (%)",
    hovermode="x unified",
    height=350,
    margin=dict(t=30, b=40),
)
st.plotly_chart(fig_ret, use_container_width=True)

# ── Drawdown ─────────────────────────────────────────────────────
st.subheader("Drawdown")
fig_dd = go.Figure()
for variant in selected:
    vdf = snapshots[snapshots["variant"] == variant].sort_values("date")
    if vdf.empty:
        continue
    eq = vdf["equity"].values
    dates = vdf["date"].values
    peak = 0.0
    dd_series = []
    for e in eq:
        if e > peak:
            peak = e
        dd_series.append(-((peak - e) / peak * 100) if peak > 0 else 0)
    fig_dd.add_trace(go.Scatter(
        x=dates,
        y=dd_series,
        fill="tozeroy",
        name=variant,
        line=dict(color=VARIANT_COLORS.get(variant)),
    ))
fig_dd.update_layout(
    xaxis_title="Date",
    yaxis_title="Drawdown (%)",
    hovermode="x unified",
    height=300,
    margin=dict(t=30, b=40),
)
st.plotly_chart(fig_dd, use_container_width=True)

# ── Trade activity ───────────────────────────────────────────────
st.subheader("Trade Activity")
trades = query_all_variants(lambda db: db.get_recent_trades(limit=500))
if trades.empty or "timestamp" not in trades.columns:
    st.info("No trades yet.")
else:
    # Trades table has two timestamp flavours:
    #   - "YYYY-MM-DD HH:MM:SS[.ffffff]" from orchestrator live inserts
    #   - "YYYY-MM-DDTHH:MM:SS[+00:00]" from reconciler bracket-leg backfill
    # format="mixed" handles both; errors="coerce" drops anything broken.
    trades["timestamp"] = pd.to_datetime(
        trades["timestamp"], format="mixed", errors="coerce", utc=True,
    )
    trades = trades.dropna(subset=["timestamp"])
    # Strip tz for resample grouping (all are UTC after utc=True).
    trades["timestamp"] = trades["timestamp"].dt.tz_localize(None)
    trades = trades[trades["variant"].isin(selected)]
    if not trades.empty:
        fig_act = go.Figure()
        for variant in selected:
            vtrades = trades[trades["variant"] == variant].set_index("timestamp")
            if vtrades.empty:
                continue
            counts = vtrades.resample(freq).size()
            if counts.empty:
                continue
            fig_act.add_trace(go.Bar(
                x=counts.index,
                y=counts.values,
                name=variant,
                marker_color=VARIANT_COLORS.get(variant),
            ))
        fig_act.update_layout(
            barmode="group",
            xaxis_title="Date",
            yaxis_title="Trade Count",
            hovermode="x unified",
            height=300,
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_act, use_container_width=True)
    else:
        st.info("No trades for selected variants.")
