"""Trade Journal — per-trade cards with LLM analysis, P&L distribution, pattern cuts."""

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

st.set_page_config(page_title="Trade Journal", layout="wide")
st.title("Trade Journal")

# ── Sidebar controls ─────────────────────────────────────────────
dbs = get_variant_dbs()
if not dbs:
    st.warning("No variant databases found.")
    st.stop()

all_variants = list(dbs.keys())
selected = st.sidebar.multiselect("Variants", all_variants, default=all_variants)
outcome_filter = st.sidebar.radio("Outcome", ["All", "Winners", "Losers"], index=0)

if not selected:
    st.info("Select at least one variant.")
    st.stop()

# ── Load trade outcomes ──────────────────────────────────────────
outcomes = query_all_variants(lambda db: db.get_all_trade_outcomes())
if outcomes.empty or "return_pct" not in outcomes.columns:
    st.info("No closed trades yet. Data will appear after trades are exited.")
    st.stop()

outcomes = outcomes[outcomes["variant"].isin(selected)]
if outcomes.empty:
    st.info("No trade outcomes for selected variants.")
    st.stop()

# Sidebar symbol filter
all_symbols = sorted(outcomes["symbol"].dropna().unique())
symbol_filter = st.sidebar.multiselect("Symbols", all_symbols, default=[])
if symbol_filter:
    outcomes = outcomes[outcomes["symbol"].isin(symbol_filter)]

# Win/loss filter
if outcome_filter == "Winners":
    outcomes = outcomes[outcomes["return_pct"] > 0]
elif outcome_filter == "Losers":
    outcomes = outcomes[outcomes["return_pct"] <= 0]

if outcomes.empty:
    st.info("No trades match the current filters.")
    st.stop()

# ── Summary metrics ──────────────────────────────────────────────
total = len(outcomes)
winners = (outcomes["return_pct"] > 0).sum()
win_rate = winners / total if total > 0 else 0
avg_return = outcomes["return_pct"].mean()
avg_winner = outcomes.loc[outcomes["return_pct"] > 0, "return_pct"].mean() if winners > 0 else 0
losers_count = total - winners
avg_loser = outcomes.loc[outcomes["return_pct"] <= 0, "return_pct"].mean() if losers_count > 0 else 0

mcols = st.columns(5)
mcols[0].metric("Total Trades", str(total))
mcols[1].metric("Win Rate", f"{win_rate:.1%}")
mcols[2].metric("Avg Return", f"{avg_return:+.2f}%")
mcols[3].metric("Avg Winner", f"{avg_winner:+.2f}%")
mcols[4].metric("Avg Loser", f"{avg_loser:+.2f}%")

# ── Trade cards ──────────────────────────────────────────────────
st.subheader("Closed Trades")


@st.cache_resource(show_spinner=False)
def _get_analyzer():
    from tradingagents.dashboard.trade_analyzer import TradeAnalyzer
    return TradeAnalyzer()


for idx, row in outcomes.iterrows():
    symbol = row.get("symbol", "?")
    ret = row.get("return_pct", 0) or 0
    color = "green" if ret > 0 else "red"
    entry_p = row.get("entry_price")
    exit_p = row.get("exit_price")
    entry_str = f"${entry_p:.2f}" if entry_p else "N/A"
    exit_str = f"${exit_p:.2f}" if exit_p else "N/A"
    variant = row.get("variant", "?")
    hold = row.get("hold_days", "?")
    exit_reason = row.get("exit_reason", "N/A")
    pattern = row.get("base_pattern", "N/A")
    regime = row.get("regime_at_entry", "N/A")
    entry_date = row.get("entry_date", "?")
    exit_date = row.get("exit_date", "?")
    outcome_id = row.get("id")

    label = f":{color}_circle: **{symbol}** ({variant}) — {ret:+.2f}% | {entry_date} → {exit_date}"
    with st.expander(label, expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.write(f"**Entry:** {entry_str} on {entry_date}")
        c2.write(f"**Exit:** {exit_str} on {exit_date}")
        c3.write(f"**Hold:** {hold} days | **Exit:** {exit_reason}")
        c4.write(f"**Pattern:** {pattern} | **Regime:** {regime}")

        # LLM analysis section
        cached_analysis = row.get("trade_analysis")
        if cached_analysis:
            st.markdown(cached_analysis)
        elif outcome_id is not None:
            button_key = f"analyze_{variant}_{outcome_id}"
            if st.button("Analyze this trade", key=button_key):
                with st.spinner("Running LLM analysis..."):
                    analyzer = _get_analyzer()
                    db = dbs.get(variant)
                    entry_signal = None
                    if db and entry_date:
                        entry_signal = db.get_entry_signal_for_trade(symbol, entry_date)
                    analysis = analyzer.analyze_trade(dict(row), entry_signal)
                    if db:
                        db.update_trade_analysis(outcome_id, analysis)
                    st.markdown(analysis)
                    st.rerun()

# ── P&L distribution ─────────────────────────────────────────────
st.subheader("P&L Distribution")
fig_pnl = go.Figure()
for variant in selected:
    vdf = outcomes[outcomes["variant"] == variant]
    if vdf.empty:
        continue
    fig_pnl.add_trace(go.Histogram(
        x=vdf["return_pct"],
        name=variant,
        marker_color=VARIANT_COLORS.get(variant),
        opacity=0.6,
    ))
fig_pnl.update_layout(
    barmode="overlay",
    xaxis_title="Return (%)",
    yaxis_title="Count",
    height=350,
    margin=dict(t=30, b=40),
)
st.plotly_chart(fig_pnl, use_container_width=True)

# ── Win rate by pattern ──────────────────────────────────────────
if outcomes["base_pattern"].notna().any():
    st.subheader("Win Rate by Pattern")
    pattern_stats = (
        outcomes.groupby(["base_pattern", "variant"])
        .agg(
            trades=("return_pct", "size"),
            win_rate=("return_pct", lambda x: (x > 0).mean() * 100),
        )
        .reset_index()
    )
    pattern_stats = pattern_stats[pattern_stats["trades"] >= 2]
    if not pattern_stats.empty:
        fig_pat = go.Figure()
        for variant in selected:
            vdf = pattern_stats[pattern_stats["variant"] == variant]
            if vdf.empty:
                continue
            fig_pat.add_trace(go.Bar(
                y=vdf["base_pattern"],
                x=vdf["win_rate"],
                name=variant,
                marker_color=VARIANT_COLORS.get(variant),
                orientation="h",
                text=vdf["trades"].apply(lambda n: f"n={n}"),
                textposition="auto",
            ))
        fig_pat.update_layout(
            barmode="group",
            xaxis_title="Win Rate (%)",
            height=max(300, len(pattern_stats["base_pattern"].unique()) * 40 + 100),
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_pat, use_container_width=True)

# ── Win rate by regime ───────────────────────────────────────────
if outcomes["regime_at_entry"].notna().any():
    st.subheader("Win Rate by Regime")
    regime_stats = (
        outcomes.groupby(["regime_at_entry", "variant"])
        .agg(
            trades=("return_pct", "size"),
            win_rate=("return_pct", lambda x: (x > 0).mean() * 100),
        )
        .reset_index()
    )
    regime_stats = regime_stats[regime_stats["trades"] >= 2]
    if not regime_stats.empty:
        fig_reg = go.Figure()
        for variant in selected:
            vdf = regime_stats[regime_stats["variant"] == variant]
            if vdf.empty:
                continue
            fig_reg.add_trace(go.Bar(
                y=vdf["regime_at_entry"],
                x=vdf["win_rate"],
                name=variant,
                marker_color=VARIANT_COLORS.get(variant),
                orientation="h",
                text=vdf["trades"].apply(lambda n: f"n={n}"),
                textposition="auto",
            ))
        fig_reg.update_layout(
            barmode="group",
            xaxis_title="Win Rate (%)",
            height=max(300, len(regime_stats["regime_at_entry"].unique()) * 40 + 100),
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_reg, use_container_width=True)
