"""Portfolio Risk — cross-strategy diagnostics in plain English.

Pure read-only. NO live sizing changes. Surface the math; sizing decisions
stay manual.

Each section answers ONE specific question with auto-generated narrative
interpretation. The narratives are RULE-BASED (deterministic, free, instant).
A single optional 'AI Synthesis' button at the top calls a strong LLM
across all the data — that's the only place this page hits an external API.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from tradingagents.dashboard.multi_variant import (
    VARIANT_COLORS,
    get_variant_dbs,
    query_all_variants,
)

st.set_page_config(page_title="Portfolio Risk", layout="wide")
st.title("Portfolio Risk Diagnostics")

st.markdown(
    "**What this page tells you**: how independent your 7 strategies "
    "actually are, where your portfolio's daily swings come from, which "
    "strategies excel in which market environments, and which entry "
    "patterns within each strategy actually make money. "
    "**Read-only — nothing here changes live trading.** "
    "Sizing decisions remain manual via `paper_launch_v2.yaml`."
)

# ─── Sidebar controls ──────────────────────────────────────────────────

dbs = get_variant_dbs()
if not dbs:
    st.warning("No variant databases found.")
    st.stop()

all_variants = list(dbs.keys())
selected = st.sidebar.multiselect("Strategies to compare", all_variants, default=all_variants)
window_days = st.sidebar.slider(
    "Recent window (days)", min_value=20, max_value=180, value=60, step=10,
    help="How many recent trading days to use for the 'recent' charts. "
         "Older days are still in the 'full history' view.",
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data source: `daily_pl_pct` from each strategy's daily snapshot. "
    "Days where any strategy has no snapshot are dropped from comparison."
)

if not selected or len(selected) < 2:
    st.info("Pick at least 2 strategies in the sidebar to compare.")
    st.stop()


# ─── AI synthesis (on-demand, cached) ─────────────────────────────────


@st.cache_data(ttl=3600, show_spinner=False)
def _ai_synthesize(prompt_hash: str, prompt: str) -> dict:
    """Call gpt-5.4-pro to synthesize across all sections. Cached by
    prompt hash so refreshes don't re-pay until the underlying data
    changes (TTL 1h as a safety net). Returns dict with text + cost +
    timestamp + model.
    """
    from tradingagents.llm_clients.factory import create_llm_client
    client = create_llm_client(provider="openai", model="gpt-5.4-pro")
    llm = client.get_llm()
    started = datetime.now(timezone.utc)
    result = llm.invoke(prompt)
    duration = (datetime.now(timezone.utc) - started).total_seconds()
    # Extract text — handle both string and list-of-content-block shapes
    # (gpt-5.4-pro emits the latter via the responses API).
    raw = result.content
    if isinstance(raw, list):
        text = "\n".join(
            block.get("text", "") for block in raw
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        text = str(raw)
    return {
        "text": text,
        "model": "gpt-5.4-pro",
        "duration_s": round(duration, 1),
        "generated_at": started.isoformat(timespec="seconds"),
        # Rough cost estimate (matching pead_llm_analyzer's _RATE_PER_1K):
        # ~5k input + ~1k output for this kind of synthesis. Real usage
        # arrives via OpenAI's response.usage if the langchain client
        # surfaces it; for now use the rough estimate.
        "cost_estimate_usd": 0.05,
    }


def _build_synthesis_prompt(
    selected_variants: list,
    coverage_days: dict,
    full_corr: pd.DataFrame,
    recent_corr: Optional[pd.DataFrame],
    sizing_table: Optional[pd.DataFrame],
    portfolio_sigma_annual: Optional[float],
    regime_sharpe_pivot: Optional[pd.DataFrame],
    pattern_summary: Optional[pd.DataFrame],
) -> str:
    """Compose a structured prompt with all the page's diagnostic data."""
    parts = [
        "You are a quantitative portfolio analyst reviewing live paper-trading "
        "data for a multi-strategy book. Be terse, specific, and actionable. "
        "Cross-reference between sections — that's the value-add over the "
        "rule-based per-section narratives the user already sees on the page.",
        "",
        f"## Portfolio scope",
        f"- Strategies in scope: {selected_variants}",
        f"- Days of data per strategy: {coverage_days}",
        "",
        "## Pairwise correlations (full history)",
        full_corr.round(2).to_string(),
    ]
    if recent_corr is not None and not recent_corr.empty:
        parts += [
            "",
            "## Pairwise correlations (recent window only)",
            recent_corr.round(2).to_string(),
        ]
    if sizing_table is not None and not sizing_table.empty:
        parts += [
            "",
            "## Current weights, vol, and risk-parity-suggested weights",
            sizing_table.to_string(),
        ]
    if portfolio_sigma_annual is not None:
        parts += ["", f"## Portfolio annualized vol (current weights): {portfolio_sigma_annual:.1f}%"]
    if regime_sharpe_pivot is not None and not regime_sharpe_pivot.empty:
        parts += [
            "",
            "## Risk-adjusted return (annualized Sharpe) per strategy per SPY regime",
            regime_sharpe_pivot.round(2).to_string(),
        ]
    if pattern_summary is not None and not pattern_summary.empty:
        parts += [
            "",
            "## Per-strategy entry-pattern P&L breakdown",
            pattern_summary.to_string(),
        ]
    parts += [
        "",
        "## Your task",
        "Write a 4-7 sentence portfolio-level synthesis that:",
        "1. Cross-references between sections (e.g. 'the redundant pair from "
        "section 1 is also the top swing-driver from section 3, so this "
        "redundancy is high-stakes')",
        "2. Calls out anything genuinely surprising or actionable",
        "3. Names ONE specific decision the user should make based on this data",
        "4. If the data is too sparse for firm conclusions, say so explicitly "
        "rather than confabulating",
        "Do NOT just rehash what the rule-based per-section narratives "
        "already say. Add value via synthesis or skip the point.",
    ]
    return "\n".join(parts)


with st.container(border=True):
    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.markdown(
            "**AI synthesis (on-demand)** — gpt-5.4-pro reads all six "
            "sections together and writes 4-7 sentences cross-referencing "
            "what each in isolation can't see. ~$0.05 per generation, "
            "cached for 1 hour or until the data changes."
        )
    with col_right:
        ai_clicked = st.button(
            "Generate AI synthesis",
            type="primary",
            use_container_width=True,
        )
    # Placeholder for the synthesis output — we populate it after computing
    # all the other sections (so we have the data to feed into the prompt).
    ai_placeholder = st.empty()

# Captures populated by later sections so the AI synthesis (rendered
# at end of page into ai_placeholder) can see all the diagnostic data.
_ai_capture: dict = {
    "sizing_table": None,
    "portfolio_sigma_annual": None,
    "recent_corr": None,
    "regime_sharpe_pivot": None,
    "pattern_summary": None,
}

# ─── Load daily returns matrix ────────────────────────────────────────

snapshots = query_all_variants(lambda db: db.get_all_snapshots())
if snapshots.empty or "date" not in snapshots.columns:
    st.info("No daily snapshots yet. Data will appear after the first trading day.")
    st.stop()

snapshots["date"] = pd.to_datetime(snapshots["date"])
snapshots = snapshots[snapshots["variant"].isin(selected)]
returns_wide = snapshots.pivot_table(
    index="date", columns="variant", values="daily_pl_pct", aggfunc="first",
).sort_index()
returns_wide = returns_wide.dropna(how="all")

if returns_wide.shape[0] < 5:
    st.warning(
        f"Only {returns_wide.shape[0]} day(s) of overlapping data — need ~10+ "
        "for meaningful conclusions. Come back after more trading days "
        "have accumulated."
    )
    st.stop()

# Coverage report — how many days each strategy has
coverage = returns_wide.notna().sum().to_dict()
total_days = returns_wide.shape[0]
st.caption(
    f"Data on hand: **{total_days} trading days** total. Per strategy: "
    + " · ".join(f"`{v}`={n}d" for v, n in coverage.items())
)

# ─── Section 1: cross-strategy correlation — full history ─────────────

st.divider()
st.header("1. How independent are these strategies, really?")
st.markdown(
    "**The question**: if I'm running 7 strategies but they all move together, "
    "I effectively have 1 strategy with 7× the operational complexity. The "
    "matrix below shows pairwise daily-return correlations. "
    "Closer to **+1** (red) means the two strategies make and lose money on "
    "the same days. Closer to **0** (white) means they're independent — "
    "real diversification. **−1** (blue) means they're natural hedges."
)

full_corr = returns_wide.corr()
fig_full = px.imshow(
    full_corr.round(2),
    text_auto=True,
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    aspect="auto",
    title="Pairwise correlation — full history",
)
fig_full.update_layout(
    height=400,
    margin=dict(t=40, b=20),
    coloraxis_colorbar=dict(title="Correlation"),
)
st.plotly_chart(fig_full, use_container_width=True)

# Auto-narrative
def _summarize_correlations(corr: pd.DataFrame) -> str:
    """Render a plain-English summary of the correlation matrix."""
    pairs = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pairs.append((a, b, float(corr.loc[a, b])))
    pairs.sort(key=lambda p: -abs(p[2]))
    lines = []
    redundant = [p for p in pairs if p[2] > 0.6]
    diversified = [p for p in pairs if abs(p[2]) < 0.2]
    hedges = [p for p in pairs if p[2] < -0.3]
    if redundant:
        lines.append(
            f"**Redundant pairs** (>0.6 correlation, basically the same trade): "
            + ", ".join(f"`{a} ↔ {b}` ({c:+.2f})" for a, b, c in redundant[:5])
        )
    if diversified:
        lines.append(
            f"**Genuinely independent pairs** (|correlation| < 0.2): "
            + ", ".join(f"`{a} ↔ {b}` ({c:+.2f})" for a, b, c in diversified[:5])
        )
    if hedges:
        lines.append(
            f"**Natural hedges** (negative correlation, one wins when the other loses): "
            + ", ".join(f"`{a} ↔ {b}` ({c:+.2f})" for a, b, c in hedges[:3])
        )
    if not lines:
        lines.append(
            "Most pairs are in the 0.2-0.6 range — moderate diversification, "
            "neither redundant nor truly independent."
        )
    return "\n\n".join(lines)


with st.container(border=True):
    st.markdown("**What this matrix says about your portfolio:**")
    st.markdown(_summarize_correlations(full_corr))
    n_pairs = len(selected) * (len(selected) - 1) // 2
    high_count = sum(
        1 for i, a in enumerate(selected) for b in selected[i + 1:]
        if abs(full_corr.loc[a, b]) > 0.6
    )
    st.caption(
        f"Total unique pairs: {n_pairs}. Pairs with strong correlation (|r|>0.6): "
        f"**{high_count}**. The more of these you have, the less your "
        "7-strategy portfolio is actually 7 strategies."
    )

# ─── Section 2: recent vs historical — has the relationship shifted? ──

st.divider()
st.header("2. Has the relationship between strategies shifted recently?")
st.markdown(
    f"**The question**: maybe your strategies were independent in 2024 but "
    f"have become correlated in 2026. The matrix below shows the same "
    f"correlations using only the last **{window_days} trading days**. If "
    f"these numbers are very different from the full-history matrix above, "
    f"a regime shift is in progress and the recent numbers are more relevant "
    f"for today's sizing decisions."
)

recent = returns_wide.tail(window_days)
if recent.shape[0] >= 5:
    recent_corr = recent.corr()
    fig_recent = px.imshow(
        recent_corr.round(2),
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        title=f"Last {window_days} days only",
    )
    fig_recent.update_layout(height=400, margin=dict(t=40, b=20))
    st.plotly_chart(fig_recent, use_container_width=True)
    _ai_capture["recent_corr"] = recent_corr

    delta = (recent_corr - full_corr).round(2)
    big_shifts = []
    cols = delta.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if abs(delta.loc[a, b]) > 0.2:
                big_shifts.append(
                    f"`{a} ↔ {b}`: {full_corr.loc[a, b]:+.2f} → "
                    f"{recent_corr.loc[a, b]:+.2f} ({'stronger' if delta.loc[a, b] > 0 else 'weaker'})"
                )
    if big_shifts:
        with st.container(border=True):
            st.markdown("**Pairs that shifted by more than 0.2:**")
            for s in big_shifts[:8]:
                st.markdown(f"- {s}")
            st.caption(
                "These pairs' relationships are not stable. Use the recent "
                "number for current decisions; consider whether the shift is "
                "permanent or a temporary regime artifact."
            )
    else:
        st.success(
            "All pairs are stable within ±0.2 — no major regime shift in the "
            "recent window. The full-history correlations are still a good "
            "guide for current sizing."
        )
else:
    st.info(f"Not enough days in last-{window_days} window yet.")

# ─── Section 3: where do the daily swings actually come from? ─────────

st.divider()
st.header("3. Which strategies cause your portfolio's daily swings?")
st.markdown(
    "**The question**: with $780k spread across strategies, on a typical "
    "down day, where did the loss come from? If 80% of your daily swings "
    "are caused by 2 of the 7 strategies, those 2 dominate your risk profile "
    "regardless of how the capital is split. The table below quantifies this. "
    "It also suggests **alternative weights** — if you wanted every strategy "
    "to contribute equally to portfolio swings (called *risk parity*), how "
    "much should you shift?"
)

returns_for_math = recent if recent.shape[0] >= 10 else returns_wide
returns_for_math = returns_for_math.dropna()
if returns_for_math.empty:
    st.info("Not enough fully-paired days yet for the risk math.")
else:
    latest_eq = (
        snapshots.sort_values("date").groupby("variant").tail(1)
        .set_index("variant")["equity"]
        .reindex(returns_for_math.columns).fillna(0)
    )
    if latest_eq.sum() <= 0:
        st.info("No equity data available.")
    else:
        current_weights = latest_eq / latest_eq.sum()
        cov = returns_for_math.cov()
        w = current_weights.values
        portfolio_var = float(w @ cov.values @ w)
        portfolio_sigma = np.sqrt(portfolio_var) if portfolio_var > 0 else 0.0
        if portfolio_sigma > 0:
            mcr = (w * (cov.values @ w)) / portfolio_sigma
            pct_contrib = mcr / mcr.sum() if mcr.sum() != 0 else mcr
        else:
            pct_contrib = np.zeros(len(w))
        inv_vol = 1.0 / np.sqrt(np.diag(cov.values))
        rp_weights = inv_vol / inv_vol.sum()

        sizing_table = pd.DataFrame({
            "Strategy": returns_for_math.columns,
            "Current $": [f"${v:,.0f}" for v in latest_eq.values],
            "Current weight": [f"{w*100:.1f}%" for w in current_weights.values],
            "Daily volatility": [f"{v:.2f}%" for v in np.sqrt(np.diag(cov.values))],
            "Causes this % of portfolio swings": [f"{p*100:.0f}%" for p in pct_contrib],
            "Risk-parity weight (suggested)": [f"{w*100:.1f}%" for w in rp_weights],
            "Shift to risk-parity": [
                f"{(rp - cur) * 100:+.1f}pp"
                for rp, cur in zip(rp_weights, current_weights.values)
            ],
        })
        st.dataframe(sizing_table.set_index("Strategy"), use_container_width=True)
        _ai_capture["sizing_table"] = sizing_table.set_index("Strategy")
        _ai_capture["portfolio_sigma_annual"] = portfolio_sigma * np.sqrt(252)

        # Auto-narrative
        contrib_series = pd.Series(pct_contrib, index=returns_for_math.columns)
        top_2 = contrib_series.nlargest(2)
        top_2_share = top_2.sum() * 100
        with st.container(border=True):
            st.markdown(
                f"**Bottom line**: The 2 strategies causing the biggest "
                f"swings — `{top_2.index[0]}` ({top_2.iloc[0]*100:.0f}%) and "
                f"`{top_2.index[1]}` ({top_2.iloc[1]*100:.0f}%) — together "
                f"are responsible for **{top_2_share:.0f}%** of your portfolio's "
                f"daily volatility."
            )
            st.markdown(
                f"**Annualized portfolio volatility**: ~{portfolio_sigma * np.sqrt(252):.1f}% "
                f"per year. (For reference: the S&P 500 historically runs "
                "around 15-20% annualized vol.)"
            )
            biggest_shift = sorted(
                [(returns_for_math.columns[i], (rp_weights[i] - w[i]) * 100)
                 for i in range(len(w))],
                key=lambda x: -abs(x[1])
            )[:3]
            shift_text = ", ".join(
                f"{s} would change by {d:+.1f}pp"
                for s, d in biggest_shift
            )
            st.markdown(
                f"**If you wanted equal-risk contribution** (the suggested "
                f"weights above), the biggest changes would be: {shift_text}. "
                "Those numbers are first-order estimates — treat them as a "
                "sanity check, not a prescription."
            )

# ─── Section 4: rolling correlation for one pair ──────────────────────

st.divider()
st.header("4. Has any specific pair's relationship moved over time?")
st.markdown(
    "Pick any two strategies. The chart shows how their correlation has "
    "evolved across the rolling window. **If the line bounces between, "
    "say, +0.7 and -0.3**, that pair's diversification is regime-dependent "
    "and you should size for the worst case (the high-correlation periods)."
)

if len(selected) >= 2:
    pair_options = [
        f"{a} ↔ {b}" for i, a in enumerate(selected) for b in selected[i + 1:]
    ]
    pair_str = st.selectbox("Pick a pair", pair_options)
    a, b = pair_str.split(" ↔ ")
    pair_returns = returns_wide[[a, b]].dropna()
    if pair_returns.shape[0] >= window_days:
        rolling_corr = pair_returns[a].rolling(window_days).corr(pair_returns[b])
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=rolling_corr.index, y=rolling_corr.values,
            mode="lines", name=f"{window_days}-day correlation",
            line=dict(color="#0097a7", width=2),
        ))
        fig_roll.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Independent")
        fig_roll.add_hline(y=0.6, line_dash="dot", line_color="red",
                           annotation_text="Effectively the same trade")
        fig_roll.add_hline(y=-0.6, line_dash="dot", line_color="green",
                           annotation_text="Natural hedge")
        fig_roll.update_layout(
            yaxis=dict(range=[-1, 1], title="Correlation"),
            xaxis_title="Date",
            height=350,
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_roll, use_container_width=True)
        # Auto-narrative
        cur_corr = rolling_corr.dropna().iloc[-1] if len(rolling_corr.dropna()) else None
        rolling_range = rolling_corr.dropna()
        if cur_corr is not None and len(rolling_range) > 5:
            min_c, max_c = rolling_range.min(), rolling_range.max()
            with st.container(border=True):
                st.markdown(
                    f"**Right now**: `{a}` and `{b}` are correlated at "
                    f"**{cur_corr:+.2f}** "
                    f"({'redundant' if abs(cur_corr) > 0.6 else 'moderately related' if abs(cur_corr) > 0.3 else 'mostly independent'})."
                )
                if max_c - min_c > 0.4:
                    st.markdown(
                        f"**Historically unstable**: this pair has swung from "
                        f"{min_c:+.2f} to {max_c:+.2f}. Don't trust either "
                        "extreme — size assuming the high-correlation case "
                        f"({max_c:+.2f})."
                    )
                else:
                    st.markdown(
                        f"**Historically stable**: range {min_c:+.2f} to {max_c:+.2f}. "
                        "The current number is a reliable guide."
                    )
    else:
        st.info(
            f"Need at least {window_days} paired days for rolling view; "
            f"only have {pair_returns.shape[0]}."
        )

# ─── Section 5: regime-conditional performance ────────────────────────

st.divider()
st.header("5. Which strategies do well in which markets?")
st.markdown(
    "**The question**: PEAD might be a star in volatile/range-bound markets "
    "but mediocre in confirmed trends; mechanical might be the opposite. "
    "If you can identify which strategies excel in which environments, "
    "you have the input for **Phase-3 fusion** — dynamically weighting "
    "strategies based on current market regime.\n\n"
    "**How regime is classified** (Minervini's standard SPY-based rules):\n"
    "- *Confirmed uptrend*: SPY above its 50-day AND 200-day moving "
    "average, AND the 200-day MA is rising. The strongest bull regime.\n"
    "- *Uptrend under pressure*: SPY still above its 200-day, but signs "
    "of weakness (broke 50-day or 200-day flat).\n"
    "- *Market correction*: SPY below 200-day. Defensive regime.\n\n"
    "The heatmap shows **risk-adjusted return** (annualized Sharpe) per "
    "strategy per regime. Higher = better. Empty cells mean no data yet."
)


@st.cache_data(ttl=300, show_spinner=False)
def _load_spy_regime() -> Optional[pd.DataFrame]:
    try:
        import duckdb
    except ImportError:
        return None
    db_path = "research_data/market_data.duckdb"
    try:
        con = duckdb.connect(db_path, read_only=True)
        try:
            spy = con.execute(
                "SELECT trade_date, close FROM daily_bars "
                "WHERE symbol='SPY' ORDER BY trade_date"
            ).df()
        finally:
            con.close()
    except Exception:
        return None
    if spy.empty:
        return None
    spy["trade_date"] = pd.to_datetime(spy["trade_date"])
    spy = spy.set_index("trade_date")
    spy["sma_50"] = spy["close"].rolling(50).mean()
    spy["sma_200"] = spy["close"].rolling(200).mean()
    spy["sma_200_20d_ago"] = spy["sma_200"].shift(20)
    above_50 = spy["close"] > spy["sma_50"]
    above_200 = spy["close"] > spy["sma_200"]
    rising_200 = spy["sma_200"] > spy["sma_200_20d_ago"]
    spy["regime"] = "market correction"
    spy.loc[above_200, "regime"] = "uptrend under pressure"
    spy.loc[above_50 & above_200 & rising_200, "regime"] = "confirmed uptrend"
    return spy[["regime"]].dropna()


regime_df = _load_spy_regime()
if regime_df is None or regime_df.empty:
    st.info("SPY history unavailable — regime panel skipped.")
else:
    tagged = returns_wide.copy()
    tagged.index.name = "date"
    tagged = tagged.merge(regime_df, left_index=True, right_index=True, how="inner")
    if tagged.empty:
        st.info("No overlap between snapshots and SPY regime data.")
    else:
        long = tagged.melt(
            id_vars="regime",
            value_vars=[c for c in tagged.columns if c != "regime"],
            var_name="variant", value_name="daily_pl_pct",
        ).dropna()
        agg = long.groupby(["variant", "regime"]).agg(
            n_days=("daily_pl_pct", "count"),
            mean_pct=("daily_pl_pct", "mean"),
            std_pct=("daily_pl_pct", "std"),
            win_rate=("daily_pl_pct", lambda s: (s > 0).mean()),
        ).reset_index()
        agg["risk_adj_return"] = (
            (agg["mean_pct"] / agg["std_pct"].replace(0, pd.NA)) * (252 ** 0.5)
        ).round(2)
        pivot_sharpe = agg.pivot(
            index="variant", columns="regime", values="risk_adj_return",
        )
        pivot_n = agg.pivot(index="variant", columns="regime", values="n_days")
        text = pivot_sharpe.copy().astype(str)
        for v in pivot_sharpe.index:
            for r in pivot_sharpe.columns:
                s = pivot_sharpe.loc[v, r]
                n = pivot_n.loc[v, r]
                text.loc[v, r] = (
                    f"{s:+.1f}<br>over {int(n) if pd.notna(n) else 0}d"
                    if pd.notna(s) else "no data"
                )
        fig_regime = go.Figure(go.Heatmap(
            z=pivot_sharpe.values,
            x=pivot_sharpe.columns,
            y=pivot_sharpe.index,
            text=text.values,
            texttemplate="%{text}",
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Risk-adjusted<br>return"),
        ))
        fig_regime.update_layout(
            height=350,
            margin=dict(t=20, b=40),
            xaxis_title="Market regime (SPY-based)",
            yaxis_title="Strategy",
        )
        st.plotly_chart(fig_regime, use_container_width=True)
        _ai_capture["regime_sharpe_pivot"] = pivot_sharpe

        # Auto-narrative
        with st.container(border=True):
            insights = []
            for variant in pivot_sharpe.index:
                row = pivot_sharpe.loc[variant].dropna()
                if len(row) < 2:
                    continue
                best_regime = row.idxmax()
                worst_regime = row.idxmin()
                if row[best_regime] > 0.5 and row[worst_regime] < -0.5:
                    insights.append(
                        f"**`{variant}`** is regime-dependent: thrives in "
                        f"*{best_regime}* (Sharpe {row[best_regime]:+.1f}), "
                        f"struggles in *{worst_regime}* (Sharpe {row[worst_regime]:+.1f})."
                    )
                elif row.min() > 0:
                    insights.append(
                        f"**`{variant}`** is regime-stable: positive in every "
                        f"regime tested. Hold across the cycle."
                    )
                elif row.max() < 0:
                    insights.append(
                        f"**`{variant}`** is negative in every regime tested. "
                        f"Pruning candidate (need more data first)."
                    )
            if insights:
                st.markdown("**What this reveals:**\n\n" + "\n\n".join(insights))
            else:
                st.markdown(
                    "Not enough days yet to draw firm conclusions. Come back "
                    "after 60+ trading days have accumulated across multiple "
                    "regimes."
                )
        small = agg[agg["n_days"] < 10]
        if not small.empty:
            st.caption(
                f"Note: {len(small)} (strategy, regime) cells have fewer than 10 days. "
                "Treat those numbers as directional only."
            )
        # Detail table for the curious
        with st.expander("See raw numbers"):
            agg["mean_pct"] = agg["mean_pct"].round(3)
            agg["std_pct"] = agg["std_pct"].round(3)
            agg["win_rate"] = (agg["win_rate"] * 100).round(1)
            st.dataframe(
                agg[["variant", "regime", "n_days", "mean_pct", "std_pct", "win_rate", "risk_adj_return"]]
                  .rename(columns={
                      "variant": "Strategy", "regime": "Regime", "n_days": "Days",
                      "mean_pct": "Avg daily %", "std_pct": "Daily volatility %",
                      "win_rate": "Win rate %", "risk_adj_return": "Risk-adj return",
                  }),
                use_container_width=True, hide_index=True,
            )

# ─── Section 6: per-pattern attribution ───────────────────────────────

st.divider()
st.header("6. Within each strategy, which entry pattern actually makes money?")
st.markdown(
    "**The question**: A strategy can be profitable overall but driven "
    "entirely by one entry pattern (e.g., `cup-with-handle`) while other "
    "patterns it fires on (`flat-base`, `leader_continuation`) are net "
    "losers dragging it down. If you can identify those losing patterns, "
    "you can either fix them, gate them, or remove them — making the "
    "strategy cleaner without giving up alpha.\n\n"
    "Each row below is one (strategy, pattern) combination across all "
    "closed trades. **`Total return %`** is what matters — that's the "
    "actual P&L contribution from this pattern."
)


def _load_outcomes(db) -> list:
    rows = db.conn.execute(
        "SELECT base_pattern, return_pct, hold_days, exit_reason "
        "FROM trade_outcomes WHERE return_pct IS NOT NULL"
    ).fetchall()
    return [dict(zip(["base_pattern", "return_pct", "hold_days", "exit_reason"], r)) for r in rows]


outcomes = query_all_variants(_load_outcomes)
if outcomes.empty:
    st.info(
        "No closed trades yet across selected strategies. This panel fills "
        "in as trades close (mechanical/llm/v2 emit `cup_with_handle`, "
        "`flat_base`, etc.; PEAD emits `earnings_surprise`)."
    )
else:
    outcomes["base_pattern"] = outcomes["base_pattern"].fillna("(unknown)").astype(str)
    grouped = outcomes.groupby(["variant", "base_pattern"]).agg(
        n_trades=("return_pct", "count"),
        avg_return_pct=("return_pct", "mean"),
        median_return_pct=("return_pct", "median"),
        win_rate=("return_pct", lambda s: (s > 0).mean()),
        total_return_pct=("return_pct", "sum"),
        avg_hold_days=("hold_days", "mean"),
    ).reset_index()
    grouped["win_rate"] = (grouped["win_rate"] * 100).round(1)
    grouped["avg_return_pct"] = grouped["avg_return_pct"].round(2)
    grouped["median_return_pct"] = grouped["median_return_pct"].round(2)
    grouped["total_return_pct"] = grouped["total_return_pct"].round(2)
    grouped["avg_hold_days"] = grouped["avg_hold_days"].round(1)
    grouped = grouped.sort_values(["variant", "total_return_pct"], ascending=[True, False])

    display = grouped.rename(columns={
        "variant": "Strategy",
        "base_pattern": "Pattern",
        "n_trades": "Trades",
        "avg_return_pct": "Avg %",
        "median_return_pct": "Median %",
        "win_rate": "Win rate %",
        "total_return_pct": "Total %",
        "avg_hold_days": "Avg held (days)",
    })
    st.dataframe(display, use_container_width=True, hide_index=True)
    _ai_capture["pattern_summary"] = grouped[
        ["variant", "base_pattern", "n_trades", "win_rate", "total_return_pct"]
    ].set_index(["variant", "base_pattern"])

    # Auto-narrative: for each variant, identify the alpha source + the drag
    insights = []
    for variant in grouped["variant"].unique():
        sub = grouped[grouped["variant"] == variant]
        if sub["n_trades"].sum() < 5:
            continue
        best = sub.loc[sub["total_return_pct"].idxmax()]
        worst = sub.loc[sub["total_return_pct"].idxmin()]
        share_pct = (
            best["total_return_pct"] / sub["total_return_pct"].abs().sum() * 100
            if sub["total_return_pct"].abs().sum() > 0 else 0
        )
        msg = (
            f"**`{variant}`**: alpha is concentrated in pattern "
            f"`{best['base_pattern']}` ({best['n_trades']} trades, "
            f"+{best['total_return_pct']:.1f}% total)."
        )
        if worst["total_return_pct"] < 0 and worst["base_pattern"] != best["base_pattern"]:
            msg += (
                f" The pattern `{worst['base_pattern']}` is a net drag "
                f"({worst['n_trades']} trades, {worst['total_return_pct']:+.1f}% total) — "
                "candidate for review."
            )
        insights.append(msg)
    if insights:
        with st.container(border=True):
            st.markdown("**What this table reveals:**\n\n" + "\n\n".join(insights))

    # Bar chart visualization
    st.markdown("**Visual**: P&L per pattern, faceted by strategy.")
    fig_pat = px.bar(
        grouped,
        x="base_pattern", y="total_return_pct",
        color="variant",
        color_discrete_map=VARIANT_COLORS,
        facet_col="variant",
        facet_col_wrap=3,
        labels={"total_return_pct": "Total return % from this pattern",
                "base_pattern": "Pattern"},
        height=400,
    )
    fig_pat.update_layout(showlegend=False, margin=dict(t=30, b=40))
    fig_pat.for_each_xaxis(lambda x: x.update(tickangle=45))
    fig_pat.for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[-1])  # clean facet titles
    )
    st.plotly_chart(fig_pat, use_container_width=True)

    small = grouped[grouped["n_trades"] < 5]
    if not small.empty:
        st.caption(
            f"Note: {len(small)} (strategy, pattern) cells have fewer than 5 trades. "
            "Treat their numbers as anecdotes, not evidence."
        )

# ─── Section 7: LLM cost monitor ──────────────────────────────────────

st.divider()
st.header("7. How much are LLM calls actually costing you?")
st.markdown(
    "Per-day LLM spend across the PEAD-LLM AMC + BMO batches (the only "
    "production LLM consumers right now — the on-demand AI synthesis "
    "above adds ~$0.05 per click on top, not tracked here). "
    "**Watchdog alerts at >$20/day.**"
)


@st.cache_data(ttl=300, show_spinner=False)
def _load_llm_cost_history(days: int = 60) -> pd.DataFrame:
    """Daily sum of cost_estimate_usd from earnings_llm_decisions."""
    try:
        import duckdb
    except ImportError:
        return pd.DataFrame()
    try:
        con = duckdb.connect("research_data/earnings_data.duckdb", read_only=True)
        try:
            df = con.execute(
                """
                SELECT CAST(analyzed_at AS DATE) AS day,
                       SUM(cost_estimate_usd) AS daily_cost,
                       COUNT(*) AS n_analyses,
                       SUM(CASE WHEN llm_decision IS NULL THEN 1 ELSE 0 END) AS n_errors
                FROM earnings_llm_decisions
                WHERE analyzed_at IS NOT NULL
                GROUP BY day
                ORDER BY day
                """
            ).df()
        finally:
            con.close()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df["day"] = pd.to_datetime(df["day"])
    return df.tail(days)


cost_df = _load_llm_cost_history()
if cost_df.empty:
    st.info(
        "No PEAD-LLM analyses on record yet. Cost panel populates after "
        "the first AMC batch fires (Sun 17:30 CDT or Mon 06:30 BMO)."
    )
else:
    # Three KPIs
    c1, c2, c3 = st.columns(3)
    today_cost = float(cost_df.iloc[-1]["daily_cost"]) if len(cost_df) else 0.0
    last_7d = float(cost_df.tail(7)["daily_cost"].sum())
    monthly_run_rate = float(cost_df.tail(30)["daily_cost"].mean()) * 30 if len(cost_df) >= 7 else None
    c1.metric("Most recent day", f"${today_cost:.2f}")
    c2.metric("Last 7 days", f"${last_7d:.2f}")
    if monthly_run_rate is not None:
        c3.metric("Monthly run-rate", f"${monthly_run_rate:.0f}")
    else:
        c3.metric("Monthly run-rate", "—")

    # Sparkline
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Bar(
        x=cost_df["day"], y=cost_df["daily_cost"],
        name="Daily cost (USD)",
        marker_color=["#d32f2f" if c > 20 else "#0097a7" for c in cost_df["daily_cost"]],
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<br>%{customdata[0]} analyses<br>%{customdata[1]} errors<extra></extra>",
        customdata=cost_df[["n_analyses", "n_errors"]].values,
    ))
    fig_cost.add_hline(y=20, line_dash="dot", line_color="red",
                       annotation_text="$20/day watchdog threshold")
    fig_cost.update_layout(
        xaxis_title="Date",
        yaxis_title="USD",
        height=300,
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    # Audit table
    with st.expander("Daily breakdown"):
        display_cost = cost_df.copy()
        display_cost["daily_cost"] = display_cost["daily_cost"].round(3)
        display_cost.columns = ["Day", "Cost (USD)", "Analyses", "Errors"]
        st.dataframe(display_cost, use_container_width=True, hide_index=True)

    # Cost-overrun warning surfaced inline (the watchdog handles it via
    # alert; this surfaces it visually for someone looking at the page).
    overruns = cost_df[cost_df["daily_cost"] > 20]
    if not overruns.empty:
        st.error(
            f"**Cost overrun**: {len(overruns)} day(s) above $20/day threshold "
            f"in the visible window. Most recent: {overruns.iloc[-1]['day'].date()} "
            f"at ${overruns.iloc[-1]['daily_cost']:.2f}. "
            "Investigate model selection or candidate-count spike."
        )


# ─── AI synthesis dispatch (rendered into the placeholder at top) ─────

if ai_clicked:
    prompt = _build_synthesis_prompt(
        selected_variants=selected,
        coverage_days=coverage,
        full_corr=full_corr.round(2),
        recent_corr=_ai_capture["recent_corr"],
        sizing_table=_ai_capture["sizing_table"],
        portfolio_sigma_annual=_ai_capture["portfolio_sigma_annual"],
        regime_sharpe_pivot=_ai_capture["regime_sharpe_pivot"],
        pattern_summary=_ai_capture["pattern_summary"],
    )
    prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    with ai_placeholder.container(border=True):
        with st.spinner("gpt-5.4-pro is reading the data..."):
            try:
                result = _ai_synthesize(prompt_hash, prompt)
            except Exception as exc:
                st.error(f"AI synthesis failed: {exc}")
                result = None
        if result:
            st.markdown(f"**AI synthesis** — `{result['model']}` · "
                        f"generated {result['generated_at']} · "
                        f"took {result['duration_s']}s · "
                        f"~${result['cost_estimate_usd']:.2f}")
            st.markdown(result["text"])
            with st.expander("View prompt sent to model (for audit)"):
                st.code(prompt, language="markdown")

# ─── Footer ───────────────────────────────────────────────────────────

st.divider()
with st.expander("Caveats and known limitations"):
    st.markdown(
        """
        - **Sample size**: with under 30 days of snapshots per strategy,
          correlations have wide confidence intervals. Take signs and
          rough magnitudes; ignore precision.
        - **A new strategy** (e.g., `pead_llm` if added recently) won't
          have full-window data and gets dropped from paired comparisons.
          The coverage line at the top shows each strategy's data count.
        - **PEAD's daily P&L** comes from the live Alpaca PEAD account;
          the LLM-gated dryrun arm has no broker P&L until promoted to
          live, so its returns are zero-noise during the A/B forward test.
        - **Risk-parity weights** shown are first-order inverse-volatility
          approximations. The full iterative solution accounting for
          correlations gives slightly different numbers but is in the
          same ballpark for sanity-check purposes.
        - **No transaction-cost model**. Re-balancing toward the suggested
          weights would itself cost slippage, not modeled here.
        - **Regime classification** uses SPY-based daily rules (Minervini's
          standard). Different regime definitions (VIX-based, breadth-based,
          factor-rotation-based) would give different splits.
        """
    )
