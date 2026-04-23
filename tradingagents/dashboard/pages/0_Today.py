"""Today — one-pane overview of all variants' closed trades and live equity.

Shows for each variant:
  - Today's daily P&L (equity delta from snapshots)
  - Closed-trade count (wins/losses)
  - Avg return per trade
  - MFE-capture ratio (avg_return / avg_MFE) — how much of the favorable
    move did we actually book? Low values mean we're cutting early or the
    stop is too tight.
  - Link to the Reviews page filtered to today + variant
Plus:
  - Combined P&L across all variants (portfolio lens)
  - Cross-variant equity strip chart (last 30 days, normalized)
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tradingagents.dashboard.multi_variant import VARIANT_COLORS
from tradingagents.storage.database import TradingDatabase


def _fresh_dbs() -> dict:
    """Open fresh TradingDatabase instances. See pages/4_Proposals.py for why
    we avoid the cached `get_variant_dbs` helper here — newly-added helper
    methods may be missing on the cached instances after a code edit."""
    from tradingagents.testing.ab_config import load_experiment
    import os
    yaml_path = os.getenv("EXPERIMENT_CONFIG_PATH", "experiments/paper_launch_v2.yaml")
    if not Path(yaml_path).exists():
        return {}
    exp = load_experiment(yaml_path)
    return {
        v.name: TradingDatabase(v.db_path)
        for v in exp.variants
        if v.db_path and Path(v.db_path).exists()
    }


st.set_page_config(page_title="Today — TradingBot", layout="wide")
st.title(f"Today · {date.today().isoformat()}")

dbs = _fresh_dbs()
if not dbs:
    st.warning("No variant databases found.")
    st.stop()

today_iso = date.today().isoformat()


@st.cache_data(ttl=30)
def _today_outcomes(db_path: str) -> pd.DataFrame:
    """Load today's closed trade outcomes for a variant DB."""
    from tradingagents.storage.database import TradingDatabase
    db = TradingDatabase(db_path)
    rows = db.get_trade_outcomes_in_range(today_iso, today_iso)
    db.close()
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data(ttl=30)
def _recent_snapshots(db_path: str, days: int = 30) -> pd.DataFrame:
    from tradingagents.storage.database import TradingDatabase
    db = TradingDatabase(db_path)
    start = (date.today() - timedelta(days=days)).isoformat()
    rows = db.get_snapshots_in_range(start, today_iso)
    db.close()
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _variant_card(variant: str, db) -> dict:
    """Compute today's stats for one variant."""
    outcomes = _today_outcomes(db.db_path if hasattr(db, "db_path") else str(db))
    # Fall back to directly accessing conn when cached function receives db obj
    # (it'll receive a path in practice via get_variant_dbs keys → .db_path).
    snaps = _recent_snapshots(db.db_path if hasattr(db, "db_path") else str(db))

    card = {
        "variant": variant,
        "trades_closed": 0,
        "wins": 0, "losses": 0,
        "avg_return": None,
        "avg_mfe": None,
        "mfe_capture": None,
        "daily_pl": None,
        "daily_pl_pct": None,
        "equity": None,
    }

    if not snaps.empty:
        last = snaps.iloc[-1]
        card["equity"] = float(last.get("equity") or 0)
        card["daily_pl"] = float(last.get("daily_pl") or 0)
        card["daily_pl_pct"] = float(last.get("daily_pl_pct") or 0)

    if not outcomes.empty and "return_pct" in outcomes.columns:
        rets = outcomes["return_pct"].dropna()
        card["trades_closed"] = int(len(outcomes))
        card["wins"] = int((rets > 0).sum())
        card["losses"] = int((rets < 0).sum())
        card["avg_return"] = float(rets.mean()) if not rets.empty else None
        if "max_favorable_excursion" in outcomes.columns:
            mfes = outcomes["max_favorable_excursion"].dropna()
            if not mfes.empty:
                card["avg_mfe"] = float(mfes.mean())
                if card["avg_mfe"] and card["avg_mfe"] > 0 and card["avg_return"] is not None:
                    card["mfe_capture"] = card["avg_return"] / card["avg_mfe"]
    return card


# ───── Per-variant cards ─────────────────────────────────────────────

cols = st.columns(len(dbs))
cards = []
for (variant, db), col in zip(dbs.items(), cols):
    # get_variant_dbs returns TradingDatabase instances; our cached function
    # needs a path argument (not a DB handle) to hash properly.
    db_path = getattr(db, "db_path", None) or ""
    if not db_path:
        # Fallback: use the DB's underlying conn str (best-effort).
        db_path = db.conn.execute("PRAGMA database_list").fetchone()[2]
    outcomes_df = _today_outcomes(db_path)
    snaps_df = _recent_snapshots(db_path)

    card = {"variant": variant, "trades_closed": 0, "wins": 0, "losses": 0,
            "avg_return": None, "avg_mfe": None, "mfe_capture": None,
            "daily_pl": None, "daily_pl_pct": None, "equity": None}
    if not snaps_df.empty:
        last = snaps_df.iloc[-1]
        card["equity"] = float(last.get("equity") or 0)
        card["daily_pl"] = float(last.get("daily_pl") or 0)
        card["daily_pl_pct"] = float(last.get("daily_pl_pct") or 0)
    if not outcomes_df.empty and "return_pct" in outcomes_df.columns:
        rets = outcomes_df["return_pct"].dropna()
        card["trades_closed"] = int(len(outcomes_df))
        card["wins"] = int((rets > 0).sum())
        card["losses"] = int((rets < 0).sum())
        card["avg_return"] = float(rets.mean()) if not rets.empty else None
        if "max_favorable_excursion" in outcomes_df.columns:
            mfes = outcomes_df["max_favorable_excursion"].dropna()
            if not mfes.empty and mfes.mean() != 0:
                card["avg_mfe"] = float(mfes.mean())
                if card["avg_return"] is not None and card["avg_mfe"] > 0:
                    card["mfe_capture"] = card["avg_return"] / card["avg_mfe"]
    cards.append(card)

    with col:
        pl_color = "#26a69a" if (card["daily_pl"] or 0) >= 0 else "#ef5350"
        equity_str = f"${card['equity']:,.0f}" if card["equity"] else "—"
        pl_pct = card["daily_pl_pct"]
        pl_str = (
            f"{card['daily_pl']:+,.0f}  ({pl_pct:+.2%})" if pl_pct is not None and card["daily_pl"] is not None
            else "—"
        )
        st.markdown(
            f"<div style='border:1px solid #333;border-radius:6px;padding:10px 12px;"
            f"background:#101418;'>"
            f"<div style='font-size:12px;color:#888;text-transform:uppercase;letter-spacing:.05em;'>{variant}</div>"
            f"<div style='font-size:20px;font-weight:600;margin-top:2px;'>{equity_str}</div>"
            f"<div style='color:{pl_color};font-size:13px;margin-top:2px;'>{pl_str}</div>"
            f"<div style='font-size:12px;color:#aaa;margin-top:8px;'>"
            f"<b>{card['trades_closed']}</b> closed today · "
            f"<span style='color:#26a69a'>{card['wins']}W</span> / "
            f"<span style='color:#ef5350'>{card['losses']}L</span>"
            f"</div>"
            f"<div style='font-size:12px;color:#aaa;margin-top:4px;'>"
            f"avg {'%+.2f%%' % (card['avg_return']*100) if card['avg_return'] is not None else '—'} · "
            f"MFE capture {'%.2f' % card['mfe_capture'] if card['mfe_capture'] is not None else '—'}"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if card["trades_closed"] > 0:
            st.caption(f"[See reviews →](/Reviews) · pick variant `{variant}`")


# ───── Combined portfolio strip ─────────────────────────────────────

st.divider()
st.subheader("Combined")
total_pl = sum((c["daily_pl"] or 0) for c in cards)
total_equity = sum((c["equity"] or 0) for c in cards)
total_closed = sum(c["trades_closed"] for c in cards)
total_wins = sum(c["wins"] for c in cards)
total_losses = sum(c["losses"] for c in cards)

ec1, ec2, ec3, ec4 = st.columns(4)
ec1.metric("Total equity", f"${total_equity:,.0f}")
ec2.metric(
    "Today P&L",
    f"${total_pl:+,.0f}",
    delta_color="normal",
)
ec3.metric("Trades closed today", f"{total_closed}")
ec4.metric("W/L", f"{total_wins} / {total_losses}")


# ───── Cross-variant equity strip (normalized to 100) ───────────────

st.divider()
st.subheader("Last 30 days — normalized equity")

all_snaps: list[pd.DataFrame] = []
for variant, db in dbs.items():
    db_path = getattr(db, "db_path", "")
    snaps = _recent_snapshots(db_path)
    if snaps.empty:
        continue
    snaps = snaps.copy()
    snaps["variant"] = variant
    # Normalize to 100 at first point in window.
    base = float(snaps.iloc[0]["equity"])
    if base > 0:
        snaps["equity_norm"] = snaps["equity"] / base * 100.0
        all_snaps.append(snaps[["date", "variant", "equity_norm"]])

if all_snaps:
    combined = pd.concat(all_snaps, ignore_index=True)
    fig = go.Figure()
    for variant in dbs.keys():
        sub = combined[combined["variant"] == variant]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["date"], y=sub["equity_norm"],
                mode="lines", name=variant,
                line=dict(width=1.5, color=VARIANT_COLORS.get(variant)),
            )
        )
    fig.add_hline(
        y=100, line=dict(color="#555", width=1, dash="dot"),
        annotation_text="launch", annotation_position="right",
    )
    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=40, r=20, t=10, b=30),
        xaxis_title="",
        yaxis_title="Equity (norm = 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No daily snapshots yet for any variant.")
