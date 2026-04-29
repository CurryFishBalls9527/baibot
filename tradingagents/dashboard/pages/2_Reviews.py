"""Trade Reviews — cross-variant overview + daily/weekly post-mortems.

Top of page: multi-variant overview (per-variant cards, combined
metrics, 30-day normalized equity strip). Below: drill-down into
individual reviews — daily per-trade post-mortems or weekly strategy
reviews — read from `results/daily_reviews/` and
`results/weekly_reviews/`.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from tradingagents.dashboard.multi_variant import VARIANT_COLORS, get_variant_dbs
from tradingagents.storage.database import TradingDatabase


st.set_page_config(page_title="Reviews", layout="wide")
st.title("Reviews")

RESULTS = Path("results")


# ── Helpers ──────────────────────────────────────────────────────

def _iso_week(d: date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


@st.cache_data(ttl=30)
def _read_md(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


@st.cache_data(ttl=30)
def _read_html(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def _daily_dir(d: date, include_dry_run: bool) -> Path:
    live = RESULTS / "daily_reviews" / d.isoformat()
    if live.exists() and any(live.iterdir()):
        return live
    if include_dry_run:
        dry = RESULTS / "daily_reviews_dryrun" / d.isoformat()
        if dry.exists():
            return dry
    return live


def _weekly_dir(d: date, include_dry_run: bool) -> Path:
    iso = _iso_week(d)
    live = RESULTS / "weekly_reviews" / iso
    if live.exists() and any(live.iterdir()):
        return live
    if include_dry_run:
        dry = RESULTS / "weekly_reviews_dryrun" / iso
        if dry.exists():
            return dry
    return live


# ── Cross-variant overview ───────────────────────────────────────


@st.cache_data(ttl=30, show_spinner=False)
def _today_outcomes(db_path: str, today_iso: str) -> pd.DataFrame:
    db = TradingDatabase(db_path)
    rows = db.get_trade_outcomes_in_range(today_iso, today_iso)
    db.close()
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data(ttl=30, show_spinner=False)
def _recent_snapshots(db_path: str, days: int, today_iso: str) -> pd.DataFrame:
    db = TradingDatabase(db_path)
    start = (date.fromisoformat(today_iso) - timedelta(days=days)).isoformat()
    rows = db.get_snapshots_in_range(start, today_iso)
    db.close()
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_variant_card(variant: str, db_path: str, today_iso: str) -> dict:
    snaps = _recent_snapshots(db_path, 30, today_iso)
    outcomes = _today_outcomes(db_path, today_iso)
    card = {
        "variant": variant, "trades_closed": 0, "wins": 0, "losses": 0,
        "avg_return": None, "avg_mfe": None, "mfe_capture": None,
        "daily_pl": None, "daily_pl_pct": None, "equity": None,
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
            if not mfes.empty and mfes.mean() != 0:
                card["avg_mfe"] = float(mfes.mean())
                if card["avg_return"] is not None and card["avg_mfe"] > 0:
                    card["mfe_capture"] = card["avg_return"] / card["avg_mfe"]
    return card


def _render_variant_card(card: dict) -> None:
    variant = card["variant"]
    pl_color = "#26a69a" if (card["daily_pl"] or 0) >= 0 else "#ef5350"
    equity_str = f"${card['equity']:,.0f}" if card["equity"] else "—"
    pl_pct = card["daily_pl_pct"]
    pl_str = (
        f"{card['daily_pl']:+,.0f}  ({pl_pct:+.2%})"
        if pl_pct is not None and card["daily_pl"] is not None else "—"
    )
    if card["trades_closed"] > 0:
        activity_line = (
            f"<b>{card['trades_closed']}</b> closed today · "
            f"<span style='color:#26a69a'>{card['wins']}W</span> / "
            f"<span style='color:#ef5350'>{card['losses']}L</span>"
        )
        avg_ret_str = f"{card['avg_return']*100:+.2f}%" if card['avg_return'] is not None else "—"
        mfe_str = f"{card['mfe_capture']:.2f}" if card['mfe_capture'] is not None else "—"
        metrics_line = f"avg {avg_ret_str} · MFE capture {mfe_str}"
    else:
        activity_line = "<span style='color:#777;'>No trades closed today</span>"
        metrics_line = (
            "<span style='color:#666;font-size:11px;'>"
            "(positions may be open — reviews only fire on exits)</span>"
        )
    st.markdown(
        f"<div style='border:1px solid #333;border-radius:6px;padding:10px 12px;"
        f"background:#101418;'>"
        f"<div style='font-size:12px;color:#888;text-transform:uppercase;"
        f"letter-spacing:.05em;'>{variant}</div>"
        f"<div style='font-size:20px;font-weight:600;margin-top:2px;'>{equity_str}</div>"
        f"<div style='color:{pl_color};font-size:13px;margin-top:2px;'>{pl_str}</div>"
        f"<div style='font-size:12px;color:#aaa;margin-top:8px;'>{activity_line}</div>"
        f"<div style='font-size:12px;color:#aaa;margin-top:4px;'>{metrics_line}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _normalized_equity_fig(variant_paths: Dict[str, str], today_iso: str):
    frames: List[pd.DataFrame] = []
    for variant, db_path in variant_paths.items():
        snaps = _recent_snapshots(db_path, 30, today_iso)
        if snaps.empty:
            continue
        snaps = snaps.copy()
        base = float(snaps.iloc[0]["equity"])
        if base <= 0:
            continue
        snaps["variant"] = variant
        snaps["equity_norm"] = snaps["equity"] / base * 100.0
        frames.append(snaps[["date", "variant", "equity_norm"]])
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    fig = go.Figure()
    for variant in variant_paths.keys():
        sub = combined[combined["variant"] == variant]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["equity_norm"], mode="lines", name=variant,
            line=dict(width=1.5, color=VARIANT_COLORS.get(variant)),
        ))
    fig.add_hline(
        y=100, line=dict(color="#555", width=1, dash="dot"),
        annotation_text="launch", annotation_position="right",
    )
    fig.update_layout(
        template="plotly_dark", height=320,
        margin=dict(l=40, r=20, t=10, b=30),
        xaxis_title="", yaxis_title="Equity (norm = 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


_today_iso = date.today().isoformat()
_dbs = get_variant_dbs()
_variant_paths: Dict[str, str] = {
    v: getattr(db, "db_path", "") for v, db in _dbs.items()
    if getattr(db, "db_path", "")
}
if _variant_paths:
    cards = [
        _build_variant_card(v, p, _today_iso) for v, p in _variant_paths.items()
    ]
    card_cols = st.columns(len(cards))
    for col, card in zip(card_cols, cards):
        with col:
            _render_variant_card(card)

    total_pl = sum((c["daily_pl"] or 0) for c in cards)
    total_equity = sum((c["equity"] or 0) for c in cards)
    total_closed = sum(c["trades_closed"] for c in cards)
    total_wins = sum(c["wins"] for c in cards)
    total_losses = sum(c["losses"] for c in cards)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total equity", f"${total_equity:,.0f}")
    m2.metric("Today P&L", f"${total_pl:+,.0f}")
    m3.metric("Trades closed today", f"{total_closed}")
    m4.metric("W / L", f"{total_wins} / {total_losses}")

    norm_fig = _normalized_equity_fig(_variant_paths, _today_iso)
    if norm_fig is not None:
        st.markdown(
            "<div style='color:#9ca3af;font-size:0.72rem;letter-spacing:0.22em;"
            "font-weight:600;margin:0.6rem 0 0.4rem;'>"
            "LAST 30 DAYS — NORMALIZED EQUITY</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(norm_fig, use_container_width=True,
                        config={"displayModeBar": False})

st.divider()


# ── Sidebar ──────────────────────────────────────────────────────

mode = st.sidebar.radio("Mode", ["Daily per-trade", "Weekly strategy"], index=0)
today = date.today()

# Hide dry-run output by default; show as opt-in.
include_dry = st.sidebar.checkbox("Include dry-run outputs", value=False)

if mode == "Daily per-trade":
    chosen_date = st.sidebar.date_input("Date", value=today, max_value=today)
    directory = _daily_dir(chosen_date, include_dry)
    st.caption(f"Directory: `{directory}` · exists={directory.exists()}")
    if not directory.exists():
        st.info("No reviews found for this date.")
        st.stop()

    # Discover variants from md file names of shape "{variant}_{symbol}.md".
    md_files = sorted(directory.glob("*.md"))
    # Exclude per-day summaries.
    md_files = [p for p in md_files if not p.name.endswith("_summary.md")]
    # Partition: *_HELD.md are the daily health checks for still-open
    # positions; everything else is a closed-trade review.
    held_files = [p for p in md_files if p.stem.endswith("_HELD")]
    closed_files = [p for p in md_files if not p.stem.endswith("_HELD")]
    # Group by variant prefix (shared logic for both partitions).
    # Order matters: longer-prefix variants must come before shorter ones so
    # `chan_daily_*.md` matches "chan_daily" before "chan", and similarly for
    # `mechanical_v2_*.md` vs "mechanical".
    KNOWN = [
        "mechanical_v2", "chan_daily", "chan_v2", "mechanical", "chan",
        "llm", "intraday_mechanical",
    ]

    def _variant_of(fname: str) -> str:
        for v in KNOWN:
            if fname.startswith(f"{v}_"):
                return v
        return fname.split("_", 1)[0]

    by_variant: dict[str, list[Path]] = {}
    held_by_variant: dict[str, list[Path]] = {}
    for p in closed_files:
        by_variant.setdefault(_variant_of(p.name), []).append(p)
    for p in held_files:
        held_by_variant.setdefault(_variant_of(p.name), []).append(p)

    # Union variant set — a variant might have HELD files but no closed
    # trades, or vice versa.
    variants = sorted(set(by_variant.keys()) | set(held_by_variant.keys()))
    if not variants:
        st.info("No variant files found.")
        st.stop()

    chosen_variant = st.sidebar.selectbox("Variant", variants)
    files = sorted(by_variant.get(chosen_variant, []))
    held = sorted(held_by_variant.get(chosen_variant, []))

    # Summary banner.
    summary_path = directory / f"{chosen_variant}_summary.md"
    if summary_path.exists():
        with st.expander("Daily summary", expanded=True):
            st.markdown(_read_md(str(summary_path)))

    st.subheader(
        f"{chosen_variant} — {chosen_date} — "
        f"{len(files)} closed · {len(held)} held"
    )

    if files:
        st.markdown("### Closed today")
        for md_path in files:
            stem = md_path.stem
            if stem.startswith(f"{chosen_variant}_"):
                symbol = stem[len(chosen_variant) + 1:]
            else:
                symbol = stem
            md = _read_md(str(md_path))
            headline = symbol
            for line in md.splitlines():
                if line.startswith("# "):
                    headline = line[2:].strip()
                    break
            with st.expander(headline, expanded=False):
                chart_path = directory / "charts" / f"{chosen_variant}_{symbol}.html"
                if chart_path.exists():
                    html = _read_html(str(chart_path))
                    components.html(html, height=620, scrolling=True)
                else:
                    st.caption("No chart available for this trade.")
                st.markdown(md)

    if held:
        st.markdown("### Held positions (open)")
        _health_colors = {
            "HEALTHY": "#66bb6a",
            "WATCH": "#ffa726",
            "WARNING": "#ef5350",
        }
        for md_path in held:
            stem = md_path.stem  # ends with "_HELD"
            inner = stem[: -len("_HELD")]
            if inner.startswith(f"{chosen_variant}_"):
                symbol = inner[len(chosen_variant) + 1:]
            else:
                symbol = inner
            md = _read_md(str(md_path))

            # Extract `### Health: X` line for the badge.
            health = None
            for line in md.splitlines():
                if line.startswith("### Health:"):
                    health = line.split(":", 1)[1].strip()
                    break
            headline = symbol
            for line in md.splitlines():
                if line.startswith("# "):
                    headline = line[2:].strip()
                    break

            header_parts = [headline]
            if health:
                color = _health_colors.get(health.upper(), "#888")
                header_parts.append(
                    f":material/circle: {health}"
                )  # streamlit ignores unknown icons but keeps text
            with st.expander(" · ".join(header_parts), expanded=False):
                if health:
                    color = _health_colors.get(health.upper(), "#888")
                    st.markdown(
                        f"<span style='background:{color};color:#000;"
                        f"padding:2px 10px;border-radius:4px;font-weight:600;"
                        f"font-size:12px;'>{health}</span>",
                        unsafe_allow_html=True,
                    )
                st.markdown(md)

    if not files and not held:
        st.info("No trades or held-position reviews for this variant on this date.")

elif mode == "Weekly strategy":
    chosen_date = st.sidebar.date_input("Week of", value=today, max_value=today)
    directory = _weekly_dir(chosen_date, include_dry)
    iso = _iso_week(chosen_date)
    st.caption(f"ISO week: `{iso}` · directory: `{directory}` · exists={directory.exists()}")
    if not directory.exists():
        st.info("No weekly reviews found for this week.")
        st.stop()

    # Variants = md files minus the index.
    md_files = sorted(p for p in directory.glob("*.md") if p.name != "_index.md")
    if not md_files:
        st.info("No per-variant reviews written for this week.")
        st.stop()

    variants = [p.stem for p in md_files]
    chosen_variant = st.sidebar.selectbox("Variant", variants)
    target = directory / f"{chosen_variant}.md"
    if not target.exists():
        st.error("File missing.")
        st.stop()

    # Index banner if present.
    idx = directory / "_index.md"
    if idx.exists():
        with st.expander("Week index", expanded=False):
            st.markdown(_read_md(str(idx)))

    st.subheader(f"{chosen_variant} — {iso}")

    # Red-team verdict badge — parsed from the final line of the markdown.
    # Format: "**RED-TEAM VERDICT: <support | partial-support | reject-with-reason>**"
    md_body = _read_md(str(target))
    verdict_color = {"support": "#66bb6a",
                     "partial-support": "#ffa726",
                     "reject": "#ef5350"}
    verdict = None
    for line in md_body.splitlines():
        if "RED-TEAM VERDICT" in line.upper():
            verdict = line.split(":", 1)[-1].strip().strip("*").strip()
            break
    if verdict:
        key = verdict.split("-with")[0].strip().lower()  # "reject-with-reason" → "reject"
        key = key.split()[0] if " " in key else key
        color = verdict_color.get(key, "#888")
        st.markdown(
            f"<div style='display:inline-block;background:{color};color:#000;"
            f"padding:4px 12px;border-radius:6px;font-weight:600;font-size:13px;"
            f"margin-bottom:12px;'>"
            f"RED-TEAM VERDICT: {verdict}</div>",
            unsafe_allow_html=True,
        )

    st.markdown(md_body)
