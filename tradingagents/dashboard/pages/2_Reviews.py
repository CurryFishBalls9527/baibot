"""Trade Reviews — browse daily per-trade post-mortems + weekly strategy reviews.

Reads markdown + chart HTML from `results/daily_reviews/` and
`results/weekly_reviews/` — both produced by the automation cron jobs.
Dry-run variants from `*_dryrun/` dirs are surfaced too via a toggle.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Trade Reviews", layout="wide")
st.title("Trade Reviews")

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
    # Group by variant prefix.
    by_variant: dict[str, list[Path]] = {}
    for p in md_files:
        # Expect VARIANT_SYMBOL.md; split on the LAST underscore would misparse
        # variants that contain underscores (e.g. mechanical_v2). Prefer a
        # known variant-prefix match.
        KNOWN = [
            "mechanical_v2", "chan_v2", "mechanical", "chan",
            "llm", "intraday_mechanical",
        ]
        matched = None
        for v in KNOWN:
            if p.name.startswith(f"{v}_"):
                matched = v
                break
        if matched is None:
            # Fallback: first token.
            matched = p.name.split("_", 1)[0]
        by_variant.setdefault(matched, []).append(p)

    variants = sorted(by_variant.keys())
    if not variants:
        st.info("No variant files found.")
        st.stop()

    chosen_variant = st.sidebar.selectbox("Variant", variants)
    files = sorted(by_variant[chosen_variant])

    # Summary banner.
    summary_path = directory / f"{chosen_variant}_summary.md"
    if summary_path.exists():
        with st.expander("Daily summary", expanded=True):
            st.markdown(_read_md(str(summary_path)))

    if not files:
        st.info("No per-trade reviews for this variant on this date.")
        st.stop()

    st.subheader(f"{chosen_variant} — {chosen_date} — {len(files)} trades")
    for md_path in files:
        # symbol = filename after the variant prefix, minus .md
        stem = md_path.stem
        if stem.startswith(f"{chosen_variant}_"):
            symbol = stem[len(chosen_variant) + 1 :]
        else:
            symbol = stem
        md = _read_md(str(md_path))
        # Pull out the top-line headline (first H1) so the expander is skimmable.
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
