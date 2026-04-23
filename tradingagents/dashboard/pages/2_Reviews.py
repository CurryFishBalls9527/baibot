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
    # Partition: *_HELD.md are the daily health checks for still-open
    # positions; everything else is a closed-trade review.
    held_files = [p for p in md_files if p.stem.endswith("_HELD")]
    closed_files = [p for p in md_files if not p.stem.endswith("_HELD")]
    # Group by variant prefix (shared logic for both partitions).
    KNOWN = [
        "mechanical_v2", "chan_v2", "mechanical", "chan",
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
