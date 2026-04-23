"""Shared daily-trade-review module.

Driven by `Orchestrator.run_daily_trade_review`, `ChanOrchestrator.run_daily_trade_review`,
and `IntradayOrchestrator.run_daily_trade_review`. For each trade closed
on the review date, fetches price bars + structured setup context, builds
a Plotly chart, asks an LLM (gpt-4o-mini) to write a teaching-oriented
markdown post-mortem, and writes both to `results/daily_reviews/`.

Idempotent and fails soft — no single trade's failure can block the
others. Kill-switched via `daily_trade_review_enabled` config flag
(default False).
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ───────────────────────── variant classification ─────────────────────────

def _is_minervini(variant: str) -> bool:
    return variant in {"mechanical", "llm", "mechanical_v2"}


def _is_chan(variant: str) -> bool:
    return variant in {"chan", "chan_v2"}


def _is_intraday(variant: str) -> bool:
    return variant.startswith("intraday")


def _timeframe_for_variant(variant: str) -> str:
    if _is_intraday(variant):
        return "15m"
    if _is_chan(variant):
        return "30m"
    return "1d"


# ───────────────────────── data fetching ─────────────────────────

def _fetch_bars(broker, symbol: str, variant: str, entry_dt: datetime, exit_dt: datetime):
    """Return an OHLCV DataFrame for the symbol's chart window.

    Window: 20 bars before entry for daily, 2 sessions before for intraday,
    extending to exit + 1 day. Uses IEX feed — paper-account credentials
    aren't authorized for recent SIP data, and IEX is authoritative enough
    for review-chart purposes.
    """
    import pandas as pd
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tf_str = _timeframe_for_variant(variant)
    if tf_str == "1d":
        tf = TimeFrame.Day
        padding_start = timedelta(days=30)
        padding_end = timedelta(days=2)
    elif tf_str == "30m":
        tf = TimeFrame(30, TimeFrameUnit.Minute)
        # Chan structures are computed over ~90 days so the chart window
        # must match or the BI/ZS/BSP timestamps fall outside the
        # visible range and silently don't render.
        padding_start = timedelta(days=90)
        padding_end = timedelta(days=1)
    else:  # 15m intraday
        tf = TimeFrame(15, TimeFrameUnit.Minute)
        padding_start = timedelta(days=2)
        padding_end = timedelta(days=1)

    data_client = getattr(broker, "data_client", None)
    if data_client is None:
        return pd.DataFrame()
    try:
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=tf,
            start=entry_dt - padding_start,
            end=exit_dt + padding_end,
            feed="iex",
        )
        bars = data_client.get_stock_bars(req)
    except Exception as e:
        logger.warning("trade_review._fetch_bars(%s) failed: %s", symbol, e)
        return pd.DataFrame()
    series = getattr(bars, "data", {}).get(symbol) or []
    if not series:
        return pd.DataFrame()
    rows = [
        {
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": float(b.volume),
            "ts": pd.Timestamp(b.timestamp),
        }
        for b in series
    ]
    df = pd.DataFrame(rows).set_index("ts").sort_index()
    return df


def _latest_setup_candidate(db, symbol: str, entry_date: str) -> Optional[dict]:
    """Most recent setup_candidates row for the symbol on or before entry_date."""
    try:
        row = db.conn.execute(
            """SELECT * FROM setup_candidates
               WHERE symbol = ? AND screen_date <= ?
               ORDER BY screen_date DESC LIMIT 1""",
            (symbol, entry_date),
        ).fetchone()
    except Exception:
        return None
    return dict(row) if row else None


def _entry_signal_metadata(db, symbol: str, entry_date: str) -> Optional[str]:
    """Raw signal_metadata JSON string for the entry signal (NULL-tolerant)."""
    try:
        row = db.conn.execute(
            """SELECT s.signal_metadata
               FROM signals s
               JOIN trades t ON t.signal_id = s.id
               WHERE t.symbol = ? AND date(t.timestamp) = ? AND t.side = 'buy'
               ORDER BY t.timestamp DESC LIMIT 1""",
            (symbol, entry_date),
        ).fetchone()
    except Exception:
        return None
    return row["signal_metadata"] if row else None


def _trades_for_marker(db, symbol: str, entry_date: str, exit_date: str) -> list[dict]:
    """Buy + sell trade rows in the chart window for marker placement."""
    try:
        rows = db.conn.execute(
            """SELECT timestamp, symbol, side, qty, filled_qty, filled_price, reasoning
               FROM trades
               WHERE symbol = ? AND date(timestamp) BETWEEN ? AND ?
                 AND status LIKE '%filled%'""",
            (symbol, entry_date, exit_date),
        ).fetchall()
    except Exception:
        return []
    return [
        {
            "timestamp": r["timestamp"],
            "symbol": r["symbol"],
            "side": r["side"],
            "qty": r["filled_qty"],
            "filled_price": r["filled_price"],
            "reasoning": r["reasoning"],
        }
        for r in rows
    ]


def _parse_payload(setup_row: Optional[dict]) -> dict:
    """Unpack payload_json blob from a setup_candidates row (or {})."""
    if not setup_row:
        return {}
    payload = setup_row.get("payload_json") or ""
    if not payload:
        return {}
    try:
        return json.loads(payload) or {}
    except Exception:
        return {}


def _minervini_criteria_checklist(setup_row: dict, payload: dict) -> list[tuple[str, str, bool]]:
    """Evaluate classic Minervini-template criteria against captured metrics.

    Returns a list of (criterion, value-as-shown, passes?). Same 8-point
    check Mark Minervini publishes in Trade Like a Stock Market Wizard,
    plus a few volume / accumulation checks that matter for breakouts.
    Passes are tri-state via None for "no data"; for the view we collapse
    to bool just for display.
    """
    sma50 = payload.get("sma_50")
    sma150 = payload.get("sma_150")
    sma200 = payload.get("sma_200")
    price = payload.get("close")
    hi52 = payload.get("52w_high")
    lo52 = payload.get("52w_low")
    rs = payload.get("rs_percentile")
    stage = payload.get("stage_number")
    adx = payload.get("adx_14")
    rsi = payload.get("rsi_14")
    vol_ratio = payload.get("breakout_volume_ratio")
    tmpl_score = payload.get("template_score")
    roc60 = payload.get("roc_60")
    roc120 = payload.get("roc_120")

    def _ok(cond):  # `None`-aware bool coercion
        return bool(cond) if cond is not None else False

    items = []
    if price is not None and sma150 is not None:
        items.append(("Price > 150-DMA", f"${price:.2f} vs ${sma150:.2f}", price > sma150))
    if price is not None and sma200 is not None:
        items.append(("Price > 200-DMA", f"${price:.2f} vs ${sma200:.2f}", price > sma200))
    if sma150 is not None and sma200 is not None:
        items.append(("150-DMA > 200-DMA", f"${sma150:.2f} vs ${sma200:.2f}", sma150 > sma200))
    if sma50 is not None and sma150 is not None and sma200 is not None:
        items.append(("50-DMA > 150 > 200 (stack)", "", sma50 > sma150 > sma200))
    if price is not None and sma50 is not None:
        items.append(("Price > 50-DMA", f"${price:.2f} vs ${sma50:.2f}", price > sma50))
    if price is not None and hi52 is not None:
        dist = (hi52 - price) / hi52
        items.append(("Within 25% of 52w high", f"{dist:.1%} below", dist <= 0.25))
    if price is not None and lo52 is not None and lo52 > 0:
        r = (price - lo52) / lo52
        items.append(("≥ 30% above 52w low", f"{r:.1%}", r >= 0.30))
    if rs is not None:
        items.append(("RS ≥ 70th pct", f"{rs:.1f}", rs >= 70))
    if stage is not None:
        items.append(("Stage 2 (1-3)", f"stage {stage:.0f}", stage <= 3))
    if vol_ratio is not None:
        items.append(("Breakout vol ≥ 1.4×", f"{vol_ratio:.2f}×", vol_ratio >= 1.4))
    if adx is not None:
        items.append(("ADX ≥ 20 (trend)", f"{adx:.1f}", adx >= 20))
    if rsi is not None:
        items.append(("RSI 40-80", f"{rsi:.1f}", 40 <= rsi <= 80))
    if roc60 is not None:
        items.append(("3-mo return ≥ 10%", f"{roc60:+.1%}", roc60 >= 0.10))
    if roc120 is not None:
        items.append(("6-mo return ≥ 25%", f"{roc120:+.1%}", roc120 >= 0.25))
    if tmpl_score is not None:
        items.append(("Template score ≤ 8", f"score {tmpl_score}", tmpl_score <= 8))
    return items


def _build_setup_context(variant: str, setup_row: Optional[dict],
                        signal_metadata_str: Optional[str],
                        entry_signal: Optional[dict] = None,
                        chan_structures: Optional[dict] = None) -> str:
    """Human-readable setup summary for the LLM prompt.

    Much richer than the v1: pulls the full payload_json from
    setup_candidates (SMAs, ROC, ADX, RSI, fundamentals, sector), surfaces
    the untruncated entry-signal reasoning, the Minervini criteria
    checklist, and — for Chan variants — a summary of the Chan
    structures (BI count, ZS/中枢 bounds, active BSP) visible at entry.
    """
    lines = []
    if _is_minervini(variant) and setup_row:
        payload = _parse_payload(setup_row)
        lines.append("Minervini setup (row from setup_candidates):")
        for k in (
            "base_label", "stage_number", "rs_percentile",
            "pivot_price", "buy_point", "buy_limit_price",
            "initial_stop_price", "initial_stop_pct",
            "candidate_status", "breakout_signal",
            "rule_entry_candidate", "rule_watch_candidate",
        ):
            v = setup_row.get(k)
            if v is not None:
                lines.append(f"  - {k}: {v}")
        if payload:
            lines.append("")
            lines.append("Market / fundamentals / technicals at entry screen:")
            for k in (
                "sector", "industry",
                "template_score", "passed_template",
                "sma_50", "sma_150", "sma_200",
                "52w_high", "52w_low",
                "base_depth_pct", "handle_depth_pct",
                "rs_score", "rsi_14", "adx_14", "atr_pct_14",
                "roc_20", "roc_60", "roc_120",
                "breakout_volume_ratio", "close_range_pct",
                "revenue_growth", "eps_growth",
                "revenue_acceleration", "eps_acceleration",
                "return_on_equity",
                "earnings_days_away",
                "market_regime", "market_confirmed_uptrend",
            ):
                v = payload.get(k)
                if v is not None:
                    lines.append(f"  - {k}: {v}")
            criteria = _minervini_criteria_checklist(setup_row, payload)
            if criteria:
                lines.append("")
                lines.append("Minervini template criteria (at entry screen):")
                for name, val, ok in criteria:
                    mark = "✓" if ok else "✗"
                    tail = f" ({val})" if val else ""
                    lines.append(f"  {mark} {name}{tail}")
    if _is_intraday(variant) and signal_metadata_str:
        try:
            meta = json.loads(signal_metadata_str)
        except Exception:
            meta = None
        if meta:
            lines.append("Intraday setup (row from signal_metadata):")
            for k in (
                "setup_family", "opening_range_high", "opening_range_low",
                "vwap", "distance_from_vwap_pct", "volume_ratio",
                "prior_session_high", "prior_session_close",
                "prior_session_is_nr", "candidate_score",
                "breakout_distance_pct",
            ):
                v = meta.get(k)
                if v is not None:
                    lines.append(f"  - {k}: {v}")
    if _is_chan(variant):
        lines.append(
            "Chan setup: entries fire on BSP (buy-structure-point) of type "
            "T1/T2/T2S off a ZS (central symmetry zone). Exits on structural "
            "ZS-low break or opposite BSP."
        )
        if signal_metadata_str:
            try:
                meta = json.loads(signal_metadata_str)
            except Exception:
                meta = {}
            for k in ("t_types", "bsp_reason", "bi_low", "confidence",
                      "regime_at_entry", "market_score"):
                v = meta.get(k)
                if v is not None:
                    lines.append(f"  - {k}: {v}")
        # Chan structures visible at entry — the 中枢 (ZS) + BI + BSP view
        # the trader was actually looking at.
        if chan_structures:
            bi_list = chan_structures.get("bi_list") or []
            zs_list = chan_structures.get("zs_list") or []
            bsp_list = chan_structures.get("bsp_list") or []
            lines.append("")
            lines.append("Chan structures visible at entry (30m):")
            lines.append(f"  - {len(bi_list)} bi, {len(zs_list)} zs (中枢), {len(bsp_list)} bsp")
            if zs_list:
                for i, zs in enumerate(zs_list[-3:], 1):
                    lines.append(
                        f"  - ZS{i}: [{zs.get('begin_time')} → {zs.get('end_time')}] "
                        f"low=${zs.get('low'):.2f} high=${zs.get('high'):.2f}"
                    )
            if bsp_list:
                for bsp in bsp_list[-2:]:
                    side = "BUY" if bsp.get("is_buy") else "SELL"
                    lines.append(
                        f"  - {side} BSP ({bsp.get('types')}) at "
                        f"{bsp.get('time')} · ${bsp.get('price'):.2f}"
                    )
    # Full entry reasoning — untruncated. Critical for "why did we enter?"
    if entry_signal:
        reasoning = (
            entry_signal.get("reasoning") or entry_signal.get("full_analysis") or ""
        )
        if reasoning:
            lines.append("")
            lines.append("Raw entry-signal reasoning (orchestrator log):")
            lines.append(reasoning)
    return "\n".join(lines) if lines else ""


def _extract_chan_structures_safely(
    symbol: str, entry_dt, exit_dt, db_path: str = None,
    history_days: int = 90,
) -> Optional[dict]:
    """Extract Chan structures for the trade review.

    Window: [entry_date - history_days, entry_date]. Deliberately cuts at
    entry so the rendered BI / ZS (中枢) / BSP match what the trader saw
    when they bought — no post-exit hindsight. Chan needs ~60+ bars to
    build meaningful pivots, which is why the lookback is weeks, not days.

    `db_path` is the DuckDB 30m bars file — same one the live chan
    orchestrator uses. Default is the broad intraday DB.
    """
    from datetime import timedelta
    try:
        from tradingagents.dashboard.chan_structures import extract_chan_structures
        path = db_path or "research_data/intraday_30m_broad.duckdb"
        if hasattr(entry_dt, "date"):
            entry_date = entry_dt.date() if not isinstance(entry_dt, date) else entry_dt
        else:
            from dateutil.parser import parse as pd
            entry_date = pd(str(entry_dt)).date()
        begin = (entry_date - timedelta(days=history_days)).strftime("%Y-%m-%d")
        # Cap at entry_date — the rendered structures reflect what was
        # visible at BUY time, not what we know now with the outcome.
        end = entry_date.strftime("%Y-%m-%d")
        return extract_chan_structures(symbol, begin, end, path)
    except Exception as e:
        logger.warning(
            "chan_structures(%s) failed [%s]: %s",
            symbol, type(e).__name__, e,
        )
        return None


# ───────────────────────── main entry ─────────────────────────

def run_daily_review(
    *,
    db,
    broker,
    variant_name: str,
    config: dict,
    review_date: Optional[date] = None,
) -> dict:
    """Review all trades that closed on `review_date` (defaults to today).

    Returns a summary dict: counts, errors, output paths.
    """
    if not config.get("daily_trade_review_enabled", False):
        return {"variant": variant_name, "status": "disabled"}

    from dateutil.parser import parse as parse_date
    import pandas as pd

    review_date = review_date or date.today()
    iso_day = review_date.isoformat()

    dry_run = bool(config.get("daily_trade_review_dry_run", False))
    base_dir = Path("results/daily_reviews_dryrun" if dry_run else "results/daily_reviews")
    out_dir = base_dir / iso_day
    chart_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    budget = int(config.get("daily_review_max_calls", 50))

    # Trades closed today for this variant's DB.
    outcomes = db.get_trade_outcomes_in_range(iso_day, iso_day)
    if not outcomes:
        summary = {
            "variant": variant_name,
            "date": iso_day,
            "closed_trades": 0,
            "analyzed": 0,
            "skipped_budget": 0,
            "errors": 0,
            "dry_run": dry_run,
        }
        logger.info("trade_review[%s]: no closed trades on %s", variant_name, iso_day)
        return summary

    # Lazy-import to keep this module lightweight at import time.
    from tradingagents.dashboard.trade_analyzer import TradeAnalyzer
    from tradingagents.automation.chart_for_review import build_trade_chart

    analyzer = TradeAnalyzer()

    analyzed = 0
    errors = 0
    skipped_budget = 0
    failed_llm = 0
    results = []

    for oc in outcomes:
        if analyzed >= budget:
            skipped_budget += 1
            continue
        sym = oc["symbol"]
        try:
            entry_dt = parse_date(oc["entry_date"])
            exit_dt = parse_date(oc["exit_date"]) + timedelta(days=1)

            setup_row = (
                _latest_setup_candidate(db, sym, oc["entry_date"])
                if _is_minervini(variant_name)
                else None
            )
            signal_metadata = _entry_signal_metadata(db, sym, oc["entry_date"])
            entry_signal = db.get_entry_signal_for_trade(sym, oc["entry_date"])
            trades_marker = _trades_for_marker(
                db, sym, oc["entry_date"], oc["exit_date"]
            )

            bars = _fetch_bars(broker, sym, variant_name, entry_dt, exit_dt)
            chan_struct = None
            if _is_chan(variant_name) and not bars.empty:
                chan_struct = _extract_chan_structures_safely(
                    sym, entry_dt, exit_dt,
                    db_path=config.get(
                        "chan_intraday_db",
                        "research_data/intraday_30m_broad.duckdb",
                    ),
                )

            fig = build_trade_chart(
                symbol=sym,
                variant=variant_name,
                bars=bars,
                outcome=oc,
                trades_for_marker=trades_marker,
                signal_metadata=signal_metadata,
                setup_row=setup_row,
                chan_structures=chan_struct,
            )
            chart_path = chart_dir / f"{variant_name}_{sym}.html"
            if fig is not None:
                fig.write_html(str(chart_path), include_plotlyjs="cdn")
            else:
                chart_path = None

            # LLM post-mortem via enriched outcome dict. Pass the full
            # entry-signal so the prompt can cite its untruncated reasoning
            # rather than the 80-char tooltip.
            oc_enriched = dict(oc)
            oc_enriched["_setup_context"] = _build_setup_context(
                variant_name, setup_row, signal_metadata,
                entry_signal=entry_signal,
                chan_structures=chan_struct,
            )
            oc_enriched["_variant"] = variant_name

            try:
                body = analyzer.analyze_trade(oc_enriched, entry_signal)
            except Exception as e:
                failed_llm += 1
                body = f"*LLM analysis failed: {e}*"
                logger.warning(
                    "trade_review[%s] LLM failed for %s: %s",
                    variant_name, sym, e,
                )

            # Build the markdown file.
            md_path = out_dir / f"{variant_name}_{sym}.md"
            md = _compose_markdown(
                variant_name, oc, entry_signal, setup_row, signal_metadata,
                body, chart_path, chan_structures=chan_struct,
            )
            md_path.write_text(md, encoding="utf-8")

            # Persist to trade_outcomes.trade_analysis (skip in dry-run).
            if not dry_run:
                try:
                    db.update_trade_analysis(oc["id"], body)
                except Exception as e:
                    logger.warning(
                        "trade_review[%s] update_trade_analysis failed %s: %s",
                        variant_name, sym, e,
                    )

            analyzed += 1
            results.append({
                "symbol": sym,
                "return_pct": oc.get("return_pct"),
                "md": str(md_path),
                "chart": str(chart_path) if chart_path else None,
            })
        except Exception as e:
            errors += 1
            logger.error(
                "trade_review[%s] unhandled error for %s: %s",
                variant_name, sym, e, exc_info=True,
            )

    # Per-day summary file.
    try:
        _write_day_summary(out_dir, variant_name, iso_day, outcomes, results)
    except Exception as e:
        logger.warning("trade_review[%s] summary write failed: %s", variant_name, e)

    summary = {
        "variant": variant_name,
        "date": iso_day,
        "closed_trades": len(outcomes),
        "analyzed": analyzed,
        "skipped_budget": skipped_budget,
        "failed_llm": failed_llm,
        "errors": errors,
        "dry_run": dry_run,
        "out_dir": str(out_dir),
    }
    logger.info(
        "trade_review[%s] %s: %d trades, %d analyzed, %d errors",
        variant_name, iso_day, len(outcomes), analyzed, errors,
    )
    return summary


# ───────────────────────── markdown composition ─────────────────────────

def _compose_markdown(
    variant: str,
    outcome: dict,
    entry_signal: Optional[dict],
    setup_row: Optional[dict],
    signal_metadata_str: Optional[str],
    llm_body: str,
    chart_path: Optional[Path],
    chan_structures: Optional[dict] = None,
) -> str:
    sym = outcome.get("symbol", "?")
    ret = outcome.get("return_pct") or 0.0
    win_loss = "WIN" if ret > 0 else ("LOSS" if ret < 0 else "FLAT")
    mfe = outcome.get("max_favorable_excursion")
    mae = outcome.get("max_adverse_excursion")

    header = (
        f"# {sym} ({variant}) — "
        f"{outcome.get('entry_date')} → {outcome.get('exit_date')}  "
        f"[{win_loss}] {ret:+.2%}\n"
    )
    link = ""
    if chart_path is not None:
        link = f"[Interactive chart]({Path('charts') / chart_path.name})\n\n"

    setup_lines = ["## The Setup (why we bought)"]
    payload = _parse_payload(setup_row)

    if _is_minervini(variant) and setup_row:
        setup_lines.append(
            f"- Pattern: **{setup_row.get('base_label') or 'n/a'}** · "
            f"stage {setup_row.get('stage_number')} · "
            f"RS {setup_row.get('rs_percentile')}"
        )
        if setup_row.get("pivot_price"):
            setup_lines.append(
                f"- Pivot **${setup_row.get('pivot_price'):.2f}** · "
                f"Buy point **${setup_row.get('buy_point') or 0:.2f}** · "
                f"Stop **${setup_row.get('initial_stop_price') or 0:.2f}**"
            )
        # Sector + earnings proximity are load-bearing for swing trades.
        sector = payload.get("sector")
        industry = payload.get("industry")
        if sector:
            setup_lines.append(f"- Sector: **{sector}**"
                               + (f" · {industry}" if industry else ""))
        edays = payload.get("earnings_days_away")
        if edays is not None:
            setup_lines.append(
                f"- Earnings: **{edays:.0f} days** away"
                + (" ⚠️ earnings risk" if 0 <= edays <= 15 else "")
            )
    if _is_intraday(variant) and signal_metadata_str:
        try:
            meta = json.loads(signal_metadata_str)
        except Exception:
            meta = {}
        setup_lines.append(f"- Setup family: **{meta.get('setup_family','?')}**")
        if meta.get("opening_range_high") is not None:
            setup_lines.append(
                f"- ORB high/low: **${meta.get('opening_range_high'):.2f}** / "
                f"**${meta.get('opening_range_low'):.2f}**"
            )
        if meta.get("vwap") is not None:
            setup_lines.append(f"- VWAP at entry: **${meta.get('vwap'):.2f}**")
        if meta.get("volume_ratio") is not None:
            setup_lines.append(f"- Volume ratio: {meta.get('volume_ratio'):.2f}")
    if _is_chan(variant) and signal_metadata_str:
        try:
            meta = json.loads(signal_metadata_str)
        except Exception:
            meta = {}
        t_types = meta.get("t_types") or outcome.get("base_pattern") or "?"
        setup_lines.append(f"- Chan buy-structure-point: **{t_types}**")
        if meta.get("bi_low") is not None:
            setup_lines.append(
                f"- BI low (structural stop anchor): **${meta['bi_low']:.2f}**"
            )
        if meta.get("confidence") is not None:
            setup_lines.append(
                f"- Signal confidence: {meta['confidence']:.2f}"
            )
        if meta.get("market_score") is not None:
            setup_lines.append(
                f"- Market regime score at entry: {meta['market_score']}/10"
            )
    elif _is_chan(variant):
        setup_lines.append(
            f"- Chan buy-structure-point: **{outcome.get('base_pattern') or '?'}**"
        )
    setup_lines.append(
        f"- Market regime at entry: {outcome.get('regime_at_entry') or 'n/a'}"
    )
    setup_block = "\n".join(setup_lines) + "\n\n"

    # Minervini criteria checklist — one line per rule, ✓ or ✗, shown above
    # the LLM narrative so you can scan "which filters actually fired".
    criteria_block = ""
    if _is_minervini(variant) and setup_row and payload:
        items = _minervini_criteria_checklist(setup_row, payload)
        if items:
            criteria_block = "## Criteria at entry (Minervini template)\n"
            for name, val, ok in items:
                mark = "✅" if ok else "❌"
                tail = f" — {val}" if val else ""
                criteria_block += f"- {mark} {name}{tail}\n"
            criteria_block += "\n"

    # Raw entry reasoning from the orchestrator — untruncated.
    reasoning = ""
    if entry_signal:
        reasoning = (
            entry_signal.get("reasoning") or entry_signal.get("full_analysis") or ""
        )
    reasoning_block = ""
    if reasoning:
        reasoning_block = (
            "## Entry signal (orchestrator log)\n"
            f"> {reasoning.strip()}\n\n"
        )

    # Chan 中枢 / BI / BSP summary — the structure the trader saw at entry,
    # computed over [entry_date - 90d, entry_date] so no post-exit hindsight.
    chan_block = ""
    if _is_chan(variant) and chan_structures:
        bi_list = chan_structures.get("bi_list") or []
        zs_list = chan_structures.get("zs_list") or []
        bsp_list = chan_structures.get("bsp_list") or []
        chan_block = "## Chan structures at entry (30m, last 90d)\n"
        chan_block += (
            f"- {len(bi_list)} bi · {len(zs_list)} **中枢 (ZS)** · "
            f"{len(bsp_list)} BSP\n"
        )
        if zs_list:
            chan_block += "\n**中枢 (last 3):**\n"
            for i, zs in enumerate(zs_list[-3:], 1):
                chan_block += (
                    f"- ZS{i} [{zs.get('begin_time')} → {zs.get('end_time')}] "
                    f"range **${zs.get('low'):.2f} – ${zs.get('high'):.2f}**\n"
                )
        if bsp_list:
            chan_block += "\n**Buy/Sell structure points (last 3):**\n"
            for bsp in bsp_list[-3:]:
                side = "BUY" if bsp.get("is_buy") else "SELL"
                chan_block += (
                    f"- {side} BSP `{bsp.get('types')}` at "
                    f"{bsp.get('time')} · ${bsp.get('price'):.2f}\n"
                )
        chan_block += "\n"

    happened_block = (
        "## What Happened\n"
        f"- Entry **${outcome.get('entry_price', 0):.2f}** on "
        f"{outcome.get('entry_date')} "
        f"({outcome.get('hold_days')} days held)\n"
    )
    if mfe is not None or mae is not None:
        excur = []
        if mfe is not None:
            excur.append(f"MFE **{mfe:+.2%}**")
        if mae is not None:
            excur.append(f"MAE **{mae:+.2%}**")
        happened_block += f"- Excursion: {' · '.join(excur)}\n"
    happened_block += (
        f"- Exit **${outcome.get('exit_price', 0):.2f}** via "
        f"_{outcome.get('exit_reason') or 'unknown'}_\n\n"
    )

    return (
        header + link
        + setup_block
        + criteria_block
        + chan_block
        + reasoning_block
        + happened_block
        + (llm_body or "")
        + "\n"
    )


def run_held_position_review(
    *,
    db,
    broker,
    variant_name: str,
    config: dict,
    review_date: Optional[date] = None,
    features_fn=None,
) -> dict:
    """Produce a short health-check markdown for each held position today.

    One small LLM call per held symbol, output to
    `results/daily_reviews/YYYY-MM-DD/{variant}_{symbol}_HELD.md`. Parallel
    to `run_daily_review` (which covers CLOSED trades). Kill-switched by
    `held_position_review_enabled` (default True).

    `features_fn` is an optional callable `(symbol) -> dict` that the
    orchestrator supplies (it has Minervini preflight cache etc). If
    omitted the health read degrades gracefully — no RS-now / SMA-50.
    """
    if not config.get("held_position_review_enabled", True):
        return {"variant": variant_name, "status": "disabled"}

    from tradingagents.automation.position_health import (
        collect_position_snapshot,
        render_health_prompt,
        compose_health_markdown,
    )

    review_date = review_date or date.today()
    iso_day = review_date.isoformat()
    dry_run = bool(config.get("held_position_review_dry_run", False))
    base_dir = Path("results/daily_reviews_dryrun" if dry_run else "results/daily_reviews")
    out_dir = base_dir / iso_day
    out_dir.mkdir(parents=True, exist_ok=True)

    budget = int(config.get("held_review_max_calls", 30))

    try:
        positions = broker.get_positions()
    except Exception as e:
        logger.error("held_position_review[%s] get_positions failed: %s",
                     variant_name, e)
        return {"variant": variant_name, "status": "error", "error": str(e)}

    # Only stock positions — skip SPY overlay / options / cash equivalents.
    stock_positions = [
        p for p in positions
        if getattr(p, "qty", 0) and str(getattr(p, "side", "")).lower() == "long"
    ]

    if not stock_positions:
        logger.info("held_position_review[%s] %s: no held positions",
                    variant_name, iso_day)
        return {"variant": variant_name, "date": iso_day, "held": 0,
                "analyzed": 0, "dry_run": dry_run}

    from tradingagents.llm_clients.factory import create_llm_client
    llm_client = create_llm_client(
        provider=config.get("llm_provider", "openai"),
        model=config.get("quick_think_llm", "gpt-4o-mini"),
    )
    llm = llm_client.get_llm()

    analyzed = 0
    failed_llm = 0
    skipped_budget = 0
    errors = 0

    for pos in stock_positions:
        if analyzed >= budget:
            skipped_budget += 1
            continue
        try:
            snapshot = collect_position_snapshot(
                db=db, symbol=pos.symbol, position=pos,
                variant=variant_name, features_fn=features_fn,
            )
            prompt = render_health_prompt(snapshot)
            try:
                body = llm.invoke(prompt).content
            except Exception as e:
                failed_llm += 1
                body = f"*LLM health read failed: {e}*"
                logger.warning(
                    "held_position_review[%s] %s LLM failed: %s",
                    variant_name, pos.symbol, e,
                )
            md = compose_health_markdown(snapshot, body)
            md_path = out_dir / f"{variant_name}_{pos.symbol}_HELD.md"
            md_path.write_text(md, encoding="utf-8")
            analyzed += 1
        except Exception as e:
            errors += 1
            logger.error(
                "held_position_review[%s] %s failed: %s",
                variant_name, pos.symbol, e, exc_info=True,
            )

    summary = {
        "variant": variant_name, "date": iso_day,
        "held": len(stock_positions),
        "analyzed": analyzed,
        "failed_llm": failed_llm,
        "skipped_budget": skipped_budget,
        "errors": errors,
        "dry_run": dry_run,
        "out_dir": str(out_dir),
    }
    logger.info(
        "held_position_review[%s] %s: %d held, %d analyzed, %d errors",
        variant_name, iso_day, len(stock_positions), analyzed, errors,
    )
    return summary


def _write_day_summary(out_dir: Path, variant: str, iso_day: str,
                      outcomes: list, results: list) -> None:
    wins = sum(1 for o in outcomes if (o.get("return_pct") or 0) > 0)
    losses = sum(1 for o in outcomes if (o.get("return_pct") or 0) < 0)
    flat = len(outcomes) - wins - losses
    total_return = sum((o.get("return_pct") or 0) for o in outcomes)
    summary_path = out_dir / f"{variant}_summary.md"
    body = [
        f"# {variant} — {iso_day}",
        "",
        f"- Trades closed: **{len(outcomes)}** "
        f"(wins {wins} · losses {losses} · flat {flat})",
        f"- Sum of returns: **{total_return:+.2%}**",
        "",
        "## Per-trade",
    ]
    for r in results:
        body.append(
            f"- [{r['symbol']}]({variant}_{r['symbol']}.md) — "
            f"{(r['return_pct'] or 0):+.2%}"
        )
    summary_path.write_text("\n".join(body) + "\n", encoding="utf-8")
