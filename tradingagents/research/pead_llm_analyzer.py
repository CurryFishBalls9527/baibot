"""LLM gate for PEAD entries via the existing 13-agent pipeline.

Wraps :class:`tradingagents.graph.trading_graph.TradingAgentsGraph` to
analyze each PEAD candidate (symbol that just reported with a 5-50%
EPS surprise). The LLM's BUY/HOLD/SELL decision is cached in
``earnings_data.duckdb.earnings_llm_decisions`` and PEAD's morning fire
INNER JOINs against the cache, entering only ``BUY`` rows.

## Standalone-strategy guardrail (do NOT change this contract)

PEAD intentionally lives OUTSIDE the swing scheduler / ABRunner. The
swing reconciler's orphan-import branch (``reconciler.py:462-522``)
fabricates an 8%-below-entry stop on any broker position lacking a
matching ``position_states`` row. Plugging PEAD into ABRunner once
already corrupted the PEAD Alpaca account on 2026-05-01 (3 unwanted
positions worth ~$50k opened by a default Minervini Orchestrator that
got built because PEAD was added to ``paper_launch_v2.yaml``).

This analyzer therefore:
  * builds its own ``TradingAgentsGraph`` instance (NOT via ABRunner)
  * never imports from ``tradingagents.testing.ab_runner`` or
    instantiates any ``Orchestrator`` subclass
  * writes only to its own cache table + dashboard mirror; no SQLite
    ``position_states`` write, no broker stop placement
  * never reads ``paper_launch_v2.yaml``

Future contributors: do NOT short-circuit any of the above. The
``test_pead_llm_analyzer.py`` lint test enforces the ABRunner-import
prohibition.

## Cache contract

``earnings_llm_decisions`` PRIMARY KEY ``(symbol, event_date)``. Re-runs
``INSERT OR REPLACE`` so the table stays at one row per analyzed event.
Failed analyses still write a row (``llm_decision = NULL``,
``error = <repr>``) so PEAD's INNER JOIN naturally drops them while
leaving an auditable trail.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import duckdb

logger = logging.getLogger(__name__)


# Lives in earnings_data.duckdb (the split-off DB owned by ingest crons +
# PEAD reads). Created lazily on first write — no separate migration.
EARNINGS_DB_DEFAULT = "research_data/earnings_data.duckdb"

# Default model selection — see plan ethereal-strolling-rocket.md §B.
# OpenAI-only (no Anthropic key). gpt-5.4-pro is the current OpenAI flagship
# (one tick above the gpt-5.2 default in default_config.py); gpt-5-mini is
# the cheap workhorse for the 11 quick-think calls per analysis.
PEAD_DEEP_MODEL_DEFAULT = os.environ.get("PEAD_DEEP_MODEL", "gpt-5.4-pro")
PEAD_QUICK_MODEL_DEFAULT = os.environ.get("PEAD_QUICK_MODEL", "gpt-5-mini")


_CREATE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS earnings_llm_decisions (
    symbol             VARCHAR    NOT NULL,
    event_date         DATE       NOT NULL,
    analyzed_at        TIMESTAMP  NOT NULL,
    llm_decision       VARCHAR,           -- BUY|SELL|HOLD; NULL on error
    llm_confidence     VARCHAR,           -- low|medium|high
    final_signal_text  VARCHAR,           -- final_trade_decision text
    trader_plan_text   VARCHAR,           -- trader_investment_plan text
    screener_context   VARCHAR,           -- the EARNINGS CATALYST CONTEXT block we sent
    deep_think_model   VARCHAR    NOT NULL,
    quick_think_model  VARCHAR    NOT NULL,
    config_hash        VARCHAR    NOT NULL,
    duration_seconds   DOUBLE,
    total_llm_calls    INTEGER,
    cost_estimate_usd  DOUBLE,
    state_log_path     VARCHAR,
    error              VARCHAR,
    PRIMARY KEY (symbol, event_date)
);
CREATE INDEX IF NOT EXISTS idx_eld_event_date ON earnings_llm_decisions(event_date);
"""


# ─── PEAD-tuned config ─────────────────────────────────────────────────


def build_pead_config(
    deep_model: str = PEAD_DEEP_MODEL_DEFAULT,
    quick_model: str = PEAD_QUICK_MODEL_DEFAULT,
    project_dir: Optional[str] = None,
    results_dir: str = "results/pead/llm",
) -> dict:
    """Build the config dict passed to TradingAgentsGraph.

    Overrides only what's PEAD-specific. Everything else inherits from
    ``tradingagents.default_config.DEFAULT_CONFIG``.
    """
    from tradingagents.default_config import DEFAULT_CONFIG  # lazy
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({
        "llm_provider": "openai",
        "deep_think_llm": deep_model,
        "quick_think_llm": quick_model,
        # Single-round debate to control latency + cost. PEAD analyses run
        # in batch with a 25-min hard ceiling for the BMO window.
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "online_tools": True,
        "results_dir": results_dir,
    })
    if project_dir:
        cfg["project_dir"] = project_dir
    return cfg


def _config_hash(cfg: dict) -> str:
    """SHA-1 of the PEAD-relevant config keys, for cache reproducibility audits."""
    keys = (
        "llm_provider", "deep_think_llm", "quick_think_llm",
        "max_debate_rounds", "max_risk_discuss_rounds", "online_tools",
    )
    payload = json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


# ─── Earnings-catalyst context ─────────────────────────────────────────


@dataclass
class EarningsEvent:
    """Row from earnings_events.surprise-filtered query."""
    symbol: str
    event_datetime: datetime
    eps_estimate: Optional[float]
    reported_eps: Optional[float]
    surprise_pct: Optional[float]
    revenue_average: Optional[float]
    time_hint: Optional[str]
    source: Optional[str]

    @property
    def event_date(self) -> date:
        return self.event_datetime.date()


def build_catalyst_context(ev: EarningsEvent) -> str:
    """Format the EARNINGS CATALYST CONTEXT block injected into both
    ``screener_context`` (trader sees it) AND the initial messages list
    (analysts see it as their first user message).
    """
    def _fmt(v: Optional[float], dollar: bool = False) -> str:
        if v is None:
            return "n/a"
        return f"${v:,.2f}" if dollar else f"{v:+.2f}%"

    return (
        "EARNINGS CATALYST CONTEXT (your primary focus for this analysis):\n"
        f"  Symbol:           {ev.symbol}\n"
        f"  Report date/time: {ev.event_datetime.strftime('%Y-%m-%d %H:%M')} "
        f"({ev.time_hint or 'unknown'})\n"
        f"  EPS estimate:     {_fmt(ev.eps_estimate, dollar=True)}\n"
        f"  EPS reported:     {_fmt(ev.reported_eps, dollar=True)}\n"
        f"  EPS surprise:     {_fmt(ev.surprise_pct)}\n"
        f"  Revenue estimate: {_fmt(ev.revenue_average, dollar=True)}\n"
        f"  Source:           {ev.source or 'unknown'}\n"
        "\n"
        "PEAD thesis: this analysis is to decide whether to LONG the post-\n"
        "earnings drift over the next ~20 trading days. The surprise%\n"
        "above ALREADY passed our 5%-50% gate — the question is whether\n"
        "the drift is likely to PERSIST or FADE. Weigh:\n"
        "  (a) beat quality — operational vs one-time items, accruals\n"
        "  (b) forward guidance direction (press release / call)\n"
        "  (c) management tone — confidence, specificity, hedging\n"
        "  (d) sector/macro context — is this stock-specific or sector tailwind?\n"
        "  (e) post-print analyst reaction if available\n"
        "Filter HOLD/SELL if the catalyst is low-quality or guidance is\n"
        "negative even when EPS beat — those are the PEAD failure modes."
    )


# ─── Cost estimation ───────────────────────────────────────────────────

# Rough per-1k-token rates for cost estimation. Update when OpenAI revises
# pricing. Underestimate is fine; overestimate biases us to upgrade-test
# Sonnet as A/B-promotion criterion.
_RATE_PER_1K = {
    "gpt-5.4-pro":  {"in": 0.015, "out": 0.060},
    "gpt-5.4":      {"in": 0.010, "out": 0.030},
    "gpt-5.2":      {"in": 0.005, "out": 0.015},
    "gpt-5.1":      {"in": 0.004, "out": 0.012},
    "gpt-5":        {"in": 0.003, "out": 0.010},
    "gpt-5-mini":   {"in": 0.0008, "out": 0.003},
    "gpt-5-nano":   {"in": 0.0002, "out": 0.0008},
    "gpt-4.1":      {"in": 0.003, "out": 0.012},
    "gpt-4.1-mini": {"in": 0.0008, "out": 0.003},
    "gpt-4.1-nano": {"in": 0.0002, "out": 0.0008},
}


def estimate_cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int,
) -> float:
    rate = _RATE_PER_1K.get(model)
    if not rate:
        return 0.0
    return (prompt_tokens * rate["in"] + completion_tokens * rate["out"]) / 1000.0


# ─── Cache I/O ─────────────────────────────────────────────────────────


def ensure_schema(db_path: str = EARNINGS_DB_DEFAULT) -> None:
    """Create the earnings_llm_decisions table if missing. Idempotent."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    try:
        con.execute(_CREATE_TABLE_DDL)
    finally:
        con.close()


@dataclass
class AnalysisResult:
    symbol: str
    event_date: date
    llm_decision: Optional[str]                # BUY|SELL|HOLD or None
    llm_confidence: Optional[str] = None
    final_signal_text: Optional[str] = None
    trader_plan_text: Optional[str] = None
    screener_context: Optional[str] = None
    deep_think_model: str = ""
    quick_think_model: str = ""
    config_hash: str = ""
    duration_seconds: float = 0.0
    total_llm_calls: int = 0
    cost_estimate_usd: float = 0.0
    state_log_path: Optional[str] = None
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


def write_result(result: AnalysisResult, db_path: str = EARNINGS_DB_DEFAULT) -> None:
    """UPSERT one analysis row. Replaces any existing row for the same
    (symbol, event_date) — re-runs are idempotent."""
    ensure_schema(db_path)
    con = duckdb.connect(db_path)
    try:
        con.execute(
            """
            INSERT OR REPLACE INTO earnings_llm_decisions (
                symbol, event_date, analyzed_at,
                llm_decision, llm_confidence,
                final_signal_text, trader_plan_text,
                screener_context,
                deep_think_model, quick_think_model, config_hash,
                duration_seconds, total_llm_calls, cost_estimate_usd,
                state_log_path, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                result.symbol, result.event_date,
                datetime.now(timezone.utc),
                result.llm_decision, result.llm_confidence,
                result.final_signal_text, result.trader_plan_text,
                result.screener_context,
                result.deep_think_model, result.quick_think_model,
                result.config_hash,
                result.duration_seconds, result.total_llm_calls,
                result.cost_estimate_usd,
                result.state_log_path, result.error,
            ],
        )
    finally:
        con.close()


# ─── Decision parsing ──────────────────────────────────────────────────

# Trader/risk_manager convention: response ends with
# "FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**" (per
# tradingagents/agents/trader/trader.py:49). We pick the LAST occurrence
# in the final_trade_decision text (risk_manager output) since that's the
# definitive judgment downstream of the trader.
import re

# Multiple decision markers, in priority order:
#  1. trader's strict format `FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`
#  2. risk_manager natural output `Recommendation: SELL` / `Final ...: SELL`
#  3. markdown decision heading `## **SELL**` / `## SELL`
# We pick the LAST occurrence across ALL patterns since the risk_manager
# is downstream of the trader and renders the definitive judgment.
_DECISION_PATTERNS = [
    re.compile(
        r"FINAL\s+TRANSACTION\s+PROPOSAL\s*:?\s*\*{0,2}\s*"
        r"(BUY|HOLD|SELL)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:Final\s+)?Recommendation\s*:\s*\*{0,2}\s*"
        r"(BUY|HOLD|SELL)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^#{1,4}\s*\*{0,2}\s*"
        r"(BUY|HOLD|SELL)\s*\*{0,2}\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
]


def _coerce_to_text(value: Any) -> str:
    """Reasoning models (e.g. gpt-5.4-pro via /v1/responses) return
    ``final_trade_decision`` as a list of content blocks
    ``[{'type': 'reasoning', ...}, {'type': 'text', 'text': '...'}, ...]``
    instead of a plain string. Concatenate any text blocks; ignore the
    reasoning ones (they're internal chain-of-thought, not the answer).
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for block in value:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(value)


def parse_decision(text: Any) -> Optional[str]:
    """Extract BUY|HOLD|SELL from a trader/risk_manager response. Returns
    None if no marker found (analyst was non-conformant — caller should
    treat as a soft failure). Accepts plain string or list of content
    blocks (the reasoning-model output shape).

    Tries multiple patterns (FINAL TRANSACTION PROPOSAL, Recommendation,
    markdown heading) and returns the LAST match across all of them since
    the risk_manager renders downstream of the trader.
    """
    text_str = _coerce_to_text(text)
    if not text_str:
        return None
    # Collect (position, decision) tuples across all patterns.
    matches: list[tuple[int, str]] = []
    for pat in _DECISION_PATTERNS:
        for m in pat.finditer(text_str):
            matches.append((m.start(), m.group(1).upper()))
    if not matches:
        return None
    matches.sort()
    return matches[-1][1]


# ─── Main entry point ─────────────────────────────────────────────────


def analyze_event(
    ev: EarningsEvent,
    db_path: str = EARNINGS_DB_DEFAULT,
    deep_model: str = PEAD_DEEP_MODEL_DEFAULT,
    quick_model: str = PEAD_QUICK_MODEL_DEFAULT,
    write_cache: bool = True,
) -> AnalysisResult:
    """Run the full multi-agent pipeline on one earnings event.

    Always returns an AnalysisResult (never raises). On exception, returns
    a result with ``llm_decision=None`` and ``error`` populated; the row
    is still written to cache (if ``write_cache=True``) so the failure is
    auditable and PEAD's INNER JOIN drops the symbol.
    """
    # Lazy imports — keep this module importable in test environments
    # that don't have all of TradingAgents' deep deps installed yet.
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    cfg = build_pead_config(deep_model=deep_model, quick_model=quick_model)
    catalyst = build_catalyst_context(ev)
    cfg_hash = _config_hash(cfg)

    result = AnalysisResult(
        symbol=ev.symbol,
        event_date=ev.event_date,
        llm_decision=None,
        screener_context=catalyst,
        deep_think_model=deep_model,
        quick_think_model=quick_model,
        config_hash=cfg_hash,
    )

    started = time.time()
    try:
        # NOTE: `screener_context` is currently consumed only by the trader
        # node (trader.py:26-32) — it appears verbatim in the trader's
        # prompt. The 4 analysts and bull/bear/manager nodes don't read it.
        # To get the catalyst in front of the analysts too, we rely on
        # the analyzer also seeding the initial messages list with a
        # leading human message containing the catalyst (see
        # _patch_initial_messages below — applied via a one-shot
        # propagator subclass). For now, screener_context covers the
        # trader (which sees the most-condensed final argument) and
        # the catalyst flows transitively via the analyst reports the
        # trader receives.
        graph = TradingAgentsGraph(debug=False, config=cfg)
        _patch_initial_messages(graph, ev, catalyst)

        final_state = None
        processed_signal = None
        try:
            final_state, processed_signal = graph.propagate(
                ev.symbol,
                str(ev.event_date),
                screener_context=catalyst,
            )
        except Exception as propagate_exc:
            # Recovery for the known reasoning-model issue: with
            # gpt-5.4-pro as deep_think, the SignalProcessor's final
            # `quick_thinking_llm.invoke(...)` rejects the prior turn's
            # `reasoning` content blocks (langchain forwards them; OpenAI's
            # /chat/completions API only accepts text|image_url|...). The
            # graph itself completes BEFORE this post-extraction step and
            # writes final_state to graph.curr_state — so we can salvage
            # the decision from there.
            curr = getattr(graph, "curr_state", None)
            if curr and curr.get("final_trade_decision"):
                logger.warning(
                    "PEAD-LLM SignalProcessor post-extraction failed for %s "
                    "(%s); recovering decision from graph.curr_state.",
                    ev.symbol, type(propagate_exc).__name__,
                )
                final_state = curr
                # Stash the post-extraction failure in extra so the audit
                # trail captures it without poisoning result.error.
                result.extra["post_extraction_error"] = str(propagate_exc)[:500]
            else:
                # Graph itself crashed — bubble up to the outer except.
                raise

        # At this point final_state is populated either way.
        # _coerce_to_text handles both plain-string outputs (gpt-5-mini,
        # gpt-5.2) AND the list-of-content-blocks shape that reasoning
        # models like gpt-5.4-pro emit via the /v1/responses endpoint.
        result.final_signal_text = _coerce_to_text(
            final_state.get("final_trade_decision")
        )
        result.trader_plan_text = _coerce_to_text(
            final_state.get("trader_investment_plan")
        )
        result.llm_decision = parse_decision(result.final_signal_text)
        # Fallback: signal_processor's own extraction (only if it ran).
        if result.llm_decision is None and processed_signal:
            sig_str = str(processed_signal).upper()
            if "BUY" in sig_str:
                result.llm_decision = "BUY"
            elif "SELL" in sig_str:
                result.llm_decision = "SELL"
            elif "HOLD" in sig_str:
                result.llm_decision = "HOLD"
        # state_log_path: TradingAgentsGraph._log_state writes here.
        result.state_log_path = (
            f"eval_results/{ev.symbol}/TradingAgentsStrategy_logs/"
            f"full_states_log_{ev.event_date}.json"
        )
        # TODO: token usage / cost is harder to estimate without
        # instrumenting the LLM clients with callbacks. For phase-1, use
        # a fixed approximation: 13 calls × ~2000-token avg = ~26k tokens
        # per analysis, mostly on quick model. Refined via callbacks
        # once the analyzer is in production.
        approx_in = 26000
        approx_out = 4000
        result.total_llm_calls = 13
        result.cost_estimate_usd = (
            estimate_cost_usd(deep_model, approx_in // 6, approx_out // 6) +
            estimate_cost_usd(quick_model, approx_in * 5 // 6, approx_out * 5 // 6)
        )
    except Exception as exc:
        logger.warning(
            "PEAD-LLM analyze failed for %s on %s: %s",
            ev.symbol, ev.event_date, exc,
        )
        result.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[:1500]}"
    finally:
        result.duration_seconds = time.time() - started

    if write_cache:
        try:
            write_result(result, db_path=db_path)
        except Exception as exc:
            logger.error(
                "PEAD-LLM cache write failed for %s on %s: %s — DECISION LOST",
                ev.symbol, ev.event_date, exc,
            )

    return result


def _patch_initial_messages(graph, ev: EarningsEvent, catalyst: str) -> None:
    """Inject the catalyst block into the initial messages list so the
    FIRST analyst node sees it as its first user-message turn (the
    analysts use ``MessagesPlaceholder(variable_name="messages")`` so the
    leading human turn shows up in their prompt context).

    Implemented by monkey-patching the graph's propagator's
    ``create_initial_state`` for this single graph instance. We do NOT
    modify the shared Propagator class globally.
    """
    original = graph.propagator.create_initial_state

    def patched(company_name, trade_date, **kwargs):
        state = original(company_name, trade_date, **kwargs)
        # Prepend the catalyst as a leading human message; the existing
        # ("human", company_name) seed becomes the second message.
        state["messages"] = [
            ("human", catalyst),
            ("human", f"Symbol: {company_name}. Trade date: {trade_date}."),
        ]
        return state

    graph.propagator.create_initial_state = patched


# ─── Batch helpers ────────────────────────────────────────────────────


def load_window_events(
    window: str,                              # "amc" or "bmo"
    today: Optional[date] = None,
    db_path: str = EARNINGS_DB_DEFAULT,
    surprise_min: float = 5.0,
    surprise_max: float = 50.0,
) -> list[EarningsEvent]:
    """Pull the earnings events to analyze in a given batch window.

    AMC: events with ``event_datetime >= today - 1d 15:00`` and ``< today 06:00``
    (yesterday after-close + overnight). The 17:30 CDT batch handles these.

    BMO: events with ``event_datetime >= today 06:00`` and ``<= today 09:30``.
    The 08:10 CDT batch handles these.

    In both cases we already filter by surprise gate so we don't waste
    LLM analyses on symbols PEAD couldn't trade anyway.
    """
    from datetime import datetime as _dt, timedelta as _td
    if today is None:
        today = date.today()
    if window == "amc":
        start = _dt.combine(today - _td(days=1), datetime.min.time()).replace(hour=15)
        end = _dt.combine(today, datetime.min.time()).replace(hour=6)
    elif window == "bmo":
        start = _dt.combine(today, datetime.min.time()).replace(hour=6)
        end = _dt.combine(today, datetime.min.time()).replace(hour=9, minute=30)
    else:
        raise ValueError(f"Unknown window: {window!r} (expected 'amc' or 'bmo')")

    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            """
            SELECT symbol, event_datetime, eps_estimate, reported_eps,
                   surprise_pct, revenue_average, time_hint, source
            FROM earnings_events
            WHERE event_datetime >= ?
              AND event_datetime <= ?
              AND surprise_pct IS NOT NULL
              AND surprise_pct >= ?
              AND surprise_pct <= ?
              AND is_future = false
            ORDER BY event_datetime, symbol
            """,
            [start, end, surprise_min, surprise_max],
        ).fetchall()
    finally:
        con.close()
    return [
        EarningsEvent(
            symbol=r[0], event_datetime=r[1],
            eps_estimate=r[2], reported_eps=r[3], surprise_pct=r[4],
            revenue_average=r[5], time_hint=r[6], source=r[7],
        )
        for r in rows
    ]
