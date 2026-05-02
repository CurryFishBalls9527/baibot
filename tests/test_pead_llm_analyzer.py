"""Tests for the PEAD-LLM analyzer + cache + JOIN semantics.

Coverage:
  * Cache schema round-trip (idempotent UPSERT on (symbol, event_date) PK)
  * Catalyst-context preamble builder
  * Decision parsing (BUY/SELL/HOLD extraction from trader/risk_manager output)
  * Failure-row write (analyzer never raises; on exception writes a row
    with llm_decision=NULL + error populated; PEAD's INNER JOIN drops it)
  * PEAD INNER JOIN behavior (BUY surfaces; HOLD/SELL/error/missing skipped)
  * Lint guard: analyzer module does NOT import from
    tradingagents.testing.ab_runner or instantiate any Orchestrator
    subclass — enforces the standalone-strategy guardrail that prevented
    the 2026-05-01 PEAD account contamination.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path

import duckdb
import pytest


# Prewarm to dodge the broker __init__ circular import seen elsewhere.
def _prewarm_imports():
    if "tradingagents.automation.events" not in sys.modules:
        stub = types.ModuleType("tradingagents.automation.events")
        class _Cat:
            ORDER_REJECT = "ORDER_REJECT"
            WASH_TRADE_REJECT = "WASH_TRADE_REJECT"
        stub.Categories = _Cat
        stub.emit_event = lambda *a, **k: None
        sys.modules["tradingagents.automation.events"] = stub


_prewarm_imports()


# ─── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def isolated_db(tmp_path):
    """Per-test DuckDB at a tmp path. Pre-creates earnings_events with one
    canonical fixture row so JOIN tests have something to match against."""
    from tradingagents.research.pead_llm_analyzer import ensure_schema
    db_path = str(tmp_path / "earnings_data.duckdb")
    # Create earnings_events table for JOIN tests
    con = duckdb.connect(db_path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS earnings_events (
            symbol VARCHAR NOT NULL,
            event_datetime TIMESTAMP NOT NULL,
            eps_estimate DOUBLE,
            reported_eps DOUBLE,
            surprise_pct DOUBLE,
            revenue_average DOUBLE,
            is_future BOOLEAN,
            source VARCHAR,
            updated_at TIMESTAMP,
            time_hint VARCHAR,
            PRIMARY KEY (symbol, event_datetime)
        )
        """
    )
    # Three fixture events on 2026-04-30 (mirrors today's actual PEAD data)
    for sym, surp, hint in (
        ("WDC", 13.57, "amc"),
        ("TEAM", 30.99, "amc"),
        ("RDDT", 26.13, "amc"),
    ):
        con.execute(
            "INSERT INTO earnings_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [sym, datetime(2026, 4, 30, 21, 0), 1.0, 1.5, surp, None,
             False, "yfinance", datetime.now(), hint],
        )
    con.close()
    ensure_schema(db_path)
    return db_path


# ─── Schema + idempotency ─────────────────────────────────────────────


def test_ensure_schema_creates_table(tmp_path):
    """Schema lands on first call and second call is a no-op."""
    from tradingagents.research.pead_llm_analyzer import ensure_schema
    db_path = str(tmp_path / "x.duckdb")
    ensure_schema(db_path)
    ensure_schema(db_path)  # idempotent
    con = duckdb.connect(db_path, read_only=True)
    cols = [c[0] for c in con.execute("DESCRIBE earnings_llm_decisions").fetchall()]
    con.close()
    expected = {
        "symbol", "event_date", "analyzed_at", "llm_decision",
        "llm_confidence", "final_signal_text", "trader_plan_text",
        "screener_context", "deep_think_model", "quick_think_model",
        "config_hash", "duration_seconds", "total_llm_calls",
        "cost_estimate_usd", "state_log_path", "error",
    }
    assert expected.issubset(set(cols))


def test_write_result_upsert_idempotent(isolated_db):
    """Two writes with the same (symbol, event_date) leave one row."""
    from tradingagents.research.pead_llm_analyzer import (
        AnalysisResult, write_result,
    )
    r1 = AnalysisResult(
        symbol="WDC", event_date=date(2026, 4, 30),
        llm_decision="BUY",
        deep_think_model="gpt-5.4-pro", quick_think_model="gpt-5-mini",
        config_hash="h1",
    )
    r2 = AnalysisResult(
        symbol="WDC", event_date=date(2026, 4, 30),
        llm_decision="HOLD",  # changed
        deep_think_model="gpt-5.4-pro", quick_think_model="gpt-5-mini",
        config_hash="h2",
    )
    write_result(r1, db_path=isolated_db)
    write_result(r2, db_path=isolated_db)
    con = duckdb.connect(isolated_db, read_only=True)
    rows = con.execute(
        "SELECT llm_decision, config_hash FROM earnings_llm_decisions "
        "WHERE symbol='WDC'"
    ).fetchall()
    con.close()
    assert len(rows) == 1
    assert rows[0] == ("HOLD", "h2"), "second write should replace first"


# ─── Catalyst context + decision parsing ──────────────────────────────


def test_build_catalyst_context_includes_thesis():
    from tradingagents.research.pead_llm_analyzer import (
        EarningsEvent, build_catalyst_context,
    )
    ev = EarningsEvent(
        symbol="META", event_datetime=datetime(2026, 4, 29, 21, 0),
        eps_estimate=6.74, reported_eps=7.22, surprise_pct=7.20,
        revenue_average=42_100_000_000.0, time_hint="amc", source="yfinance",
    )
    ctx = build_catalyst_context(ev)
    # Hard checks: structured fields
    assert "EARNINGS CATALYST CONTEXT" in ctx
    assert "META" in ctx
    assert "+7.20%" in ctx
    assert "$6.74" in ctx
    assert "amc" in ctx
    # Thesis hint mentions the 5 evaluation factors so analysts focus on them
    for letter in ("(a)", "(b)", "(c)", "(d)", "(e)"):
        assert letter in ctx


def test_parse_decision_handles_all_three():
    from tradingagents.research.pead_llm_analyzer import parse_decision
    assert parse_decision("FINAL TRANSACTION PROPOSAL: **BUY**") == "BUY"
    assert parse_decision("blah FINAL TRANSACTION PROPOSAL: SELL") == "SELL"
    assert parse_decision("FINAL TRANSACTION PROPOSAL: **HOLD**") == "HOLD"


def test_parse_decision_picks_last_occurrence():
    """The risk_manager output is downstream of the trader — both may
    contain the FINAL TRANSACTION PROPOSAL marker. We pick the LAST one
    (the risk_manager's definitive call)."""
    from tradingagents.research.pead_llm_analyzer import parse_decision
    text = (
        "Trader said: FINAL TRANSACTION PROPOSAL: **BUY**\n"
        "Risk Manager: After review, FINAL TRANSACTION PROPOSAL: HOLD"
    )
    assert parse_decision(text) == "HOLD"


def test_parse_decision_returns_none_when_format_missing():
    from tradingagents.research.pead_llm_analyzer import parse_decision
    assert parse_decision("I think we should buy") is None
    assert parse_decision("") is None
    assert parse_decision(None) is None


def test_parse_decision_recommendation_pattern():
    """The risk_manager output uses 'Recommendation: SELL' rather than the
    trader's 'FINAL TRANSACTION PROPOSAL' format. Discovered in the
    2026-05-02 live test on META."""
    from tradingagents.research.pead_llm_analyzer import parse_decision
    text = (
        "**Recommendation: SELL**\n\n"
        "This is a risk-management SELL, not an emotional anti-Meta call."
    )
    assert parse_decision(text) == "SELL"
    assert parse_decision("Final Recommendation: BUY") == "BUY"


def test_parse_decision_markdown_heading():
    """risk_manager often closes with '## **SELL**' as the final verdict."""
    from tradingagents.research.pead_llm_analyzer import parse_decision
    text = "# Final judgment\n\n## **SELL**\n\nBut execute it intelligently:"
    assert parse_decision(text) == "SELL"
    assert parse_decision("### BUY") == "BUY"
    # Don't false-match section headings that mention multiple decisions
    assert parse_decision("## Discussion of BUY vs SELL options") is None


def test_parse_decision_picks_LAST_across_patterns():
    """Trader says BUY (FINAL TRANSACTION PROPOSAL), risk_manager
    overrides with HOLD (Recommendation:) downstream → HOLD wins."""
    from tradingagents.research.pead_llm_analyzer import parse_decision
    text = (
        "Trader: FINAL TRANSACTION PROPOSAL: **BUY**\n\n"
        "Risk Manager downstream review:\n"
        "Recommendation: HOLD"
    )
    assert parse_decision(text) == "HOLD"


def test_parse_decision_reasoning_model_content_blocks():
    """gpt-5.4-pro returns final_trade_decision as a list of content
    blocks: [{'type': 'reasoning', ...}, {'type': 'text', 'text': '...'}].
    parse_decision must accept this shape (extract text, ignore reasoning)."""
    from tradingagents.research.pead_llm_analyzer import parse_decision
    blocks = [
        {"id": "rs_xyz", "type": "reasoning", "summary": []},
        {"type": "text", "text": "Long analysis...\nFINAL TRANSACTION PROPOSAL: **SELL**"},
    ]
    assert parse_decision(blocks) == "SELL"


def test_coerce_to_text_handles_all_shapes():
    from tradingagents.research.pead_llm_analyzer import _coerce_to_text
    assert _coerce_to_text("hello") == "hello"
    assert _coerce_to_text(None) == ""
    assert _coerce_to_text([{"type": "text", "text": "a"},
                             {"type": "reasoning"},
                             {"type": "text", "text": "b"}]) == "a\nb"
    assert _coerce_to_text(["x", "y"]) == "x\ny"  # bare strings in list
    assert _coerce_to_text([{"type": "reasoning"}]) == ""  # no text blocks


# ─── Failure-row write ───────────────────────────────────────────────


def test_analyze_event_writes_failure_row_on_exception(isolated_db, monkeypatch):
    """If TradingAgentsGraph blows up, analyzer still writes a row with
    error populated and llm_decision=NULL."""
    from tradingagents.research import pead_llm_analyzer as pla

    # Mock TradingAgentsGraph to raise immediately
    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("simulated network error")

    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.TradingAgentsGraph",
        _Boom,
    )

    ev = pla.EarningsEvent(
        symbol="ZZZ", event_datetime=datetime(2026, 4, 30, 21, 0),
        eps_estimate=1.0, reported_eps=1.5, surprise_pct=10.0,
        revenue_average=None, time_hint="amc", source="yfinance",
    )
    result = pla.analyze_event(ev, db_path=isolated_db)
    # Result reports the failure but doesn't raise
    assert result.error is not None
    assert "RuntimeError" in result.error
    assert result.llm_decision is None
    # Cache row was written
    con = duckdb.connect(isolated_db, read_only=True)
    row = con.execute(
        "SELECT llm_decision, error FROM earnings_llm_decisions "
        "WHERE symbol='ZZZ'"
    ).fetchone()
    con.close()
    assert row is not None, "failure row must be written for audit trail"
    assert row[0] is None
    assert "RuntimeError" in row[1]


def test_analyze_event_recovers_from_post_extraction_crash(isolated_db, monkeypatch):
    """Discovered in 2026-05-02 live test: with gpt-5.4-pro as deep_think,
    SignalProcessor's `quick_thinking_llm.invoke(messages)` rejects the
    'reasoning' content blocks the responses-API forwarded. The graph
    itself completes (final_state populated, log written) — only the
    post-extraction step crashes. We must recover the decision from
    graph.curr_state and treat as SUCCESS, not failure.
    """
    from tradingagents.research import pead_llm_analyzer as pla

    class _FakeGraph:
        def __init__(self, *a, **k):
            self.curr_state = None
            self.propagator = type("P", (), {
                "create_initial_state": lambda *a, **k: {},
            })()

        def propagate(self, symbol, date, screener_context="", **k):
            # Simulate successful graph.invoke + state save, then failure
            # in the post-extraction process_signal call.
            self.curr_state = {
                "final_trade_decision": "Recommendation: SELL\n\nWeak beat quality...",
                "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: **SELL**",
            }
            raise RuntimeError(
                "BadRequestError: Invalid value: 'reasoning'. "
                "Supported values are: 'text', 'image_url', ..."
            )

    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.TradingAgentsGraph",
        _FakeGraph,
    )

    ev = pla.EarningsEvent(
        symbol="META", event_datetime=datetime(2026, 4, 29, 21, 0),
        eps_estimate=6.82, reported_eps=7.31, surprise_pct=7.20,
        revenue_average=None, time_hint="amc", source="yfinance",
    )
    result = pla.analyze_event(ev, db_path=isolated_db)
    # Recovered: decision present, error NOT set on the result row, but
    # post_extraction_error captured in extra for audit.
    assert result.llm_decision == "SELL", (
        "Decision must be recovered from graph.curr_state when only "
        "post-extraction fails."
    )
    assert result.error is None, "Recovered runs are SUCCESS, not failure."
    assert "post_extraction_error" in result.extra
    assert "Invalid value" in result.extra["post_extraction_error"]
    # And the cache row reflects success (decision present, error NULL)
    con = duckdb.connect(isolated_db, read_only=True)
    row = con.execute(
        "SELECT llm_decision, error FROM earnings_llm_decisions "
        "WHERE symbol='META'"
    ).fetchone()
    con.close()
    assert row == ("SELL", None)


# ─── PEAD JOIN semantics ─────────────────────────────────────────────


@pytest.mark.parametrize("decision,expected_count", [
    ("BUY", 1),     # passes
    ("HOLD", 0),    # filtered
    ("SELL", 0),    # filtered
    (None, 0),      # error row → no decision → filtered (failure-mode policy)
])
def test_pead_join_filters_by_decision(isolated_db, decision, expected_count):
    """Whether the LLM said BUY/HOLD/SELL/None, the JOIN should only
    surface BUY symbols when require_llm_buy=True."""
    from tradingagents.research.pead_llm_analyzer import (
        AnalysisResult, write_result,
    )
    write_result(
        AnalysisResult(
            symbol="WDC", event_date=date(2026, 4, 30),
            llm_decision=decision,
            deep_think_model="x", quick_think_model="y", config_hash="z",
            error="x" if decision is None else None,
        ),
        db_path=isolated_db,
    )
    # Hand-roll the JOIN like find_new_signals does (avoids importing
    # PEADTrader and its broker dependencies)
    con = duckdb.connect(isolated_db, read_only=True)
    rows = con.execute(
        """
        SELECT ee.symbol FROM earnings_events ee
        INNER JOIN earnings_llm_decisions eld
          ON eld.symbol = ee.symbol
         AND eld.event_date = CAST(ee.event_datetime AS DATE)
         AND eld.llm_decision = 'BUY'
        WHERE ee.symbol = 'WDC'
        """
    ).fetchall()
    con.close()
    assert len(rows) == expected_count


def test_pead_join_off_returns_all(isolated_db):
    """Without the gate, all earnings_events surprise rows pass."""
    con = duckdb.connect(isolated_db, read_only=True)
    rows = con.execute(
        "SELECT symbol FROM earnings_events WHERE surprise_pct >= 5"
    ).fetchall()
    con.close()
    assert len(rows) == 3  # WDC, TEAM, RDDT fixtures


# ─── Standalone-strategy guardrail (lint test) ────────────────────────


def test_analyzer_does_not_import_abrunner_or_orchestrator():
    """LOAD-BEARING: the analyzer must NEVER touch ABRunner or any
    Orchestrator subclass. This prevented the 2026-05-01 contamination
    incident from recurring at the import-graph level. See
    pead_llm_analyzer.py module docstring + CLAUDE.md "PEAD LLM gate"
    section + reconciler.py:462-522 orphan-import trap.
    """
    src = (
        Path(__file__).resolve().parent.parent
        / "tradingagents/research/pead_llm_analyzer.py"
    ).read_text()
    forbidden = (
        "from tradingagents.testing.ab_runner",
        "from tradingagents.testing import ab_runner",
        "import tradingagents.testing.ab_runner",
        "from tradingagents.automation.orchestrator import Orchestrator",
        "from tradingagents.automation.chan_orchestrator",
        "from tradingagents.automation.chan_daily_orchestrator",
        "from tradingagents.automation.intraday_orchestrator",
    )
    for needle in forbidden:
        assert needle not in src, (
            f"pead_llm_analyzer.py must NOT contain {needle!r} — would "
            "open a path back to the swing reconciler's orphan-import "
            "trap. See CLAUDE.md 'PEAD LLM gate' guardrail section."
        )
