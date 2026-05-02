"""Weekly per-strategy review with SPY/QQQ benchmark comparison + gpt-5.2.

Called by scheduler Sat 09:00 ET. Produces one markdown file per variant
under `results/weekly_reviews/YYYY-WW/`. Each file contains:
  - Scope (what was analyzed vs NOT)
  - Performance (returns vs SPY/QQQ, sharpe, drawdown, correlation)
  - Regime / pattern / day-of-week breakdowns
  - Lessons from this week's agent_memories
  - Open-ended improvement proposals with validate/risk fields

Uses `gpt-5.2` (deep-think) via a fresh LLMClient — does not mutate global
automation config. Kill-switched by `weekly_strategy_review_enabled`.
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


RED_TEAM_PROMPT = """You are an adversarial strategy reviewer. The weekly
analyst below wrote the PRIMARY REVIEW. Your job is to find where their
confidence exceeds the evidence. Do not cheerlead. Do not rehash stats.

Focus on:
1. Small-sample claims dressed as conclusions (N < 15 is weak, N < 5 is noise).
2. Negative data that was mentioned but not weighed.
3. Proposals whose validation plan is vague or easily gamed.
4. Correlation mistaken for causation.
5. Contradictions with prior-week proposals (if provided in the prompt).

If the primary review is sound, say so explicitly under a single line and
move on. Do NOT invent problems just to fill space.

Output format — exactly these Markdown H3 headers in order:

### Small-Sample Warning
Any conclusion drawn from N < 15 trades or from a single week's noise?
Cite the specific section. If none, write "N/A — sample sizes adequate."

### Ignored / Contradictory Data
What in the raw stats (below) was omitted or minimised?

### Challenge to Proposals
For each of the primary review's proposals, challenge its validation plan
and its risk/reward assumption.

Under 350 words total. End on a single line:
**RED-TEAM VERDICT: <support | partial-support | reject-with-reason>**

---

## RAW STATS (same inputs as primary review)
{raw_stats}

---

## PRIMARY REVIEW TO CHALLENGE
{primary_review}
"""


BIAS_AUDIT_STANZA = """
BIAS-AUDIT RULES (from the repo's CLAUDE.md — apply to all recommendations):
1. State scope explicitly: list what you ANALYZED AND what you did NOT.
2. If any R/DD > 2.0 on a multi-period backtest, flag "RE-AUDIT NEEDED" — do not celebrate it.
3. Prior experiments found: subtractive changes accepted, additive rejected.
   Lean toward removing noise, not adding complexity. See memory file
   `project_minervini_optimization_ceiling.md`.
4. Do NOT recommend changes whose success depends on expanding to
   survivorship-biased universes. See `project_partial_exit_survivorship.md`.
5. Memory insights below are point-in-time; any recommendation citing them
   must be testable against current code before acting.

ALREADY-TESTED DEAD ENDS — DO NOT propose these. Multi-period broad-universe
backtests have rejected them; the memory files cited carry the evidence:
6. Volume-based gates on the leader_continuation entry path (any flavor).
   - Daily-bar `breakout_volume_ratio >= 1.4 / 1.5` → -54pp / -44pp 2023_25
     (`project_volume_gate_null.md`).
   - Time-of-day `morning_volume_ratio` (09:30-13:30 ET) at 1.0×/1.2×/1.5×
     → -37 to -57pp 2023_25 (`project_morning_volume_gate_null.md`).
   - Mature leaders' relative volume reverts toward 1.0× as the rolling
     baseline catches up. Continuation entries are SUPPOSED to lack volume
     surge — that's the structural feature, not a bug.
7. ADX/RSI/MACD/single-indicator additive gates on entry. 20+ rejects in
   `project_minervini_optimization_ceiling.md`; ADX≥20 specifically tested
   and confirmed null in `project_w18_verification_batch.md`.
8. Tighter trailing stops (3-5% flat or aggressive regime variants below
   10/10/7). Universe-ambiguous failures: `project_a_buildout_mfe_lock_candidate.md`
   covers full sweep. The shipped `mfe_lock_10_5` (lock at entry+5% once up
   +10%) is the tested winner; do not propose alternatives without
   multi-period evidence.
9. ATR-based stops on intraday_mechanical: `project_atr_stops_null.md` — 1×
   ATR / 2× trail worse on ALL 3 periods, WR -20pp, trades +30%. Do not
   propose tightening intraday stops absent a regime-conditional gate.

ENTRY-PATH DESIGN — DO NOT propose "fixes" for these intentional behaviors:
10. `base_pattern = "none"` and `stage_number > 3` on Minervini entries
    are CORRECT for the leader_continuation entry path (an established
    leader running in stage 4+ has no fresh base by definition). The
    continuation path has its own gates (RS, regime, close_range≥0.15,
    6% stop) and intentionally bypasses the base-pattern stage gate. See
    `orchestrator.py:1389-1400` (live) and `backtester.py:_row_passes_leader_continuation_entry`.
    If you see "none" / late-stage entries, that's not a data-quality
    problem — it's the continuation path operating as designed.
11. `template_score` is a count of ~25 boolean conditions, NOT out of 10.
    Values like 17/18 are HIGH scores that PASS the typical >8 threshold.
"""


# Closed-trade thresholds below which the LLM is told to suppress
# interpretive sections (regime/setup/day-of-week breakdowns) and confine
# proposals to process/instrumentation rather than strategy tuning. The
# review still emits all raw numbers — this is an interpretive guard, not
# a data-display gate.
MIN_SAMPLE_FOR_INTERPRETATION = 15
MIN_SAMPLE_FOR_BREAKDOWN = 30


def _iso_week(d: date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def _last_n_bdays(d: date, n: int) -> list[date]:
    out = []
    cur = d
    while len(out) < n:
        if cur.weekday() < 5:
            out.append(cur)
        cur -= timedelta(days=1)
    return list(reversed(out))


def _compute_trade_stats(outcomes: list) -> dict:
    if not outcomes:
        return {"count": 0}
    rets = [o.get("return_pct") or 0.0 for o in outcomes]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    mfes = [o.get("max_favorable_excursion") for o in outcomes if o.get("max_favorable_excursion") is not None]
    maes = [o.get("max_adverse_excursion") for o in outcomes if o.get("max_adverse_excursion") is not None]
    total_pnl = sum(rets)
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else float("inf") if wins else 0.0
    return {
        "count": len(outcomes),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(outcomes) if outcomes else 0.0,
        "total_return": total_pnl,
        "avg_return": total_pnl / len(outcomes) if outcomes else 0.0,
        "avg_winner": sum(wins) / len(wins) if wins else 0.0,
        "avg_loser": sum(losses) / len(losses) if losses else 0.0,
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "avg_mfe": sum(mfes) / len(mfes) if mfes else None,
        "avg_mae": sum(maes) / len(maes) if maes else None,
        "avg_hold_days": sum((o.get("hold_days") or 0) for o in outcomes) / len(outcomes),
    }


def _breakdown_by(outcomes: list, key: str) -> dict:
    buckets: dict = {}
    for o in outcomes:
        k = o.get(key) or "n/a"
        buckets.setdefault(k, []).append(o)
    return {k: _compute_trade_stats(v) for k, v in buckets.items()}


def _exit_reason_breakdown(outcomes: list) -> dict:
    """Group by exit_reason. Captures the intraday-vs-close question
    proposal #3 asks for: bracket_stop_loss / bracket_take_profit fire
    intraday during RTH; eod_flatten and rule-driven exits (dead_money,
    regime_exit, partial_profit, lost_50dma) fire at scheduled checkpoints.
    """
    buckets: dict = {}
    for o in outcomes:
        k = o.get("exit_reason") or "unknown"
        buckets.setdefault(k, []).append(o)
    return {k: _compute_trade_stats(v) for k, v in buckets.items()}


def _giveback_stats(outcomes: list) -> Optional[dict]:
    """Per-trade `giveback = max_favorable_excursion - return_pct`. Surfaces
    whether exits systematically leave money on the table (high giveback)
    or trades reverse hard from MFE (also high giveback). Returns None when
    no trades have MFE populated.
    """
    pairs = [
        (o.get("max_favorable_excursion"), o.get("return_pct"))
        for o in outcomes
        if o.get("max_favorable_excursion") is not None
        and o.get("return_pct") is not None
    ]
    if not pairs:
        return None
    givebacks = [mfe - ret for mfe, ret in pairs]
    givebacks_sorted = sorted(givebacks)
    n = len(givebacks_sorted)
    median = givebacks_sorted[n // 2]
    p90_idx = max(0, int(n * 0.9) - 1)
    p90 = givebacks_sorted[p90_idx]
    avg = sum(givebacks) / n
    # Exit efficiency = realized/MFE, clamped — only meaningful when MFE>0.
    efficiencies = [ret / mfe for mfe, ret in pairs if mfe and mfe > 0]
    median_eff = (
        sorted(efficiencies)[len(efficiencies) // 2] if efficiencies else None
    )
    return {
        "n": n,
        "avg": avg,
        "median": median,
        "p90": p90,
        "median_efficiency": median_eff,
    }


def _dow_breakdown(outcomes: list) -> dict:
    from dateutil.parser import parse as parse_date
    buckets: dict = {}
    for o in outcomes:
        try:
            d = parse_date(o["entry_date"]).date()
            dow = d.strftime("%A")
        except Exception:
            dow = "unknown"
        buckets.setdefault(dow, []).append(o)
    return {dow: _compute_trade_stats(v) for dow, v in buckets.items()}


def _memories_this_week(db, start_iso: str) -> list[dict]:
    """Agent memories added on or after the week's start."""
    try:
        rows = db.conn.execute(
            """SELECT memory_name, situation, recommendation, created_at
               FROM agent_memories
               WHERE created_at >= ?
               ORDER BY created_at DESC
               LIMIT 40""",
            (start_iso,),
        ).fetchall()
    except Exception:
        return []
    return [dict(r) for r in rows]


def _strategy_daily_returns(db, start: date, end: date) -> list[float]:
    """Pull daily_pl_pct from daily_snapshots in the window."""
    try:
        rows = db.get_snapshots_in_range(start.isoformat(), end.isoformat())
    except Exception:
        return []
    out = []
    for r in rows:
        pct = r.get("daily_pl_pct")
        if pct is None:
            continue
        out.append(float(pct))
    return out


_PROPOSAL_SECTION_HEADERS = (
    "## Proposals (Open-Ended)",
    "## Proposals",
    "## Proposals (Open Ended)",
)


def _parse_and_store_proposals(
    db, *, variant: str, iso_week: str, review_md: str,
) -> int:
    """Parse proposals from the LLM-produced weekly review markdown.

    Expected structure:

        ## Proposals (Open-Ended)

        ### Proposal: <title>
        - **What:** <text>
        - **Why:** <text>
        - **How to validate:** <text>
        - **Risk:** <text>

        ### Proposal: <next title>
        ...

    Tolerant to formatting drift — missing bullets become NULL, unusually
    titled proposal headers still match by starting with "### Proposal:".
    Returns the number of rows inserted.
    """
    if not review_md:
        return 0

    # Locate the proposals section by header match.
    section = ""
    for header in _PROPOSAL_SECTION_HEADERS:
        idx = review_md.find(header)
        if idx != -1:
            section = review_md[idx + len(header):]
            break
    if not section:
        return 0
    # Trim at the next H2 header if any.
    next_h2 = section.find("\n## ")
    if next_h2 != -1:
        section = section[:next_h2]

    # Split into per-proposal blocks on "### Proposal:"
    blocks = []
    cursor = 0
    while True:
        start = section.find("### Proposal:", cursor)
        if start == -1:
            break
        nxt = section.find("### Proposal:", start + 1)
        block = section[start:nxt] if nxt != -1 else section[start:]
        blocks.append(block)
        cursor = start + 1
        if nxt == -1:
            break

    inserted = 0
    for block in blocks:
        title_line, _, body = block.partition("\n")
        title = title_line.replace("### Proposal:", "").strip()[:200] or "Untitled"
        fields = {"what": None, "why": None, "how_to_validate": None, "risk": None}

        # Bullet extractor — handles "**What:** text" and "- **What:** text".
        # Only strip "-" and whitespace on the left, NOT "*" (which is a
        # markdown bold marker the alias explicitly includes).
        for line in body.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("- "):
                stripped = stripped[2:].lstrip()
            lower = stripped.lower()
            for key, alias in (
                ("what", "**what:**"),
                ("why", "**why:**"),
                ("how_to_validate", "**how to validate:**"),
                ("risk", "**risk:**"),
            ):
                if lower.startswith(alias):
                    fields[key] = stripped[len(alias):].strip() or None
                    break
        try:
            db.insert_proposal(
                variant=variant, iso_week=iso_week, title=title, **fields,
            )
            inserted += 1
        except Exception as e:
            logger.warning(
                "insert_proposal(%s, %s) failed: %s", variant, title[:40], e,
            )
    if inserted:
        logger.info(
            "weekly_review[%s]: parsed + stored %d proposal(s)",
            variant, inserted,
        )
    return inserted


def _prior_proposals_block(db, variant: str, max_weeks: int = 8) -> str:
    """Compact summary of prior proposals for this variant — fed into next week's prompt.

    Shows status for everything in the last N weeks, plus any still-open
    proposal regardless of age. Keeps the prompt context bounded.
    """
    try:
        rows = db.get_proposals(variant=variant, limit=50)
    except Exception:
        return ""
    if not rows:
        return ""
    # Filter: keep if open/accepted/tested OR created in last max_weeks.
    cutoff = (date.today() - timedelta(weeks=max_weeks)).isoformat()
    kept = [
        r for r in rows
        if r.get("status") in ("open", "accepted", "tested")
        or (r.get("created_at") or "") >= cutoff
    ]
    if not kept:
        return ""
    lines = ["## Prior proposals (status carried from previous weeks)"]
    lines.append(
        "Use this to avoid re-proposing the same idea; acknowledge what "
        "was tested and whether it worked."
    )
    for r in kept[:20]:
        status = r.get("status", "open")
        title = (r.get("title") or "?")[:80]
        week = r.get("iso_week", "")
        outcome = (r.get("outcome_summary") or "").replace("\n", " ")[:160]
        if outcome:
            lines.append(f"- [{status}] {week} · {title} → {outcome}")
        else:
            lines.append(f"- [{status}] {week} · {title}")
    return "\n".join(lines)


def _build_prompt(
    variant_name: str, window_start: date, window_end: date,
    stats: dict, regime_stats: dict, pattern_stats: dict, dow_stats: dict,
    spy_cmp: dict, qqq_cmp: dict, memories: list[dict],
    prior_proposals_block: str = "",
    exit_reason_stats: Optional[dict] = None,
    giveback: Optional[dict] = None,
) -> str:
    def _fmt_stats(s: dict) -> str:
        if not s or s.get("count", 0) == 0:
            return "  (no trades)"
        return (
            f"  trades={s['count']} · "
            f"win_rate={s['win_rate']:.0%} · "
            f"sum_trade_returns_pct={s['total_return']:+.2%} · "
            f"avg={s['avg_return']:+.2%} · "
            f"PF={s['profit_factor']}"
        )

    def _fmt_buckets(buckets: dict) -> str:
        lines = []
        for k, s in sorted(buckets.items(), key=lambda kv: -(kv[1].get("count") or 0)):
            lines.append(f"- **{k}** — {s.get('count', 0)} trades, "
                         f"WR {s.get('win_rate', 0):.0%}, "
                         f"avg {s.get('avg_return', 0):+.2%}")
        return "\n".join(lines) if lines else "  (n/a)"

    if exit_reason_stats:
        exit_lines = []
        for reason, s in sorted(
            exit_reason_stats.items(), key=lambda kv: -(kv[1].get("count") or 0)
        ):
            exit_lines.append(
                f"- **{reason}** — {s.get('count', 0)} trades, "
                f"WR {s.get('win_rate', 0):.0%}, avg {s.get('avg_return', 0):+.2%}"
            )
        exit_block = "\n".join(exit_lines)
    else:
        exit_block = "  (n/a)"

    if giveback:
        giveback_block = (
            f"  n={giveback['n']} · avg={giveback['avg']:+.2%} · "
            f"median={giveback['median']:+.2%} · p90={giveback['p90']:+.2%}"
        )
        if giveback.get("median_efficiency") is not None:
            giveback_block += f" · median_efficiency={giveback['median_efficiency']:.2f}"
    else:
        giveback_block = "  (no trades with MFE/return populated)"

    memory_block = ""
    if memories:
        lines = []
        for m in memories[:20]:
            sit = (m.get("situation") or "")[:200].replace("\n", " ")
            rec = (m.get("recommendation") or "")[:200].replace("\n", " ")
            lines.append(f"- [{m.get('memory_name')}] situation={sit} · rec={rec}")
        memory_block = "\n".join(lines)
    else:
        memory_block = "(no memories logged this week)"

    n_trades = stats.get("count", 0) if stats else 0
    if n_trades < MIN_SAMPLE_FOR_INTERPRETATION:
        sample_banner = (
            f"\n## SAMPLE-SIZE NOTICE — N={n_trades} closed trades\n"
            f"This window has fewer than {MIN_SAMPLE_FOR_INTERPRETATION} trades. "
            "DO NOT draw directional conclusions from regime / setup / day-of-week "
            "breakdowns; the breakdowns are below for reference only. Profit-factor "
            "values like `inf` or huge numbers at this N are math artifacts, not edge.\n"
            "Proposals MUST be limited to: (a) process/instrumentation, (b) measurement "
            "additions, (c) noise reduction. NO strategy-knob tuning at this N — even "
            "subtractive changes need a multi-period backtest first.\n"
        )
    elif n_trades < MIN_SAMPLE_FOR_BREAKDOWN:
        sample_banner = (
            f"\n## SAMPLE-SIZE NOTICE — N={n_trades} closed trades\n"
            f"Aggregate-level reads are tentative; sub-bucket breakdowns "
            f"(regime/setup/day-of-week) require N≥{MIN_SAMPLE_FOR_BREAKDOWN} for any "
            "directional claim. Treat per-bucket lines as observational only.\n"
        )
    else:
        sample_banner = ""

    return f"""You are a trading-strategy post-mortem analyst writing for a
human trader who can read code but wants high-signal improvement proposals.
Write the weekly review for ONE strategy variant.

{BIAS_AUDIT_STANZA}
{sample_banner}
Variant: **{variant_name}**
Window: {window_start} → {window_end}

## Aggregate stats (closed-trade level)
{_fmt_stats(stats)}
avg_MFE={stats.get('avg_mfe')}  avg_MAE={stats.get('avg_mae')}  avg_hold_days={stats.get('avg_hold_days'):.1f}

## Benchmark comparison (portfolio-equity level)
SPY: strategy_return={spy_cmp.get('strategy_return'):+.2%}, benchmark_return={spy_cmp.get('benchmark_return'):+.2%}, excess={spy_cmp.get('excess_return'):+.2%}, correlation={spy_cmp.get('correlation')}
QQQ: benchmark_return={qqq_cmp.get('benchmark_return'):+.2%}, excess={qqq_cmp.get('excess_return'):+.2%}, correlation={qqq_cmp.get('correlation')}

> **Metric definitions — these are NOT the same number.**
> - `sum_trade_returns_pct` (Aggregate stats): sum of per-trade %-returns for trades that **closed** in the window. Each trade's return is `(exit_price - entry_price) / entry_price` weighted equally per trade. Ignores positions opened before the window or still open at window end. Ignores cash drag and position sizing.
> - `strategy_return` (Benchmark comparison): compounded daily portfolio-equity returns from `daily_snapshots.daily_pl_pct`, which is Alpaca's `(equity - last_equity) / last_equity`. Reflects the **whole account**, including unrealized P&L on still-open positions, cash drag, and position-sizing effects.
> They legitimately diverge when (a) trades open before the window are still appreciating/declining inside it, (b) positions are sized small relative to total equity, or (c) trades cross the window boundary. Do NOT compute "excess return" by subtracting `sum_trade_returns_pct` from `benchmark_return` — only the `strategy_return` line is comparable to benchmarks.
>
> **Important: the gap is mostly arithmetic, not a bug.** With `max_single_position_pct = 0.10` (hard risk gate per CLAUDE.md), a trade returning +15% contributes ~+1.5% to portfolio. So a 3-trade week with `sum_trade_returns_pct = +45%` translates to roughly +4.5% portfolio impact — that's expected, not "sizing/cash drag mystery." Only flag a sizing/exposure concern if (i) avg gross exposure across the window is materially below the implicit `max_total_exposure ≈ 80%` cap **and** (ii) max_positions count was NOT being hit. Otherwise the variant is operating as designed under its risk gates.

## By regime
{_fmt_buckets(regime_stats)}

## By entry pattern / setup
{_fmt_buckets(pattern_stats)}

## By day-of-week
{_fmt_buckets(dow_stats)}

## By exit reason
{exit_block}
> `bracket_stop_loss` and `bracket_take_profit` fire intraday (Alpaca OCO leg).
> `eod_flatten` is the 15:55 ET intraday-strategy close. Rule-driven exits like
> `dead_money`, `regime_exit`, `partial_profit`, `lost_50dma` fire at scheduled
> daily-scan or 5-min-cron checkpoints. This row IS the answer to "intraday vs
> close" — no separate flag is needed.

## Giveback (MFE − realized return)
{giveback_block}
> High median giveback ⇒ exits leaving money on the table or trades reversing
> hard from MFE. Low giveback + low realized return ⇒ trades not running far
> enough. Low giveback + high realized return ⇒ exits well-timed.

## Lessons logged this week (from agent_memories)
{memory_block}

{prior_proposals_block}

## OUTPUT CONTRACT
Write the review using exactly these Markdown H2 headers, in order:

## Scope
List what was analyzed (trade window, variant, data sources) AND what was
NOT (e.g. not checked: open-position reflection, LLM entry quality, slippage
vs backtest, pre-trade universe).

## Performance
Plain-English read of returns vs SPY and QQQ, Sharpe/DD posture, correlation.
Is this variant beating buy-and-hold on this window? What drove the gap?

## Regime Breakdown
One line per regime bucket with observation. If aggregate N is below
{MIN_SAMPLE_FOR_BREAKDOWN}, write "Insufficient sample for directional reads (N={n_trades})"
and list the buckets without commentary.

## Entry Pattern Breakdown
Call out any setup with enough trades to draw even a tentative signal.
Same insufficient-sample rule as above.

## Day-of-Week
Observational only — do not over-read ≤ 2 trades per day. If aggregate N is
below {MIN_SAMPLE_FOR_BREAKDOWN}, mark this section "noise" and skip
inferences.

## Lessons from this week's memories
Synthesize the logged lessons into ≤ 5 themes. Flag any that contradict
or were ignored.

## Proposals (Open-Ended)
1-3 concrete, testable changes. For each, use this template:

### Proposal: <short name>
- **What:** (specific config knob / code path / filter change)
- **Why:** (what the data shows — cite stats above)
- **How to validate:** (backtest config, paper-test gate, what "pass" looks like)
- **Risk:** (what could go wrong, how to revert)
"""


def run_weekly_review(
    *,
    ab_runner,
    config: dict,
    review_date: Optional[date] = None,
) -> dict:
    """Generate a weekly per-variant review.

    `ab_runner` provides access to each variant's `db` and `broker` via
    `ab_runner.orchestrators[variant].db/broker`.
    """
    if not config.get("weekly_strategy_review_enabled", False):
        return {"status": "disabled"}

    from tradingagents.analytics.benchmark import compute_benchmark_comparison
    from tradingagents.llm_clients.factory import create_llm_client

    review_date = review_date or date.today()
    # 7-day trading window ending today — Mon-Fri only → last 5 bdays.
    bdays = _last_n_bdays(review_date, 5)
    start, end = bdays[0], bdays[-1]
    iso_week = _iso_week(review_date)

    dry_run = bool(config.get("weekly_review_dry_run", False))
    base_dir = Path("results/weekly_reviews_dryrun" if dry_run else "results/weekly_reviews")
    out_dir = base_dir / iso_week
    out_dir.mkdir(parents=True, exist_ok=True)

    budget = int(config.get("weekly_review_max_calls", 10))

    # Use the most capable available model for weekly (deep-think). Never
    # mutate global config — `weekly_review_model` is the dedicated knob.
    # 2026-05-02: bumped default to gpt-5.5 after API availability check.
    provider = config.get("llm_provider", "openai")
    deep_model = config.get("weekly_review_model", "gpt-5.5")

    # Shared credentials across variants assumed; if individual variants need
    # different ALPACA creds they're already on the orchestrator's broker.
    summary = {
        "iso_week": iso_week, "variants": {},
        "start": start.isoformat(), "end": end.isoformat(),
    }

    orchestrators = getattr(ab_runner, "orchestrators", {}) or {}
    llm_calls = 0
    for name, orch in orchestrators.items():
        if llm_calls >= budget:
            summary["variants"][name] = {"status": "skipped_budget"}
            continue
        try:
            db = orch.db
            broker = orch.broker
            # Credentials path — try to get from orchestrator's broker.
            api_key = getattr(broker, "_api_key", None) or os.environ.get(
                f"ALPACA_{name.upper()}_API_KEY"
            )
            secret_key = getattr(broker, "_secret_key", None) or os.environ.get(
                f"ALPACA_{name.upper()}_SECRET_KEY"
            )

            outcomes = db.get_trade_outcomes_in_range(start.isoformat(), end.isoformat())
            stats = _compute_trade_stats(outcomes)
            regime_stats = _breakdown_by(outcomes, "regime_at_entry")
            pattern_stats = _breakdown_by(outcomes, "base_pattern")
            dow_stats = _dow_breakdown(outcomes)
            exit_reason_stats = _exit_reason_breakdown(outcomes)
            giveback = _giveback_stats(outcomes)

            strat_returns = _strategy_daily_returns(db, start, end)

            spy_cmp = _safe_bench(
                "SPY", start, end, strat_returns, (api_key, secret_key)
            )
            qqq_cmp = _safe_bench(
                "QQQ", start, end, strat_returns, (api_key, secret_key)
            )

            memories = _memories_this_week(db, start.isoformat())
            prior_block = _prior_proposals_block(db, variant=name)

            prompt = _build_prompt(
                variant_name=name,
                window_start=start, window_end=end,
                stats=stats, regime_stats=regime_stats,
                pattern_stats=pattern_stats, dow_stats=dow_stats,
                spy_cmp=spy_cmp, qqq_cmp=qqq_cmp,
                memories=memories,
                prior_proposals_block=prior_block,
                exit_reason_stats=exit_reason_stats,
                giveback=giveback,
            )

            client = create_llm_client(provider=provider, model=deep_model)
            llm = client.get_llm()
            body = llm.invoke(prompt).content
            llm_calls += 1

            # Red-team pass: second gpt-5.2 call that challenges the primary
            # review. Gated by `weekly_red_team_enabled` (default True).
            # Output appended as a "## Red-Team Counter-Review" section.
            red_team_md = ""
            if config.get("weekly_red_team_enabled", True) and llm_calls < budget:
                try:
                    red_prompt = RED_TEAM_PROMPT.format(
                        raw_stats=prompt.split("## OUTPUT CONTRACT", 1)[0],
                        primary_review=body,
                    )
                    red_client = create_llm_client(provider=provider, model=deep_model)
                    red_body = red_client.get_llm().invoke(red_prompt).content
                    red_team_md = (
                        "\n\n---\n\n## Red-Team Counter-Review\n"
                        + red_body.strip() + "\n"
                    )
                    llm_calls += 1
                except Exception as e:
                    logger.warning(
                        "weekly_review[%s] red-team pass failed: %s", name, e,
                    )
                    red_team_md = (
                        "\n\n---\n\n## Red-Team Counter-Review\n"
                        f"_(red-team pass failed: {e})_\n"
                    )

            out_path = out_dir / f"{name}.md"
            header = (
                f"# Weekly Review — {name} — {iso_week}\n"
                f"_{start} → {end} · generated {review_date}_\n\n"
            )
            out_path.write_text(header + body + red_team_md, encoding="utf-8")

            # Parse the `## Proposals (Open-Ended)` section and persist each
            # proposal so the loop actually closes. Tolerant: log on drift,
            # never raise. Skip in dry-run so test runs don't pollute DB.
            proposals_inserted = 0
            if not dry_run:
                try:
                    proposals_inserted = _parse_and_store_proposals(
                        db, variant=name, iso_week=iso_week, review_md=body,
                    )
                except Exception as e:
                    logger.warning(
                        "weekly_review[%s] proposal parse failed: %s", name, e,
                    )

            summary["variants"][name] = {
                "trades": stats.get("count", 0),
                "output": str(out_path),
                "proposals_inserted": proposals_inserted,
            }
            logger.info(
                "weekly_review[%s] %s: %d trades, review saved to %s",
                name, iso_week, stats.get("count", 0), out_path,
            )
        except Exception as e:
            logger.error(
                "weekly_review[%s] failed: %s", name, e, exc_info=True,
            )
            summary["variants"][name] = {"error": str(e)}

    # Write an index file so the dashboard can list variants for this week.
    try:
        index_lines = [f"# {iso_week} · weekly strategy reviews\n"]
        for name, info in summary["variants"].items():
            if "output" in info:
                fname = Path(info["output"]).name
                index_lines.append(f"- [{name}]({fname}) — {info.get('trades', 0)} trades")
            else:
                index_lines.append(
                    f"- {name} — {info.get('status') or info.get('error') or 'skipped'}"
                )
        (out_dir / "_index.md").write_text(
            "\n".join(index_lines) + "\n", encoding="utf-8"
        )
    except Exception as e:
        logger.warning("weekly_review index write failed: %s", e)

    return summary


def _safe_bench(ticker, start, end, strat_returns, creds):
    from tradingagents.analytics.benchmark import compute_benchmark_comparison
    try:
        return compute_benchmark_comparison(
            ticker=ticker, start=start, end=end,
            strategy_daily_returns=strat_returns,
            alpaca_credentials=creds if creds[0] else None,
        )
    except Exception as e:
        logger.warning("benchmark %s fetch failed: %s", ticker, e)
        return {
            "ticker": ticker, "num_days": 0,
            "strategy_return": sum(strat_returns) if strat_returns else 0.0,
            "benchmark_return": 0.0, "excess_return": 0.0, "correlation": None,
        }
