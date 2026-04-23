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
"""


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


def _build_prompt(
    variant_name: str, window_start: date, window_end: date,
    stats: dict, regime_stats: dict, pattern_stats: dict, dow_stats: dict,
    spy_cmp: dict, qqq_cmp: dict, memories: list[dict],
) -> str:
    def _fmt_stats(s: dict) -> str:
        if not s or s.get("count", 0) == 0:
            return "  (no trades)"
        return (
            f"  trades={s['count']} · "
            f"win_rate={s['win_rate']:.0%} · "
            f"total_return={s['total_return']:+.2%} · "
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

    return f"""You are a trading-strategy post-mortem analyst writing for a
human trader who can read code but wants high-signal improvement proposals.
Write the weekly review for ONE strategy variant.

{BIAS_AUDIT_STANZA}

Variant: **{variant_name}**
Window: {window_start} → {window_end}

## Aggregate stats
{_fmt_stats(stats)}
avg_MFE={stats.get('avg_mfe')}  avg_MAE={stats.get('avg_mae')}  avg_hold_days={stats.get('avg_hold_days'):.1f}

## Benchmark comparison
SPY: strategy_return={spy_cmp.get('strategy_return'):+.2%}, benchmark_return={spy_cmp.get('benchmark_return'):+.2%}, excess={spy_cmp.get('excess_return'):+.2%}, correlation={spy_cmp.get('correlation')}
QQQ: benchmark_return={qqq_cmp.get('benchmark_return'):+.2%}, excess={qqq_cmp.get('excess_return'):+.2%}, correlation={qqq_cmp.get('correlation')}

## By regime
{_fmt_buckets(regime_stats)}

## By entry pattern / setup
{_fmt_buckets(pattern_stats)}

## By day-of-week
{_fmt_buckets(dow_stats)}

## Lessons logged this week (from agent_memories)
{memory_block}

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
One line per regime bucket with observation.

## Entry Pattern Breakdown
Call out any setup with enough trades to draw even a tentative signal.

## Day-of-Week
Observational only — do not over-read ≤ 2 trades per day.

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

    # Use gpt-5.2 (deep-think) specifically for weekly, never mutate global.
    provider = config.get("llm_provider", "openai")
    deep_model = config.get("weekly_review_model", "gpt-5.2")

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

            strat_returns = _strategy_daily_returns(db, start, end)

            spy_cmp = _safe_bench(
                "SPY", start, end, strat_returns, (api_key, secret_key)
            )
            qqq_cmp = _safe_bench(
                "QQQ", start, end, strat_returns, (api_key, secret_key)
            )

            memories = _memories_this_week(db, start.isoformat())

            prompt = _build_prompt(
                variant_name=name,
                window_start=start, window_end=end,
                stats=stats, regime_stats=regime_stats,
                pattern_stats=pattern_stats, dow_stats=dow_stats,
                spy_cmp=spy_cmp, qqq_cmp=qqq_cmp,
                memories=memories,
            )

            client = create_llm_client(provider=provider, model=deep_model)
            llm = client.get_llm()
            body = llm.invoke(prompt).content
            llm_calls += 1

            out_path = out_dir / f"{name}.md"
            header = (
                f"# Weekly Review — {name} — {iso_week}\n"
                f"_{start} → {end} · generated {review_date}_\n\n"
            )
            out_path.write_text(header + body, encoding="utf-8")

            summary["variants"][name] = {
                "trades": stats.get("count", 0),
                "output": str(out_path),
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
