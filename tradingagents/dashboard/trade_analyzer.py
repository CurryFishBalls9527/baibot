"""LLM-powered analysis of individual closed trades."""

from __future__ import annotations

import logging

from tradingagents.automation.config import build_config
from tradingagents.llm_clients.factory import create_llm_client

logger = logging.getLogger(__name__)


# Tells the daily reviewer to stop reproposing strategy changes that have
# already been swept null on multi-period broad-universe evidence. Without
# this, the LLM keeps suggesting tighter stops / single-indicator filters /
# partial-exits week after week because each individual trade looks like
# it would have benefited — but the counterfactual has been measured and
# rejected. Per-trade observations are still fine; what's blocked is
# turning a pattern-recognition hunch into a strategy proposal.
PRIOR_NULL_FINDINGS_STANZA = """
KNOWN DEAD ENDS — DO NOT propose these as strategy changes in the Lesson
section. Multi-period broad-universe backtests have already rejected them.
You may still note "this trade would have benefited from X" as observation,
but DO NOT phrase it as "we should add X" or "tighten X for future trades":

1. Tighter / earlier / trailing stops or breakeven locks. Swept null in 5+
   memories: ATR-based stops worse on all 3 periods on intraday_mechanical;
   21 chan_v2 exit/sizing variants all worse-or-wash; ORB breakeven-lock
   -30 to -45pp on 2023_25; partial-exit edge collapsed on broad universe;
   Minervini exit lineage at local optimum. Individual trades that "would
   have benefited" from a tighter stop are anchoring on the realized
   outcome — the counterfactual on the trades that ran is the negative
   evidence.

2. Additive single-indicator entry filters (ADX≥X, RSI/MACD confirmation,
   stochastic, etc.). 20+ additive filters rejected in the Minervini
   optimization-ceiling sweep. The repo prior is "subtractive accepted,
   additive rejected" — do not propose adding a new indicator gate.
   Volume-based gates are CONCLUSIVELY null across THREE metric formulations
   AND across all three live entry paths (rule, leader_continuation,
   pyramid add-on):
     - Daily-bar `breakout_volume_ratio` ≥ 1.4× / 1.5×: -54pp / -44pp on
       2023_25 research, leader_continuation (`project_volume_gate_null.md`).
     - Time-of-day `morning_volume_ratio` (09:30-13:30 ET / 20-day same-
       window avg) ≥ 1.0× / 1.2× / 1.5×: -37 to -57pp on 2023_25 across
       both flavors (`project_morning_volume_gate_null.md`).
     - Bar-level `max_bar_rvol_20d` (today's biggest 30m bar / 20-day same-
       slot baseline) ≥ 1.5× / 2.0× AND `max_bar_rvol_intraday` (within-day
       max/mean concentration) ≥ 3.5× / 4.0× on the pyramid add-on path:
       all 4 variants null on broad-evidence sweep, with research +19pp /
       live -0.2pp split being a textbook universe-construction-bias
       fingerprint (`project_max_bar_volume_gate_null.md`).
   The mechanism: mature leaders have above-average ABSOLUTE volume but
   below-average RELATIVE volume vs their own rolling baseline (the 20-day
   MA has caught up). The continuation strategy specifically targets these
   stocks. Gating on relative volume of ANY flavor (daily, summed-window,
   bar-level cross-day, bar-level within-day) filters precisely the trades
   the strategy is designed to capture. The "weak volume" you may see on
   continuation OR pyramid add-on entries is a mechanical truth about
   mature leaders, not a flaw to fix. Do not propose another volume-gate
   variation without a fundamentally new mechanism (e.g. regime-conditional
   gating, gate combined with a separate signal — and even then, very
   high prior of null).

3. "Add partial profit-taking" or "add pyramid scaling." Both are
   universe-ambiguous (positive on seed, null/negative on broad). The
   pyramid path is already wired observational-only on mechanical_v2.

4. "stage=13/16 is too late" or "base_pattern=none is missing data" or
   "template_score is below required threshold." These are NOT anomalies.
   The Minervini lineage has TWO valid entry paths:
   (a) Rule entry — fresh stage-1/2 base with a labeled pattern
       (consolidation/flat_base/vcp/cup_handle), max_stage_number=3 enforced.
   (b) Leader continuation — already-running stage-4+ leader, no fresh
       base, base_label="none" by design, stage gate explicitly BYPASSED
       (orchestrator.py:1389), uses its own gates (close_range≥0.15,
       6% stop, RS still required, regime still checked).
   template_score is a count of passing conditions out of ~25, NOT out of
   10 — values like 17/18 are HIGH scores that pass. Do not treat
   continuation entries as bugs; they are intentional and gated separately.

What IS welcome in the Lesson section: process/instrumentation observations,
specific data-quality flags (impossible numbers, missing fields), and
concrete subtractive proposals (remove a filter, widen a threshold, disable
a sleeve in a specific regime) — but only when the trade in front of you
clearly motivates them.
"""


class TradeAnalyzer:
    def __init__(self):
        config = build_config()
        client = create_llm_client(
            provider=config.get("llm_provider", "openai"),
            # Dedicated knob mirroring weekly_review_model. Defaults to
            # gpt-5.2 so daily post-mortems get the same model strength as
            # the weekly review without forcing the LLM trader (13-agent
            # graph) off its cheaper quick_think_llm.
            model=config.get("daily_review_model", "gpt-5.2"),
        )
        self.llm = client.get_llm()

    def analyze_trade(self, outcome: dict, entry_signal: dict | None) -> str:
        """Generate LLM analysis for a closed trade. Returns markdown string."""
        prompt = self._build_prompt(outcome, entry_signal)
        result = self.llm.invoke(prompt)
        return result.content

    def _build_prompt(self, outcome: dict, entry_signal: dict | None) -> str:
        symbol = outcome.get("symbol", "?")
        entry_price = outcome.get("entry_price")
        exit_price = outcome.get("exit_price")
        return_pct = outcome.get("return_pct")
        hold_days = outcome.get("hold_days")
        exit_reason = outcome.get("exit_reason", "unknown")
        base_pattern = outcome.get("base_pattern", "N/A")
        regime = outcome.get("regime_at_entry", "N/A")
        entry_date = outcome.get("entry_date", "?")
        exit_date = outcome.get("exit_date", "?")

        # Optional enriched context — populated by the automation review loop
        # (backwards compatible: omitted when called from the UI with a plain
        # trade_outcomes row).
        mfe = outcome.get("max_favorable_excursion")
        mae = outcome.get("max_adverse_excursion")
        setup_context = outcome.get("_setup_context") or ""
        variant = outcome.get("_variant", "")

        excursion_section = ""
        if mfe is not None or mae is not None:
            excursion_section = (
                "\nExcursion (hourly bars over trade window):\n"
                f"- Max Favorable (MFE): {mfe:+.2%}\n" if mfe is not None else ""
            )
            if mae is not None:
                excursion_section += f"- Max Adverse (MAE): {mae:+.2%}\n"

        setup_section = ""
        if setup_context:
            setup_section = f"\nSetup Context:\n{setup_context}\n"

        variant_section = f"- Strategy variant: {variant}\n" if variant else ""

        signal_section = ""
        if entry_signal:
            reasoning = entry_signal.get("reasoning") or entry_signal.get("full_analysis") or ""
            confidence = entry_signal.get("confidence")
            action = entry_signal.get("action")
            signal_section = f"""
Entry Signal:
- Action: {action}
- Confidence: {confidence}
- Reasoning: {reasoning[:1000]}
"""

        return f"""You are a trading journal analyst writing a post-mortem for a human trader who is learning technical analysis. Explain this closed trade in plain, concrete, teaching-oriented language. Reference specific levels, indicators, or price events — no vague platitudes.

{PRIOR_NULL_FINDINGS_STANZA}

Trade Details:
- Symbol: {symbol}
{variant_section}- Entry: ${entry_price:.2f} on {entry_date}
- Exit: ${exit_price:.2f} on {exit_date}
- Return: {return_pct:+.2f}%
- Hold Duration: {hold_days} days
- Exit Reason: {exit_reason}
- Base Pattern: {base_pattern}
- Market Regime at Entry: {regime}
{excursion_section}{setup_section}{signal_section}
Write in 4 sections (use markdown H3 headers):

### Entry Thesis
Why did we buy? Name the specific setup and the TA indicators that fired (pivot / ORB level / Chan BSP type / SMA stack / VWAP / volume surge — whichever apply). Be concrete about the price levels.

### What Happened
Narrate the price path from entry to exit in 2-3 short beats. Reference MFE/MAE if they tell a story (e.g. "ran to +8% before giving back").

### Assessment
Did this trade {"succeed" if return_pct and return_pct > 0 else "fail"} for the reason we expected, or by luck? What market condition helped/hurt? Contrast against what a textbook outcome would look like.

### Lesson
One actionable, testable takeaway for future trades with this setup. Avoid generic advice.

Keep total response under 350 words. Quote prices and indicators precisely."""
