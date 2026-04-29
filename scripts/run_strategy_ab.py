#!/usr/bin/env python3
"""Run a strategy A/B comparison for one treatment vs both baselines.

Loads the frozen baseline (live and/or research), builds the treatment by
applying overrides to the baseline's BacktestConfig, and runs the paired
comparison across all configured periods. Emits JSON to
research_data/ab_runs/<change_id>_<flavor>.json.

Usage:
    # Test Change #1 (chandelier) against both baselines
    python scripts/run_strategy_ab.py --change chandelier_exit

    # Only against research baseline (faster iteration)
    python scripts/run_strategy_ab.py --change chandelier_exit --flavor research

    # Custom overrides via KEY=VALUE
    python scripts/run_strategy_ab.py --change my_test \\
        --set use_chandelier_exit=true --set chandelier_atr_multiple=2.5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradingagents.research.backtester import BacktestConfig
from tradingagents.research.seed_universe import load_seed_universe
from tradingagents.research.strategy_ab_runner import (
    DEFAULT_PERIODS,
    PeriodSpec,
    StrategyABRunner,
    compare_period,
    run_single_period,
    write_json,
)
from tradingagents.research.walk_forward import WalkForwardConfig
from tradingagents.research.warehouse import MarketDataWarehouse

_BT_FIELDS = set(BacktestConfig.__dataclass_fields__.keys())
_WF_FIELDS = set(WalkForwardConfig.__dataclass_fields__.keys())

from scripts.freeze_baseline import build_current_live_config, build_research_config, build_wf_config


# Registered changes: each is a dict of BacktestConfig field overrides.
# New changes added here as the rollout progresses (#2, #3, ...).
CHANGES = {
    # Post-B4L cleanup sweep: change timing of candidate admission rather than
    # tightening screens. Hypothesis: weekly snapshots may miss fast-moving
    # breakout/leader-continuation setups that age out before Friday.
    "rebalance_daily": {"rebalance_frequency": "daily"},
    "rebalance_monthly": {"rebalance_frequency": "monthly"},
    # Post-B4L regime-dependent entries/exposure: align the research posture in
    # "uptrend under pressure" with the live automation setting (0.48 instead
    # of the backtester default 0.60). This reduces aggression in the regime
    # where false breakouts are most common without adding more entry filters.
    "pressure_exposure_48": {"target_exposure_uptrend_under_pressure": 0.48},
    # Post-B4L sizing test: use the dormant realized-vol targeting machinery to
    # scale the whole strategy's exposure inversely with recent realized vol.
    # This changes sizing, not entry selection.
    "vol_target_15": {"vol_target_enabled": True, "vol_target_annual": 0.15},
    "chandelier_exit": {"use_chandelier_exit": True, "chandelier_atr_multiple": 3.0},
    "chandelier_2_5x": {"use_chandelier_exit": True, "chandelier_atr_multiple": 2.5},
    "chandelier_3_5x": {"use_chandelier_exit": True, "chandelier_atr_multiple": 3.5},
    "chandelier_2_0x": {"use_chandelier_exit": True, "chandelier_atr_multiple": 2.0},
    # Inverse-accepted changes: used to re-validate prior accept verdicts
    # against the new B2L baseline. `dead_money_off` turns off the B1
    # acceptance; `partial_33_at_12` restores the pre-B2 partial profit.
    "dead_money_off": {"use_dead_money_stop": False},
    "partial_33_at_12": {"partial_profit_trigger_pct": 0.12, "partial_profit_fraction": 0.33},
    # Change #2: dead-money time stop
    "dead_money_5_15": {"use_dead_money_stop": True, "dead_money_min_gain_pct": 0.05, "dead_money_max_days": 15},
    "dead_money_3_10": {"use_dead_money_stop": True, "dead_money_min_gain_pct": 0.03, "dead_money_max_days": 10},
    "dead_money_7_20": {"use_dead_money_stop": True, "dead_money_min_gain_pct": 0.07, "dead_money_max_days": 20},
    # Change #3: breakeven fix (trigger +8%, lock entry+1%)
    "breakeven_8_101": {"breakeven_trigger_pct": 0.08, "breakeven_lock_offset_pct": 0.01},
    # Change #5: industry group RS filter (Minervini leaders-in-leading-groups).
    # min_group_rank_pct=60 => require top 40% group. Lives in WalkForwardConfig.
    "industry_group_60": {"min_group_rank_pct": 60.0},
    "industry_group_50": {"min_group_rank_pct": 50.0},
    "industry_group_70": {"min_group_rank_pct": 70.0},
    # Change #6: RVOL gate on breakouts (volume >= N x 50-day avg)
    "rvol_1_5x": {"min_breakout_volume_ratio": 1.5},
    "rvol_1_25x": {"min_breakout_volume_ratio": 1.25},
    "rvol_2_0x": {"min_breakout_volume_ratio": 2.0},
    # Change #7: partial profit params
    "partial_25_at_20": {"partial_profit_trigger_pct": 0.20, "partial_profit_fraction": 0.25},
    "partial_off": {"partial_profit_fraction": 0.0},
    "partial_50_at_15": {"partial_profit_trigger_pct": 0.15, "partial_profit_fraction": 0.50},
    # Change #8: 50-DMA exit
    "no_50dma": {"use_50dma_exit": False},
    # Change #9: pyramiding (already on in portfolio_backtester with 0.30/0.20
    # fractions). Test its marginal contribution by disabling.
    "pyramiding_off": {"add_on_fraction_1": 0.0, "add_on_fraction_2": 0.0},
    # Change #17 (Wave 2): adaptive dead-money — regime-dependent hold window.
    # B2 already has dead_money on at 10 days / 3% min_gain. This variant lets
    # winners breathe in confirmed uptrends (15d) and cuts stuck names faster
    # in corrections (7d).
    "adaptive_dm_15_10_7": {
        "adaptive_dead_money": True,
        "dead_money_max_days_uptrend": 15,
        "dead_money_max_days_pressure": 10,
        "dead_money_max_days_correction": 7,
    },
    # Stricter min_gain variant — run only if 15_10_7 passes.
    "adaptive_dm_15_10_7_5pct": {
        "adaptive_dead_money": True,
        "dead_money_max_days_uptrend": 15,
        "dead_money_max_days_pressure": 10,
        "dead_money_max_days_correction": 7,
        "dead_money_min_gain_pct": 0.05,
    },
    # Reversed hypothesis: cut dogs faster in confirmed uptrends (they should be
    # moving with the market; if not, exit). Loosen in corrections where chop
    # may temporarily mask a real setup.
    "adaptive_dm_7_10_15": {
        "adaptive_dead_money": True,
        "dead_money_max_days_uptrend": 7,
        "dead_money_max_days_pressure": 10,
        "dead_money_max_days_correction": 15,
    },
    # Change #14 (Wave 2): max positions cap. Diagnostic at B3L shows p95 hits
    # cap (6 live / 12 research) on every period. Bump proportionally 1.67x.
    "cap_10": {"max_positions": 10},  # for live flavor
    "cap_20": {"max_positions": 20},  # for research flavor (scales w/ 12 base)
    "cap_8": {"max_positions": 8},    # milder bump for live
    "cap_16": {"max_positions": 16},  # milder bump for research
    # Change #11 (Wave 2): continuation entry — pick up trending names on EMA
    # pullbacks that never form a fresh base. Raw defaults (template≥7, ≤10%
    # from high, ROC60≥10%, ATR≤6%) produce +460 trades on 2023_2025 but blow
    # DD 2%→34% per diagnostics. Tight variant below restricts to top-quality
    # continuations near the pivot with high momentum and low volatility.
    "continuation_tight": {
        "allow_continuation_entry": True,
        "continuation_min_template_score": 8,
        "continuation_min_roc_60": 0.15,
        "continuation_max_distance_from_high": 0.05,
        "continuation_max_atr_pct": 0.04,
    },
    # B2L parity: add the live `leader_continuation` entry path to the
    # backtester. 7 of 8 live mechanical trades since 2026-04-13 came from
    # this path; B0/B1/B2 never modeled it. Gate params mirror
    # automation/config.py defaults. See plan doc for parity caveat on
    # per-row RS (walk-forward floor applies instead of per-path 75).
    # Change #16 (Wave 2): composite entry scoring. Same slot budget, same
    # entry gates — just re-ranks candidates by a 0-22 composite (template +
    # RS band + base depth + vol contraction + stage + RVOL + ROC60). Target:
    # improve win rate when candidate count exceeds max_positions.
    "composite_on": {"use_composite_scoring": True},
    # v2: drop stage/RS bands, double-weight template + ROC60. Avoids
    # demoting leader_continuation entries (Stage 2 by definition).
    "composite_v2": {"use_composite_scoring": True, "composite_variant": "v2"},
    # Change #15 (Wave 2): RSI-2 mean-reversion sleeve on SPY. Only active
    # when main strategy exposure < 50% (deploys idle capital in flat years).
    # Connors defaults: entry RSI2<10, exit above 5-DMA OR hold >=5d, 20%
    # position, 200-DMA long-term regime filter.
    "rsi2_sleeve": {"rsi2_sleeve_enabled": True},
    # Larger position for stronger signal test
    "rsi2_sleeve_30": {"rsi2_sleeve_enabled": True, "rsi2_sleeve_position_pct": 0.30},
    # Tighter entry threshold (RSI2<5 = more extreme pullback)
    "rsi2_sleeve_tight": {"rsi2_sleeve_enabled": True, "rsi2_sleeve_entry_threshold": 5.0},
    # Smaller position to cut stacked-DD contribution in flat years
    "rsi2_sleeve_10": {"rsi2_sleeve_enabled": True, "rsi2_sleeve_position_pct": 0.10},
    # Combo: smaller + tighter (fewer, smaller bets)
    "rsi2_sleeve_10tight": {
        "rsi2_sleeve_enabled": True,
        "rsi2_sleeve_position_pct": 0.10,
        "rsi2_sleeve_entry_threshold": 5.0,
    },
    # Lower exposure gate: only fire when main is near-fully sidelined.
    # Theoretical Connors play — sleeve and main shouldn't overlap in time.
    "rsi2_sleeve_20low": {
        "rsi2_sleeve_enabled": True,
        "rsi2_sleeve_exposure_threshold": 0.25,
    },
    "rsi2_sleeve_10low": {
        "rsi2_sleeve_enabled": True,
        "rsi2_sleeve_position_pct": 0.10,
        "rsi2_sleeve_exposure_threshold": 0.25,
    },
    # Tightest-DD variant: 10% pos + low exposure + 3-day max hold to cut
    # losing sleeves before they compound a flat-year drawdown.
    # Change #12 (Wave 2): regime-dependent trail stop. Widths tuned per
    # regime — wider in uptrends (let winners breathe), baseline in pressure,
    # tighter in correction (protect profits).
    "regime_trail_12_10_7": {
        "regime_aware_trail": True,
        "trail_stop_pct_uptrend": 0.12,
        "trail_stop_pct_pressure": 0.10,
        "trail_stop_pct_correction": 0.07,
    },
    "regime_trail_15_10_5": {
        "regime_aware_trail": True,
        "trail_stop_pct_uptrend": 0.15,
        "trail_stop_pct_pressure": 0.10,
        "trail_stop_pct_correction": 0.05,
    },
    # Asymmetric: only loosen uptrend, keep correction at 0.10 baseline
    "regime_trail_loose_up": {
        "regime_aware_trail": True,
        "trail_stop_pct_uptrend": 0.15,
        "trail_stop_pct_pressure": 0.10,
        "trail_stop_pct_correction": 0.10,
    },
    # Asymmetric: only tighten correction, keep uptrend at 0.10 baseline
    # Change #13 (Wave 2): cross-asset rotation into TLT when main strategy
    # is light AND regime is not confirmed_uptrend. Exit on regime flip back
    # to uptrend. Targets flat-year idle-cash gap with non-equity exposure.
    "cross_asset_tlt": {"cross_asset_enabled": True},
    "cross_asset_tlt_25": {"cross_asset_enabled": True, "cross_asset_position_pct": 0.25},
    "cross_asset_tlt_75": {"cross_asset_enabled": True, "cross_asset_position_pct": 0.75},
    # IEF is shorter-duration, less sensitive to rates (gentler DD)
    "cross_asset_ief": {"cross_asset_enabled": True, "cross_asset_symbol": "IEF"},
    "cross_asset_ief_25": {
        "cross_asset_enabled": True,
        "cross_asset_symbol": "IEF",
        "cross_asset_position_pct": 0.25,
    },
    "cross_asset_ief_75": {
        "cross_asset_enabled": True,
        "cross_asset_symbol": "IEF",
        "cross_asset_position_pct": 0.75,
    },
    "regime_trail_tight_corr": {
        "regime_aware_trail": True,
        "trail_stop_pct_uptrend": 0.10,
        "trail_stop_pct_pressure": 0.10,
        "trail_stop_pct_correction": 0.07,
    },
    "rsi2_sleeve_tightdd": {
        "rsi2_sleeve_enabled": True,
        "rsi2_sleeve_position_pct": 0.10,
        "rsi2_sleeve_exposure_threshold": 0.25,
        "rsi2_sleeve_max_hold_days": 3,
    },
    "b2l_parity": {
        "use_leader_continuation_entry": True,
        "leader_cont_min_rs_percentile": 75.0,
        "leader_cont_min_adx_14": 12.0,
        "leader_cont_min_close_range_pct": 0.15,
        "leader_cont_min_roc_60": 0.0,
        "leader_cont_min_roc_120": 0.0,
        "leader_cont_max_below_52w_high": 0.30,
        "leader_cont_max_extension_pct": 0.07,
        "leader_cont_max_pullback_pct": 0.08,
    },
    # Upstream-inspired "don't chase" gate: block new entries when QQQ is
    # stretched > 5% above its 21EMA OR its 5-day ROC exceeds 5%. Entry-time
    # only; existing positions unaffected. Pair with the future-blanked probe
    # variant to bias-audit before accepting.
    "market_extension_5_5": {
        "market_extension_filter_enabled": True,
        "market_extension_max_qqq_above_ema21_pct": 0.05,
        "market_extension_max_qqq_roc_5": 0.05,
    },
    # Looser threshold: gate only when QQQ is meaningfully extended.
    "market_extension_8_7": {
        "market_extension_filter_enabled": True,
        "market_extension_max_qqq_above_ema21_pct": 0.08,
        "market_extension_max_qqq_roc_5": 0.07,
    },
    # Future-blanked probe: same 5/5 thresholds, evaluated against QQQ
    # lagged by 1 bar. If this materially changes vs market_extension_5_5,
    # the same-bar variant was reading same-day info (lookahead).
    "market_extension_5_5_lag1": {
        "market_extension_filter_enabled": True,
        "market_extension_max_qqq_above_ema21_pct": 0.05,
        "market_extension_max_qqq_roc_5": 0.05,
        "market_extension_lag_bars": 1,
    },
}

# Overrides applied to EVERY control + treatment to build the accumulated baseline.
# As changes are accepted, add their overrides here.
# B2L = B2 + leader_continuation entry path (parity with live orchestrator).
# Discovered 2026-04-16 that 7 of 8 live mechanical trades came from the
# leader_continuation path that B0/B1/B2 backtests never modeled. All prior
# Wave 1 accept/reject verdicts were measured on an incomplete model of live.
ACCEPTED_CHANGES = {
    "use_dead_money_stop": True,
    "dead_money_min_gain_pct": 0.03,
    "dead_money_max_days": 10,
    # Change #7: disable partial profit (held full positions outperform on both flavors)
    "partial_profit_fraction": 0.0,
    # B2L parity with live leader_continuation path.
    "use_leader_continuation_entry": True,
    "leader_cont_min_rs_percentile": 75.0,
    "leader_cont_min_adx_14": 12.0,
    "leader_cont_min_close_range_pct": 0.15,
    "leader_cont_min_roc_60": 0.0,
    "leader_cont_min_roc_120": 0.0,
    "leader_cont_max_below_52w_high": 0.30,
    "leader_cont_max_extension_pct": 0.07,
    "leader_cont_max_pullback_pct": 0.08,
    # B3L: Change #3 breakeven fix (trigger +8%, lock entry+1%). Rejected at B1
    # for zero-delta (too few trades for breakeven to matter). At B2L, leader-
    # continuation path produces enough winners in the +5-12% window that lock
    # matters. Live 2/4 (+8.5pp 2023_2025, +2.5pp 2018), research 2/4 (+1.4pp
    # 2023_2025, +3.2pp 2018), safety gates green both flavors. Accepted 2026-04-16.
    "breakeven_trigger_pct": 0.08,
    "breakeven_lock_offset_pct": 0.01,
    # B4L: Change #12 regime-dependent trail — tighten only in correction
    # (0.07 vs baseline 0.10), keep uptrend/pressure at 0.10. Live PASS 3/4
    # (+0.2/+0.1/-0.1/+0.4pp), research PASS 2/4 (+5.8pp 2023_2025, +0.5pp
    # 2020). Asymmetric variants that widened uptrend hurt 2023_2025 -6.8pp.
    # Accepted 2026-04-17.
    "regime_aware_trail": True,
    "trail_stop_pct_uptrend": 0.10,
    "trail_stop_pct_pressure": 0.10,
    "trail_stop_pct_correction": 0.07,
}
CURRENT_BASELINE_ID = "B4L"


def _coerce(value: str):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def split_overrides(overrides: dict) -> tuple[dict, dict]:
    bt, wf = {}, {}
    for k, v in overrides.items():
        if k in _WF_FIELDS:
            wf[k] = v
        elif k in _BT_FIELDS:
            bt[k] = v
        else:
            raise ValueError(f"Unknown override key '{k}' (not in BacktestConfig or WalkForwardConfig)")
    return bt, wf


def apply_overrides(base: BacktestConfig, overrides: dict) -> BacktestConfig:
    data = asdict(base)
    data.update(overrides)
    return BacktestConfig(**data)


def apply_wf_overrides(base: WalkForwardConfig, overrides: dict) -> WalkForwardConfig:
    data = asdict(base)
    data.update(overrides)
    return WalkForwardConfig(**data)


def parse_sets(raw_sets):
    overrides = {}
    for s in raw_sets or []:
        if "=" not in s:
            raise ValueError(f"Invalid --set '{s}'; expected KEY=VALUE")
        k, v = s.split("=", 1)
        overrides[k.strip()] = _coerce(v.strip())
    return overrides


def pick_db(period_name: str) -> str:
    # 2023_2025 requires main DB; earlier periods use historical DB.
    # When the live scheduler holds exclusive lock on market_data.duckdb, copy it
    # to /tmp/market_data_copy.duckdb and point the env var AB_MARKET_DB at it.
    # AB_HISTORICAL_DB overrides the pre-2022 DB path (useful when the
    # historical DB needs an augmented schema, e.g. earnings_events backfill).
    import os
    if period_name == "2023_2025":
        return os.environ.get("AB_MARKET_DB", "research_data/market_data.duckdb")
    return os.environ.get("AB_HISTORICAL_DB", "research_data/historical_2014_2021.duckdb")


def run_change_for_flavor(
    change_id: str,
    overrides: dict,
    flavor: str,
    period_filter: list | None,
    out_dir: Path,
) -> dict:
    seed_symbols = load_seed_universe(
        os.environ.get("AB_SEED_UNIVERSE", "research_data/seed_universe.json")
    )
    raw_wf = build_wf_config(flavor)
    raw_base = build_research_config() if flavor == "research" else build_current_live_config()

    accepted_bt, accepted_wf = split_overrides(ACCEPTED_CHANGES)
    change_bt, change_wf = split_overrides(overrides)

    # Accumulated baseline (control): base + all prior accepted changes
    base_config = apply_overrides(raw_base, accepted_bt)
    base_wf = apply_wf_overrides(raw_wf, accepted_wf)
    # Treatment: accumulated baseline + this change's overrides
    treatment = apply_overrides(base_config, change_bt)
    treatment_wf = apply_wf_overrides(base_wf, change_wf)

    periods = [p for p in DEFAULT_PERIODS if not period_filter or p.name in period_filter]
    by_db: dict[str, list[PeriodSpec]] = {}
    for p in periods:
        by_db.setdefault(pick_db(p.name), []).append(p)

    per_period = []
    for db_path, db_periods in by_db.items():
        warehouse = MarketDataWarehouse(db_path=db_path, read_only=True)
        for p in db_periods:
            c_res = run_single_period(warehouse, seed_symbols, p, base_config, base_wf)
            t_res = run_single_period(warehouse, seed_symbols, p, treatment, treatment_wf)
            per_period.append(compare_period(p, c_res, t_res))

    # Rebuild aggregate verdict across all periods
    all_safe = all(p["safety_gates"]["passed"] for p in per_period)
    n_improved = sum(1 for p in per_period if p["delta"]["total_return_pp"] > 0)
    n_degraded_hard = sum(1 for p in per_period if p["delta"]["total_return_pp"] < -3.0)
    verdict = {
        "passed": all_safe and n_improved >= 2 and n_degraded_hard == 0,
        "n_periods": len(per_period),
        "n_improved": n_improved,
        "n_degraded_hard": n_degraded_hard,
        "all_safety_gates_passed": all_safe,
    }

    out = {
        "change_id": change_id,
        "flavor": flavor,
        "baseline_id": CURRENT_BASELINE_ID,
        "accepted_changes": ACCEPTED_CHANGES,
        "overrides": overrides,
        "treatment_config": asdict(treatment),
        "control_config": asdict(base_config),
        "treatment_wf_config": asdict(treatment_wf),
        "control_wf_config": asdict(base_wf),
        "periods": per_period,
        "verdict": verdict,
    }
    out_path = out_dir / f"{change_id}_{flavor}.json"
    write_json(out, out_path)
    return out


def summarize(result: dict) -> str:
    lines = [f"  {result['change_id']} [{result['flavor']}]:"]
    for p in result["periods"]:
        n = p["period"]["name"]
        d = p["delta"]
        c = p["control"]
        t = p["treatment"]
        gates = "✓" if p["safety_gates"]["passed"] else "✗ " + ";".join(p["safety_gates"]["failures"])
        paired = p["paired_t_test"]
        pval = f"p={paired['p_value']:.3f}" if paired.get("p_value") is not None else "p=n/a"
        lines.append(
            f"    {n:10s}: Δreturn={d['total_return_pp']:+5.1f}pp  ΔDD={d['max_drawdown_pp']:+5.1f}pp  "
            f"Δtrades={d['trade_count']:+3d}  ΔwinRate={d['win_rate_pp']:+4.1f}pp  "
            f"(C:{c['total_return']*100:+5.1f}%→T:{t['total_return']*100:+5.1f}%)  {pval}  {gates}"
        )
    v = result["verdict"]
    status = "PASS" if v["passed"] else "FAIL"
    lines.append(
        f"  verdict: {status}  (improved {v['n_improved']}/{v['n_periods']}, "
        f"hard-degraded {v['n_degraded_hard']}, safety_all={v['all_safety_gates_passed']})"
    )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Run a strategy A/B comparison.")
    p.add_argument("--change", required=True, help=f"Registered change ID or arbitrary label. Known: {list(CHANGES)}")
    p.add_argument("--flavor", choices=["live", "research", "both"], default="both")
    p.add_argument("--periods", nargs="*", default=None, help="Subset of periods to run (default: all)")
    p.add_argument("--set", dest="sets", action="append", help="Override BacktestConfig field: KEY=VALUE")
    p.add_argument("--out-dir", default="research_data/ab_runs")
    p.add_argument("--log", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log), format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    overrides = dict(CHANGES.get(args.change, {}))
    overrides.update(parse_sets(args.sets))
    if not overrides:
        logger.error("No overrides supplied. Either use a registered --change or pass --set.")
        return 2

    logger.info("Change '%s' overrides: %s", args.change, overrides)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flavors = ["live", "research"] if args.flavor == "both" else [args.flavor]
    summaries = []
    for flavor in flavors:
        logger.info("=== Running %s flavor ===", flavor)
        result = run_change_for_flavor(args.change, overrides, flavor, args.periods, out_dir)
        summaries.append(summarize(result))

    print()
    print("=" * 80)
    for s in summaries:
        print(s)
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
