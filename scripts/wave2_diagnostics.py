#!/usr/bin/env python3
"""Wave 2 pre-diagnostics (mandatory gate, read-only).

Runs the current B2 config across the 4 mandatory periods and emits four
diagnostic JSONs + a markdown rollup. No A/B, no code changes to live or
backtester logic — just instrumentation via a method wrapper.

Diagnostics:
  1. Position utilization per period (p50/p95 concurrent positions)
  2. Continuation path fire rate (observe-only with allow_continuation_entry=True)
  3. Cash-idle window measurement (% days with gross exposure < 50%)
  4. Top-N PnL concentration (top-3/5/10 trade share per period)

Outputs under research_data/wave2_diagnostics/.

Usage:
    python scripts/wave2_diagnostics.py               # both flavors, all periods
    python scripts/wave2_diagnostics.py --flavor live # live only
    python scripts/wave2_diagnostics.py --periods 2023_2025 2020
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradingagents.research.backtester import BacktestConfig
from tradingagents.research.portfolio_backtester import PortfolioMinerviniBacktester
from tradingagents.research.seed_universe import load_seed_universe
from tradingagents.research.strategy_ab_runner import (
    DEFAULT_PERIODS,
    PeriodSpec,
    run_single_period,
)
from tradingagents.research.walk_forward import WalkForwardConfig
from tradingagents.research.warehouse import MarketDataWarehouse

from scripts.freeze_baseline import (
    build_current_live_config,
    build_research_config,
    build_wf_config,
)
from scripts.run_strategy_ab import ACCEPTED_CHANGES, apply_overrides, apply_wf_overrides, split_overrides


logger = logging.getLogger(__name__)


def pick_db(period_name: str) -> str:
    if period_name == "2023_2025":
        return "research_data/market_data.duckdb"
    return "research_data/historical_2014_2021.duckdb"


def build_b2_configs(flavor: str) -> tuple[BacktestConfig, WalkForwardConfig]:
    """Return the accumulated B2 configs (raw base + ACCEPTED_CHANGES)."""
    raw_wf = build_wf_config(flavor)
    raw_base = build_research_config() if flavor == "research" else build_current_live_config()
    accepted_bt, accepted_wf = split_overrides(ACCEPTED_CHANGES)
    return (
        apply_overrides(raw_base, accepted_bt),
        apply_wf_overrides(raw_wf, accepted_wf),
    )


def compute_utilization(daily_state) -> Dict:
    if daily_state.empty or "positions" not in daily_state.columns:
        return {"p50": 0, "p95": 0, "max": 0, "mean": 0.0, "n_days": 0}
    pos = daily_state["positions"].astype(int).tolist()
    pos_sorted = sorted(pos)
    n = len(pos_sorted)
    p50 = pos_sorted[n // 2]
    p95_idx = min(n - 1, int(0.95 * n))
    p95 = pos_sorted[p95_idx]
    return {
        "p50": int(p50),
        "p95": int(p95),
        "max": int(max(pos)),
        "mean": round(sum(pos) / n, 3),
        "n_days": n,
        "histogram": _histogram(pos),
    }


def _histogram(values: List[int]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for v in values:
        key = str(v)
        hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: int(kv[0])))


def compute_cash_idle(equity_curve) -> Dict:
    if equity_curve.empty or "exposure" not in equity_curve.columns:
        return {
            "pct_days_below_50": 0.0,
            "pct_days_below_25": 0.0,
            "pct_days_below_10": 0.0,
            "mean_exposure": 0.0,
            "n_days": 0,
        }
    exposures = equity_curve["exposure"].astype(float).tolist()
    n = len(exposures)
    return {
        "pct_days_below_50": round(sum(1 for e in exposures if e < 0.50) / n, 4),
        "pct_days_below_25": round(sum(1 for e in exposures if e < 0.25) / n, 4),
        "pct_days_below_10": round(sum(1 for e in exposures if e < 0.10) / n, 4),
        "mean_exposure": round(sum(exposures) / n, 4),
        "n_days": n,
    }


def compute_pnl_concentration(trades_df) -> Dict:
    if trades_df.empty or "pnl" not in trades_df.columns:
        return {
            "total_pnl": 0.0,
            "n_trades": 0,
            "top_1_share": 0.0,
            "top_3_share": 0.0,
            "top_5_share": 0.0,
            "top_10_share": 0.0,
        }
    pnls = trades_df["pnl"].astype(float).tolist()
    total = sum(pnls)
    top = sorted(pnls, reverse=True)

    def share(n: int) -> float:
        if not top or total == 0:
            return 0.0
        return round(sum(top[:n]) / total, 4)

    return {
        "total_pnl": round(total, 2),
        "n_trades": len(pnls),
        "top_1_share": share(1),
        "top_3_share": share(3),
        "top_5_share": share(5),
        "top_10_share": share(10),
        "top_1_pnl": round(top[0], 2) if top else 0.0,
        "top_3_pnl": round(sum(top[:3]), 2),
        "top_5_pnl": round(sum(top[:5]), 2),
    }


def run_b2_baseline_for_period(
    period: PeriodSpec,
    flavor: str,
    seed_symbols: List[str],
) -> Dict:
    """Run B2 as-is over one period and compute diagnostics 1/3/4 from captured state."""
    bt_config, wf_config = build_b2_configs(flavor)
    warehouse = MarketDataWarehouse(db_path=pick_db(period.name), read_only=True)
    logger.info("[%s/%s] running B2 baseline...", flavor, period.name)
    result = run_single_period(warehouse, seed_symbols, period, bt_config, wf_config)
    pr = result.portfolio_result
    return {
        "period": period.name,
        "flavor": flavor,
        "utilization": compute_utilization(pr.daily_state),
        "cash_idle": compute_cash_idle(pr.equity_curve),
        "pnl_concentration": compute_pnl_concentration(pr.trades),
        "summary": pr.summary,
    }


def run_continuation_observe_for_period(
    period: PeriodSpec,
    flavor: str,
    seed_symbols: List[str],
) -> Dict:
    """Run B2 with allow_continuation_entry=True, counting fires via a method wrapper.

    The wrapper counts every call to _row_passes_continuation_entry that returned
    True, keyed by (symbol, date). Because continuation entries can displace
    breakout entries within max_positions, this measures *opportunities*, not
    *additional trades*. The additional-trade count is derived by comparing
    total trades vs the B2 baseline.
    """
    bt_config, wf_config = build_b2_configs(flavor)
    from dataclasses import replace
    bt_config = replace(bt_config, allow_continuation_entry=True)

    fires: List[Dict] = []
    original = PortfolioMinerviniBacktester._row_passes_continuation_entry

    def wrapper(self, row, price):
        ok = original(self, row, price)
        if ok:
            # row.name is the trade date when row is a Series from a DataFrame
            fires.append({
                "date": row.name.date().isoformat() if hasattr(row.name, "date") else str(row.name),
            })
        return ok

    PortfolioMinerviniBacktester._row_passes_continuation_entry = wrapper
    result = None
    crash_msg = None
    try:
        warehouse = MarketDataWarehouse(db_path=pick_db(period.name), read_only=True)
        logger.info("[%s/%s] running B2 + continuation observe...", flavor, period.name)
        try:
            result = run_single_period(warehouse, seed_symbols, period, bt_config, wf_config)
        except Exception as e:
            crash_msg = f"{type(e).__name__}: {e}"
            logger.warning(
                "[%s/%s] continuation-observe simulation crashed late (%s); "
                "fire count still captured from monkey-patch.",
                flavor, period.name, crash_msg,
            )
    finally:
        PortfolioMinerviniBacktester._row_passes_continuation_entry = original

    n_trades = 0
    summary: Dict = {}
    if result is not None:
        pr = result.portfolio_result
        if not pr.trades.empty:
            n_trades = len(pr.trades)
        if pr.summary:
            summary = {
                "total_return": float(pr.summary.get("total_return", 0.0)),
                "benchmark_return": float(pr.summary.get("benchmark_return", 0.0)),
                "max_drawdown": float(pr.summary.get("max_drawdown", 0.0)),
                "sharpe_ratio": float(pr.summary.get("sharpe_ratio", 0.0)),
                "trade_win_rate": float(pr.summary.get("trade_win_rate", 0.0)),
                "avg_trade_return": float(pr.summary.get("avg_trade_return", 0.0)),
                "avg_exposure_pct": float(pr.summary.get("avg_exposure_pct", 0.0)),
            }
    return {
        "period": period.name,
        "flavor": flavor,
        "fire_count": len(fires),
        "unique_fire_days": len({f["date"] for f in fires}),
        "n_trades_with_continuation": n_trades,
        "simulation_completed": result is not None,
        "crash_msg": crash_msg,
        "continuation_summary": summary,
        "fires_sample": fires[:20],
    }


def run_all(flavors: List[str], period_names: List[str], out_dir: Path) -> Dict:
    periods = [p for p in DEFAULT_PERIODS if not period_names or p.name in period_names]
    seed_symbols = load_seed_universe("research_data/seed_universe.json")

    by_diagnostic: Dict[str, Dict] = {
        "utilization": {"by_period": {}},
        "cash_idle": {"by_period": {}},
        "pnl_concentration": {"by_period": {}},
        "continuation_fires": {"by_period": {}},
    }

    for flavor in flavors:
        for period in periods:
            key = f"{flavor}/{period.name}"

            # Baseline pass: diagnostics 1, 3, 4
            baseline = run_b2_baseline_for_period(period, flavor, seed_symbols)
            by_diagnostic["utilization"]["by_period"][key] = baseline["utilization"]
            by_diagnostic["cash_idle"]["by_period"][key] = baseline["cash_idle"]
            by_diagnostic["pnl_concentration"]["by_period"][key] = baseline["pnl_concentration"]

            baseline_trade_count = baseline["summary"].get("total_trades", 0)

            # Continuation observe pass: diagnostic 2
            cont = run_continuation_observe_for_period(period, flavor, seed_symbols)
            cont["baseline_trade_count"] = baseline_trade_count
            cont["baseline_summary"] = baseline["summary"]
            cont["additional_trades_vs_baseline"] = (
                cont["n_trades_with_continuation"] - baseline_trade_count
                if cont["simulation_completed"] else None
            )
            by_diagnostic["continuation_fires"]["by_period"][key] = cont

            logger.info(
                "[%s/%s] done: pos p50=%d p95=%d | idle<50=%.1f%% | top3=%.1f%% | cont_fires=%d",
                flavor,
                period.name,
                baseline["utilization"]["p50"],
                baseline["utilization"]["p95"],
                baseline["cash_idle"]["pct_days_below_50"] * 100,
                baseline["pnl_concentration"]["top_3_share"] * 100,
                cont["fire_count"],
            )

    for name, payload in by_diagnostic.items():
        out = out_dir / f"{name}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info("wrote %s", out)

    return by_diagnostic


def write_rollup(out_dir: Path, results: Dict) -> None:
    """One-page markdown summary of all 4 diagnostics."""
    lines = ["# Wave 2 Pre-diagnostics Rollup", ""]
    lines.append("Source: `scripts/wave2_diagnostics.py` on B2 (accepted: dead_money 3%/10d + partial_off).")
    lines.append("")

    # Diagnostic 1
    lines.append("## 1. Position utilization (gates #14)")
    lines.append("")
    lines.append("| flavor/period | p50 | p95 | max | mean | n_days |")
    lines.append("|---|---|---|---|---|---|")
    for k, v in results["utilization"]["by_period"].items():
        lines.append(f"| {k} | {v['p50']} | {v['p95']} | {v['max']} | {v['mean']} | {v['n_days']} |")
    lines.append("")
    lines.append("**Verdict:** #14 (cap 6→10) is material if p95 hits the cap (6 live / 12 research) regularly; no-op if p95 stays well below.")
    lines.append("")

    # Diagnostic 2
    lines.append("## 2. Continuation path fire rate + return delta (gates #11)")
    lines.append("")
    lines.append("| flavor/period | fires | extra_trades | B2_return | cont_return | Δreturn | B2_DD | cont_DD | B2_wr | cont_wr |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, v in results["continuation_fires"]["by_period"].items():
        extra = v["additional_trades_vs_baseline"]
        extra_str = f"{extra:+d}" if isinstance(extra, int) else "n/a"
        bs = v.get("baseline_summary", {}) or {}
        cs = v.get("continuation_summary", {}) or {}
        b_ret = bs.get("total_return", 0.0)
        c_ret = cs.get("total_return", 0.0) if cs else None
        d_ret = (c_ret - b_ret) if c_ret is not None else None
        b_dd = bs.get("max_drawdown", 0.0)
        c_dd = cs.get("max_drawdown", 0.0) if cs else None
        b_wr = bs.get("trade_win_rate", 0.0)
        c_wr = cs.get("trade_win_rate", 0.0) if cs else None
        fmt_pct = lambda x: f"{x*100:+.1f}%" if x is not None else "n/a"
        fmt_pos = lambda x: f"{x*100:.1f}%" if x is not None else "n/a"
        lines.append(
            f"| {k} | {v['fire_count']} | {extra_str} | "
            f"{fmt_pct(b_ret)} | {fmt_pct(c_ret)} | {fmt_pct(d_ret)} | "
            f"{fmt_pos(b_dd)} | {fmt_pos(c_dd)} | "
            f"{fmt_pos(b_wr)} | {fmt_pos(c_wr)} |"
        )
    lines.append("")
    lines.append("**Note:** continuation run uses `allow_continuation_entry=True` on top of B2; no other knobs changed. Δreturn here is a viability signal, not an A/B verdict.")
    lines.append("")

    # Diagnostic 3
    lines.append("## 3. Cash-idle windows (gates #13, #15)")
    lines.append("")
    lines.append("| flavor/period | %days<10% | %days<25% | %days<50% | mean_exposure |")
    lines.append("|---|---|---|---|---|")
    for k, v in results["cash_idle"]["by_period"].items():
        lines.append(
            f"| {k} | {v['pct_days_below_10']*100:.1f}% | "
            f"{v['pct_days_below_25']*100:.1f}% | "
            f"{v['pct_days_below_50']*100:.1f}% | "
            f"{v['mean_exposure']*100:.1f}% |"
        )
    lines.append("")
    lines.append("**Verdict:** #13 (cross-asset) / #15 (RSI-2 sleeve) have real ceiling if `%days<50%` is large in flat years (2015/2018).")
    lines.append("")

    # Diagnostic 4
    lines.append("## 4. Top-N PnL concentration (gates #16)")
    lines.append("")
    lines.append("| flavor/period | n_trades | top1 | top3 | top5 | top10 | total_pnl |")
    lines.append("|---|---|---|---|---|---|---|")
    for k, v in results["pnl_concentration"]["by_period"].items():
        lines.append(
            f"| {k} | {v['n_trades']} | {v['top_1_share']*100:.1f}% | "
            f"{v['top_3_share']*100:.1f}% | {v['top_5_share']*100:.1f}% | "
            f"{v['top_10_share']*100:.1f}% | ${v['total_pnl']:,.0f} |"
        )
    lines.append("")
    lines.append("**Verdict:** #16 (composite scoring) has high ceiling if top-3 share is very large (few winners carry the book); low ceiling if PnL is diffuse.")
    lines.append("")

    out = out_dir / "rollup.md"
    out.write_text("\n".join(lines) + "\n")
    logger.info("wrote %s", out)


def main() -> int:
    p = argparse.ArgumentParser(description="Wave 2 pre-diagnostics (read-only)")
    p.add_argument("--flavor", choices=["live", "research", "both"], default="both")
    p.add_argument("--periods", nargs="*", default=None)
    p.add_argument("--out-dir", default="research_data/wave2_diagnostics")
    p.add_argument("--log", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log), format="%(asctime)s [%(levelname)s] %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flavors = ["live", "research"] if args.flavor == "both" else [args.flavor]
    results = run_all(flavors, args.periods or [], out_dir)
    write_rollup(out_dir, results)
    logger.info("all diagnostics written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
