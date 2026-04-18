"""Strategy A/B runner.

Runs two BacktestConfig variants (control, treatment) through the
WalkForwardBacktester on identical data and emits a structured comparison
that includes safety gates and a paired t-test on per-trade returns.

Not to be confused with tradingagents/testing/ab_runner.py, which orchestrates
LIVE paper-trading variants. This module is research-only.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .backtester import BacktestConfig
from .walk_forward import WalkForwardBacktester, WalkForwardConfig, WalkForwardResult
from .warehouse import MarketDataWarehouse

logger = logging.getLogger(__name__)


@dataclass
class PeriodSpec:
    name: str
    start_date: str
    end_date: str
    trade_start_date: Optional[str] = None


@dataclass
class SafetyGates:
    max_dd_relative_worsen: float = 0.15  # treatment DD cannot be >15% worse than control DD
    min_trade_count_ratio: float = 0.50  # treatment trade count cannot be <50% of control
    max_return_degradation_pp: float = 3.0  # no period may degrade by >3pp


def run_single_period(
    warehouse: MarketDataWarehouse,
    seed_symbols: List[str],
    period: PeriodSpec,
    config: BacktestConfig,
    wf_config: Optional[WalkForwardConfig] = None,
) -> WalkForwardResult:
    """Run one BacktestConfig through the walk-forward backtester for one period."""
    wfb = WalkForwardBacktester(
        warehouse=warehouse,
        wf_config=wf_config,
        backtest_config=config,
    )
    return wfb.run(
        seed_symbols=seed_symbols,
        start_date=period.start_date,
        end_date=period.end_date,
        trade_start_date=period.trade_start_date,
    )


def _extract_summary(result: WalkForwardResult) -> Dict:
    s = result.portfolio_result.summary or {}
    return {
        "total_return": float(s.get("total_return", 0.0)),
        "benchmark_return": float(s.get("benchmark_return", 0.0)),
        "max_drawdown": float(s.get("max_drawdown", 0.0)),
        "sharpe_ratio": float(s.get("sharpe_ratio", 0.0)),
        "total_trades": int(s.get("total_trades", 0)),
        "trade_win_rate": float(s.get("trade_win_rate", 0.0)),
        "avg_trade_return": float(s.get("avg_trade_return", 0.0)),
        "avg_exposure_pct": float(s.get("avg_exposure_pct", 0.0)),
    }


def _paired_t_test(control_trades: pd.DataFrame, treatment_trades: pd.DataFrame) -> Dict:
    """Paired t-test on per-trade returns. Pairs by (symbol, entry_date)."""
    if control_trades.empty or treatment_trades.empty:
        return {"n_pairs": 0, "mean_diff": 0.0, "t_stat": 0.0, "p_value": None}

    # Normalize columns we depend on
    key_cols = [c for c in ("symbol", "entry_date") if c in control_trades.columns]
    if len(key_cols) < 2:
        return {"n_pairs": 0, "mean_diff": 0.0, "t_stat": 0.0, "p_value": None,
                "note": "missing key columns for pairing"}

    merged = control_trades.merge(
        treatment_trades,
        on=key_cols,
        suffixes=("_c", "_t"),
        how="inner",
    )
    if merged.empty or "return_pct_c" not in merged.columns:
        return {"n_pairs": 0, "mean_diff": 0.0, "t_stat": 0.0, "p_value": None}

    diffs = (merged["return_pct_t"] - merged["return_pct_c"]).dropna()
    n = len(diffs)
    if n < 2:
        return {"n_pairs": int(n), "mean_diff": float(diffs.mean()) if n else 0.0,
                "t_stat": 0.0, "p_value": None}

    mean_d = float(diffs.mean())
    sd_d = float(diffs.std(ddof=1))
    if sd_d == 0:
        return {"n_pairs": int(n), "mean_diff": mean_d, "t_stat": 0.0, "p_value": 1.0}
    t_stat = mean_d / (sd_d / math.sqrt(n))

    # Two-sided p-value via scipy if present; else approximate via normal tail.
    try:
        from scipy.stats import t as _t
        p_value = float(2 * (1 - _t.cdf(abs(t_stat), df=n - 1)))
    except ImportError:
        # Normal approximation (acceptable for n > 30)
        p_value = float(math.erfc(abs(t_stat) / math.sqrt(2)))

    return {"n_pairs": int(n), "mean_diff": mean_d, "t_stat": float(t_stat), "p_value": p_value}


def _check_safety_gates(control: Dict, treatment: Dict, gates: SafetyGates) -> Dict:
    failures: List[str] = []

    c_dd = control["max_drawdown"]
    t_dd = treatment["max_drawdown"]
    if c_dd > 0 and t_dd > c_dd * (1 + gates.max_dd_relative_worsen):
        failures.append(
            f"DD worsened by {(t_dd - c_dd) / c_dd:.1%} (gate: {gates.max_dd_relative_worsen:.0%})"
        )

    c_n = control["total_trades"]
    t_n = treatment["total_trades"]
    if c_n > 0 and t_n < c_n * gates.min_trade_count_ratio:
        failures.append(f"Trade count dropped to {t_n}/{c_n} (gate: ≥{gates.min_trade_count_ratio:.0%})")

    return_delta_pp = (treatment["total_return"] - control["total_return"]) * 100
    if return_delta_pp < -gates.max_return_degradation_pp:
        failures.append(
            f"Return degraded by {return_delta_pp:.1f}pp (gate: ≥-{gates.max_return_degradation_pp:.1f}pp)"
        )

    return {"passed": len(failures) == 0, "failures": failures}


def compare_period(
    period: PeriodSpec,
    control_result: WalkForwardResult,
    treatment_result: WalkForwardResult,
    gates: Optional[SafetyGates] = None,
) -> Dict:
    gates = gates or SafetyGates()
    c = _extract_summary(control_result)
    t = _extract_summary(treatment_result)
    paired = _paired_t_test(
        control_result.portfolio_result.trades,
        treatment_result.portfolio_result.trades,
    )
    safety = _check_safety_gates(c, t, gates)
    return {
        "period": asdict(period),
        "control": c,
        "treatment": t,
        "delta": {
            "total_return_pp": round((t["total_return"] - c["total_return"]) * 100, 2),
            "max_drawdown_pp": round((t["max_drawdown"] - c["max_drawdown"]) * 100, 2),
            "sharpe": round(t["sharpe_ratio"] - c["sharpe_ratio"], 3),
            "trade_count": t["total_trades"] - c["total_trades"],
            "win_rate_pp": round((t["trade_win_rate"] - c["trade_win_rate"]) * 100, 2),
        },
        "paired_t_test": paired,
        "safety_gates": safety,
    }


class StrategyABRunner:
    """Runs (control, treatment) pair across multiple backtest periods."""

    def __init__(
        self,
        warehouse: MarketDataWarehouse,
        seed_symbols: List[str],
        wf_config: Optional[WalkForwardConfig] = None,
        gates: Optional[SafetyGates] = None,
    ):
        self.warehouse = warehouse
        self.seed_symbols = seed_symbols
        self.wf_config = wf_config
        self.gates = gates or SafetyGates()

    def run(
        self,
        control: BacktestConfig,
        treatment: BacktestConfig,
        periods: List[PeriodSpec],
    ) -> Dict:
        per_period: List[Dict] = []
        for p in periods:
            logger.info("=== Period %s (%s → %s) ===", p.name, p.start_date, p.end_date)
            c_res = run_single_period(self.warehouse, self.seed_symbols, p, control, self.wf_config)
            t_res = run_single_period(self.warehouse, self.seed_symbols, p, treatment, self.wf_config)
            per_period.append(compare_period(p, c_res, t_res, self.gates))
        return {
            "periods": per_period,
            "verdict": self._verdict(per_period),
        }

    def run_baseline(
        self,
        config: BacktestConfig,
        periods: List[PeriodSpec],
    ) -> Dict:
        """Run only control config (no comparison) to freeze a baseline artifact."""
        out: List[Dict] = []
        for p in periods:
            logger.info("=== Baseline period %s (%s → %s) ===", p.name, p.start_date, p.end_date)
            res = run_single_period(self.warehouse, self.seed_symbols, p, config, self.wf_config)
            summary = _extract_summary(res)
            trades = res.portfolio_result.trades
            trades_dict = trades.to_dict(orient="records") if not trades.empty else []
            out.append({"period": asdict(p), "summary": summary, "trades": trades_dict})
        return {"periods": out, "config": asdict(config)}

    def _verdict(self, per_period: List[Dict]) -> Dict:
        """Aggregate pass/fail across periods per the plan's criteria:
        - No period may degrade return by >3pp
        - Improve ≥2 of N periods
        - All periods pass safety gates
        """
        all_safe = all(p["safety_gates"]["passed"] for p in per_period)
        n_improved = sum(1 for p in per_period if p["delta"]["total_return_pp"] > 0)
        n_degraded_hard = sum(
            1 for p in per_period if p["delta"]["total_return_pp"] < -self.gates.max_return_degradation_pp
        )
        passed = all_safe and n_improved >= 2 and n_degraded_hard == 0
        return {
            "passed": passed,
            "n_periods": len(per_period),
            "n_improved": n_improved,
            "n_degraded_hard": n_degraded_hard,
            "all_safety_gates_passed": all_safe,
        }


def write_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=str)
    logger.info("Wrote %s", path)


DEFAULT_PERIODS: Tuple[PeriodSpec, ...] = (
    PeriodSpec("2023_2025", "2022-01-01", "2025-12-31", "2023-01-01"),
    PeriodSpec("2020", "2019-01-01", "2020-12-31", "2020-01-01"),
    PeriodSpec("2018", "2017-01-01", "2018-12-31", "2018-01-01"),
    PeriodSpec("2015", "2014-01-01", "2015-12-31", "2015-01-01"),
)
