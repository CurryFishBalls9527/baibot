#!/usr/bin/env python3
"""PEAD edge-or-no-edge validation backtest.

Question being answered: at production config (5% surprise gate, 20-day
hold, 5%/name, 10 concurrent), does PEAD show clearly positive net
return after costs across multiple market regimes?

Honest about its limits:
  * Data source is yfinance/AV — survivor-biased. Failed/delisted names
    are silently absent from the universe. This INFLATES apparent edge.
  * BUT: our LIVE deployment universe (broad250, top-250 by liquidity)
    is also a survivor universe by construction. Survivor-biased
    backtest approximately matches deployment context.
  * ASYMMETRIC information: if PEAD shows null/negative on biased data,
    it almost certainly has no real edge (bias only works in one direction
    — toward inflating positive results). If PEAD shows positive, real
    edge is likely ~50-70% of measured magnitude.
  * Compounding inflation: total net% over multi-year periods compounds
    unrealistically without capacity caps. We rely on `avg_per_trade_pct`
    and WR as the compounding-independent signals.

Verdict (binary edge gate, matches `run_strategy_ab.py` style):
  PASS = ≥60% of periods net-positive AND avg WR > 55%
         AND no period worse than -10% net AND avg_per_trade_pct > 0
  Anything else = FAIL → recommend killing PEAD live, reclaim Alpaca slot.

Usage:
    .venv/bin/python scripts/run_pead_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_pead_mvp import PEADConfig, run_period  # noqa: E402

# Production config — matches scripts/run_pead_paper.py + the live
# `com.baibot.pead` cron flags.
PROD = dict(
    min_positive_surprise_pct=5.0,
    max_positive_surprise_pct=50.0,
    hold_days=20,
    position_pct=0.05,
    max_concurrent_positions=10,
    entry_lag_days=1,
    half_spread_bps=1.0,
    slippage_bps=2.0,
)

# 7 periods spanning multiple regimes:
# - 2014-2017: pre-bull, choppy
# - 2018:      bear-half (Oct-Dec selloff)
# - 2019:      bull
# - 2020:      V-shaped covid recovery
# - 2021:      late-bull, meme/spec spike
# - 2022:      bear (rate-hike cycle)
# - 2023_2025: bull (AI rally)
PERIODS = [
    ("2014_2017", "2014-01-01", "2017-12-31"),
    ("2018",      "2018-01-01", "2018-12-31"),
    ("2019",      "2019-01-01", "2019-12-31"),
    ("2020",      "2020-01-01", "2020-12-31"),
    ("2021",      "2021-01-01", "2021-12-31"),
    ("2022",      "2022-01-01", "2022-12-31"),
    ("2023_2025", "2023-01-01", "2025-12-31"),
]

DAILY_DB = "research_data/market_data.duckdb"
EARNINGS_DB = "research_data/earnings_data.duckdb"


def main() -> int:
    print("=" * 80)
    print("PEAD validation backtest — production config across 7 regimes")
    print("=" * 80)
    print(f"  config: {PROD}")
    print()
    print("CAVEAT: data is survivor-biased (yfinance/AV exclude delisted).")
    print("Bias INFLATES apparent edge. A null/negative result here is")
    print("STRONGER evidence than a positive one. See script docstring.")
    print()

    cfg = PEADConfig(**PROD)
    results = []
    for label, b, e in PERIODS:
        r = run_period(label, DAILY_DB, b, e, cfg, earnings_db=EARNINGS_DB)
        results.append(r)

    # Per-trade decomposition (compounding-independent).
    for r in results:
        r["avg_per_trade"] = (r["net_pct"] / r["trades"]) if r["trades"] else 0.0

    print(f"{'period':<12} {'events':>7} {'trades':>7} "
          f"{'net%':>9} {'avg/trade%':>11} {'WR':>6}")
    print("-" * 60)
    for r in results:
        print(f"{r['label']:<12} {r['events']:>7} {r['trades']:>7} "
              f"{r['net_pct']:>+9.2f} {r['avg_per_trade']:>+11.4f} "
              f"{r['win_rate']:>6.2%}")
    print()

    # ─── Verdict ──────────────────────────────────────────────────────
    n = len(results)
    n_positive = sum(1 for r in results if r["net_pct"] > 0)
    n_strongly_negative = sum(1 for r in results if r["net_pct"] < -10)
    avg_wr = sum(r["win_rate"] for r in results) / n
    avg_per_trade = sum(r["avg_per_trade"] for r in results) / n
    worst_net = min(r["net_pct"] for r in results)

    pass_pct_periods = n_positive / n >= 0.60
    pass_avg_wr = avg_wr > 0.55
    pass_no_disaster = n_strongly_negative == 0
    pass_avg_per_trade = avg_per_trade > 0

    verdict = (
        pass_pct_periods and pass_avg_wr and pass_no_disaster and pass_avg_per_trade
    )

    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print(f"  Positive in {n_positive}/{n} periods "
          f"({'PASS' if pass_pct_periods else 'FAIL'} ≥60% rule)")
    print(f"  Average WR: {avg_wr:.2%}  "
          f"({'PASS' if pass_avg_wr else 'FAIL'} >55% rule)")
    print(f"  Worst period: {worst_net:+.2f}%  "
          f"({'PASS' if pass_no_disaster else 'FAIL'} no period <-10% rule)")
    print(f"  Avg per-trade return: {avg_per_trade:+.4f}%  "
          f"({'PASS' if pass_avg_per_trade else 'FAIL'} >0 rule)")
    print()
    print(f"OVERALL: {'PASS — keep PEAD running' if verdict else 'FAIL — kill PEAD, free the slot'}")
    print()

    # ─── Surprise-bucket analysis (cheap LLM-gate proxy) ──────────────
    print("=" * 80)
    print("Surprise-bucket subgroup analysis (rough proxy for selectivity)")
    print("=" * 80)
    print("Hypothesis: if a simple 'higher surprise = better' rule beats")
    print("flat 5-50% gating, that's evidence selectivity matters and an")
    print("LLM gate could plausibly add value.")
    print()
    for low, high, label in [
        (5.0, 10.0, "5-10%  (marginal)"),
        (10.0, 20.0, "10-20% (clear beat)"),
        (20.0, 50.0, "20-50% (huge beat)"),
    ]:
        bucket_cfg = PEADConfig(**{**PROD,
                                   "min_positive_surprise_pct": low,
                                   "max_positive_surprise_pct": high})
        bucket_results = []
        for plabel, b, e in PERIODS:
            r = run_period(plabel, DAILY_DB, b, e, bucket_cfg,
                           earnings_db=EARNINGS_DB)
            bucket_results.append(r)
        total_trades = sum(r["trades"] for r in bucket_results)
        avg_wr_bucket = (
            sum(r["win_rate"] * r["trades"] for r in bucket_results)
            / total_trades if total_trades else 0
        )
        avg_per_trade_bucket = (
            sum((r["net_pct"] / r["trades"]) * r["trades"]
                for r in bucket_results if r["trades"])
            / total_trades if total_trades else 0
        )
        print(f"  surprise {label:<22} "
              f"{total_trades:>5} trades  "
              f"WR {avg_wr_bucket:>6.2%}  "
              f"avg/trade {avg_per_trade_bucket:>+.4f}%")
    print()
    print("If the higher buckets show clearly higher avg/trade, the LLM gate")
    print("'BUY only on high-quality beats' has a chance to work.")
    print("If they're flat, surprise % alone doesn't carry the signal.")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
