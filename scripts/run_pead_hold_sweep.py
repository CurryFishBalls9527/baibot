#!/usr/bin/env python3
"""PEAD hold-period sweep — backtest the strategy across multiple hold
windows (5/10/15/20/30/45/60 trading days) on multiple time periods to
validate (or refute) the academic 20-day default for OUR universe.

⚠⚠⚠ SURVIVORSHIP-BIASED — DO NOT USE OUTPUT TO PICK A HOLD WINDOW ⚠⚠⚠

When run 2026-05-02, this sweep showed a textbook survivor-bias
fingerprint: WR climbed monotonically from 5d → 60d in EVERY regime,
and 2023_25 60d showed an absurd +8418% net. That's not "PEAD edge
peaks at 60 days"; that's "longer holds capture more of the surviving
mega-caps' compounding bull run, while failures aren't in the data
to drag returns down."

The data source (yfinance/AV) excludes delisted names. 0/15 known-
failed stocks are present in earnings_events. Longer holds favor
survivors disproportionately, which produces the misleading
"longer = better" pattern.

DO NOT use this script's output to change `PEADConfig.hold_days` from
the production 20-day default. For proper validation, need PIT data
source (Polygon $99/mo or similar). See
`memory/project_pead_validation_failed.md`.

────────────────────────────────────────────────────────────────────

Original intent (kept for reference if/when we have clean data):

Why this exists: production PEAD currently uses `hold_days=20` (academic
PEAD literature default). That assumption was inherited, never validated.
With the LLM-gated A/B treatment arm now armed and waiting to accumulate
60-90 days of live data, we want to lock in the right hold window BEFORE
the verdict gets baked into the comparison. If e.g. 15 days is materially
better, both arms (control + treatment) should run on 15 days during the
A/B; otherwise we're testing the LLM gate on a suboptimal substrate.

Reuses `run_pead_mvp.py`'s machinery (PEADConfig, run_period, simulate)
directly — same signal logic, same cost model, same lookahead probe path.

Usage:
    .venv/bin/python scripts/run_pead_hold_sweep.py
    .venv/bin/python scripts/run_pead_hold_sweep.py --include-shorts
    .venv/bin/python scripts/run_pead_hold_sweep.py --periods 2023_2025 2024_2025
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reuse the MVP infrastructure directly — no duplication.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_pead_mvp import PEADConfig, run_period  # noqa: E402

# Production-aligned config (matches scripts/run_pead_paper.py defaults).
PROD_CONFIG_BASE = dict(
    min_positive_surprise_pct=5.0,
    max_positive_surprise_pct=50.0,
    position_pct=0.05,
    max_concurrent_positions=10,
    entry_lag_days=1,
    half_spread_bps=1.0,
    slippage_bps=2.0,
)

HOLD_WINDOWS = [5, 10, 15, 20, 30, 45, 60]

DEFAULT_PERIODS = [
    ("2018",       "2018-01-01", "2018-12-31"),
    ("2020",       "2020-01-01", "2020-12-31"),
    ("2023_2025",  "2023-01-01", "2025-12-31"),
    ("2014_2017",  "2014-01-01", "2017-12-31"),  # extra OOS — different regime
]


def main() -> int:
    p = argparse.ArgumentParser()
    # daily_bars and earnings_events live in different DBs after 2026-05-01 split.
    p.add_argument("--daily-db", default="research_data/market_data.duckdb",
                   help="DB containing daily_bars (default market_data.duckdb)")
    p.add_argument("--earnings-db", default="research_data/earnings_data.duckdb",
                   help="DB containing earnings_events (default earnings_data.duckdb)")
    p.add_argument("--include-shorts", action="store_true",
                   help="Add L/S variants alongside long-only.")
    p.add_argument("--periods", nargs="*", default=None,
                   help=f"Subset of period names. Default all: "
                        f"{[p[0] for p in DEFAULT_PERIODS]}")
    p.add_argument("--hold-windows", nargs="*", type=int, default=None,
                   help=f"Override hold-day list. Default: {HOLD_WINDOWS}")
    args = p.parse_args()

    periods = (
        DEFAULT_PERIODS
        if not args.periods
        else [p for p in DEFAULT_PERIODS if p[0] in args.periods]
    )
    holds = args.hold_windows or HOLD_WINDOWS

    print("PEAD HOLD-PERIOD SWEEP")
    print(f"  config:   surprise [{PROD_CONFIG_BASE['min_positive_surprise_pct']:.0f}%, "
          f"{PROD_CONFIG_BASE['max_positive_surprise_pct']:.0f}%] · "
          f"size {PROD_CONFIG_BASE['position_pct']*100:.0f}%/name · "
          f"max {PROD_CONFIG_BASE['max_concurrent_positions']} concurrent · "
          f"entry_lag {PROD_CONFIG_BASE['entry_lag_days']}d")
    print(f"  hold:     {holds}")
    print(f"  periods:  {[p[0] for p in periods]}")
    print(f"  costs:    half_spread={PROD_CONFIG_BASE['half_spread_bps']}bps "
          f"slippage={PROD_CONFIG_BASE['slippage_bps']}bps")
    if args.include_shorts:
        print(f"  L/S enabled (mirrors symmetric on negative surprises)")
    print()

    # Per-period table layout
    header_fmt = (
        f"{'period':<11} {'hold_d':>6} {'events':>7} {'trades':>7} "
        f"{'gross%':>9} {'net%':>9} {'WR':>6} {'avg_ret_per_trade%':>20}"
    )
    row_fmt = (
        "{period:<11} {hold:>6} {events:>7} {trades:>7} "
        "{gross:>+9.2f} {net:>+9.2f} {wr:>6.2%} {avg:>20.2f}"
    )

    # Collect all results so we can also emit a cross-period summary
    # at the end.
    results: dict[str, dict[int, dict]] = {}  # period → hold → result

    for plabel, b, e in periods:
        print(f"\n=== Period {plabel} ({b} → {e}) ===")
        print(header_fmt)
        print("-" * 95)
        results[plabel] = {}
        for hold in holds:
            cfg = PEADConfig(hold_days=hold, **PROD_CONFIG_BASE)
            r = run_period(plabel, args.daily_db, b, e, cfg,
                           earnings_db=args.earnings_db)
            results[plabel][hold] = r
            avg_per_trade = r["net_pct"] / r["trades"] if r["trades"] else 0.0
            print(row_fmt.format(
                period=plabel, hold=hold, events=r["events"],
                trades=r["trades"], gross=r["gross_pct"],
                net=r["net_pct"], wr=r["win_rate"], avg=avg_per_trade,
            ))

    # ─── Cross-period summary: best hold per period + overall ────────
    print("\n=== Cross-period summary — net% by hold window ===")
    print(f"{'hold_d':<8} " + " ".join(f"{p[0]:>11}" for p in periods)
          + f" {'avg':>9} {'min':>9}")
    print("-" * (8 + 12 * len(periods) + 22))
    for hold in holds:
        nets = [results[p[0]][hold]["net_pct"] for p in periods]
        avg = sum(nets) / len(nets)
        worst = min(nets)
        line = f"{hold:<8} " + " ".join(f"{n:>+11.2f}" for n in nets)
        print(f"{line} {avg:>+9.2f} {worst:>+9.2f}")

    # Pick the best by both criteria
    best_by_avg = max(holds, key=lambda h: sum(results[p[0]][h]["net_pct"]
                                               for p in periods))
    best_by_min = max(holds, key=lambda h: min(results[p[0]][h]["net_pct"]
                                               for p in periods))
    prod_hold = 20  # current production
    prod_avg = sum(results[p[0]][prod_hold]["net_pct"] for p in periods) / len(periods)
    best_avg_score = sum(results[p[0]][best_by_avg]["net_pct"]
                         for p in periods) / len(periods)
    print()
    print(f"  Production hold ({prod_hold}d) avg net %: {prod_avg:+.2f}")
    print(f"  Best by AVG net %:  hold={best_by_avg}d (avg {best_avg_score:+.2f}, "
          f"delta vs prod {best_avg_score - prod_avg:+.2f}pp)")
    print(f"  Best by WORST-PERIOD net %: hold={best_by_min}d "
          f"(worst {min(results[p[0]][best_by_min]['net_pct'] for p in periods):+.2f}, "
          f"vs prod-worst {min(results[p[0]][prod_hold]['net_pct'] for p in periods):+.2f})")
    print()
    print("  Recommendation: prefer 'best by worst-period' for robustness "
          "(survivorship-bias safer than 'best by avg' which can chase outliers).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
