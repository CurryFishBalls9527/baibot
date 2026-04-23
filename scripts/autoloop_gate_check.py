#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

YEARS = {"2023_2025": 3.0, "2020": 1.0, "2018": 1.0}
RET_FLOORS = {"2023_2025": 72.8, "2020": 20.0, "2018": 15.0}

def annualized(total_return_pct: float, years: float) -> float:
    return (1 + total_return_pct / 100.0) ** (1 / years) - 1

def check_row(row: dict) -> tuple[bool, str, dict]:
    period = row["period"]
    years = YEARS[period]
    total_return = float(row["total_return_pct"])
    max_dd = float(row["max_drawdown_pct"])
    trades = int(row["total_trades"])
    ann = annualized(total_return, years)
    calmar_floor_dd = total_return / years
    if total_return < RET_FLOORS[period]:
        return False, f"gate_{period}_return", {"annualized": ann, "calmar_floor_dd": calmar_floor_dd}
    if max_dd > calmar_floor_dd:
        return False, f"gate_{period}_calmar", {"annualized": ann, "calmar_floor_dd": calmar_floor_dd}
    if trades < 100:
        return False, f"gate_{period}_trades", {"annualized": ann, "calmar_floor_dd": calmar_floor_dd}
    return True, "pass", {"annualized": ann, "calmar_floor_dd": calmar_floor_dd}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_json")
    args = ap.parse_args()
    payload = json.loads(Path(args.run_json).read_text())
    rows = payload.get("rows", [])
    by_period = {row["period"]: row for row in rows}
    results = []
    overall = True
    for period in ["2023_2025", "2020", "2018"]:
        if period not in by_period:
            print(f"FAIL missing_{period}")
            return 2
        ok, reason, extra = check_row(by_period[period])
        results.append((period, ok, reason, extra, by_period[period]))
        overall &= ok
    for period, ok, reason, extra, row in results:
        print(json.dumps({
            "period": period,
            "ok": ok,
            "reason": reason,
            "total_return_pct": row["total_return_pct"],
            "max_drawdown_pct": row["max_drawdown_pct"],
            "total_trades": row["total_trades"],
            "annualized_return": extra["annualized"],
            "max_allowed_drawdown_pct": extra["calmar_floor_dd"],
        }))
    print("PASS" if overall else "FAIL")
    return 0 if overall else 1

if __name__ == "__main__":
    raise SystemExit(main())
