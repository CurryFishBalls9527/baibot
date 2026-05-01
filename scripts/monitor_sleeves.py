#!/usr/bin/env python3
"""Monitor for the standalone strategy sleeves (PEAD, CAF, XASSET).

Reads JSONL state from each sleeve's results dir + queries Alpaca for live
account state. Designed to be run interactively or piped to ntfy/Telegram.

Usage:
    ./.venv/bin/python scripts/monitor_sleeves.py             # all sleeves
    ./.venv/bin/python scripts/monitor_sleeves.py PEAD        # one sleeve
    ./.venv/bin/python scripts/monitor_sleeves.py --json      # machine-readable
    ./.venv/bin/python scripts/monitor_sleeves.py --since 2h  # only recent fills
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tradingagents.automation.config  # noqa: E402,F401  (pre-warm broker imports)
from tradingagents.broker.alpaca_broker import AlpacaBroker  # noqa: E402


# Sleeve registry: which env-var prefix maps to which JSONL log dir.
SLEEVES = {
    "PEAD": {
        "log_dir": "results/pead/paper",
        "fills": "fills.jsonl",
        "state": "positions.json",
        "activity": "activity.jsonl",
        "description": "post-earnings drift (≥5% surprise, 20d hold)",
    },
    "CAF": {
        "log_dir": "results/intraday_xsection/caf",
        "fills": "fills.jsonl",
        "state": None,
        "activity": "rebalances.jsonl",
        "description": "closing-auction fade (once-daily 14:30, 15min hold)",
    },
    "XSECTION": {
        "log_dir": "results/intraday_xsection/paper",
        "fills": "fills.jsonl",
        "state": None,
        "activity": "rebalances.jsonl",
        "description": "intraday xsection (morning + closing-auction combined)",
    },
    "XASSET": {
        "log_dir": "results/xasset_rotation/paper",
        "fills": "fills.jsonl",
        "state": None,
        "activity": "rebalances.jsonl",
        "description": "cross-asset rotation (5 ETFs, monthly)",
    },
}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().strip().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _broker_for(prefix: str) -> AlpacaBroker | None:
    api_key = os.environ.get(f"ALPACA_{prefix}_API_KEY")
    secret = os.environ.get(f"ALPACA_{prefix}_SECRET_KEY")
    if not api_key or not secret:
        return None
    if api_key == "your_paper_key_here" or secret == "your_paper_secret_here":
        return None
    try:
        return AlpacaBroker(api_key=api_key, secret_key=secret, paper=True)
    except Exception:
        return None


def _parse_since(spec: str | None) -> datetime | None:
    if not spec:
        return None
    spec = spec.strip().lower()
    if spec.endswith("h"):
        hours = int(spec[:-1])
        return datetime.now(timezone.utc) - timedelta(hours=hours)
    if spec.endswith("d"):
        days = int(spec[:-1])
        return datetime.now(timezone.utc) - timedelta(days=days)
    return None


def collect(prefix: str, since: datetime | None = None) -> dict:
    cfg = SLEEVES[prefix]
    log_dir = Path(cfg["log_dir"])
    summary: dict = {
        "sleeve": prefix,
        "description": cfg["description"],
        "log_dir": str(log_dir),
        "log_dir_exists": log_dir.exists(),
        "alpaca_account": "(not configured)",
        "open_positions_alpaca": None,
        "equity": None,
        "cash": None,
        "buying_power": None,
        "fills_total": 0,
        "fills_recent": 0,
        "median_slippage_bps_recent": None,
        "last_activity": None,
        "open_positions_state": None,
        "errors_recent": [],
    }
    # Alpaca live state
    broker = _broker_for(prefix)
    if broker is not None:
        try:
            acct = broker.get_account()
            positions = broker.trading_client.get_all_positions()
            summary["alpaca_account"] = "ACTIVE"
            summary["equity"] = round(float(acct.equity), 2)
            summary["cash"] = round(float(acct.cash), 2)
            summary["buying_power"] = round(float(acct.buying_power), 2)
            summary["open_positions_alpaca"] = len(positions)
            if positions:
                summary["alpaca_position_symbols"] = sorted(
                    p.symbol for p in positions
                )
        except Exception as exc:
            summary["alpaca_account"] = f"ERROR: {exc}"
    # JSONL fills
    fills = _load_jsonl(log_dir / cfg["fills"]) if cfg["fills"] else []
    summary["fills_total"] = len(fills)
    if since is not None:
        recent = [
            r for r in fills
            if (
                "submit_ts" in r
                and r["submit_ts"]
                and _safe_dt(r["submit_ts"]) is not None
                and _safe_dt(r["submit_ts"]) >= since
            )
        ]
    else:
        recent = fills
    summary["fills_recent"] = len(recent)
    slips = [
        r["slippage_bps"] for r in recent
        if r.get("slippage_bps") is not None
    ]
    if slips:
        summary["median_slippage_bps_recent"] = round(
            statistics.median(slips), 2
        )
        summary["mean_slippage_bps_recent"] = round(
            statistics.mean(slips), 2
        )
    # Errors
    errors = [
        {"ts": r.get("submit_ts"), "symbol": r.get("symbol"),
         "status": r.get("fill_status"), "error": r.get("error")}
        for r in recent
        if r.get("error") or (
            r.get("fill_status") and "fail" in str(r.get("fill_status")).lower()
        )
    ]
    summary["errors_recent"] = errors[-5:]  # last 5
    # State (PEAD has positions.json with explicit exit dates)
    if cfg["state"]:
        state_path = log_dir / cfg["state"]
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
                summary["open_positions_state"] = len(state.get("positions", []))
                summary["state_saved_at"] = state.get("saved_at")
                summary["state_position_summary"] = [
                    {"symbol": p["symbol"],
                     "entry_date": p.get("entry_date"),
                     "exit_target_date": p.get("exit_target_date"),
                     "shares": p.get("shares"),
                     "surprise_pct": p.get("surprise_pct")}
                    for p in state.get("positions", [])
                ]
            except Exception as exc:
                summary["state_error"] = str(exc)
    # Activity (last entry from rebalances/activity log)
    if cfg["activity"]:
        activity = _load_jsonl(log_dir / cfg["activity"])
        if activity:
            summary["last_activity"] = activity[-1]
            summary["activity_total"] = len(activity)
    return summary


def _safe_dt(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def render_text(s: dict) -> str:
    lines = []
    lines.append(f"━━━ {s['sleeve']:<8} ━━━ {s['description']}")
    if s["alpaca_account"] != "ACTIVE":
        lines.append(f"  account: {s['alpaca_account']}")
        lines.append("")
        return "\n".join(lines)
    eq = s["equity"] or 0
    lines.append(
        f"  account: equity=${eq:>10,.2f}  cash=${s['cash']:>10,.2f}  "
        f"BP=${s['buying_power']:>10,.2f}  open={s['open_positions_alpaca']}"
    )
    if s.get("alpaca_position_symbols"):
        lines.append(f"  alpaca-side symbols: {s['alpaca_position_symbols']}")
    if s["open_positions_state"] is not None:
        lines.append(
            f"  state file: {s['open_positions_state']} positions tracked "
            f"(saved {s.get('state_saved_at', '?')})"
        )
        for p in (s.get("state_position_summary") or [])[:8]:
            lines.append(
                f"    {p['symbol']:>6} entry={p['entry_date']} "
                f"exit≥{p['exit_target_date']} qty={p['shares']} "
                f"surprise={p['surprise_pct']:+.1f}%"
            )
    lines.append(
        f"  fills: total={s['fills_total']} recent={s['fills_recent']}  "
        f"median_slip={s.get('median_slippage_bps_recent', 'n/a')} bps  "
        f"mean_slip={s.get('mean_slippage_bps_recent', 'n/a')} bps"
    )
    if s.get("errors_recent"):
        lines.append(f"  ⚠ recent errors ({len(s['errors_recent'])}):")
        for e in s["errors_recent"]:
            lines.append(f"    {e}")
    if s.get("last_activity"):
        a = s["last_activity"]
        lines.append(
            f"  last activity: action={a.get('action')}  "
            f"date={a.get('date') or a.get('rebalance_ts') or a.get('rebalance_date') or '?'}"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sleeves", nargs="*",
                        help="Optional sleeve names (PEAD CAF XSECTION XASSET). Default: all.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--since", default=None,
                        help="Only count fills since (e.g. '2h', '1d'). Default: all-time.")
    args = parser.parse_args()
    targets = args.sleeves or list(SLEEVES.keys())
    since = _parse_since(args.since)

    summaries = [collect(s.upper(), since=since) for s in targets if s.upper() in SLEEVES]
    if args.json:
        print(json.dumps(summaries, indent=2, default=str))
    else:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"Sleeve monitor — {ts}")
        if since:
            print(f"  fills filtered to since={args.since}")
        print()
        for s in summaries:
            print(render_text(s))


if __name__ == "__main__":
    raise SystemExit(main())
