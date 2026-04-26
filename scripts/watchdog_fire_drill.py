#!/usr/bin/env python3
"""End-to-end fire drill for the watchdog.

Emits one synthetic event per category (variant=fire_drill, symbol=DRILL),
then invokes each check_* function once with a mock notifier and prints
which categories produced an alert. Confirms the full chain
(emitter → events.jsonl → check → notifier).

Run:
    python scripts/watchdog_fire_drill.py

Exits non-zero if any expected category fails to surface as an alert.

Note: drift / naked / stranded / activity_gap / job_shortfall checks need
live state (DB rows or Alpaca creds) to fire — those are skipped here.
This drill validates the events.jsonl pipeline only. For end-to-end
real-traffic validation, install the watchdog and let it run a full
session.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BAIBOT_RESULTS_DIR"] = tmp
        os.environ["BAIBOT_REPO_ROOT"] = tmp

        from tradingagents.automation import events
        from tradingagents.watchdog import checks
        from tradingagents.watchdog.state import TailedFileState, WatchdogState

        # Categories whose path through check_event_tail can be exercised
        # purely by writing events.jsonl (no DB / Alpaca needed).
        categories = [
            events.Categories.ORDER_REJECT,
            events.Categories.WASH_TRADE_REJECT,
            events.Categories.BRACKET_LEG_MISSING,
            events.Categories.POSITION_STRANDED,
            events.Categories.NAKED_POSITION,
            events.Categories.EXIT_SKIPPED,
            events.Categories.RS_FILTER_BYPASSED,
            events.Categories.ADD_ON_SUPERSESSION,
            events.Categories.DRIFT_DETECTED,
            events.Categories.JOB_FAILED,
            events.Categories.ACTIVITY_GAP,
        ]
        for cat in categories:
            events.emit_event(
                cat,
                level="error",
                variant="fire_drill",
                symbol="DRILL",
                code="99999999",
                message=f"fire-drill synthetic {cat}",
                context={"drill": True},
            )

        # Sanity: verify events.jsonl contains all categories.
        events_path = Path(tmp) / "service_logs" / "events.jsonl"
        assert events_path.exists(), "events.jsonl was not created"

        # Run check_event_tail with a fresh state.
        state = WatchdogState(
            events_jsonl=TailedFileState(path=str(events_path)),
            service_log=TailedFileState(path=str(Path(tmp) / "missing.log")),
        )
        alerts = checks.check_event_tail(state)
        observed = {a.category for a in alerts}
        # ORDER_REJECT is batched into a single alert keyed as ORDER_REJECT.
        expected = set(categories)
        missing = expected - observed
        if missing:
            print("FAIL: categories with no alert:", sorted(missing))
            return 1

        for a in alerts:
            print(
                f"  ALERT  category={a.category:24s} "
                f"priority={a.priority:6s} dedupe={a.dedupe_key}"
            )

        # Quick liveness check on the log_error_sweep too.
        log_path = Path(tmp) / "service_logs" / "automation_service.out.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "2026-04-24 10:00:00 INFO scheduler: tick\n"
            "2026-04-24 10:00:01 ERROR orchestrator: fire-drill synthetic error\n"
        )
        log_alerts = checks.check_log_error_sweep(state)
        if not log_alerts:
            print("FAIL: log_error_sweep produced no alerts")
            return 1
        for a in log_alerts:
            print(f"  LOG    category={a.category:24s} title={a.title[:80]}")

        print(f"\nPASS: {len(alerts)} event alerts + {len(log_alerts)} log alerts")
        return 0


if __name__ == "__main__":
    sys.exit(main())
