"""Watchdog checks. Each function returns a list of Alert objects.

A check is read-only against the world: it inspects DB / Alpaca / log
files / state but never mutates the trading system. Mutations live in
the scheduler-side reconciler and orchestrator.

Strict-silence rule: a healthy state must produce zero alerts. Any
threshold here that fires routinely is a misconfiguration to tune down,
not a habit to suppress with broader dedupe.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from tradingagents.automation.events import Categories
from .state import TailedFileState, WatchdogState

logger = logging.getLogger(__name__)

ET = ZoneInfo("US/Eastern")

# ── Variant inventory ─────────────────────────────────────────────────
# (variant_name, db_filename, alpaca_api_env, alpaca_secret_env)
# Mirrors scripts/drift_report.py:VARIANTS — kept in sync intentionally.
VARIANTS: Sequence[Tuple[str, str, str, str]] = (
    ("mechanical",          "trading_mechanical.db",          "ALPACA_MECHANICAL_API_KEY",      "ALPACA_MECHANICAL_SECRET_KEY"),
    ("llm",                 "trading_llm.db",                 "ALPACA_LLM_API_KEY",             "ALPACA_LLM_SECRET_KEY"),
    ("mechanical_v2",       "trading_mechanical_v2.db",       "ALPACA_MECHANICAL_V2_API_KEY",   "ALPACA_MECHANICAL_V2_SECRET_KEY"),
    ("chan_v2",             "trading_chan_v2.db",             "ALPACA_CHAN_V2_API_KEY",         "ALPACA_CHAN_V2_SECRET_KEY"),
    ("intraday_mechanical", "trading_intraday_mechanical.db", "ALPACA_V2_INTRADAY_API_KEY",     "ALPACA_V2_INTRADAY_SECRET_KEY"),
    # chan v1 retired 2026-04-27 (paper_launch_v2.yaml). chan_daily replaces it
    # in the active mix and uses its own Alpaca paper account + DB.
    ("chan_daily",          "trading_chan_daily.db",          "ALPACA_CHAN_DAILY_API_KEY",      "ALPACA_CHAN_DAILY_SECRET_KEY"),
)


# ── Alert dataclass ───────────────────────────────────────────────────


@dataclass
class Alert:
    category: str
    title: str
    body: str
    dedupe_key: str
    priority: str = "high"
    tags: List[str] = field(default_factory=lambda: ["warning"])
    ttl_hours: int = 24


# ── Path helpers ──────────────────────────────────────────────────────


def _repo_root() -> Path:
    base = os.environ.get("BAIBOT_REPO_ROOT")
    if base:
        return Path(base).resolve()
    return Path(__file__).resolve().parents[2]


def _results_dir() -> Path:
    base = os.environ.get("BAIBOT_RESULTS_DIR")
    if base:
        return Path(base).resolve()
    return _repo_root() / "results"


def events_jsonl_path() -> Path:
    return _results_dir() / "service_logs" / "events.jsonl"


def service_log_path() -> Path:
    return _results_dir() / "service_logs" / "automation_service.out.log"


def now_et() -> datetime:
    return datetime.now(ET)


def is_market_hours(now: Optional[datetime] = None) -> bool:
    now = now or now_et()
    if now.weekday() >= 5:
        return False
    market_open = time(9, 30)
    market_close = time(16, 5)
    return market_open <= now.time() <= market_close


def is_scheduler_active_window(now: Optional[datetime] = None) -> bool:
    """True if the scheduler is expected to be running jobs frequently.

    Outside this window the scheduler is intentionally idle (overnight,
    weekend), so log mtime ages without indicating failure. Liveness
    alerts that gate on log-staleness must skip this window or they fire
    false positives every off-hours bucket.

    The scheduler does run a handful of pre-market and post-close jobs
    on weekdays (warehouse refresh, daily review, weekly heartbeat at
    Sat 11:00 ET), but log gaps of several hours between them are
    normal. We pick a generous weekday window — pre-market cron starts
    around 04:00 ET and after-hours work wraps by ~18:00 ET.
    """
    now = now or now_et()
    if now.weekday() >= 5:
        return False
    return time(4, 0) <= now.time() <= time(18, 0)


# NYSE full-close holidays. Refresh annually from
# https://www.nyse.com/markets/hours-calendars. Maintained as a static
# set so the watchdog has no network/dep on a market-calendars library.
_NYSE_HOLIDAYS: frozenset = frozenset({
    # 2026
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed; 7/4 = Saturday)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
    # 2027
    date(2027, 1, 1),
    date(2027, 1, 18),
    date(2027, 2, 15),
    date(2027, 3, 26),   # Good Friday
    date(2027, 5, 31),
    date(2027, 6, 18),   # Juneteenth observed (6/19 = Saturday)
    date(2027, 7, 5),    # Independence Day observed (7/4 = Sunday)
    date(2027, 9, 6),
    date(2027, 11, 25),
    date(2027, 12, 24),  # Christmas observed (12/25 = Saturday)
})


def is_nyse_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in _NYSE_HOLIDAYS


def _scheduler_launchd_alive() -> Optional[bool]:
    """Best-effort check that launchd holds com.tradingagents.scheduler with a live PID.

    Returns True/False if confident, None if launchctl is unavailable —
    callers should treat None as "unknown" and fall back to log-staleness.
    """
    try:
        out = subprocess.run(
            ["launchctl", "list", "com.tradingagents.scheduler"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return False
    m = re.search(r'"PID"\s*=\s*(\d+);', out.stdout)
    if not m:
        return False
    return int(m.group(1)) > 0


# ── File tailing ──────────────────────────────────────────────────────


def _read_new_lines(file_state: TailedFileState) -> Tuple[List[str], int, Optional[int]]:
    """Read lines appended to ``file_state.path`` since ``offset``.

    Returns (lines, new_offset, new_inode). Detects rotation/truncation
    via inode mismatch and resets offset to 0 in that case.
    """
    path = Path(file_state.path)
    if not path.exists():
        return [], 0, None
    try:
        st = path.stat()
    except OSError:
        return [], file_state.offset, file_state.inode

    new_inode = st.st_ino
    start = file_state.offset
    if file_state.inode is not None and new_inode != file_state.inode:
        # File was rotated/replaced — start over.
        start = 0
    elif start > st.st_size:
        # Truncated.
        start = 0

    try:
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read()
            new_offset = f.tell()
    except OSError as e:
        logger.warning("watchdog: read %s failed: %s", path, e)
        return [], file_state.offset, file_state.inode

    text = data.decode("utf-8", errors="replace")
    if not text:
        return [], new_offset, new_inode
    lines = text.splitlines()
    # Drop a trailing partial line — we'll pick it up next tick.
    if not text.endswith("\n") and lines:
        # Roll back offset to the start of the partial line.
        last_line_bytes = len(lines[-1].encode("utf-8")) + 0
        new_offset -= last_line_bytes
        lines = lines[:-1]
    return lines, new_offset, new_inode


# ── Check 1: scheduler liveness ───────────────────────────────────────


def check_scheduler_liveness(state: WatchdogState) -> List[Alert]:
    log_path = service_log_path()
    if not log_path.exists():
        return [
            Alert(
                category="scheduler_liveness",
                title="Scheduler log missing",
                body=f"{log_path} does not exist. Scheduler may not have started.",
                dedupe_key=f"liveness:scheduler_log_missing:{date.today().isoformat()}",
                priority="urgent",
                tags=["warning", "rotating_light"],
                ttl_hours=6,
            )
        ]

    # Non-trading day: the log naturally idles for 24h+ between Friday's
    # post-close jobs and Monday's pre-market jobs (and longer around
    # holidays). Skip the staleness check entirely as long as launchd
    # confirms the scheduler is still loaded with a live PID. If
    # launchctl is unavailable we get None back and fall through to the
    # log-based check, which keeps existing CI behavior.
    if not is_nyse_trading_day(now_et().date()) and _scheduler_launchd_alive() is True:
        return []
    try:
        mtime = datetime.fromtimestamp(log_path.stat().st_mtime, tz=ET)
    except OSError:
        return []

    age_min = (now_et() - mtime).total_seconds() / 60.0
    market_hrs = is_market_hours()
    active_window = is_scheduler_active_window()

    last_done = _last_job_done(log_path)
    last_done_age_min: Optional[float] = None
    if last_done is not None:
        last_done_age_min = (now_et() - last_done).total_seconds() / 60.0

    # Outside the scheduler's active window (weekends, overnight) the log
    # is expected to be quiet. Only escalate if it's been *really* long
    # since the last JOB DONE — that catches a process that died Friday
    # afternoon and is still down Monday morning.
    if not active_window:
        if last_done_age_min is None or last_done_age_min < 14 * 60:
            return []

    threshold_min = 10 if market_hrs else 60
    if age_min <= threshold_min:
        return []

    if market_hrs and last_done_age_min is not None and last_done_age_min < 30:
        # Active recently enough — log mtime may have stalled but jobs are running.
        return []

    # Off-market hours inside the active window (e.g. 04:00–09:30 ET): the
    # scheduler can be idle for an hour+ between pre-market cron jobs even
    # though it's running fine. If launchctl confirms a live PID and the
    # last JOB DONE is within the same ~14h cutoff used for non-trading
    # days, treat as healthy. Without this we fired false positives at
    # 03:00 / 05:00 ET on a healthy scheduler.
    if not market_hrs and _scheduler_launchd_alive() is True:
        if last_done_age_min is not None and last_done_age_min < 14 * 60:
            return []

    bucket = now_et().hour // 6
    return [
        Alert(
            category="scheduler_liveness",
            title=f"Scheduler stale ({age_min:.0f}m)",
            body=(
                f"Last log mtime: {mtime.isoformat()}\n"
                f"Last JOB DONE: {last_done.isoformat() if last_done else 'unknown'}\n"
                f"Threshold: {threshold_min}m. Scheduler may be hung or launchd died."
            ),
            dedupe_key=f"liveness:scheduler:{date.today().isoformat()}:{bucket}",
            priority="urgent",
            tags=["warning", "rotating_light"],
            ttl_hours=6,
        )
    ]


_JOB_DONE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*JOB DONE: (.+?)( —|$)")


def _last_job_done(log_path: Path, label: Optional[str] = None) -> Optional[datetime]:
    """Tail the last ~500 lines and return the most recent JOB DONE timestamp."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()[-500:]
    for line in reversed(lines):
        m = _JOB_DONE_RE.match(line)
        if not m:
            continue
        if label is not None and m.group(2).strip() != label:
            continue
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=ET)
        except ValueError:
            continue
    return None


# ── Check 2: event tail ───────────────────────────────────────────────


# Categories we always escalate (≥1 occurrence triggers an alert).
_EVENT_FIRE_ON_ONE = {
    Categories.WASH_TRADE_REJECT,
    Categories.BRACKET_LEG_MISSING,
    Categories.POSITION_STRANDED,
    Categories.NAKED_POSITION,
    Categories.EXIT_SKIPPED,
    Categories.RS_FILTER_BYPASSED,
    Categories.ADD_ON_SUPERSESSION,
    Categories.JOB_FAILED,
    Categories.DRIFT_DETECTED,
    Categories.ACTIVITY_GAP,
}

_EVENT_PRIORITIES = {
    Categories.BRACKET_LEG_MISSING: "urgent",
    Categories.POSITION_STRANDED: "urgent",
    Categories.NAKED_POSITION: "urgent",
    Categories.ADD_ON_SUPERSESSION: "urgent",
    Categories.WASH_TRADE_REJECT: "high",
    Categories.EXIT_SKIPPED: "high",
    Categories.RS_FILTER_BYPASSED: "high",
    Categories.JOB_FAILED: "high",
    Categories.DRIFT_DETECTED: "high",
    Categories.ACTIVITY_GAP: "high",
}


def check_event_tail(state: WatchdogState) -> List[Alert]:
    state.events_jsonl.path = str(events_jsonl_path())
    lines, new_offset, new_inode = _read_new_lines(state.events_jsonl)
    state.events_jsonl.offset = new_offset
    if new_inode is not None:
        state.events_jsonl.inode = new_inode

    alerts: List[Alert] = []
    today = date.today().isoformat()
    reject_count = 0
    for raw_line in lines:
        if not raw_line.strip():
            continue
        try:
            ev = json.loads(raw_line)
        except Exception:
            continue
        if ev.get("level") == "info":
            continue
        cat = ev.get("category") or ""
        variant = ev.get("variant") or "-"
        symbol = ev.get("symbol") or ""
        code = ev.get("code") or ""

        if cat == Categories.ORDER_REJECT:
            reject_count += 1
            continue  # batched after the loop

        if cat not in _EVENT_FIRE_ON_ONE:
            continue

        dedupe_parts = [cat, variant]
        if symbol:
            dedupe_parts.append(symbol)
        if code:
            dedupe_parts.append(str(code))
        dedupe_parts.append(today)
        dedupe_key = ":".join(dedupe_parts)

        title = f"{cat} [{variant}]"
        if symbol:
            title += f" {symbol}"
        body = json.dumps(ev, indent=2)[:1500]

        alerts.append(
            Alert(
                category=cat,
                title=title,
                body=body,
                dedupe_key=dedupe_key,
                priority=_EVENT_PRIORITIES.get(cat, "high"),
                tags=["warning", "rotating_light"]
                if _EVENT_PRIORITIES.get(cat) == "urgent"
                else ["warning"],
                ttl_hours=24,
            )
        )

    if reject_count:
        # Batch order_reject so a flurry produces one alert, not N.
        bucket = now_et().strftime("%Y-%m-%dT%H")
        alerts.append(
            Alert(
                category=Categories.ORDER_REJECT,
                title=f"Order rejects: {reject_count} this hour",
                body=f"{reject_count} new order_reject event(s) since last sweep.",
                dedupe_key=f"order_reject:hourly:{bucket}",
                priority="high",
                ttl_hours=1,
            )
        )
    return alerts


# ── Check 3: drift ────────────────────────────────────────────────────


def check_drift(state: WatchdogState) -> List[Alert]:
    """Run scripts/drift_report.py --json and alert per drifted (variant, symbol)."""
    script = _repo_root() / "scripts" / "drift_report.py"
    if not script.exists():
        return []
    try:
        # sys.executable points at the watchdog's own venv python — bare
        # "python" doesn't resolve under launchd on macOS Sequoia (Apple
        # removed /usr/bin/python; only python3 is on PATH). Using
        # sys.executable also guarantees the script runs against the same
        # site-packages the watchdog is using.
        proc = subprocess.run(
            [sys.executable, str(script), "--json"],
            cwd=str(_repo_root()),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return [
            Alert(
                category="drift_check",
                title="drift_report.py timed out",
                body="120s exceeded. Alpaca API may be degraded.",
                dedupe_key=f"drift_check:timeout:{now_et().strftime('%Y-%m-%dT%H')}",
                priority="high",
                ttl_hours=1,
            )
        ]
    except Exception as e:
        logger.warning("watchdog: drift_report invocation failed: %s", e)
        return []

    if proc.returncode not in (0, 1):
        # 0 = clean, 1 = drift detected, anything else = bug
        return [
            Alert(
                category="drift_check",
                title=f"drift_report.py exit={proc.returncode}",
                body=(proc.stderr or proc.stdout)[:1500],
                dedupe_key=f"drift_check:bad_exit:{date.today().isoformat()}",
                priority="high",
                ttl_hours=12,
            )
        ]

    try:
        data = json.loads(proc.stdout or "{}")
    except Exception:
        return []

    today = date.today().isoformat()
    alerts: List[Alert] = []
    for variant, rows in (data or {}).items():
        for row in rows or []:
            issues = row.get("issues") or []
            if not issues:
                continue
            symbol = row.get("symbol") or "-"
            issues_csv = ",".join(sorted(set(issues)))
            alerts.append(
                Alert(
                    category="drift",
                    title=f"Drift {variant}: {symbol} ({issues_csv})",
                    body=json.dumps(row, indent=2)[:1500],
                    dedupe_key=f"drift:{variant}:{symbol}:{issues_csv}:{today}",
                    priority="high",
                    ttl_hours=24,
                )
            )
    return alerts


# ── Check 4: naked positions ─────────────────────────────────────────


def check_naked_positions(state: WatchdogState) -> List[Alert]:
    today = date.today().isoformat()
    alerts: List[Alert] = []
    for variant, dbfile, kenv, senv in VARIANTS:
        api = os.environ.get(kenv)
        sec = os.environ.get(senv)
        if not api or not sec:
            continue
        dbpath = _repo_root() / dbfile
        if not dbpath.exists():
            continue

        try:
            con = sqlite3.connect(str(dbpath))
            con.row_factory = sqlite3.Row
            db_naked = {
                row["symbol"]: dict(row)
                for row in con.execute(
                    "SELECT symbol, entry_price, current_stop, stop_order_id "
                    "FROM position_states "
                    "WHERE current_stop IS NOT NULL AND stop_order_id IS NULL"
                ).fetchall()
            }
            con.close()
        except sqlite3.Error as e:
            logger.warning("watchdog: sqlite read %s failed: %s", dbpath, e)
            continue

        if not db_naked:
            continue

        try:
            from alpaca.trading.client import TradingClient
            client = TradingClient(api, sec, paper=True)
            live_positions = {p.symbol for p in client.get_all_positions()}
        except Exception as e:
            logger.warning("watchdog: Alpaca query for %s failed: %s", variant, e)
            continue

        for symbol, row in db_naked.items():
            if symbol not in live_positions:
                continue
            alerts.append(
                Alert(
                    category=Categories.NAKED_POSITION,
                    title=f"Naked position {variant}/{symbol}",
                    body=(
                        f"DB has stop={row.get('current_stop')} but "
                        f"stop_order_id is NULL. Live broker position present.\n"
                        f"Hand-attach a stop or run the reconciler."
                    ),
                    dedupe_key=f"naked:{variant}:{symbol}:{today}",
                    priority="urgent",
                    tags=["warning", "rotating_light"],
                    ttl_hours=12,
                )
            )
    return alerts


# ── Check 5: stranded intraday ──────────────────────────────────────


def check_stranded_intraday(state: WatchdogState) -> List[Alert]:
    """At 16:10 ET on weekdays: flag any open positions on the intraday account."""
    api = os.environ.get("ALPACA_V2_INTRADAY_API_KEY")
    sec = os.environ.get("ALPACA_V2_INTRADAY_SECRET_KEY")
    if not api or not sec:
        return []
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api, sec, paper=True)
        positions = client.get_all_positions()
    except Exception as e:
        logger.warning("watchdog: stranded intraday query failed: %s", e)
        return []

    today = date.today().isoformat()
    alerts: List[Alert] = []
    for p in positions:
        alerts.append(
            Alert(
                category=Categories.POSITION_STRANDED,
                title=f"Stranded intraday: {p.symbol}",
                body=(
                    f"qty={p.qty} avg_entry=${p.avg_entry_price} "
                    f"unrealized={p.unrealized_pl}. "
                    f"flatten_all did not close this — close manually."
                ),
                dedupe_key=f"stranded:intraday_mechanical:{p.symbol}:{today}",
                priority="urgent",
                tags=["warning", "rotating_light"],
                ttl_hours=12,
            )
        )
    return alerts


# ── Check 6: job execution sanity ───────────────────────────────────


# Minimum expected JOB DONE counts per label by 16:30 ET on a market weekday.
_EXPECTED_JOB_COUNTS = {
    "Daily Swing Analysis": 1,
    "Market Open Snapshot": 1,
    "Exit Check Pass": 80,
    # Scheduler logs this label as "Order Reconciliation" (not "Reconciler
    # Pass") — keep this string in sync with run_trading.py / scheduler.py
    # JOB DONE markers or the count silently shows 0 and fires false alerts.
    "Order Reconciliation": 60,
    "Daily Trade Review": 1,
}


def check_job_execution_sanity(state: WatchdogState) -> List[Alert]:
    """Count `JOB DONE: {label}` markers since 00:00 ET today; alert on shortfalls."""
    log_path = service_log_path()
    if not log_path.exists():
        return []
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    today = now_et().date()
    cutoff = datetime.combine(today, time(0, 0), tzinfo=ET)
    counts = {}
    for line in text.splitlines():
        m = _JOB_DONE_RE.match(line)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=ET)
        except ValueError:
            continue
        if ts < cutoff:
            continue
        label = m.group(2).strip()
        counts[label] = counts.get(label, 0) + 1

    alerts: List[Alert] = []
    today_iso = today.isoformat()
    for label, expected in _EXPECTED_JOB_COUNTS.items():
        actual = counts.get(label, 0)
        if actual >= expected:
            continue
        alerts.append(
            Alert(
                category="job_shortfall",
                # Stay ASCII-only: ntfy puts the title in an HTTP header
                # which urllib encodes as latin-1; characters like ≥ blow up
                # the whole notification with a UnicodeEncodeError.
                title=f"Job shortfall: {label} ran {actual}x (expected >={expected})",
                body=(
                    f"Cron '{label}' fired only {actual} times by 16:30 ET. "
                    f"Variant cron may be silently broken."
                ),
                dedupe_key=f"job_shortfall:{label}:{today_iso}",
                priority="high",
                ttl_hours=24,
            )
        )
    return alerts


# ── Check 7: daily activity sanity ──────────────────────────────────


def check_daily_activity_sanity(state: WatchdogState) -> List[Alert]:
    """Verify each variant wrote a daily_snapshots row for today."""
    today = date.today()
    today_iso = today.isoformat()
    if today.weekday() >= 5:
        return []
    alerts: List[Alert] = []
    for variant, dbfile, kenv, _senv in VARIANTS:
        if not os.environ.get(kenv):
            continue
        dbpath = _repo_root() / dbfile
        if not dbpath.exists():
            continue
        try:
            con = sqlite3.connect(str(dbpath))
            row = con.execute(
                "SELECT COUNT(*) FROM daily_snapshots WHERE date = ?",
                (today_iso,),
            ).fetchone()
            con.close()
        except sqlite3.Error as e:
            logger.warning(
                "watchdog: sqlite query %s daily_snapshots failed: %s", dbpath, e
            )
            continue
        count = int(row[0] if row else 0)
        if count > 0:
            continue
        alerts.append(
            Alert(
                category=Categories.ACTIVITY_GAP,
                title=f"No daily_snapshots row for {variant} ({today_iso})",
                body=(
                    f"{variant} did not write a daily_snapshots row by post-close. "
                    "Repro of the 2026-04-13 chan-silence incident."
                ),
                dedupe_key=f"activity_gap:{variant}:{today_iso}",
                priority="high",
                ttl_hours=24,
            )
        )
    return alerts


# ── Check 9: intraday regime gate stuck ─────────────────────────────


_REGIME_BLOCKED_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}).*Minervini regime gate: skipping scan",
)


def check_intraday_regime_gate_stuck(state: WatchdogState) -> List[Alert]:
    """Alert when the intraday Minervini regime gate has blocked all scans
    on N+ recent trading days.

    Catches two failure modes that the rest of the watchdog won't:
      (1) gate mis-configured / SPY data corrupt — gate stays "blocked"
          forever even though the real market is fine
      (2) a legitimate but extended correction — the user wants to know
          the gate is binding hard so they can decide whether to override

    Healthy state: gate may block on individual days during real
    corrections; alert only fires after 3 of last 5 trading days are
    completely blocked.
    """
    log_path = service_log_path()
    if not log_path.exists():
        return []
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    blocked_dates: set[str] = set()
    for line in text.splitlines():
        m = _REGIME_BLOCKED_RE.match(line)
        if m:
            blocked_dates.add(m.group(1))

    # Last 5 trading days ending today (today excluded if it's a weekend).
    today = now_et().date()
    cur = today
    recent_trading_days: List[date] = []
    while len(recent_trading_days) < 5:
        if cur.weekday() < 5:
            recent_trading_days.append(cur)
        cur -= timedelta(days=1)

    blocked_count = sum(
        1 for d in recent_trading_days if d.isoformat() in blocked_dates
    )
    if blocked_count < 3:
        return []

    blocked_iso = sorted(d.isoformat() for d in recent_trading_days
                         if d.isoformat() in blocked_dates)
    return [
        Alert(
            category="regime_gate_stuck",
            title=f"Intraday regime gate blocked {blocked_count}/5 last trading days",
            body=(
                f"Minervini regime gate suppressed every intraday entry "
                f"scan on these recent dates: {', '.join(blocked_iso)}.\n"
                f"If the broader market is in confirmed_uptrend per your view, "
                f"investigate stale SPY data in research_data/market_data.duckdb "
                f"or the regime classifier. To temporarily disable, set "
                f"`minervini_regime_filter: null` for intraday_mechanical in "
                f"experiments/paper_launch_v2.yaml and restart the scheduler."
            ),
            dedupe_key=f"regime_gate_stuck:{date.today().isoformat()}",
            priority="high",
            ttl_hours=24,
        )
    ]


# ── Check 8: log error sweep ────────────────────────────────────────


_LOG_LEVEL_RE = re.compile(r"\b(ERROR|CRITICAL)\b")
_NUMERIC_RE = re.compile(r"\b\d{6,}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)

# Substrings that mark log-error lines as known noise — don't alert on these.
# yfinance fetcher errors fire constantly for delisted/illiquid tickers and
# are not actionable. Notifier transient HTTP failures are retried by the
# notifier itself; if a real outage develops it will surface via the
# heartbeat/event-tail paths, not via spammed log_error alerts.
_LOG_NOISE_SUBSTRINGS: Tuple[str, ...] = (
    "ERROR yfinance:",
    "[ERROR] yfinance:",
    "Failed to send ntfy notification: HTTP",
    "Failed to send Telegram notification: HTTP",
)


def _is_log_noise(line: str) -> bool:
    return any(s in line for s in _LOG_NOISE_SUBSTRINGS)


def check_log_error_sweep(state: WatchdogState) -> List[Alert]:
    state.service_log.path = str(service_log_path())
    lines, new_offset, new_inode = _read_new_lines(state.service_log)
    state.service_log.offset = new_offset
    if new_inode is not None:
        state.service_log.inode = new_inode

    fingerprints: dict[str, list[str]] = {}
    for line in lines:
        if not _LOG_LEVEL_RE.search(line):
            continue
        # Skip lines that are themselves part of the watchdog's own logs.
        if "watchdog" in line.lower():
            continue
        # Skip stack-trace continuations (indented) so we group on the head line.
        if line.startswith(" ") or line.startswith("\t"):
            continue
        if _is_log_noise(line):
            continue
        fp = _fingerprint_log(line)
        fingerprints.setdefault(fp, []).append(line)

    alerts: List[Alert] = []
    today = date.today().isoformat()
    for fp, samples in fingerprints.items():
        count = len(samples)
        # Always alert on count >= 3 in this delta; also alert on first-time
        # fingerprints (count 1 is fine to alert on a never-before-seen one).
        bucket = now_et().strftime("%Y-%m-%dT%H")
        dedupe_key = f"log_err:{fp[:24]}:{today}"
        if count < 3 and dedupe_key in state.alert_dedupe:
            continue
        sample = samples[-1][:400]
        alerts.append(
            Alert(
                category="log_error",
                title=f"Log {count}× error: {fp[:60]}",
                body=f"Last sample:\n{sample}",
                dedupe_key=dedupe_key,
                priority="high",
                ttl_hours=12,
            )
        )
    return alerts


def _fingerprint_log(line: str) -> str:
    """Collapse variable IDs/timestamps so similar messages bucket together."""
    # Drop leading timestamp.
    line = line[20:] if len(line) > 20 and line[4] == "-" and line[10] == " " else line
    line = _UUID_RE.sub("<id>", line)
    line = _NUMERIC_RE.sub("<n>", line)
    return line.strip()[:120]
