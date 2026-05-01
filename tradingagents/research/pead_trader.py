"""PEAD (post-earnings-announcement drift) live paper trader.

Daily-frequency strategy: long stocks the morning after they report a positive
EPS surprise of >= threshold, hold for N trading days, exit at close.

Why structurally separate from `xasset_rotation_trader.py`:
  * Stateful — each open position has a SPECIFIC exit_target_date (not
    a stateless rebalance to a basket). State persists across runs in
    a JSON file.
  * Many simultaneous positions (one per qualifying earnings event)
  * Universe is event-driven — symbols come from `earnings_events` table
  * No periodic rebalance — entries are event-triggered, exits are
    time-triggered

Backtest results (V2 config: ≥5% surprise, 20-day hold, long-only,
1+2 bps round-trip cost on the 104-symbol earnings universe):
  * 2018:    +0.91%
  * 2020:   +42.58%
  * 2023_25: +221.89%
  * Lookahead probe (entry_lag=2 vs 1) clean
  * Cost-stress 5x: 2023_25 still +183%

Honest discount for survivorship + AI-rally concentration: real-world
expected ~30-50%/year net at realistic Alpaca-paper costs. Highest
expected-alpha candidate from the research log.

Architecture:
  * `is_eligible_for_run(today)` — only acts on trading days
  * `find_new_signals(today)` — query earnings_events for events that
    became actionable since last run (uses prior-day boundary like
    backtester)
  * `open_positions(signals)` — submit market BUY for each new signal
  * `check_exits(today)` — for each open position whose exit_target_date
    has been reached, submit market SELL
  * State stored in JSON: list of {symbol, entry_date, entry_order_id,
    entry_price, shares, surprise_pct, exit_target_date}
  * Idempotent — re-running on the same day is safe (skips already-known
    signals, doesn't double-exit)

Safety defaults: dry_run=True, max_gross_exposure=0.5 cap, position_pct=0.05,
max_concurrent_positions=10.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import duckdb
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from tradingagents.broker.alpaca_broker import AlpacaBroker

logger = logging.getLogger(__name__)


@dataclass
class PEADConfig:
    """Strategy + safety knobs."""
    # Signal
    min_positive_surprise_pct: float = 5.0
    max_positive_surprise_pct: float = 50.0  # cap for data-noise prints
    # Holding
    hold_days: int = 20  # trading days
    # Sizing
    position_pct: float = 0.05  # of equity per name
    max_concurrent_positions: int = 10
    max_per_symbol_positions: int = 1
    max_gross_exposure: float = 0.5  # safety cap
    # Universe gate (None = all symbols in earnings_events)
    universe_path: Optional[str] = None  # e.g. "research_data/intraday_top250_universe.json"
    # Data
    daily_db_path: str = "research_data/market_data.duckdb"
    # Operational
    log_dir: Path = field(default_factory=lambda: Path("results/pead/paper"))
    dry_run: bool = True


@dataclass
class PEADPositionState:
    """Persisted record of one open trade."""
    symbol: str
    surprise_pct: float
    event_date: str        # ISO date of earnings event
    entry_date: str        # ISO date of entry
    exit_target_date: str  # ISO date — earliest session we'll exit on
    entry_order_id: Optional[str]
    entry_price: Optional[float]
    intent_price: Optional[float]
    shares: int
    dry_run: bool

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class PEADFillRecord:
    """One row per submitted order — entry or exit."""
    submit_ts: str
    fill_ts: Optional[str]
    symbol: str
    action: str            # "open_long" | "exit_long"
    side: str              # "buy" | "sell"
    qty: int
    intent_price: float
    fill_price: Optional[float]
    fill_status: str
    slippage_bps: Optional[float]
    surprise_pct: Optional[float]
    event_date: Optional[str]
    bars_held: Optional[int]
    dry_run: bool
    order_id: Optional[str]
    error: Optional[str] = None


class PEADTrader:
    def __init__(self, config: PEADConfig, broker: "AlpacaBroker"):
        self.config = config
        self.broker = broker
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.config.log_dir / "positions.json"
        self._fills_path = self.config.log_dir / "fills.jsonl"
        self._activity_path = self.config.log_dir / "activity.jsonl"
        self._universe = self._load_universe_filter()

    # ── Universe ─────────────────────────────────────────────────────

    def _load_universe_filter(self) -> Optional[set[str]]:
        if not self.config.universe_path:
            return None
        try:
            payload = json.loads(Path(self.config.universe_path).read_text())
            symbols = payload["symbols"] if isinstance(payload, dict) else payload
            return set(symbols)
        except Exception as exc:
            logger.warning("Universe load failed (%s) — accepting all symbols", exc)
            return None

    # ── Calendar ─────────────────────────────────────────────────────

    def is_trading_day(self, today: datetime) -> bool:
        """Is today a US-equity trading session per Alpaca's calendar?"""
        try:
            from alpaca.trading.requests import GetCalendarRequest

            today_date = today.date()
            cal_req = GetCalendarRequest(
                start=today_date, end=today_date,
            )
            cal = self.broker.trading_client.get_calendar(cal_req)
            return any(c.date == today_date for c in cal)
        except Exception as exc:
            logger.warning("Alpaca calendar lookup failed (%s) — falling back to weekday rule", exc)
            return today.weekday() < 5

    def trading_days_after(
        self, start: datetime, n: int,
    ) -> Optional[datetime]:
        """Return the date that is `n` trading days AFTER `start` (exclusive)."""
        try:
            from alpaca.trading.requests import GetCalendarRequest

            start_date = start.date()
            # 60 calendar days easily covers 20 trading days.
            cal_req = GetCalendarRequest(
                start=start_date, end=start_date + timedelta(days=n * 2 + 30),
            )
            cal = self.broker.trading_client.get_calendar(cal_req)
            future = sorted({c.date for c in cal if c.date > start_date})
            if len(future) < n:
                return None
            return datetime.combine(future[n - 1], datetime.min.time())
        except Exception as exc:
            logger.warning("Alpaca calendar lookup failed (%s) — using weekday approximation", exc)
            d = start
            added = 0
            while added < n:
                d = d + timedelta(days=1)
                if d.weekday() < 5:
                    added += 1
            return d

    # ── State persistence ───────────────────────────────────────────

    def load_state(self) -> list[PEADPositionState]:
        if not self._state_path.exists():
            return []
        try:
            payload = json.loads(self._state_path.read_text())
        except Exception as exc:
            logger.error("State file corrupt: %s — starting empty", exc)
            return []
        return [PEADPositionState(**row) for row in payload.get("positions", [])]

    def save_state(self, positions: list[PEADPositionState]) -> None:
        payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "positions": [p.to_dict() for p in positions],
        }
        # Atomic write
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self._state_path)

    # ── Signal: query yesterday's earnings ───────────────────────────

    def find_new_signals(self, today: datetime) -> list[dict]:
        """Find earnings events that became actionable since the last run.

        AMC events from prior trading day → actionable today
        BMO events from today → actionable today (but ideally the cron
        runs after open so the bar exists)

        Excludes symbols we already have a position in.
        """
        cfg = self.config
        # Pull events from past 3 calendar days to be safe (covers weekend + holiday gaps).
        end_dt = today
        start_dt = today - timedelta(days=4)
        try:
            con = duckdb.connect(cfg.daily_db_path, read_only=True)
            df = con.execute(
                """
                SELECT symbol, event_datetime, surprise_pct, time_hint
                FROM earnings_events
                WHERE is_future = false
                  AND surprise_pct IS NOT NULL
                  AND event_datetime >= ?
                  AND event_datetime <= ?
                ORDER BY event_datetime
                """,
                [start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                 end_dt.strftime("%Y-%m-%d %H:%M:%S")],
            ).fetchdf()
            con.close()
        except Exception as exc:
            logger.error("earnings_events query failed: %s", exc)
            return []

        if df.empty:
            return []
        df["event_datetime"] = pd.to_datetime(df["event_datetime"])
        df["event_date"] = df["event_datetime"].dt.normalize()

        # Filter: surprise within bounds
        df = df[
            (df["surprise_pct"] >= cfg.min_positive_surprise_pct)
            & (df["surprise_pct"] <= cfg.max_positive_surprise_pct)
        ]
        if df.empty:
            return []

        # Universe filter
        if self._universe is not None:
            df = df[df["symbol"].isin(self._universe)]
        if df.empty:
            return []

        # Map event → first eligible session for entry.
        today_date = today.date()
        signals = []
        for _, row in df.iterrows():
            sym = row["symbol"]
            event_date = row["event_date"].date()
            time_hint = str(row.get("time_hint", "amc"))
            # Determine which session is the "next session after the event"
            if time_hint == "bmo":
                # Before market open — the event date IS the first session
                first_eligible_session_date = event_date
            else:
                # AMC or unknown — assume after-close, first eligible is NEXT session
                # (use Alpaca calendar for accuracy)
                next_day = self.trading_days_after(
                    datetime.combine(event_date, datetime.min.time()), 1,
                )
                if next_day is None:
                    continue
                first_eligible_session_date = next_day.date()
            # Only include signals where the first eligible session is today.
            # (Past-eligible signals from earlier in the week were either taken
            # or missed; we don't enter retroactively.)
            if first_eligible_session_date != today_date:
                continue
            signals.append({
                "symbol": sym,
                "surprise_pct": float(row["surprise_pct"]),
                "event_date": event_date.isoformat(),
                "first_eligible_session": first_eligible_session_date.isoformat(),
                "time_hint": time_hint,
            })
        return signals

    # ── Sizing ───────────────────────────────────────────────────────

    def compute_target_dollars(self) -> float:
        try:
            account = self.broker.get_account()
            equity = float(account.equity)
        except Exception as exc:
            logger.warning("get_account failed: %s — assuming $100k", exc)
            equity = 100_000.0
        gross = min(self.config.position_pct, self.config.max_gross_exposure)
        return equity * gross

    # ── Order submission ────────────────────────────────────────────

    def _capture_intent(self, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}
        try:
            quotes = self.broker.get_latest_prices(symbols)
        except Exception as exc:
            logger.warning("get_latest_prices failed: %s", exc)
            return {}
        return {s: float(p) for s, p in quotes.items() if p and p > 0}

    def _submit_buy(
        self, symbol: str, qty: int, intent_price: float,
    ) -> tuple[Optional[str], Optional[float], str, Optional[str]]:
        """Submit a market BUY. Returns (order_id, fill_price, status, error)."""
        if self.config.dry_run:
            return None, None, "dry_run", None
        from tradingagents.broker.models import OrderRequest as _OrderRequest
        order = _OrderRequest(
            symbol=symbol, side="buy", qty=qty, order_type="market", time_in_force="day",
        )
        try:
            result = self.broker.submit_order(order)
        except Exception as exc:
            return None, None, "submit_failed", str(exc)
        final = self._poll(result.order_id)
        return result.order_id, final.filled_avg_price, final.status, None

    def _submit_close(
        self, symbol: str,
    ) -> tuple[Optional[str], Optional[float], str, Optional[str]]:
        """Submit a market close (Alpaca infers SELL for long position)."""
        if self.config.dry_run:
            return None, None, "dry_run", None
        try:
            result = self.broker.close_position(symbol)
        except Exception as exc:
            return None, None, "close_failed", str(exc)
        final = self._poll(result.order_id)
        return result.order_id, final.filled_avg_price, final.status, None

    def _poll(self, order_id: str, attempts: int = 12, delay: float = 1.0):
        import time as _time
        terminal = {"filled", "canceled", "cancelled", "expired", "rejected"}
        last = None
        for _ in range(attempts):
            try:
                last = self.broker.get_order(order_id)
            except Exception as exc:
                logger.warning("get_order(%s) failed: %s", order_id, exc)
                _time.sleep(delay)
                continue
            status_l = str(last.status).lower().split(".")[-1]
            if status_l in terminal or last.filled_avg_price:
                return last
            _time.sleep(delay)
        if last is None:
            from tradingagents.broker.models import OrderResult as _OrderResult
            return _OrderResult(
                order_id=order_id, symbol="?", side="?", qty=None,
                notional=None, order_type="market", status="poll_timeout",
                filled_qty=0.0, filled_avg_price=None,
            )
        return last

    @staticmethod
    def _slippage_bps(intent: Optional[float], fill: Optional[float], side: str) -> Optional[float]:
        if intent is None or fill is None or intent <= 0:
            return None
        raw = (fill - intent) / intent * 1e4
        return round(raw if side == "buy" else -raw, 2)

    # ── Logging ─────────────────────────────────────────────────────

    def append_fill(self, record: PEADFillRecord) -> None:
        with self._fills_path.open("a") as fh:
            fh.write(json.dumps(record.__dict__, default=str) + "\n")

    def append_activity(self, payload: dict) -> None:
        with self._activity_path.open("a") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")

    # ── Main entry ──────────────────────────────────────────────────

    def open_new_positions(
        self,
        signals: list[dict],
        already_open_symbols: set[str],
        target_dollars: float,
    ) -> list[PEADPositionState]:
        """Submit buys for new signals and return their state records."""
        new_states: list[PEADPositionState] = []
        if not signals:
            return new_states
        # Filter: skip symbols we already hold
        candidates = [s for s in signals if s["symbol"] not in already_open_symbols]
        # Cap by max_concurrent — leave room for existing positions
        room = max(0, self.config.max_concurrent_positions - len(already_open_symbols))
        if len(candidates) > room:
            # Prefer larger surprise % when capacity-bound
            candidates = sorted(
                candidates, key=lambda s: s["surprise_pct"], reverse=True,
            )[:room]

        intents = self._capture_intent([s["symbol"] for s in candidates])
        submit_ts = datetime.now(timezone.utc).isoformat()

        for sig in candidates:
            sym = sig["symbol"]
            intent_price = intents.get(sym)
            if intent_price is None or intent_price <= 0:
                logger.info("skipping %s — no intent price", sym)
                continue
            qty = int(target_dollars / intent_price)
            if qty <= 0:
                continue
            order_id, fill_price, status, error = self._submit_buy(sym, qty, intent_price)
            slip = self._slippage_bps(intent_price, fill_price, "buy")

            # Compute exit target = entry_date + hold_days trading days
            exit_target = self.trading_days_after(datetime.now(), self.config.hold_days)
            exit_iso = exit_target.date().isoformat() if exit_target else ""

            self.append_fill(PEADFillRecord(
                submit_ts=submit_ts, fill_ts=None,
                symbol=sym, action="open_long", side="buy", qty=qty,
                intent_price=intent_price, fill_price=fill_price,
                fill_status=status, slippage_bps=slip,
                surprise_pct=sig["surprise_pct"],
                event_date=sig["event_date"], bars_held=None,
                dry_run=self.config.dry_run, order_id=order_id, error=error,
            ))

            if status not in ("submit_failed",) and not (self.config.dry_run is False and error):
                new_states.append(PEADPositionState(
                    symbol=sym, surprise_pct=sig["surprise_pct"],
                    event_date=sig["event_date"],
                    entry_date=datetime.now().date().isoformat(),
                    exit_target_date=exit_iso,
                    entry_order_id=order_id,
                    entry_price=fill_price,
                    intent_price=intent_price,
                    shares=qty,
                    dry_run=self.config.dry_run,
                ))
        return new_states

    def close_due_positions(
        self,
        positions: list[PEADPositionState],
        today: datetime,
    ) -> tuple[list[PEADPositionState], list[PEADPositionState]]:
        """Return (still_open, just_closed) tuple."""
        today_date = today.date()
        still_open: list[PEADPositionState] = []
        just_closed: list[PEADPositionState] = []
        for pos in positions:
            try:
                exit_due = pd.Timestamp(pos.exit_target_date).date() <= today_date
            except Exception:
                exit_due = False
            if not exit_due:
                still_open.append(pos)
                continue
            # Close it
            submit_ts = datetime.now(timezone.utc).isoformat()
            intents = self._capture_intent([pos.symbol])
            intent_price = intents.get(pos.symbol, pos.entry_price or 0.0)
            order_id, fill_price, status, error = self._submit_close(pos.symbol)
            slip = self._slippage_bps(intent_price, fill_price, "sell")
            try:
                bars_held = (pd.Timestamp(today_date) - pd.Timestamp(pos.entry_date)).days
            except Exception:
                bars_held = None
            self.append_fill(PEADFillRecord(
                submit_ts=submit_ts, fill_ts=None,
                symbol=pos.symbol, action="exit_long", side="sell", qty=pos.shares,
                intent_price=intent_price, fill_price=fill_price,
                fill_status=status, slippage_bps=slip,
                surprise_pct=pos.surprise_pct, event_date=pos.event_date,
                bars_held=bars_held,
                dry_run=self.config.dry_run, order_id=order_id, error=error,
            ))
            just_closed.append(pos)
        return still_open, just_closed

    def run_once(self, today: Optional[datetime] = None) -> dict:
        """Single daily check: process exits, then process new signals."""
        today = today or datetime.now()
        if not self.is_trading_day(today):
            payload = {"action": "skip", "reason": "not_trading_day",
                       "date": today.date().isoformat()}
            self.append_activity(payload)
            return payload

        positions = self.load_state()
        # 1. Process exits first (free up capacity for new entries)
        still_open, just_closed = self.close_due_positions(positions, today)

        # 2. Find new signals
        signals = self.find_new_signals(today)
        already_open = {p.symbol for p in still_open}
        target_dollars = self.compute_target_dollars()

        # 3. Open new positions
        new_states = self.open_new_positions(signals, already_open, target_dollars)

        # 4. Persist combined state
        next_state = still_open + new_states
        self.save_state(next_state)

        summary = {
            "action": "rebalance",
            "date": today.date().isoformat(),
            "positions_before": len(positions),
            "exits_executed": len(just_closed),
            "new_signals_found": len(signals),
            "new_positions_opened": len(new_states),
            "positions_after": len(next_state),
            "target_dollars_per_name": round(target_dollars, 2),
            "dry_run": self.config.dry_run,
        }
        self.append_activity(summary)
        return summary
