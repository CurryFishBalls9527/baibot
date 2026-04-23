"""Local DB ↔ broker order reconciliation (Track P-SYNC).

Periodic job that pulls recently-submitted orders from the local SQLite
`trades` table and compares each against the broker's current state.
If the broker has advanced (e.g., PENDING_NEW → FILLED) and the local row
is stale, we update the local row in place.

Also reconciles `position_states` against live Alpaca positions — covers
the DB/broker divergence modes documented in
`memory/project_bracket_leg_id_bug.md`:
  - ghost rows (DB says open, Alpaca has no position) → delete
  - ID mismatch (DB leg IDs point to cancelled orders) → update to live
  - orphan positions (Alpaca has position, DB has no row) → insert

Additive by design at the broker level: this reads broker state and
writes ONLY to local SQLite. It never submits, cancels, or modifies orders.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional

from tradingagents.broker.base_broker import BaseBroker
from tradingagents.storage.database import TradingDatabase

logger = logging.getLogger(__name__)

# Order statuses that indicate the local row may still be waiting on broker
# state to settle. Stored in DB as the stringified alpaca-py enum, e.g.
# "OrderStatus.PENDING_NEW". We match on substring to be enum-format-agnostic.
_OPEN_STATUS_TOKENS = (
    "pending_new",
    "accepted",
    "new",
    "partially_filled",
    "pending_replace",
    "pending_cancel",
)

# If this many tokens match "filled" substring we treat as filled. "filled"
# alone (not partial) means terminal.
_TERMINAL_STATUS_TOKENS = (
    "filled",
    "canceled",
    "cancelled",
    "expired",
    "rejected",
    "replaced",
    "done_for_day",
)


def _is_open(status: str | None) -> bool:
    if not status:
        return False
    s = status.lower()
    # "filled" (terminal) should never count as open; "partially_filled" should.
    if "partially_filled" in s:
        return True
    if "filled" in s:
        return False
    return any(tok in s for tok in _OPEN_STATUS_TOKENS)


class OrderReconciler:
    """Pulls open local trades and syncs their status with the broker.

    One instance per variant DB. Safe to call repeatedly; each call is a
    bounded pass over recent open rows.
    """

    def __init__(
        self,
        broker: BaseBroker,
        db: TradingDatabase,
        variant: str | None = None,
        lookback_days: int = 30,
        drift_alert_hours: int = 1,
        notifier=None,
        config: dict | None = None,
    ):
        self.broker = broker
        self.db = db
        self.variant = variant
        self.lookback_days = lookback_days
        self.drift_alert_hours = drift_alert_hours
        self.notifier = notifier
        # config gates the bracket-fire outcome hook (excursion fetch cost +
        # paper-account IEX feed fallback). Optional so existing tests that
        # construct OrderReconciler without config keep working.
        self.config = config or {}

    def _fetch_open_trades(self) -> List[dict]:
        cutoff = (datetime.utcnow() - timedelta(days=self.lookback_days)).isoformat()
        rows = self.db.conn.execute(
            """
            SELECT id, symbol, side, qty, status, filled_qty, filled_price,
                   order_id, timestamp
            FROM trades
            WHERE order_id IS NOT NULL
              AND timestamp >= ?
            ORDER BY timestamp DESC
            """,
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows if _is_open(r["status"])]

    def _fetch_recent_buys(self) -> List[dict]:
        cutoff = (datetime.utcnow() - timedelta(days=self.lookback_days)).isoformat()
        rows = self.db.conn.execute(
            """
            SELECT id, symbol, qty, order_id, timestamp
            FROM trades
            WHERE side = 'buy' AND order_id IS NOT NULL AND timestamp >= ?
            ORDER BY timestamp DESC
            """,
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    def _leg_already_logged(self, order_id: str) -> bool:
        row = self.db.conn.execute(
            "SELECT 1 FROM trades WHERE order_id = ? LIMIT 1", (order_id,)
        ).fetchone()
        return row is not None

    def _insert_leg_fill(
        self, parent_symbol: str, label: str, leg, parent_variant: str | None
    ) -> None:
        """Insert a SELL row for a filled bracket leg.

        `label` is 'stop_loss' or 'take_profit'. Uses the leg's filled_at as
        timestamp when available so trade history reflects actual fill time,
        not reconcile time. Strips the 'OrderStatus.' enum prefix for consistency
        with how the orchestrator writes the field.
        """
        filled_qty = float(leg.filled_qty or 0)
        filled_price = (
            float(leg.filled_avg_price) if leg.filled_avg_price is not None else None
        )
        status_str = str(leg.status)
        if status_str.startswith("OrderStatus."):
            status_str = status_str.split(".", 1)[1].lower()
        ts = getattr(leg, "filled_at", None)
        timestamp = (
            ts.isoformat() if hasattr(ts, "isoformat") else datetime.utcnow().isoformat()
        )
        self.db.conn.execute(
            """
            INSERT INTO trades (
                timestamp, symbol, side, qty, notional, order_type, status,
                filled_qty, filled_price, order_id, signal_id, reasoning, variant
            ) VALUES (?, ?, 'sell', ?, NULL, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                timestamp,
                parent_symbol,
                float(leg.qty) if leg.qty else filled_qty,
                "stop" if label == "stop_loss" else "limit",
                status_str,
                filled_qty,
                filled_price,
                str(leg.order_id),
                f"bracket_{label}",
                parent_variant or self.variant,
            ),
        )
        self.db.conn.commit()
        # The trade row is now durable; pair it with a trade_outcomes row
        # while the entry context in position_states is still available
        # (the ghost-cleanup phase later in reconcile_once() will delete it).
        # Without this hook the daily/weekly review silently ignores every
        # bracket-fired exit — only ExitManager-path SELLs get outcomes.
        self._maybe_log_bracket_outcome(parent_symbol, label, leg, filled_price)

    def _maybe_log_bracket_outcome(
        self,
        parent_symbol: str,
        label: str,
        leg,
        filled_price: float | None,
    ) -> None:
        """Write a trade_outcomes row for a just-swept bracket fill.

        Runs in the same reconcile tick that writes the trade row, BEFORE
        the ghost-cleanup phase deletes position_states. Idempotent: skips
        if an outcome already exists for (symbol, entry_date, entry_price).

        Fails soft — outcome bookkeeping must never block the trade-row
        insert or the rest of reconciliation.
        """
        if filled_price is None:
            return
        try:
            pos_state = self.db.get_position_state(parent_symbol)
        except Exception as e:
            logger.warning(
                "reconciler[%s]: get_position_state(%s) failed during outcome "
                "log: %s", self.variant or "-", parent_symbol, e,
            )
            return
        if pos_state is None:
            # Already cleaned up — entry context is gone. Fall back to the
            # backfill script if the caller wants an outcome row anyway.
            return
        entry_date = pos_state.get("entry_date")
        entry_price = pos_state.get("entry_price")
        if entry_price is None:
            return
        try:
            existing = self.db.conn.execute(
                """
                SELECT id FROM trade_outcomes
                WHERE symbol = ? AND entry_date = ? AND entry_price = ?
                LIMIT 1
                """,
                (parent_symbol, entry_date, float(entry_price)),
            ).fetchone()
            if existing is not None:
                return
        except Exception as e:
            logger.warning(
                "reconciler[%s]: outcome idempotence check failed for %s: %s",
                self.variant or "-", parent_symbol, e,
            )
            return
        try:
            from tradingagents.automation.trade_outcome import log_closed_trade
            outcome_id = log_closed_trade(
                db=self.db,
                symbol=parent_symbol,
                pos_state=pos_state,
                exit_price=float(filled_price),
                exit_reason=f"bracket_{label}",
                broker=self.broker,
                excursion_enabled=bool(
                    self.config.get("trade_outcome_excursion_enabled", False)
                ),
            )
            if outcome_id is not None:
                logger.info(
                    "reconciler[%s] trade_outcome logged: %s bracket_%s "
                    "entry=$%.2f exit=$%.2f (id=%s)",
                    self.variant or "-", parent_symbol, label,
                    float(entry_price), float(filled_price), outcome_id,
                )
        except Exception as e:
            logger.warning(
                "reconciler[%s]: log_closed_trade(%s) failed: %s",
                self.variant or "-", parent_symbol, e,
            )

    def _sweep_bracket_legs(self) -> dict:
        """Pull bracket child fills into the local DB.

        Entry orders record the parent order_id only; SL/TP legs fire
        broker-side under their own order_ids and never reach SQLite. Without
        this sweep, the trades table shows buys only and the dashboard history
        is misleading. Pass is idempotent — skips any leg_id already logged.
        """
        checked = 0
        inserted = 0
        errors = 0
        for parent in self._fetch_recent_buys():
            try:
                remote = self.broker.get_order(parent["order_id"])
            except Exception as e:
                errors += 1
                logger.debug(
                    "reconciler: sweep get_order(%s) failed: %s",
                    parent["order_id"],
                    e,
                )
                continue
            for label, leg_id in (
                ("stop_loss", remote.stop_order_id),
                ("take_profit", remote.tp_order_id),
            ):
                # Defensive: broker adapters should return str or None, but
                # mocks / unexpected types shouldn't reach the sqlite bind.
                if not isinstance(leg_id, str) or not leg_id:
                    continue
                if self._leg_already_logged(leg_id):
                    continue
                try:
                    leg = self.broker.get_order(leg_id)
                except Exception as e:
                    errors += 1
                    logger.debug(
                        "reconciler: sweep get_order(leg %s) failed: %s", leg_id, e
                    )
                    continue
                checked += 1
                status_l = (leg.status or "").lower()
                is_terminal = (
                    ("filled" in status_l and "partially" not in status_l)
                    or any(
                        t in status_l
                        for t in ("canceled", "cancelled", "expired", "rejected")
                    )
                )
                if not is_terminal:
                    continue
                # Skip zero-fill cancellations (losing side of OCO / replaced
                # legs). They didn't move shares and clutter the history.
                if float(leg.filled_qty or 0) == 0 and (
                    "canceled" in status_l or "cancelled" in status_l
                ):
                    continue
                try:
                    # Use the parent row's variant to stay consistent across
                    # multi-variant DBs, falling back to reconciler-level default.
                    parent_variant = self.db.conn.execute(
                        "SELECT variant FROM trades WHERE id = ?", (parent["id"],)
                    ).fetchone()
                    variant = parent_variant[0] if parent_variant else None
                    self._insert_leg_fill(parent["symbol"], label, leg, variant)
                    inserted += 1
                    logger.info(
                        "reconciler[%s] bracket leg backfilled: %s %s %s "
                        "qty=%s @ %s",
                        self.variant or "-",
                        parent["symbol"],
                        label,
                        leg_id[:8],
                        leg.filled_qty,
                        leg.filled_avg_price,
                    )
                except Exception as e:
                    errors += 1
                    logger.warning(
                        "reconciler: insert leg fill (%s) failed: %s", leg_id, e
                    )
        return {"checked_legs": checked, "inserted": inserted, "errors": errors}

    def _find_live_sl_tp(self, symbol: str):
        """Return (sl_order, tp_order) for a symbol from live broker state.

        Uses broker.get_live_orders which includes HELD — the status OCO /
        bracket stop legs sit in while waiting for their trigger. If the
        broker adapter doesn't implement it, falls back to get_open_orders.
        Both orders are filtered to side=sell; most recent submission wins
        if multiple candidates exist (covers manual OCO resubmissions).
        """
        getter = getattr(self.broker, "get_live_orders", None)
        if getter is None:
            getter = self.broker.get_open_orders
        try:
            orders = getter(symbol)
        except Exception as e:
            logger.warning("reconciler: get_live_orders(%s) failed: %s", symbol, e)
            return (None, None)

        def _sell(o):
            side = str(getattr(o, "side", "")).lower()
            return "sell" in side

        def _is_stop(o):
            t = str(getattr(o, "order_type", "")).lower()
            return "stop" in t and "limit" not in t

        def _is_limit(o):
            t = str(getattr(o, "order_type", "")).lower()
            return "limit" in t and "stop" not in t

        # Stable sort: prefer the most recently submitted order if multiple
        # active SL/TP orders exist for the same symbol (covers manual OCO
        # resubmissions that leave the originals cancelled).
        def _key(o):
            return getattr(o, "submitted_at", None) or 0

        sls = sorted((o for o in orders if _sell(o) and _is_stop(o)), key=_key, reverse=True)
        tps = sorted((o for o in orders if _sell(o) and _is_limit(o)), key=_key, reverse=True)
        return (sls[0] if sls else None, tps[0] if tps else None)

    def _reconcile_position_states(self, dry_run: bool = False) -> dict:
        """Sync DB `position_states` rows with live Alpaca state.

        Writes to SQLite only (never to the broker). Three actions per pass:

        * **Ghost:** DB has a row but Alpaca reports no position → delete.
        * **ID drift:** DB row's stop_order_id / tp_order_id don't match the
          current live SL/TP orders (manual OCO resubmission, re-anchored
          on fill drift, etc.) → update DB IDs.
        * **Orphan:** Alpaca has a position but DB has no row → insert a
          minimal state using broker-side fields so the exit manager can
          track it going forward.

        `dry_run=True` logs what WOULD happen but doesn't touch the DB.
        """
        ghosts = 0
        id_fixes = 0
        orphans = 0
        errors = 0

        try:
            live_positions = {p.symbol: p for p in self.broker.get_positions()}
        except Exception as e:
            logger.warning("reconciler: get_positions() failed: %s", e)
            return {"ghosts": 0, "id_fixes": 0, "orphans": 0, "errors": 1}

        # Pull all DB position_states rows directly (no helper exists for
        # "list all symbols", so go raw; reads only).
        db_symbols = [
            row[0]
            for row in self.db.conn.execute(
                "SELECT symbol FROM position_states"
            ).fetchall()
        ]

        # Ghost pass: DB row with no broker position.
        for sym in db_symbols:
            if sym in live_positions:
                continue
            if dry_run:
                ghosts += 1
                logger.info(
                    "reconciler[%s] DRY-RUN: would delete ghost %s",
                    self.variant or "-", sym,
                )
                continue
            try:
                self.db.delete_position_state(sym)
                ghosts += 1
                logger.info(
                    "reconciler[%s]: ghost position_state deleted for %s "
                    "(no broker position held)",
                    self.variant or "-", sym,
                )
            except Exception as e:
                errors += 1
                logger.warning(
                    "reconciler: delete_position_state(%s) failed: %s", sym, e
                )

        # ID-drift + orphan passes: per symbol with a live broker position.
        for sym, pos in live_positions.items():
            existing = self.db.get_position_state(sym)
            sl, tp = self._find_live_sl_tp(sym)
            live_sl_id = getattr(sl, "order_id", None) if sl else None
            live_tp_id = getattr(tp, "order_id", None) if tp else None
            live_sl_price = getattr(sl, "stop_price", None) if sl else None
            # ^ OrderResult model — not every broker populates stop_price.
            # Fall back below using a second get_order if we actually need it.

            if existing is None:
                # Orphan: Alpaca has position, DB has nothing. Import using
                # broker state so future ticks can manage it.
                entry_price = float(pos.avg_entry_price)
                # Seed current_stop from the live broker SL price when
                # available — keeps DB and broker aligned from t=0. Fall back
                # to 8% below entry only if there's no live SL order at all
                # (the position is unprotected; ExitManager will ratchet from
                # here once it sees the row).
                stop_price = getattr(sl, "stop_price", None) if sl else None
                if stop_price is None:
                    stop_price = round(entry_price * 0.92, 2)
                state = {
                    "entry_price": entry_price,
                    "entry_date": date.today().isoformat(),
                    "highest_close": entry_price,
                    "current_stop": float(stop_price),
                    "partial_taken": False,
                    "stop_type": "imported",
                    "stop_order_id": live_sl_id,
                    "tp_order_id": live_tp_id,
                    "variant": self.variant,
                    "bars_held": 0,
                }
                if dry_run:
                    orphans += 1
                    logger.info(
                        "reconciler[%s] DRY-RUN: would import orphan %s "
                        "(entry=$%.2f, stop=$%.2f, qty=%s)",
                        self.variant or "-", sym, entry_price,
                        float(stop_price), pos.qty,
                    )
                    continue
                try:
                    self.db.upsert_position_state(sym, state)
                    orphans += 1
                    logger.info(
                        "reconciler[%s]: imported orphan position %s "
                        "(entry=$%.2f, stop=$%.2f, qty=%s)",
                        self.variant or "-", sym, entry_price,
                        float(stop_price), pos.qty,
                    )
                except Exception as e:
                    errors += 1
                    logger.warning(
                        "reconciler: upsert orphan %s failed: %s", sym, e
                    )
                continue

            # ID-drift pass: DB has row AND position; update leg IDs if they
            # don't match the live SL/TP orders. Covers:
            #   (a) pre-fix NULL IDs being populated now
            #   (b) manual OCO resubmissions overriding bracket IDs
            #   (c) re-anchor on drift generating new IDs
            updates = {}
            if live_sl_id and existing.get("stop_order_id") != live_sl_id:
                updates["stop_order_id"] = live_sl_id
            if live_tp_id and existing.get("tp_order_id") != live_tp_id:
                updates["tp_order_id"] = live_tp_id
            if updates:
                if dry_run:
                    id_fixes += 1
                    logger.info(
                        "reconciler[%s] DRY-RUN: would update %s IDs (%s)",
                        self.variant or "-", sym,
                        ", ".join(
                            f"{k}={str(v)[:8]}" for k, v in updates.items()
                        ),
                    )
                else:
                    try:
                        self.db.upsert_position_state(sym, updates)
                        id_fixes += 1
                        logger.info(
                            "reconciler[%s]: %s leg IDs updated (%s)",
                            self.variant or "-", sym,
                            ", ".join(
                                f"{k}={str(v)[:8]}" for k, v in updates.items()
                            ),
                        )
                    except Exception as e:
                        errors += 1
                        logger.warning(
                            "reconciler: upsert IDs %s failed: %s", sym, e
                        )

        return {
            "ghosts": ghosts,
            "id_fixes": id_fixes,
            "orphans": orphans,
            "errors": errors,
        }

    def _sync_broker_stops(
        self,
        max_tighten_pct: float = 0.30,
        dry_run: bool = False,
    ) -> dict:
        """Push locally-ratcheted DB `current_stop` values up to the broker.

        When ExitManagerV2 couldn't propagate a stop ratchet (e.g. pre-fix
        when stop_order_id was NULL, or a broker reject), the DB knows the
        stop moved but Alpaca still holds the old level. This pass detects
        DB `current_stop > broker stop` and issues `replace_order` to sync.

        Guardrails:
        * Only tightens (moves stop UP) — never loosens.
        * Only fires when the new stop is strictly below the current market
          price (can't set a stop above mkt — would trigger immediately).
        * Bounded by `max_tighten_pct` vs broker stop; any delta beyond that
          is logged and skipped as a safety fence against a corrupted DB row.
        """
        synced = 0
        skipped = 0
        errors = 0

        try:
            live_positions = {p.symbol: p for p in self.broker.get_positions()}
        except Exception as e:
            logger.warning("reconciler: get_positions() failed: %s", e)
            return {"synced": 0, "skipped": 0, "errors": 1}

        for sym, pos in live_positions.items():
            state = self.db.get_position_state(sym)
            if not state:
                continue
            db_stop = state.get("current_stop")
            stop_id = state.get("stop_order_id")
            if not db_stop or not stop_id:
                continue
            sl, _tp = self._find_live_sl_tp(sym)
            broker_stop = getattr(sl, "stop_price", None) if sl else None
            if broker_stop is None:
                continue
            # Only act when DB is clearly above broker (>0.5%) — avoid
            # churning the broker stop by a dollar each tick for noise.
            delta_pct = (float(db_stop) - float(broker_stop)) / float(broker_stop)
            if delta_pct < 0.005:
                continue
            # Safety: don't move stop above current price (would trigger now).
            current_price = float(pos.current_price)
            if float(db_stop) >= current_price:
                logger.warning(
                    "reconciler[%s]: %s DB stop $%.2f >= current price $%.2f "
                    "— NOT pushing (would instant-fill). Inspect state.",
                    self.variant or "-", sym, float(db_stop), current_price,
                )
                skipped += 1
                continue
            # Safety: reject absurd deltas that suggest corrupt DB.
            if delta_pct > max_tighten_pct:
                logger.warning(
                    "reconciler[%s]: %s DB stop $%.2f is %.1f%% above broker "
                    "stop $%.2f — exceeds max_tighten_pct=%.0f%%, SKIPPED.",
                    self.variant or "-", sym, float(db_stop),
                    delta_pct * 100, float(broker_stop), max_tighten_pct * 100,
                )
                skipped += 1
                continue

            if dry_run:
                logger.info(
                    "reconciler[%s] DRY-RUN: would push %s broker stop "
                    "$%.2f -> $%.2f (order %s)",
                    self.variant or "-", sym, float(broker_stop),
                    float(db_stop), str(stop_id)[:8],
                )
                synced += 1
                continue

            try:
                replaced = self.broker.replace_order(
                    stop_id, stop_price=round(float(db_stop), 2)
                )
                synced += 1
                new_id = getattr(replaced, "order_id", None) if replaced else None
                logger.info(
                    "reconciler[%s]: %s broker stop synced $%.2f -> $%.2f "
                    "(old=%s new=%s)",
                    self.variant or "-", sym, float(broker_stop),
                    float(db_stop), str(stop_id)[:8],
                    str(new_id)[:8] if new_id else "?",
                )
                # Alpaca's replace_order returns a NEW order ID; the old leg
                # is cancelled. Update DB immediately so subsequent ticks
                # don't see a stale ID and flag it as drift.
                if new_id and new_id != stop_id:
                    try:
                        self.db.upsert_position_state(sym, {"stop_order_id": new_id})
                    except Exception as e:
                        errors += 1
                        logger.warning(
                            "reconciler[%s]: %s failed to persist new stop_id "
                            "%s: %s",
                            self.variant or "-", sym, str(new_id)[:8], e,
                        )
            except Exception as e:
                errors += 1
                logger.warning(
                    "reconciler[%s]: %s replace_order(%s) failed: %s",
                    self.variant or "-", sym, str(stop_id)[:8], e,
                )

        return {"synced": synced, "skipped": skipped, "errors": errors}

    def reconcile_once(self) -> dict:
        """Single reconciliation pass. Returns summary dict."""
        rows = self._fetch_open_trades()
        checked = 0
        updated = 0
        drifted = 0
        errors = 0
        for row in rows:
            checked += 1
            order_id = row["order_id"]
            try:
                remote = self.broker.get_order(order_id)
            except Exception as e:
                errors += 1
                logger.debug(
                    "reconciler: get_order(%s) failed: %s", order_id, e
                )
                continue

            local_status = (row["status"] or "").lower()
            remote_status = (remote.status or "").lower()
            local_filled_qty = float(row["filled_qty"] or 0)
            remote_filled_qty = float(remote.filled_qty or 0)
            local_filled_price = row["filled_price"]
            remote_filled_price = remote.filled_avg_price

            # Detect meaningful drift on any of the three fields.
            drifted_row = (
                local_status != remote_status
                or abs(local_filled_qty - remote_filled_qty) > 1e-9
                or (
                    remote_filled_price is not None
                    and (
                        local_filled_price is None
                        or abs(float(local_filled_price) - float(remote_filled_price))
                        > 1e-6
                    )
                )
            )
            if not drifted_row:
                continue

            drifted += 1
            try:
                self.db.update_trade_status(
                    order_id=order_id,
                    status=remote.status,
                    filled_qty=remote_filled_qty,
                    filled_price=remote_filled_price,
                )
                updated += 1
                logger.info(
                    "reconciler[%s]: %s %s %s -> %s (filled %.4f @ %s)",
                    self.variant or "-",
                    row["symbol"],
                    order_id[:8],
                    local_status,
                    remote_status,
                    remote_filled_qty,
                    remote_filled_price,
                )
            except Exception as e:
                errors += 1
                logger.warning(
                    "reconciler: update_trade_status(%s) failed: %s", order_id, e
                )

            # Alert on stale rows that were drifting >N hours — we don't know
            # when the broker fill actually happened, but we know the local row
            # never updated. Use the local timestamp as a lower bound on age.
            if self.notifier is not None and self.drift_alert_hours > 0:
                try:
                    ts = datetime.fromisoformat(row["timestamp"].replace("Z", ""))
                    age = datetime.utcnow() - ts
                    if age >= timedelta(hours=self.drift_alert_hours):
                        self.notifier.send(
                            "Order drift reconciled",
                            (
                                f"{self.variant or '-'} {row['symbol']} "
                                f"{order_id[:8]} {local_status} -> {remote_status} "
                                f"(local stale {age})"
                            ),
                            priority="default",
                            tags=["warning"],
                            dedupe_key=f"reconcile-drift:{order_id}",
                        )
                except Exception:
                    pass

        leg_summary = self._sweep_bracket_legs()
        pos_summary = self._reconcile_position_states()
        # Broker-stop sync runs AFTER position-state reconcile so newly-fixed
        # leg IDs and freshly-imported orphans participate in this pass.
        stop_summary = self._sync_broker_stops()

        summary = {
            "variant": self.variant,
            "checked": checked,
            "drifted": drifted,
            "updated": updated,
            "errors": (
                errors
                + leg_summary["errors"]
                + pos_summary["errors"]
                + stop_summary["errors"]
            ),
            "leg_checked": leg_summary["checked_legs"],
            "leg_inserted": leg_summary["inserted"],
            "position_ghosts_deleted": pos_summary["ghosts"],
            "position_id_fixes": pos_summary["id_fixes"],
            "position_orphans_imported": pos_summary["orphans"],
            "broker_stops_synced": stop_summary["synced"],
            "broker_stops_skipped": stop_summary["skipped"],
        }
        if (
            checked
            or leg_summary["inserted"]
            or pos_summary["ghosts"]
            or pos_summary["id_fixes"]
            or pos_summary["orphans"]
            or stop_summary["synced"]
            or stop_summary["skipped"]
        ):
            logger.info(
                "reconciler[%s] summary: checked=%d drifted=%d updated=%d "
                "errors=%d leg_checked=%d leg_inserted=%d ghosts=%d "
                "id_fixes=%d orphans=%d stops_synced=%d stops_skipped=%d",
                self.variant or "-",
                checked, drifted, updated,
                summary["errors"],
                leg_summary["checked_legs"],
                leg_summary["inserted"],
                pos_summary["ghosts"],
                pos_summary["id_fixes"],
                pos_summary["orphans"],
                stop_summary["synced"],
                stop_summary["skipped"],
            )
        return summary
