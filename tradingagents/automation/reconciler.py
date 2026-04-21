"""Local DB ↔ broker order reconciliation (Track P-SYNC).

Periodic job that pulls recently-submitted orders from the local SQLite
`trades` table and compares each against the broker's current state.
If the broker has advanced (e.g., PENDING_NEW → FILLED) and the local row
is stale, we update the local row in place.

Additive by design: this reads the broker and writes only to the `trades`
table on confirmed drift. It never submits, cancels, or modifies orders.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List

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
        lookback_days: int = 7,
        drift_alert_hours: int = 1,
        notifier=None,
    ):
        self.broker = broker
        self.db = db
        self.variant = variant
        self.lookback_days = lookback_days
        self.drift_alert_hours = drift_alert_hours
        self.notifier = notifier

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

        summary = {
            "variant": self.variant,
            "checked": checked,
            "drifted": drifted,
            "updated": updated,
            "errors": errors + leg_summary["errors"],
            "leg_checked": leg_summary["checked_legs"],
            "leg_inserted": leg_summary["inserted"],
        }
        if checked or leg_summary["inserted"]:
            logger.info(
                "reconciler[%s] summary: checked=%d drifted=%d updated=%d "
                "errors=%d leg_checked=%d leg_inserted=%d",
                self.variant or "-",
                checked,
                drifted,
                updated,
                summary["errors"],
                leg_summary["checked_legs"],
                leg_summary["inserted"],
            )
        return summary
