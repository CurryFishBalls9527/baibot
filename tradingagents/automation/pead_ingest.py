"""One-way bridge: PEAD's JSON state → dashboard-shaped SQLite DB.

PEAD remains a standalone launchd job (separate process, separate Alpaca
account, fixed-time-exit thesis incompatible with the swing
ExitManager). To surface PEAD in the unified dashboard alongside the 6
trading variants, this module mirrors PEAD's JSON state into a SQLite
DB shaped like the per-variant trading_*.db files the dashboard
auto-discovers.

**Read-only mirror.** ``positions.json`` remains PEAD's authoritative
state. This bridge only writes to SQLite — it never modifies PEAD's JSON
files, so a sync failure cannot corrupt PEAD itself.

**Idempotent.** All inserts dedupe on a stable key (trades by
``order_id``, snapshots by ``date``, position_states by ``symbol``,
trade_outcomes by ``(symbol, entry_date, exit_date)``). Re-running on
the same day is safe.

**Why not migrate PEAD into ``position_states`` directly?** The swing
reconciler's orphan branch (``reconciler.py:462-522``) imports any
Alpaca position lacking a DB row by fabricating an 8%-below-entry stop.
PEAD's thesis is "no stop, exit by date." If PEAD wrote to the same
``position_states`` table the reconciler scans, ExitManager would start
ratcheting a fabricated stop on PEAD positions — strategy-killing bug.
By keeping PEAD in its own ``trading_pead.db`` SQLite that no
Orchestrator/reconciler is wired against, we get dashboard visibility
without coupling.

Usage:
    from tradingagents.automation.pead_ingest import sync_pead_to_sqlite
    sync_pead_to_sqlite()
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tradingagents.storage.database import TradingDatabase

logger = logging.getLogger(__name__)

VARIANT_NAME = "pead"
PEAD_BASE_PATTERN = "earnings_surprise"


@dataclass
class PEADIngestSummary:
    trades_inserted: int = 0
    trades_skipped: int = 0
    positions_upserted: int = 0
    positions_removed: int = 0
    snapshots_inserted: int = 0
    outcomes_inserted: int = 0
    errors: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "trades_inserted": self.trades_inserted,
            "trades_skipped": self.trades_skipped,
            "positions_upserted": self.positions_upserted,
            "positions_removed": self.positions_removed,
            "snapshots_inserted": self.snapshots_inserted,
            "outcomes_inserted": self.outcomes_inserted,
            "errors": self.errors,
        }


def _read_positions_json(state_dir: Path) -> List[Dict[str, Any]]:
    p = state_dir / "positions.json"
    if not p.exists():
        return []
    payload = json.loads(p.read_text())
    return payload.get("positions") or []


def _read_fills_jsonl(state_dir: Path) -> List[Dict[str, Any]]:
    p = state_dir / "fills.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning("malformed fills.jsonl line: %s — %s", line[:80], exc)
    return rows


def _existing_order_ids(db: TradingDatabase, variant: str) -> set[str]:
    rows = db.conn.execute(
        "SELECT order_id FROM trades WHERE variant = ? AND order_id IS NOT NULL",
        (variant,),
    ).fetchall()
    return {row["order_id"] for row in rows}


def _sync_trades(
    db: TradingDatabase,
    fills: List[Dict[str, Any]],
    summary: PEADIngestSummary,
) -> None:
    """One row per fill. Dedupe on order_id+side (PEAD reuses the same
    order_id semantically for one open + one close, but we prefer
    order_id alone since each Alpaca order_id is unique per submission).
    """
    existing = _existing_order_ids(db, VARIANT_NAME)
    for fill in fills:
        oid = fill.get("order_id")
        if not oid:
            summary.trades_skipped += 1
            continue
        # Skip if already mirrored
        if oid in existing:
            summary.trades_skipped += 1
            continue
        action = fill.get("action") or ""
        side = fill.get("side") or ("buy" if action == "open_long" else "sell")
        qty = float(fill.get("qty") or 0)
        fill_price = fill.get("fill_price")
        # Map fill_status string back to a canonical status the dashboard
        # understands. PEAD writes e.g. "OrderStatus.PARTIALLY_FILLED".
        raw_status = (fill.get("fill_status") or "").lower()
        if "filled" in raw_status and "partial" not in raw_status:
            status = "filled"
        elif "partial" in raw_status:
            status = "partially_filled"
        elif "rejected" in raw_status or "canceled" in raw_status:
            status = "rejected"
        else:
            status = "submitted"
        reasoning = (
            f"PEAD {action} | surprise={fill.get('surprise_pct')}% "
            f"| event={fill.get('event_date')} "
            f"| slippage={fill.get('slippage_bps')}bps"
        )
        try:
            db.log_trade(
                symbol=fill.get("symbol", ""),
                side=side,
                qty=qty,
                notional=(qty * float(fill_price)) if fill_price else None,
                order_type="market",
                status=status,
                filled_qty=qty if status in ("filled", "partially_filled") else 0,
                filled_price=fill_price,
                order_id=oid,
                reasoning=reasoning,
                variant=VARIANT_NAME,
            )
            # Override the auto-generated timestamp so the dashboard sees
            # the actual fill time, not now(). log_trade doesn't accept
            # a timestamp arg — patch in place.
            ts = fill.get("submit_ts") or fill.get("fill_ts")
            if ts:
                db.conn.execute(
                    "UPDATE trades SET timestamp = ? WHERE order_id = ?",
                    (ts.replace("T", " ").replace("Z", "")[:19], oid),
                )
                db.conn.commit()
            summary.trades_inserted += 1
        except Exception as exc:
            logger.warning("trade insert failed for %s/%s: %s",
                           fill.get("symbol"), oid, exc)
            summary.errors += 1


def _sync_position_states(
    db: TradingDatabase,
    positions: List[Dict[str, Any]],
    summary: PEADIngestSummary,
) -> None:
    """Mirror open positions into position_states.

    PEAD has no stop loss — set ``current_stop=0.0`` as a sentinel
    (dashboard renders this as $0 = clear visual cue that this strategy
    is not stop-managed). ``base_pattern='earnings_surprise'`` lets
    cross-variant pattern queries distinguish PEAD trades.
    """
    open_symbols = {p["symbol"] for p in positions if p.get("symbol")}
    # Remove rows for symbols PEAD no longer holds (closed since last sync).
    existing = db.conn.execute(
        "SELECT symbol FROM position_states WHERE variant = ?",
        (VARIANT_NAME,),
    ).fetchall()
    for row in existing:
        sym = row["symbol"]
        if sym not in open_symbols:
            try:
                db.delete_position_state(sym)
                summary.positions_removed += 1
            except Exception as exc:
                logger.warning("delete_position_state(%s) failed: %s", sym, exc)
                summary.errors += 1

    for pos in positions:
        sym = pos.get("symbol")
        if not sym:
            continue
        entry_price = pos.get("entry_price") or pos.get("intent_price") or 0.0
        try:
            db.upsert_position_state(sym, {
                "entry_price": float(entry_price),
                "entry_date": pos.get("entry_date") or date.today().isoformat(),
                "highest_close": float(entry_price),
                # PEAD uses fixed-time exits, not stops. Sentinel 0.0
                # signals "no stop" to anyone reading this row.
                "current_stop": 0.0,
                "stop_type": "pead_no_stop",
                "variant": VARIANT_NAME,
                "entry_order_id": pos.get("entry_order_id"),
                "base_pattern": PEAD_BASE_PATTERN,
                # Surprise pct stored in rs_at_entry as a structured
                # numeric carrier (no PEAD-specific column added) —
                # downstream code that knows variant=='pead' can interpret.
                "rs_at_entry": pos.get("surprise_pct"),
                "regime_at_entry": f"event_{pos.get('event_date', '')}",
            })
            summary.positions_upserted += 1
        except Exception as exc:
            logger.warning("upsert_position_state(%s) failed: %s", sym, exc)
            summary.errors += 1


def _sync_daily_snapshot(
    db: TradingDatabase,
    alpaca_api_key: Optional[str],
    alpaca_secret_key: Optional[str],
    summary: PEADIngestSummary,
) -> None:
    """Query Alpaca PEAD account → log today's daily_snapshot.

    Dedupe via UNIQUE(date) constraint + INSERT OR REPLACE in
    take_snapshot. Skipped silently if Alpaca creds missing (e.g.
    backfill mode running offline).
    """
    if not alpaca_api_key or not alpaca_secret_key:
        logger.info("alpaca PEAD creds missing; skipping daily_snapshot")
        return
    try:
        # Lazy import to keep this module importable in environments
        # without alpaca-py (e.g., a test harness).
        from tradingagents.broker.alpaca_broker import AlpacaBroker  # noqa: WPS433
        broker = AlpacaBroker(
            api_key=alpaca_api_key, secret_key=alpaca_secret_key, paper=True,
        )
        account = broker.get_account()
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        cash = float(account.cash)
        buying_power = float(getattr(account, "buying_power", 0) or 0)
        positions = broker.get_positions() if hasattr(broker, "get_positions") else []
        # Normalize positions to JSON-safe primitive dicts
        pos_payload = []
        for p in positions:
            try:
                pos_payload.append({
                    "symbol": getattr(p, "symbol", None),
                    "qty": float(getattr(p, "qty", 0) or 0),
                    "market_value": float(getattr(p, "market_value", 0) or 0),
                    "unrealized_pl": float(getattr(p, "unrealized_pl", 0) or 0),
                })
            except Exception:
                continue
        portfolio_value = sum(p["market_value"] for p in pos_payload)
        daily_pl = equity - last_equity
        daily_pl_pct = (daily_pl / last_equity * 100) if last_equity else 0
        db.take_snapshot(
            equity=equity, cash=cash, buying_power=buying_power,
            portfolio_value=portfolio_value, positions=pos_payload,
            daily_pl=daily_pl, daily_pl_pct=daily_pl_pct,
        )
        summary.snapshots_inserted += 1
    except Exception as exc:
        logger.warning("daily_snapshot sync failed: %s", exc)
        summary.errors += 1


def _sync_trade_outcomes(
    db: TradingDatabase,
    fills: List[Dict[str, Any]],
    summary: PEADIngestSummary,
) -> None:
    """Pair entries (open_long) with exits (exit_long) per symbol and
    log to trade_outcomes. Dedupe by (symbol, entry_date, exit_date)
    using a SELECT before INSERT.

    PEAD's fills.jsonl doesn't store separate entry/exit timestamps in
    a structured way, so we use the fill's submit_ts. Multiple round
    trips of the same symbol are handled by FIFO matching.
    """
    by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for f in fills:
        sym = f.get("symbol")
        if not sym:
            continue
        by_symbol.setdefault(sym, []).append(f)

    for sym, sfills in by_symbol.items():
        # Sort by submit_ts to allow FIFO matching of opens to exits.
        sfills.sort(key=lambda r: (r.get("submit_ts") or ""))
        opens: List[Dict[str, Any]] = []
        for fill in sfills:
            action = fill.get("action") or ""
            if action == "open_long":
                opens.append(fill)
            elif action == "exit_long" and opens:
                entry = opens.pop(0)
                entry_date = (entry.get("submit_ts") or "")[:10]
                exit_date = (fill.get("submit_ts") or "")[:10]
                # Dedupe: skip if outcome already logged
                already = db.conn.execute(
                    """SELECT 1 FROM trade_outcomes
                       WHERE symbol = ? AND entry_date = ? AND exit_date = ?""",
                    (sym, entry_date, exit_date),
                ).fetchone()
                if already:
                    continue
                ep = entry.get("fill_price") or entry.get("intent_price")
                xp = fill.get("fill_price") or fill.get("intent_price")
                if not ep or not xp:
                    continue
                ret_pct = (xp - ep) / ep * 100
                hold_days = None
                try:
                    if entry_date and exit_date:
                        d1 = datetime.strptime(entry_date, "%Y-%m-%d").date()
                        d2 = datetime.strptime(exit_date, "%Y-%m-%d").date()
                        hold_days = (d2 - d1).days
                except Exception:
                    pass
                try:
                    db.log_trade_outcome({
                        "symbol": sym,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "entry_price": ep,
                        "exit_price": xp,
                        "return_pct": ret_pct,
                        "hold_days": hold_days,
                        "exit_reason": "pead_time_exit",
                        "base_pattern": PEAD_BASE_PATTERN,
                        "rs_at_entry": entry.get("surprise_pct"),
                        "regime_at_entry": f"event_{entry.get('event_date', '')}",
                    })
                    summary.outcomes_inserted += 1
                except Exception as exc:
                    logger.warning(
                        "trade_outcome insert failed for %s: %s", sym, exc,
                    )
                    summary.errors += 1


def sync_pead_to_sqlite(
    state_dir: str = "results/pead/paper",
    db_path: str = "trading_pead.db",
    alpaca_api_key: Optional[str] = None,
    alpaca_secret_key: Optional[str] = None,
) -> PEADIngestSummary:
    """Mirror PEAD's JSON state into a dashboard-shaped SQLite DB.

    All four sub-syncs (trades, positions, snapshot, outcomes) run
    independently. A failure in one does not block the others; errors
    are counted in the returned summary.
    """
    sd = Path(state_dir)
    summary = PEADIngestSummary()
    db = TradingDatabase(db_path, variant=VARIANT_NAME)
    try:
        fills = _read_fills_jsonl(sd)
        positions = _read_positions_json(sd)
        _sync_trades(db, fills, summary)
        _sync_position_states(db, positions, summary)
        _sync_trade_outcomes(db, fills, summary)
        if alpaca_api_key is None:
            alpaca_api_key = os.environ.get("ALPACA_PEAD_API_KEY")
        if alpaca_secret_key is None:
            alpaca_secret_key = os.environ.get("ALPACA_PEAD_SECRET_KEY")
        _sync_daily_snapshot(db, alpaca_api_key, alpaca_secret_key, summary)
    finally:
        db.conn.close()
    return summary
