"""SQLite storage for trades, signals, and daily snapshots."""

import sqlite3
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class TradingDatabase:
    """Lightweight SQLite database for all trading records."""

    def __init__(self, db_path: str = "trading.db", variant: Optional[str] = None):
        self.db_path = db_path
        self.default_variant = variant
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info(f"TradingDatabase opened at {db_path}")

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL DEFAULT (datetime('now')),
                symbol        TEXT    NOT NULL,
                side          TEXT    NOT NULL,
                qty           REAL,
                notional      REAL,
                order_type    TEXT    NOT NULL DEFAULT 'market',
                status        TEXT    NOT NULL,
                filled_qty    REAL    DEFAULT 0,
                filled_price  REAL,
                order_id      TEXT,
                signal_id     INTEGER REFERENCES signals(id),
                reasoning     TEXT
            );

            CREATE TABLE IF NOT EXISTS signals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL DEFAULT (datetime('now')),
                symbol        TEXT    NOT NULL,
                action        TEXT    NOT NULL,
                confidence    REAL    DEFAULT 0,
                reasoning     TEXT,
                stop_loss     REAL,
                take_profit   REAL,
                timeframe     TEXT,
                full_analysis TEXT,
                executed      INTEGER DEFAULT 0,
                rejected_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_snapshots (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                date          TEXT    NOT NULL,
                equity        REAL    NOT NULL,
                cash          REAL    NOT NULL,
                buying_power  REAL,
                portfolio_value REAL,
                positions_json TEXT,
                daily_pl      REAL    DEFAULT 0,
                daily_pl_pct  REAL    DEFAULT 0,
                UNIQUE(date)
            );

            CREATE TABLE IF NOT EXISTS agent_memories (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_name   TEXT    NOT NULL,
                situation     TEXT    NOT NULL,
                recommendation TEXT   NOT NULL,
                created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS setup_candidates (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                screen_date   TEXT    NOT NULL,
                symbol        TEXT    NOT NULL,
                selected_for_analysis INTEGER DEFAULT 0,
                passed_template INTEGER DEFAULT 0,
                market_regime TEXT,
                market_confirmed_uptrend INTEGER DEFAULT 0,
                rs_percentile REAL,
                revenue_growth REAL,
                eps_growth REAL,
                revenue_acceleration REAL,
                eps_acceleration REAL,
                stage_number REAL,
                base_label TEXT,
                candidate_status TEXT,
                pivot_price REAL,
                buy_point REAL,
                buy_limit_price REAL,
                initial_stop_price REAL,
                initial_stop_pct REAL,
                distance_to_pivot_pct REAL,
                distance_to_buy_point_pct REAL,
                buy_zone_pct REAL,
                breakout_signal INTEGER DEFAULT 0,
                rule_watch_candidate INTEGER DEFAULT 0,
                rule_entry_candidate INTEGER DEFAULT 0,
                next_earnings_datetime TEXT,
                earnings_days_away REAL,
                payload_json TEXT,
                UNIQUE(screen_date, symbol)
            );

            CREATE TABLE IF NOT EXISTS screening_batches (
                screen_date   TEXT PRIMARY KEY,
                market_regime TEXT,
                market_confirmed_uptrend INTEGER DEFAULT 0,
                row_count     INTEGER DEFAULT 0,
                selected_count INTEGER DEFAULT 0,
                approved_symbols_json TEXT,
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
            CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
            CREATE INDEX IF NOT EXISTS idx_daily_snapshots_date ON daily_snapshots(date);
            CREATE INDEX IF NOT EXISTS idx_agent_memories_name ON agent_memories(memory_name);
            CREATE INDEX IF NOT EXISTS idx_setup_candidates_date ON setup_candidates(screen_date);
            CREATE INDEX IF NOT EXISTS idx_setup_candidates_symbol ON setup_candidates(symbol);

            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                config_yaml   TEXT NOT NULL,
                start_date    TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'running',
                primary_metric TEXT DEFAULT 'sharpe_ratio',
                min_trades    INTEGER DEFAULT 30,
                min_days      INTEGER DEFAULT 20,
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS experiment_snapshots (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                variant_name  TEXT NOT NULL,
                date          TEXT NOT NULL,
                equity        REAL,
                daily_return  REAL,
                total_return  REAL,
                sharpe_ratio  REAL,
                max_drawdown  REAL,
                total_trades  INTEGER,
                win_rate      REAL,
                UNIQUE(experiment_id, variant_name, date)
            );

            CREATE TABLE IF NOT EXISTS position_states (
                symbol          TEXT PRIMARY KEY,
                entry_price     REAL NOT NULL,
                entry_date      TEXT NOT NULL,
                highest_close   REAL NOT NULL,
                current_stop    REAL NOT NULL,
                partial_taken   INTEGER DEFAULT 0,
                stop_type       TEXT DEFAULT 'initial',
                updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS trade_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_date TEXT,
                exit_date TEXT,
                entry_price REAL,
                exit_price REAL,
                return_pct REAL,
                hold_days INTEGER,
                exit_reason TEXT,
                base_pattern TEXT,
                stage_at_entry REAL,
                rs_at_entry REAL,
                regime_at_entry TEXT,
                max_favorable_excursion REAL,
                max_adverse_excursion REAL,
                trade_analysis TEXT
            );

            CREATE TABLE IF NOT EXISTS proposals (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
                variant             TEXT    NOT NULL,
                iso_week            TEXT    NOT NULL,
                title               TEXT    NOT NULL,
                what                TEXT,
                why                 TEXT,
                how_to_validate     TEXT,
                risk                TEXT,
                status              TEXT    NOT NULL DEFAULT 'open',
                status_updated_at   TEXT,
                outcome_summary     TEXT,
                outcome_metrics_json TEXT
            );
        """)
        self._ensure_columns(
            "trade_outcomes",
            {"trade_analysis": "TEXT"},
        )
        # Track P: additive columns for bracket-leg tracking and variant attribution.
        # All default NULL — existing code ignores them, v2 path populates them.
        self._ensure_columns(
            "position_states",
            {
                "stop_order_id": "TEXT",
                "tp_order_id": "TEXT",
                "variant": "TEXT",
                "entry_order_id": "TEXT",
                "bars_held": "INTEGER DEFAULT 0",
                # Entry-time context captured at BUY time and read at SELL by
                # the shared trade_outcome helper. Polymorphic across variants:
                #   - Minervini: base_pattern=<base_label>, rs/stage populated.
                #   - Chan: base_pattern=<T-type: T1/T2/T2S>, rs/stage NULL.
                #   - Intraday: base_pattern=<setup_family>, rs/stage NULL.
                # regime_at_entry is optional per variant.
                "regime_at_entry": "TEXT",
                "base_pattern": "TEXT",
                "rs_at_entry": "REAL",
                "stage_at_entry": "REAL",
            },
        )
        self._ensure_columns(
            "trades",
            {
                "variant": "TEXT",
            },
        )
        # Structured entry-signal metadata (intraday ORB levels, NR4 range,
        # VWAP, Minervini pivot, Chan T-type, etc). JSON blob, NULL for older
        # rows and for any call that doesn't provide it. Readers MUST tolerate
        # NULL. See memory/project_bracket_leg_id_bug.md for the analogous
        # NULL-tolerant pattern around stop_order_id.
        self._ensure_columns(
            "signals",
            {
                "signal_metadata": "TEXT",
            },
        )
        self._ensure_columns(
            "setup_candidates",
            {
                "stage_number": "REAL",
                "candidate_status": "TEXT",
                "buy_point": "REAL",
                "buy_limit_price": "REAL",
                "initial_stop_price": "REAL",
                "initial_stop_pct": "REAL",
                "distance_to_buy_point_pct": "REAL",
                "buy_zone_pct": "REAL",
                "rule_watch_candidate": "INTEGER DEFAULT 0",
                "rule_entry_candidate": "INTEGER DEFAULT 0",
            },
        )
        self.conn.commit()

    def _ensure_columns(self, table: str, columns: Dict[str, str]):
        existing = {
            row["name"] if isinstance(row, sqlite3.Row) else row[1]
            for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, definition in columns.items():
            if name in existing:
                continue
            try:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise

    # ── Signals ──────────────────────────────────────────────────────

    def log_signal(self, symbol: str, action: str, confidence: float = 0,
                   reasoning: str = "", stop_loss: float = None,
                   take_profit: float = None, timeframe: str = "",
                   full_analysis: str = "",
                   signal_metadata: Optional[str] = None) -> int:
        """Insert a signal row.

        `signal_metadata` is an optional JSON-encoded string carrying the
        structured per-strategy entry context (pivot, ORB levels, ZS bounds,
        T-type, etc). Kept as opaque TEXT so schema stays stable as setups
        evolve. All readers must tolerate NULL.
        """
        cur = self.conn.execute(
            """INSERT INTO signals (symbol, action, confidence, reasoning,
               stop_loss, take_profit, timeframe, full_analysis,
               signal_metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol, action, confidence, reasoning,
             stop_loss, take_profit, timeframe, full_analysis,
             signal_metadata),
        )
        self.conn.commit()
        return cur.lastrowid

    def mark_signal_executed(self, signal_id: int):
        self.conn.execute("UPDATE signals SET executed = 1 WHERE id = ?", (signal_id,))
        self.conn.commit()

    def mark_signal_rejected(self, signal_id: int, reason: str):
        self.conn.execute(
            "UPDATE signals SET rejected_reason = ? WHERE id = ?", (reason, signal_id)
        )
        self.conn.commit()

    # ── Trades ───────────────────────────────────────────────────────

    def log_trade(self, symbol: str, side: str, qty: float = None,
                  notional: float = None, order_type: str = "market",
                  status: str = "submitted", filled_qty: float = 0,
                  filled_price: float = None, order_id: str = None,
                  signal_id: int = None, reasoning: str = "",
                  variant: Optional[str] = None) -> int:
        if variant is None:
            variant = self.default_variant
        cur = self.conn.execute(
            """INSERT INTO trades (symbol, side, qty, notional, order_type,
               status, filled_qty, filled_price, order_id, signal_id, reasoning,
               variant)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol, side, qty, notional, order_type, status,
             filled_qty, filled_price, order_id, signal_id, reasoning, variant),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_trade_status(self, order_id: str, status: str,
                            filled_qty: float = None, filled_price: float = None):
        fields = ["status = ?"]
        values = [status]
        if filled_qty is not None:
            fields.append("filled_qty = ?")
            values.append(filled_qty)
        if filled_price is not None:
            fields.append("filled_price = ?")
            values.append(filled_price)
        values.append(order_id)
        self.conn.execute(
            f"UPDATE trades SET {', '.join(fields)} WHERE order_id = ?", values
        )
        self.conn.commit()

    # ── Daily Snapshots ──────────────────────────────────────────────

    def take_snapshot(self, equity: float, cash: float, buying_power: float,
                      portfolio_value: float, positions: List[Dict],
                      daily_pl: float = 0, daily_pl_pct: float = 0,
                      snapshot_date: str = None):
        d = snapshot_date or date.today().isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO daily_snapshots
               (date, equity, cash, buying_power, portfolio_value,
                positions_json, daily_pl, daily_pl_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (d, equity, cash, buying_power, portfolio_value,
             json.dumps(positions), daily_pl, daily_pl_pct),
        )
        self.conn.commit()

    def get_snapshots(self, days: int = 30) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM daily_snapshots ORDER BY date DESC LIMIT ?", (days,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_starting_equity(self) -> Optional[float]:
        row = self.conn.execute(
            "SELECT equity FROM daily_snapshots ORDER BY date ASC LIMIT 1"
        ).fetchone()
        return row["equity"] if row else None

    # ── Agent Memories ───────────────────────────────────────────────

    def save_memories(self, memory_name: str, situations_and_recs: List[tuple]):
        for situation, recommendation in situations_and_recs:
            self.conn.execute(
                """INSERT INTO agent_memories (memory_name, situation, recommendation)
                   VALUES (?, ?, ?)""",
                (memory_name, situation, recommendation),
            )
        self.conn.commit()

    def load_memories(self, memory_name: str) -> List[tuple]:
        rows = self.conn.execute(
            "SELECT situation, recommendation FROM agent_memories WHERE memory_name = ?",
            (memory_name,),
        ).fetchall()
        return [(r["situation"], r["recommendation"]) for r in rows]

    # ── Minervini Setups ─────────────────────────────────────────────

    def save_setup_candidates(
        self,
        rows: List[Dict[str, Any]],
        screen_date: Optional[str] = None,
        selected_symbols: Optional[List[str]] = None,
    ):
        if not rows and screen_date is None:
            return

        selected = set(selected_symbols or [])
        batch_date = screen_date or rows[0].get("trade_date") or date.today().isoformat()
        self.conn.execute("DELETE FROM setup_candidates WHERE screen_date = ?", (batch_date,))
        if not rows:
            self.conn.commit()
            return

        for row in rows:
            symbol = row.get("symbol")
            next_earnings = row.get("next_earnings_datetime")
            payload = json.dumps(row, default=str)
            self.conn.execute(
                """
                INSERT INTO setup_candidates (
                    screen_date, symbol, selected_for_analysis, passed_template,
                    market_regime, market_confirmed_uptrend, rs_percentile,
                    revenue_growth, eps_growth, revenue_acceleration, eps_acceleration,
                    stage_number, base_label, candidate_status, pivot_price, buy_point,
                    buy_limit_price, initial_stop_price, initial_stop_pct,
                    distance_to_pivot_pct, distance_to_buy_point_pct, buy_zone_pct,
                    breakout_signal, rule_watch_candidate, rule_entry_candidate,
                    next_earnings_datetime, earnings_days_away, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_date,
                    symbol,
                    1 if symbol in selected else 0,
                    1 if row.get("passed_template") else 0,
                    row.get("market_regime"),
                    1 if row.get("market_confirmed_uptrend") else 0,
                    row.get("rs_percentile"),
                    row.get("revenue_growth"),
                    row.get("eps_growth"),
                    row.get("revenue_acceleration"),
                    row.get("eps_acceleration"),
                    row.get("stage_number"),
                    row.get("base_label"),
                    row.get("candidate_status"),
                    row.get("pivot_price"),
                    row.get("buy_point"),
                    row.get("buy_limit_price"),
                    row.get("initial_stop_price"),
                    row.get("initial_stop_pct"),
                    row.get("distance_to_pivot_pct"),
                    row.get("distance_to_buy_point_pct"),
                    row.get("buy_zone_pct"),
                    1 if row.get("breakout_signal") else 0,
                    1 if row.get("rule_watch_candidate") else 0,
                    1 if row.get("rule_entry_candidate") else 0,
                    next_earnings,
                    row.get("earnings_days_away"),
                    payload,
                ),
            )
        self.conn.commit()

    def save_screening_batch(
        self,
        screen_date: str,
        market_regime: Optional[str],
        confirmed_uptrend: bool,
        approved_symbols: Optional[List[str]] = None,
        row_count: int = 0,
    ):
        approved = approved_symbols or []
        self.conn.execute(
            """
            INSERT OR REPLACE INTO screening_batches (
                screen_date, market_regime, market_confirmed_uptrend,
                row_count, selected_count, approved_symbols_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                screen_date,
                market_regime,
                1 if confirmed_uptrend else 0,
                row_count,
                len(approved),
                json.dumps(approved),
            ),
        )
        self.conn.commit()

    # ── Queries ──────────────────────────────────────────────────────

    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trades_for_symbol(self, symbol: str) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC", (symbol,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_today_trades(self) -> List[Dict]:
        return self.get_trades_on_date()

    def get_trades_on_date(self, trade_date: Optional[str] = None) -> List[Dict]:
        today = trade_date or date.today().isoformat()
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE date(timestamp) = ?", (today,)
        ).fetchall()
        return [dict(r) for r in rows]

    def was_stopped_today(self, symbol: str, as_of: Optional[str] = None) -> bool:
        """True iff `symbol` had a filled bracket_stop_loss sell on `as_of`.

        Relies on reconciler-populated SELL rows where `reasoning='bracket_stop_loss'`.
        Callers must reconcile before using this — stop-loss legs fire broker-side and
        only reach SQLite via `OrderReconciler.reconcile_once()`.
        """
        day = as_of or date.today().isoformat()
        row = self.conn.execute(
            """
            SELECT 1 FROM trades
            WHERE symbol = ?
              AND side = 'sell'
              AND LOWER(COALESCE(reasoning, '')) LIKE '%stop_loss%'
              AND LOWER(status) LIKE '%filled%'
              AND COALESCE(filled_qty, 0) > 0
              AND date(timestamp) = ?
            LIMIT 1
            """,
            (symbol, day),
        ).fetchone()
        return row is not None

    def get_trade_summary(self, trade_date: Optional[str] = None) -> Dict[str, Any]:
        summary_date = trade_date or date.today().isoformat()
        trades = self.get_trades_on_date(summary_date)
        filled_trades = [
            t for t in trades if "fill" in (t.get("status") or "").lower()
        ]

        gross_filled_notional = sum(
            (t.get("filled_qty") or t.get("qty") or 0) * (t.get("filled_price") or 0)
            for t in filled_trades
        )

        return {
            "date": summary_date,
            "total_orders": len(trades),
            "filled_orders": len(filled_trades),
            "buy_orders": sum(1 for t in trades if t.get("side") == "buy"),
            "sell_orders": sum(1 for t in trades if t.get("side") == "sell"),
            "symbols": sorted({t.get("symbol") for t in trades if t.get("symbol")}),
            "gross_filled_notional": gross_filled_notional,
        }

    def get_today_pl(self) -> float:
        today = date.today().isoformat()
        row = self.conn.execute(
            "SELECT daily_pl FROM daily_snapshots WHERE date = ?", (today,)
        ).fetchone()
        return row["daily_pl"] if row else 0.0

    def get_setup_candidates_on_date(self, screen_date: Optional[str] = None) -> List[Dict]:
        target_date = screen_date or date.today().isoformat()
        rows = self.conn.execute(
            """
            SELECT *
            FROM setup_candidates
            WHERE screen_date = ?
            ORDER BY
                rule_entry_candidate DESC,
                selected_for_analysis DESC,
                passed_template DESC,
                rs_percentile DESC
            """,
            (target_date,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_setup_candidates(self) -> List[Dict]:
        row = self.conn.execute(
            "SELECT MAX(screen_date) AS screen_date FROM setup_candidates"
        ).fetchone()
        if row is None or row["screen_date"] is None:
            return []
        return self.get_setup_candidates_on_date(row["screen_date"])

    def get_latest_screening_batch(self) -> Optional[Dict]:
        row = self.conn.execute(
            """
            SELECT *
            FROM screening_batches
            ORDER BY screen_date DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["approved_symbols"] = json.loads(result.get("approved_symbols_json") or "[]")
        return result

    def get_win_rate(self) -> Dict[str, Any]:
        rows = self.conn.execute(
            """SELECT symbol, side, filled_price FROM trades
               WHERE status LIKE '%fill%' ORDER BY timestamp"""
        ).fetchall()
        if not rows:
            return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0}
        return {"total_trades": len(rows), "note": "detailed P&L requires position pairing"}

    # ── Position States (Exit Manager) ─────────────────────────────

    def get_position_state(self, symbol: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM position_states WHERE symbol = ?", (symbol,)
        ).fetchone()
        if row is None:
            return None
        state = dict(row)
        state["partial_taken"] = bool(state.get("partial_taken", 0))
        return state

    def upsert_position_state(self, symbol: str, state: Dict):
        # Merge with existing row so v1 callers (passing only core fields) don't
        # overwrite v2-populated columns like stop_order_id / bars_held / variant
        # or feedback-loop fields like base_pattern / regime_at_entry.
        existing = self.get_position_state(symbol) or {}
        merged = {**existing, **state}
        self.conn.execute(
            """INSERT OR REPLACE INTO position_states
               (symbol, entry_price, entry_date, highest_close, current_stop,
                partial_taken, stop_type, stop_order_id, tp_order_id,
                entry_order_id, variant, bars_held,
                regime_at_entry, base_pattern, rs_at_entry, stage_at_entry,
                updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            (
                symbol,
                merged["entry_price"],
                merged["entry_date"],
                merged["highest_close"],
                merged["current_stop"],
                1 if merged.get("partial_taken") else 0,
                merged.get("stop_type", "initial"),
                merged.get("stop_order_id"),
                merged.get("tp_order_id"),
                merged.get("entry_order_id"),
                merged.get("variant"),
                int(merged.get("bars_held") or 0),
                merged.get("regime_at_entry"),
                merged.get("base_pattern"),
                merged.get("rs_at_entry"),
                merged.get("stage_at_entry"),
            ),
        )
        self.conn.commit()

    def delete_position_state(self, symbol: str):
        self.conn.execute("DELETE FROM position_states WHERE symbol = ?", (symbol,))
        self.conn.commit()

    # ── Experiments ──────────────────────────────────────────────────

    def save_experiment(self, experiment_id: str, config_yaml: str,
                        start_date: str, primary_metric: str = "sharpe_ratio",
                        min_trades: int = 30, min_days: int = 20):
        self.conn.execute(
            """INSERT OR REPLACE INTO experiments
               (experiment_id, config_yaml, start_date, primary_metric,
                min_trades, min_days)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (experiment_id, config_yaml, start_date, primary_metric,
             min_trades, min_days),
        )
        self.conn.commit()

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_experiment_status(self, experiment_id: str, status: str):
        self.conn.execute(
            "UPDATE experiments SET status = ? WHERE experiment_id = ?",
            (status, experiment_id),
        )
        self.conn.commit()

    def save_experiment_snapshot(self, experiment_id: str, variant_name: str,
                                snapshot_date: str, metrics: Dict):
        self.conn.execute(
            """INSERT OR REPLACE INTO experiment_snapshots
               (experiment_id, variant_name, date, equity, daily_return,
                total_return, sharpe_ratio, max_drawdown, total_trades, win_rate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                experiment_id, variant_name, snapshot_date,
                metrics.get("equity"), metrics.get("daily_return"),
                metrics.get("total_return"), metrics.get("sharpe_ratio"),
                metrics.get("max_drawdown"), metrics.get("total_trades"),
                metrics.get("win_rate"),
            ),
        )
        self.conn.commit()

    # ── Trade Outcomes (Reflection) ──────────────────────────────────

    def log_trade_outcome(self, outcome: Dict) -> int:
        cur = self.conn.execute(
            """INSERT INTO trade_outcomes
               (symbol, entry_date, exit_date, entry_price, exit_price,
                return_pct, hold_days, exit_reason, base_pattern,
                stage_at_entry, rs_at_entry, regime_at_entry,
                max_favorable_excursion, max_adverse_excursion)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                outcome.get("symbol"),
                outcome.get("entry_date"),
                outcome.get("exit_date"),
                outcome.get("entry_price"),
                outcome.get("exit_price"),
                outcome.get("return_pct"),
                outcome.get("hold_days"),
                outcome.get("exit_reason"),
                outcome.get("base_pattern"),
                outcome.get("stage_at_entry"),
                outcome.get("rs_at_entry"),
                outcome.get("regime_at_entry"),
                outcome.get("max_favorable_excursion"),
                outcome.get("max_adverse_excursion"),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_pattern_stats(self) -> List[Dict]:
        rows = self.conn.execute("""
            SELECT base_pattern, regime_at_entry,
                   COUNT(*) as trades,
                   AVG(CASE WHEN return_pct > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                   AVG(return_pct) as avg_return
            FROM trade_outcomes
            GROUP BY base_pattern, regime_at_entry
            HAVING trades >= 3
        """).fetchall()
        return [dict(r) for r in rows]

    # ── Dashboard Queries ──────────────────────────────────────────

    def get_all_snapshots(self) -> List[Dict]:
        """All daily snapshots, oldest first."""
        rows = self.conn.execute(
            "SELECT * FROM daily_snapshots ORDER BY date ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_snapshots_in_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Daily snapshots in [start_date, end_date] (inclusive), oldest first."""
        rows = self.conn.execute(
            "SELECT * FROM daily_snapshots WHERE date BETWEEN ? AND ? ORDER BY date ASC",
            (start_date, end_date),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trades_in_range(self, start_date: str, end_date: str) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE date(timestamp) BETWEEN ? AND ? ORDER BY timestamp",
            (start_date, end_date),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_trade_outcomes(self) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM trade_outcomes ORDER BY exit_date DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trade_outcomes_in_range(self, start_date: str, end_date: str) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM trade_outcomes WHERE exit_date BETWEEN ? AND ? ORDER BY exit_date",
            (start_date, end_date),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_entry_signal_for_trade(self, symbol: str, entry_date: str) -> Optional[Dict]:
        """Get the entry signal reasoning for a trade by matching symbol + date."""
        row = self.conn.execute(
            """SELECT s.reasoning, s.full_analysis, s.confidence, s.action
               FROM signals s
               JOIN trades t ON t.signal_id = s.id
               WHERE t.symbol = ? AND date(t.timestamp) = ?
               AND t.side = 'buy'
               ORDER BY t.timestamp DESC LIMIT 1""",
            (symbol, entry_date),
        ).fetchone()
        return dict(row) if row else None

    def update_trade_analysis(self, outcome_id: int, analysis: str):
        """Cache LLM analysis on a trade outcome."""
        self.conn.execute(
            "UPDATE trade_outcomes SET trade_analysis = ? WHERE id = ?",
            (analysis, outcome_id),
        )
        self.conn.commit()

    def update_trade_excursion(
        self,
        outcome_id: int,
        mfe: Optional[float],
        mae: Optional[float],
    ) -> None:
        """Patch MFE/MAE onto an existing trade outcome row (used by backfill)."""
        self.conn.execute(
            """UPDATE trade_outcomes
               SET max_favorable_excursion = ?, max_adverse_excursion = ?
               WHERE id = ?""",
            (mfe, mae, outcome_id),
        )
        self.conn.commit()

    def update_trade_outcome_base_pattern(
        self,
        outcome_id: int,
        base_pattern: Optional[str],
    ) -> None:
        """Patch base_pattern on an existing trade outcome row.

        Used by `scripts/backfill_chan_base_pattern.py` to populate
        historical Chan trades whose T-type was captured on the entry
        signal but never copied into trade_outcomes.
        """
        self.conn.execute(
            "UPDATE trade_outcomes SET base_pattern = ? WHERE id = ?",
            (base_pattern, outcome_id),
        )
        self.conn.commit()

    # ── Proposals (weekly-review feedback loop) ───────────────────────

    def insert_proposal(
        self,
        *,
        variant: str,
        iso_week: str,
        title: str,
        what: Optional[str] = None,
        why: Optional[str] = None,
        how_to_validate: Optional[str] = None,
        risk: Optional[str] = None,
    ) -> int:
        """Insert one proposal row from the weekly review. Returns row id."""
        cur = self.conn.execute(
            """INSERT INTO proposals
               (variant, iso_week, title, what, why, how_to_validate, risk, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'open')""",
            (variant, iso_week, title, what, why, how_to_validate, risk),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_proposal_status(
        self,
        proposal_id: int,
        status: str,
        outcome_summary: Optional[str] = None,
    ) -> None:
        """Update a proposal's status (open/accepted/rejected/tested) + optional outcome."""
        self.conn.execute(
            """UPDATE proposals
               SET status = ?, status_updated_at = datetime('now'),
                   outcome_summary = COALESCE(?, outcome_summary)
               WHERE id = ?""",
            (status, outcome_summary, proposal_id),
        )
        self.conn.commit()

    def get_proposals(
        self,
        variant: Optional[str] = None,
        status: Optional[str] = None,
        iso_week: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Filtered read of proposals table. Most-recent-first."""
        clauses, params = [], []
        if variant:
            clauses.append("variant = ?"); params.append(variant)
        if status:
            clauses.append("status = ?"); params.append(status)
        if iso_week:
            clauses.append("iso_week = ?"); params.append(iso_week)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM proposals {where} ORDER BY created_at DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
