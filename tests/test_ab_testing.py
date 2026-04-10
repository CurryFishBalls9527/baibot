"""Tests for A/B testing harness (Phase 1)."""

import os
import tempfile
import pytest

from tradingagents.testing.ab_models import Experiment, ExperimentVariant, VariantMetrics
from tradingagents.testing.ab_config import build_variant_config, load_experiment, save_experiment
from tradingagents.testing.ab_reporter import ABReporter
from tradingagents.storage.database import TradingDatabase


class TestABModels:
    def test_experiment_defaults(self):
        exp = Experiment(experiment_id="test-1", start_date="2026-01-01")
        assert exp.min_trades == 30
        assert exp.min_days == 20
        assert exp.primary_metric == "sharpe_ratio"
        assert exp.status == "running"
        assert exp.variants == []

    def test_variant_metrics_defaults(self):
        m = VariantMetrics(variant_name="control")
        assert m.total_return == 0.0
        assert m.daily_returns == []
        assert m.t_stat is None
        assert m.p_value is None


class TestABConfig:
    def test_build_variant_config_merges(self):
        base = {"alpaca_api_key": "base_key", "db_path": "base.db", "foo": "bar"}
        variant = ExperimentVariant(
            name="test",
            description="test variant",
            alpaca_api_key="variant_key",
            db_path="variant.db",
            config_overrides={"foo": "baz", "extra": True},
        )
        result = build_variant_config(base, variant)
        assert result["alpaca_api_key"] == "variant_key"
        assert result["db_path"] == "variant.db"
        assert result["foo"] == "baz"
        assert result["extra"] is True

    def test_build_variant_config_no_override(self):
        base = {"alpaca_api_key": "base_key", "db_path": "base.db"}
        variant = ExperimentVariant(name="control", description="no changes")
        result = build_variant_config(base, variant)
        assert result["alpaca_api_key"] == "base_key"
        assert result["db_path"] == "base.db"

    def test_save_and_load_experiment(self):
        exp = Experiment(
            experiment_id="round-trip-test",
            start_date="2026-04-01",
            variants=[
                ExperimentVariant(
                    name="control", description="baseline",
                    db_path="/tmp/control.db",
                ),
                ExperimentVariant(
                    name="challenger", description="with trailing stop",
                    config_overrides={"trail_stop_pct": 0.08},
                    db_path="/tmp/challenger.db",
                ),
            ],
            min_trades=20,
            min_days=15,
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name

        try:
            save_experiment(exp, path)
            loaded = load_experiment(path)
            assert loaded.experiment_id == "round-trip-test"
            assert len(loaded.variants) == 2
            assert loaded.variants[1].config_overrides["trail_stop_pct"] == 0.08
            assert loaded.min_trades == 20
        finally:
            os.unlink(path)


class TestABReporter:
    def _make_db_with_snapshots(self, db_path, equities):
        """Create a DB with daily snapshots from a list of equity values."""
        db = TradingDatabase(db_path)
        for i, eq in enumerate(equities):
            d = f"2026-01-{i + 1:02d}"
            prev = equities[i - 1] if i > 0 else eq
            daily_pl = eq - prev
            daily_pl_pct = daily_pl / prev if prev > 0 else 0
            db.take_snapshot(
                equity=eq, cash=eq * 0.5, buying_power=eq,
                portfolio_value=eq, positions=[],
                daily_pl=daily_pl, daily_pl_pct=daily_pl_pct,
                snapshot_date=d,
            )
        return db

    def test_compute_metrics_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = TradingDatabase(db_path)
            exp = Experiment(
                experiment_id="test",
                start_date="2026-01-01",
                variants=[ExperimentVariant(name="empty", description="", db_path=db_path)],
            )
            reporter = ABReporter(exp)
            m = reporter.compute_metrics("empty")
            assert m.total_trades == 0
            assert m.total_return == 0.0
            db.close()
        finally:
            os.unlink(db_path)

    def test_compute_metrics_with_data(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            equities = [100000, 100500, 101000, 100800, 101500, 102000]
            db = self._make_db_with_snapshots(db_path, equities)
            exp = Experiment(
                experiment_id="test",
                start_date="2026-01-01",
                variants=[ExperimentVariant(name="v1", description="", db_path=db_path)],
            )
            reporter = ABReporter(exp)
            m = reporter.compute_metrics("v1")
            assert m.total_return > 0
            assert len(m.daily_returns) == 5  # 6 snapshots -> 5 returns
            assert m.max_drawdown >= 0
            db.close()
        finally:
            os.unlink(db_path)

    def test_is_promotion_ready_not_enough_trades(self):
        paths = []
        try:
            for i in range(2):
                f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
                paths.append(f.name)
                f.close()
                equities = [100000 + j * 100 for j in range(10)]
                self._make_db_with_snapshots(paths[-1], equities)

            exp = Experiment(
                experiment_id="test",
                start_date="2026-01-01",
                variants=[
                    ExperimentVariant(name="control", description="", db_path=paths[0]),
                    ExperimentVariant(name="challenger", description="", db_path=paths[1]),
                ],
                min_trades=30,
            )
            reporter = ABReporter(exp)
            ready, reason = reporter.is_promotion_ready()
            assert not ready
            assert "Not enough trades" in reason
        finally:
            for p in paths:
                os.unlink(p)


class TestDatabaseNewTables:
    def test_position_state_crud(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = TradingDatabase(db_path)

            # Initially empty
            assert db.get_position_state("AAPL") is None

            # Upsert
            db.upsert_position_state("AAPL", {
                "entry_price": 150.0,
                "entry_date": "2026-04-01",
                "highest_close": 155.0,
                "current_stop": 142.0,
                "partial_taken": False,
            })
            state = db.get_position_state("AAPL")
            assert state is not None
            assert state["entry_price"] == 150.0
            assert state["partial_taken"] is False

            # Update
            db.upsert_position_state("AAPL", {
                "entry_price": 150.0,
                "entry_date": "2026-04-01",
                "highest_close": 160.0,
                "current_stop": 150.0,
                "partial_taken": True,
            })
            state = db.get_position_state("AAPL")
            assert state["highest_close"] == 160.0
            assert state["partial_taken"] is True

            # Delete
            db.delete_position_state("AAPL")
            assert db.get_position_state("AAPL") is None

            db.close()
        finally:
            os.unlink(db_path)

    def test_experiment_crud(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = TradingDatabase(db_path)

            db.save_experiment("exp-1", "config: yaml", "2026-01-01")
            exp = db.get_experiment("exp-1")
            assert exp is not None
            assert exp["status"] == "running"

            db.update_experiment_status("exp-1", "promoted")
            exp = db.get_experiment("exp-1")
            assert exp["status"] == "promoted"

            db.close()
        finally:
            os.unlink(db_path)

    def test_trade_outcomes(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = TradingDatabase(db_path)

            # Log multiple outcomes
            for i in range(5):
                db.log_trade_outcome({
                    "symbol": "AAPL",
                    "entry_date": "2026-01-01",
                    "exit_date": "2026-02-01",
                    "entry_price": 150.0,
                    "exit_price": 160.0 if i < 3 else 145.0,
                    "return_pct": 0.067 if i < 3 else -0.033,
                    "hold_days": 30,
                    "exit_reason": "trailing_stop",
                    "base_pattern": "VCP",
                    "regime_at_entry": "confirmed_uptrend",
                })

            stats = db.get_pattern_stats()
            assert len(stats) == 1
            assert stats[0]["trades"] == 5
            assert stats[0]["base_pattern"] == "VCP"
            assert stats[0]["win_rate"] == pytest.approx(0.6, abs=0.01)

            db.close()
        finally:
            os.unlink(db_path)
