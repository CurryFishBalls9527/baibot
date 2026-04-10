"""Compare performance across experiment variants."""

import logging
from typing import Dict, Tuple

from tradingagents.storage.database import TradingDatabase

from .ab_models import Experiment, VariantMetrics

logger = logging.getLogger(__name__)


class ABReporter:
    """Compare performance across experiment variants."""

    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.dbs: Dict[str, TradingDatabase] = {
            v.name: TradingDatabase(v.db_path) for v in experiment.variants
        }

    def compute_metrics(self, variant_name: str) -> VariantMetrics:
        """Compute performance metrics from a variant's DB."""
        db = self.dbs[variant_name]
        snapshots = db.get_snapshots(days=365)

        if not snapshots:
            return VariantMetrics(variant_name=variant_name)

        equities = [s["equity"] for s in reversed(snapshots)]
        starting = db.get_starting_equity() or equities[0]

        # Total return
        total_return = (equities[-1] - starting) / starting if starting > 0 else 0.0

        # Daily returns
        daily_returns = []
        for i in range(1, len(equities)):
            if equities[i - 1] > 0:
                daily_returns.append(
                    (equities[i] - equities[i - 1]) / equities[i - 1]
                )

        # Max drawdown
        max_equity = 0.0
        max_drawdown = 0.0
        for eq in equities:
            if eq > max_equity:
                max_equity = eq
            dd = (max_equity - eq) / max_equity if max_equity > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

        # Sharpe ratio
        avg_ret = sum(daily_returns) / len(daily_returns) if daily_returns else 0
        std_ret = (
            (sum((r - avg_ret) ** 2 for r in daily_returns) / len(daily_returns))
            ** 0.5
            if daily_returns
            else 0
        )
        sharpe = (avg_ret / std_ret * (252**0.5)) if std_ret > 0 else 0

        # Sortino ratio (downside deviation only)
        neg_returns = [r for r in daily_returns if r < 0]
        downside_std = (
            (sum(r**2 for r in neg_returns) / len(neg_returns)) ** 0.5
            if neg_returns
            else 0
        )
        sortino = (avg_ret / downside_std * (252**0.5)) if downside_std > 0 else 0

        # Win rate and profit factor from trades
        trades = db.get_recent_trades(limit=10000)
        filled = [t for t in trades if "fill" in (t.get("status") or "").lower()]
        wins = sum(
            1
            for t in filled
            if t.get("side") == "sell" and (t.get("filled_price") or 0) > 0
        )
        total_trades = len(filled)
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        # Profit factor (approximate from daily returns)
        gross_profit = sum(r for r in daily_returns if r > 0)
        gross_loss = abs(sum(r for r in daily_returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return VariantMetrics(
            variant_name=variant_name,
            total_return=total_return,
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=round(profit_factor, 4),
            avg_hold_days=0.0,
            total_trades=total_trades,
            daily_returns=daily_returns,
        )

    def compare(self) -> Dict[str, VariantMetrics]:
        """Compare all variants. Returns metrics + statistical tests."""
        metrics = {
            v.name: self.compute_metrics(v.name) for v in self.experiment.variants
        }

        if len(self.experiment.variants) < 2:
            return metrics

        # Welch's t-test on daily returns between control and each challenger
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy not installed; skipping statistical tests")
            return metrics

        control = metrics[self.experiment.variants[0].name]
        for variant in self.experiment.variants[1:]:
            challenger = metrics[variant.name]
            if len(control.daily_returns) < 2 or len(challenger.daily_returns) < 2:
                continue
            t_stat, p_value = stats.ttest_ind(
                control.daily_returns, challenger.daily_returns, equal_var=False
            )
            challenger.t_stat = t_stat
            challenger.p_value = p_value
            challenger.confidence = 1 - p_value if p_value is not None else None

        return metrics

    def is_promotion_ready(self) -> Tuple[bool, str]:
        """Check if challenger beats control with sufficient confidence."""
        metrics = self.compare()
        control_name = self.experiment.variants[0].name

        for variant in self.experiment.variants[1:]:
            m = metrics[variant.name]
            if m.total_trades < self.experiment.min_trades:
                return False, (
                    f"Not enough trades ({m.total_trades}/{self.experiment.min_trades})"
                )
            if m.confidence is not None and m.confidence >= 0.80:
                primary = getattr(m, self.experiment.primary_metric, None)
                control_primary = getattr(
                    metrics[control_name], self.experiment.primary_metric, None
                )
                if primary is not None and control_primary is not None and primary > control_primary:
                    return True, (
                        f"{variant.name} beats {control_name} with "
                        f"{m.confidence:.0%} confidence "
                        f"({self.experiment.primary_metric}: "
                        f"{primary:.4f} vs {control_primary:.4f})"
                    )

        return False, "Challenger has not yet proven superior"

    def summary_table(self) -> str:
        """Return a formatted comparison table."""
        metrics = self.compare()
        lines = [
            f"{'Variant':<20} {'Return':>8} {'Sharpe':>8} {'Sortino':>8} "
            f"{'MaxDD':>8} {'Trades':>7} {'Conf':>8}",
            "-" * 79,
        ]
        for v in self.experiment.variants:
            m = metrics[v.name]
            conf_str = f"{m.confidence:.0%}" if m.confidence is not None else "n/a"
            lines.append(
                f"{v.name:<20} {m.total_return:>7.2%} {m.sharpe_ratio:>8.2f} "
                f"{m.sortino_ratio:>8.2f} {m.max_drawdown:>7.2%} "
                f"{m.total_trades:>7} {conf_str:>8}"
            )
        return "\n".join(lines)
