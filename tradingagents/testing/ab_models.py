"""Data models for A/B testing experiments."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentVariant:
    """A single variant in an A/B experiment."""

    name: str  # e.g. "control", "trailing_exit"
    description: str
    config_overrides: dict = field(default_factory=dict)
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    db_path: str = ""


@dataclass
class Experiment:
    """An A/B experiment comparing strategy variants."""

    experiment_id: str  # e.g. "exit-strategy-v1"
    start_date: str
    variants: List[ExperimentVariant] = field(default_factory=list)
    min_trades: int = 30
    min_days: int = 20
    primary_metric: str = "sharpe_ratio"
    status: str = "running"  # running | paused | completed | promoted


@dataclass
class VariantMetrics:
    """Performance metrics for a single variant."""

    variant_name: str
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_hold_days: float = 0.0
    total_trades: int = 0
    daily_returns: List[float] = field(default_factory=list)
    # Set by comparison
    t_stat: Optional[float] = None
    p_value: Optional[float] = None
    confidence: Optional[float] = None
