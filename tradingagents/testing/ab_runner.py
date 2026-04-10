"""Orchestrates parallel execution of strategy variants."""

import logging
from typing import Dict

from tradingagents.automation.orchestrator import Orchestrator

from .ab_config import build_variant_config
from .ab_models import Experiment

logger = logging.getLogger(__name__)


class ABRunner:
    """Orchestrates parallel execution of strategy variants."""

    def __init__(self, experiment: Experiment, base_config: dict):
        self.experiment = experiment
        self.orchestrators: Dict[str, Orchestrator] = {}
        for variant in experiment.variants:
            variant_config = build_variant_config(base_config, variant)
            self.orchestrators[variant.name] = Orchestrator(variant_config)

    def run_daily_analysis(self) -> Dict[str, Dict]:
        """Run all variants sequentially (same market data, different decisions)."""
        results = {}
        for name, orch in self.orchestrators.items():
            logger.info(f"Running variant '{name}'...")
            try:
                results[name] = orch.run_daily_analysis()
            except Exception as e:
                logger.error(f"Variant '{name}' failed: {e}", exc_info=True)
                results[name] = {"error": str(e)}
        return results

    def run_daily_reflection(self) -> Dict[str, Dict]:
        """Run reflection for all variants."""
        results = {}
        for name, orch in self.orchestrators.items():
            try:
                results[name] = orch.run_daily_reflection()
            except Exception as e:
                logger.error(f"Variant '{name}' reflection failed: {e}", exc_info=True)
                results[name] = {"error": str(e)}
        return results

    def take_market_snapshot(self) -> Dict[str, Dict]:
        """Take market snapshot for all variants."""
        results = {}
        for name, orch in self.orchestrators.items():
            try:
                results[name] = orch.take_market_snapshot()
            except Exception as e:
                logger.error(f"Variant '{name}' snapshot failed: {e}", exc_info=True)
                results[name] = {"error": str(e)}
        return results
