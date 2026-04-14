"""Orchestrates parallel execution of strategy variants."""

import logging
from typing import Dict

from tradingagents.automation.orchestrator import Orchestrator

from .ab_config import build_variant_config
from .ab_models import Experiment

logger = logging.getLogger(__name__)


def _build_orchestrator(variant_config: dict):
    """Build the appropriate orchestrator based on strategy_type config."""
    strategy_type = variant_config.get("strategy_type", "minervini")
    if strategy_type == "chan":
        from tradingagents.automation.chan_orchestrator import ChanOrchestrator
        return ChanOrchestrator(variant_config)
    return Orchestrator(variant_config)


class ABRunner:
    """Orchestrates parallel execution of strategy variants."""

    def __init__(self, experiment: Experiment, base_config: dict):
        self.experiment = experiment
        self.orchestrators: Dict[str, Orchestrator] = {}
        for variant in experiment.variants:
            variant_config = build_variant_config(base_config, variant)
            self.orchestrators[variant.name] = _build_orchestrator(variant_config)

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

    def run_intraday_scan(self) -> Dict[str, Dict]:
        """Run intraday scans for Chan (30m signals) and mechanical (price checks)."""
        from tradingagents.automation.chan_orchestrator import ChanOrchestrator

        results = {}
        for name, orch in self.orchestrators.items():
            if isinstance(orch, ChanOrchestrator):
                logger.info(f"Intraday scan for variant '{name}' (Chan)...")
                try:
                    results[name] = orch.run_daily_analysis()
                except Exception as e:
                    logger.error(f"Variant '{name}' intraday scan failed: {e}", exc_info=True)
                    results[name] = {"error": str(e)}
            else:
                logger.info(f"Intraday entry scan for variant '{name}'...")
                try:
                    results[name] = orch.run_intraday_entry_scan()
                except Exception as e:
                    logger.error(f"Variant '{name}' intraday scan failed: {e}", exc_info=True)
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

    def bulk_refresh_chan_data(self) -> Dict[str, Dict]:
        """Bulk-refresh 30m bars for all Chan orchestrators."""
        from tradingagents.automation.chan_orchestrator import ChanOrchestrator

        results = {}
        for name, orch in self.orchestrators.items():
            if isinstance(orch, ChanOrchestrator):
                logger.info("Bulk 30m refresh for variant '%s'...", name)
                try:
                    results[name] = orch.bulk_refresh_30m_data()
                except Exception as e:
                    logger.error("Bulk 30m refresh failed for '%s': %s", name, e, exc_info=True)
                    results[name] = {"error": str(e)}
        return results
