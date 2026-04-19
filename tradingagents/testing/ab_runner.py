"""Orchestrates parallel execution of strategy variants."""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Tuple

from tradingagents.automation.orchestrator import Orchestrator

from .ab_config import build_variant_config
from .ab_models import Experiment

logger = logging.getLogger(__name__)

# Cap on simultaneous variant workers. Matches feedback_parallel_backtests.md
# (user machine has limited compute). Variants sharing a DuckDB writer path
# are additionally serialized within a group to avoid writer-lock contention.
_MAX_WORKERS = 3


def _build_orchestrator(variant_config: dict):
    """Build the appropriate orchestrator based on strategy_type config."""
    strategy_type = variant_config.get("strategy_type", "minervini")
    if strategy_type == "chan":
        from tradingagents.automation.chan_orchestrator import ChanOrchestrator
        return ChanOrchestrator(variant_config)
    if strategy_type == "intraday_mechanical":
        from tradingagents.automation.intraday_orchestrator import IntradayOrchestrator
        return IntradayOrchestrator(variant_config)
    return Orchestrator(variant_config)


def _shared_writer_key(orch) -> str:
    """Identify the DuckDB writer path a variant contends on, if any.

    Variants returning the same key serialize against each other; variants
    with distinct keys run in parallel. Only Chan orchestrators share a
    DuckDB writer today (intraday_30m_broad.duckdb between chan + chan_v2).
    Intraday mechanical fetches bars per-scan directly from Alpaca so it has
    no shared writer.
    """
    from tradingagents.automation.chan_orchestrator import ChanOrchestrator
    if isinstance(orch, ChanOrchestrator):
        return f"chan_intraday:{orch.intraday_db}"
    return f"indep:{id(orch)}"


class ABRunner:
    """Orchestrates parallel execution of strategy variants."""

    def __init__(self, experiment: Experiment, base_config: dict):
        self.experiment = experiment
        self.orchestrators: Dict[str, Orchestrator] = {}
        for variant in experiment.variants:
            variant_config = build_variant_config(base_config, variant)
            self.orchestrators[variant.name] = _build_orchestrator(variant_config)

    def _grouped_parallel(
        self,
        call_fn: Callable[[str, object], Dict],
        label: str,
    ) -> Dict[str, Dict]:
        """Run `call_fn(name, orch)` across variants, grouped by shared DuckDB.

        Variants in the same shared-writer group run sequentially; groups run
        in parallel up to `_MAX_WORKERS`. Exceptions in one variant do not
        affect the others.
        """
        groups: Dict[str, List[Tuple[str, object]]] = {}
        for name, orch in self.orchestrators.items():
            groups.setdefault(_shared_writer_key(orch), []).append((name, orch))

        def _run_group(items: List[Tuple[str, object]]) -> Dict[str, Dict]:
            group_results: Dict[str, Dict] = {}
            for name, orch in items:
                logger.info("Variant '%s' %s starting...", name, label)
                try:
                    group_results[name] = call_fn(name, orch)
                except Exception as e:
                    logger.error(
                        "Variant '%s' %s failed: %s", name, label, e, exc_info=True
                    )
                    group_results[name] = {"error": str(e)}
            return group_results

        merged: Dict[str, Dict] = {}
        with ThreadPoolExecutor(
            max_workers=_MAX_WORKERS, thread_name_prefix="variant_group"
        ) as executor:
            for partial in executor.map(_run_group, groups.values()):
                merged.update(partial)
        return merged

    def run_daily_analysis(self) -> Dict[str, Dict]:
        """Run all variants (parallel across shared-writer groups)."""
        return self._grouped_parallel(
            lambda _name, orch: orch.run_daily_analysis(),
            "daily analysis",
        )

    def reconcile_orders(self) -> Dict[str, Dict]:
        """Run local↔broker reconciliation for all variants (Track P-SYNC)."""
        return self._grouped_parallel(
            lambda _name, orch: orch.reconcile_orders(),
            "reconcile",
        )

    def run_intraday_scan(self) -> Dict[str, Dict]:
        """Run intraday scans (Chan: 30m signals; intraday_mechanical: 15m NR4 scan;
        others: Minervini entry price checks)."""
        from tradingagents.automation.chan_orchestrator import ChanOrchestrator
        from tradingagents.automation.intraday_orchestrator import IntradayOrchestrator

        def _dispatch(_name: str, orch) -> Dict:
            if isinstance(orch, ChanOrchestrator):
                return orch.run_daily_analysis()
            if isinstance(orch, IntradayOrchestrator):
                return orch.scan()
            return orch.run_intraday_entry_scan()

        return self._grouped_parallel(_dispatch, "intraday scan")

    def flatten_all_intraday(self) -> Dict[str, Dict]:
        """EOD flatten hook for intraday_mechanical variants (15:55 ET job).

        Non-intraday variants return {"status": "not_applicable"}. This runs
        sequentially across intraday variants because the flatten path makes
        Alpaca position-close calls; if multiple intraday variants are ever
        added they're on different accounts anyway, so we could parallelize
        later — today there's one intraday variant so sequential is fine.
        """
        from tradingagents.automation.intraday_orchestrator import IntradayOrchestrator

        results: Dict[str, Dict] = {}
        for name, orch in self.orchestrators.items():
            if not isinstance(orch, IntradayOrchestrator):
                results[name] = {"status": "not_applicable"}
                continue
            logger.info("Intraday EOD flatten: variant '%s'", name)
            try:
                results[name] = orch.flatten_all()
            except Exception as e:
                logger.error(
                    "Variant '%s' flatten failed: %s", name, e, exc_info=True
                )
                results[name] = {"error": str(e)}
        return results

    def run_daily_reflection(self) -> Dict[str, Dict]:
        """Run reflection for all variants."""
        return self._grouped_parallel(
            lambda _name, orch: orch.run_daily_reflection(),
            "reflection",
        )

    def take_market_snapshot(self) -> Dict[str, Dict]:
        """Take market snapshot for all variants."""
        return self._grouped_parallel(
            lambda _name, orch: orch.take_market_snapshot(),
            "market snapshot",
        )

    def bulk_refresh_chan_data(self) -> Dict[str, Dict]:
        """Bulk-refresh 30m bars for Chan variants; dedupes on shared DB path.

        Chan and chan_v2 today share `intraday_30m_broad.duckdb`. Running both
        refreshes would duplicate ~90 days of multi-batch Alpaca fetches and
        contend on the DuckDB writer. We run the refresh once per unique
        intraday DB path; other variants sharing that path get a 'deduped'
        status surfaced in their result.
        """
        from tradingagents.automation.chan_orchestrator import ChanOrchestrator

        chan_entries = [
            (name, orch) for name, orch in self.orchestrators.items()
            if isinstance(orch, ChanOrchestrator)
        ]

        owner_by_db: Dict[str, str] = {}
        to_run: List[Tuple[str, ChanOrchestrator]] = []
        for name, orch in chan_entries:
            db_path = orch.intraday_db
            if db_path in owner_by_db:
                logger.info(
                    "Bulk 30m refresh: variant '%s' shares DB with '%s' (%s) — deduping",
                    name, owner_by_db[db_path], db_path,
                )
                continue
            owner_by_db[db_path] = name
            to_run.append((name, orch))

        results: Dict[str, Dict] = {}

        def _refresh(item: Tuple[str, ChanOrchestrator]) -> Tuple[str, Dict]:
            name, orch = item
            try:
                return name, orch.bulk_refresh_30m_data()
            except Exception as e:
                logger.error(
                    "Bulk 30m refresh failed for '%s': %s", name, e, exc_info=True
                )
                return name, {"error": str(e)}

        with ThreadPoolExecutor(
            max_workers=_MAX_WORKERS, thread_name_prefix="chan_refresh"
        ) as executor:
            for name, result in executor.map(_refresh, to_run):
                results[name] = result

        for name, orch in chan_entries:
            if name not in results:
                results[name] = {
                    "status": "deduped",
                    "shared_with": owner_by_db[orch.intraday_db],
                }
        return results
