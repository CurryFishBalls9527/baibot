"""Walk-forward backtester with dynamic Minervini screening.

Eliminates survivorship bias by screening stocks at each rebalance point
using only data available on that date. Stocks enter/leave the tradeable
universe dynamically, just like in live trading.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from .backtester import BacktestConfig
from .market_context import build_market_context
from .minervini import MinerviniConfig, MinerviniScreener
from .portfolio_backtester import PortfolioBacktestResult, PortfolioMinerviniBacktester
from .warehouse import MarketDataWarehouse

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    rebalance_frequency: str = "weekly"  # "daily", "weekly", or "monthly"
    min_template_score: int = 6
    min_rs_percentile: float = 70.0
    min_data_bars: int = 252  # require 1yr history for indicators
    max_screen_candidates: int = 50
    benchmark: str = "SPY"
    # Industry-group rank filter (Minervini "leaders in leading groups")
    sector_map_path: Optional[str] = "research_data/sector_map.json"
    group_by: str = "industry"  # "industry" or "sector"
    min_group_rank_pct: float = 0.0  # 0 = off; 60 = require top 40% group
    group_min_members: int = 3  # groups with fewer members keep their raw score
    # Gate the group filter by regime. Set to ("confirmed_uptrend",
    # "uptrend_under_pressure") to only enforce leaders-in-leading-groups
    # during trending markets. Empty tuple = apply to all regimes.
    group_filter_regimes: tuple = (
        "confirmed_uptrend",
        "uptrend_under_pressure",
    )


@dataclass
class WalkForwardResult:
    portfolio_result: PortfolioBacktestResult
    rebalance_log: pd.DataFrame
    universe_snapshots: Dict[str, List[str]]


class WalkForwardBacktester:
    """Walk-forward backtester that screens stocks dynamically at each rebalance."""

    def __init__(
        self,
        warehouse: MarketDataWarehouse,
        wf_config: WalkForwardConfig | None = None,
        backtest_config: BacktestConfig | None = None,
        screener_config: MinerviniConfig | None = None,
    ):
        self.warehouse = warehouse
        self.wf_config = wf_config or WalkForwardConfig()
        self.backtest_config = backtest_config or BacktestConfig()
        self.screener_config = screener_config or MinerviniConfig(
            require_fundamentals=False,
            require_market_uptrend=False,
        )
        self.screener = MinerviniScreener(self.screener_config)
        self.sector_map = self._load_sector_map()

    def _load_sector_map(self) -> Dict[str, Dict[str, str]]:
        path = self.wf_config.sector_map_path
        if not path:
            return {}
        p = Path(path)
        if not p.exists():
            if self.wf_config.min_group_rank_pct > 0:
                raise FileNotFoundError(
                    f"sector_map_path {path} missing but min_group_rank_pct > 0. "
                    "Run scripts/build_sector_map.py first."
                )
            return {}
        try:
            return json.loads(p.read_text())
        except Exception as e:
            logger.warning("Failed to load sector map %s: %s", path, e)
            return {}

    def run(
        self,
        seed_symbols: list[str],
        start_date: str,
        end_date: str,
        trade_start_date: str | None = None,
    ) -> WalkForwardResult:
        """Execute the full walk-forward backtest.

        Args:
            seed_symbols: Broad universe to screen from (500-700 symbols).
            start_date: Data start (includes warmup period for indicators).
            end_date: Backtest end date.
            trade_start_date: When to start trading (default: inferred from data).
        """
        benchmark = self.wf_config.benchmark

        # 1. Load all data from warehouse
        cross_asset_symbol = (
            self.backtest_config.cross_asset_symbol
            if self.backtest_config.cross_asset_enabled
            else None
        )
        extras = [benchmark] + ([cross_asset_symbol] if cross_asset_symbol else [])
        logger.info(f"Loading data for {len(seed_symbols)} symbols + {extras}...")
        all_symbols = sorted(set(seed_symbols + extras))
        all_data = self.warehouse.get_daily_bars_bulk(all_symbols, start_date, end_date)

        if benchmark not in all_data:
            raise ValueError(f"Benchmark {benchmark} not found in warehouse")

        # Filter to symbols with enough data
        benchmark_df = all_data.pop(benchmark)
        cross_asset_df = (
            all_data.pop(cross_asset_symbol, None) if cross_asset_symbol else None
        )
        data_by_symbol = {
            sym: df for sym, df in all_data.items()
            if len(df) >= self.wf_config.min_data_bars
        }
        logger.info(
            f"Loaded {len(data_by_symbol)} symbols with >= {self.wf_config.min_data_bars} bars "
            f"(benchmark: {len(benchmark_df)} bars)"
        )

        # 2. Pre-compute features for all symbols (full period)
        logger.info("Computing features for all symbols...")
        prepared_frames: dict[str, pd.DataFrame] = {}
        for symbol, df in data_by_symbol.items():
            prepared = self.screener.prepare_features(df)
            if not prepared.empty:
                prepared_frames[symbol] = prepared

        logger.info(f"Features computed for {len(prepared_frames)} symbols")

        # 3. Generate rebalance dates
        all_dates = sorted(
            {ts for frame in prepared_frames.values() for ts in frame.index}
        )
        if trade_start_date:
            trade_start_ts = pd.Timestamp(trade_start_date)
        else:
            # Default: start trading after 1 year of warmup
            warmup_idx = min(self.wf_config.min_data_bars, len(all_dates) - 1)
            trade_start_ts = all_dates[warmup_idx]

        rebalance_dates = self._generate_rebalance_dates(
            trade_start_ts, all_dates[-1]
        )
        logger.info(
            f"Trading period: {trade_start_ts.strftime('%Y-%m-%d')} to "
            f"{all_dates[-1].strftime('%Y-%m-%d')} "
            f"({len(rebalance_dates)} rebalance points)"
        )

        # Build regime frame once (needs SPY at minimum)
        regime_frame = build_market_context({benchmark: benchmark_df})

        # 4. Build candidate schedule (point-in-time screening)
        logger.info("Screening at each rebalance point (point-in-time)...")
        candidate_schedule, rebalance_log = self._build_candidate_schedule(
            prepared_frames, benchmark_df, rebalance_dates, regime_frame
        )

        # Log universe stats
        avg_approved = rebalance_log["approved_count"].mean()
        logger.info(f"Average approved candidates per rebalance: {avg_approved:.1f}")

        # 5. Run portfolio backtest with the schedule
        logger.info("Running portfolio backtest with dynamic universe...")
        backtester = PortfolioMinerviniBacktester(
            config=self.backtest_config,
            screener=self.screener,
        )

        portfolio_result = backtester.backtest_portfolio(
            data_by_symbol=data_by_symbol,
            benchmark_df=benchmark_df,
            trade_start_date=trade_start_ts.strftime("%Y-%m-%d"),
            candidate_schedule=candidate_schedule,
            cross_asset_df=cross_asset_df,
        )

        # 6. Build universe snapshots dict
        universe_snapshots = {
            row["rebalance_date"]: row["approved_symbols"].split(",")
            for _, row in rebalance_log.iterrows()
            if row["approved_symbols"]
        }

        return WalkForwardResult(
            portfolio_result=portfolio_result,
            rebalance_log=rebalance_log,
            universe_snapshots=universe_snapshots,
        )

    def _generate_rebalance_dates(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> list[pd.Timestamp]:
        """Generate rebalance dates based on frequency config."""
        freq = self.wf_config.rebalance_frequency
        if freq == "daily":
            dates = pd.date_range(start, end, freq="B")
        elif freq == "weekly":
            dates = pd.date_range(start, end, freq="W-FRI")
        elif freq == "monthly":
            dates = pd.date_range(start, end, freq="BMS")
        else:
            raise ValueError(f"Unknown rebalance frequency: {freq}")
        return list(dates)

    def _build_candidate_schedule(
        self,
        prepared_frames: dict[str, pd.DataFrame],
        benchmark_df: pd.DataFrame,
        rebalance_dates: list[pd.Timestamp],
        regime_frame: pd.DataFrame,
    ) -> tuple[dict[pd.Timestamp, set[str]], pd.DataFrame]:
        """Pre-compute approved candidates at each rebalance date.

        At each rebalance, only the latest row (as of that date) is used for
        screening. This ensures no look-ahead bias.
        """
        schedule: dict[pd.Timestamp, set[str]] = {}
        log_rows: list[dict] = []

        for i, rebalance_date in enumerate(rebalance_dates):
            regime_label = self._regime_at(regime_frame, rebalance_date)
            approved = self._screen_at_date(
                prepared_frames, benchmark_df, rebalance_date, regime_label
            )
            schedule[rebalance_date] = set(approved)
            log_rows.append({
                "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
                "universe_size": sum(
                    1 for f in prepared_frames.values()
                    if rebalance_date in f.index or any(f.index <= rebalance_date)
                ),
                "approved_count": len(approved),
                "approved_symbols": ",".join(sorted(approved)),
            })

            if (i + 1) % 10 == 0:
                logger.info(
                    f"  Screened {i + 1}/{len(rebalance_dates)}: "
                    f"{len(approved)} approved on {rebalance_date.strftime('%Y-%m-%d')}"
                )

        return schedule, pd.DataFrame(log_rows)

    @staticmethod
    def _regime_at(regime_frame: pd.DataFrame, as_of_date: pd.Timestamp) -> str:
        if regime_frame is None or regime_frame.empty:
            return "unknown"
        valid = regime_frame.loc[regime_frame.index <= as_of_date]
        if valid.empty:
            return "unknown"
        return str(valid["market_regime"].iloc[-1])

    def _screen_at_date(
        self,
        prepared_frames: dict[str, pd.DataFrame],
        benchmark_df: pd.DataFrame,
        as_of_date: pd.Timestamp,
        regime_label: str = "unknown",
    ) -> list[str]:
        """Run Minervini screen using only data available up to as_of_date.

        Computes the 8 core template conditions from pre-computed features.
        No look-ahead bias: only uses data on or before as_of_date.
        """
        candidates = []
        for symbol, frame in prepared_frames.items():
            valid = frame.loc[frame.index <= as_of_date]
            if len(valid) < self.wf_config.min_data_bars:
                continue
            row = valid.iloc[-1]

            # Compute template score from the 8 core Minervini conditions
            close = float(row.get("close", 0) or 0)
            sma_50 = float(row.get("sma_50", 0) or 0)
            sma_150 = float(row.get("sma_150", 0) or 0)
            sma_200 = float(row.get("sma_200", 0) or 0)
            sma_200_20d_ago = float(row.get("sma_200_20d_ago", 0) or 0)
            high_52w = float(row.get("52w_high", 0) or 0)
            low_52w = float(row.get("52w_low", 0) or 0)

            if close <= 0 or sma_50 <= 0 or sma_150 <= 0 or sma_200 <= 0:
                continue

            # 8 core template conditions
            score = 0
            if close > sma_150:
                score += 1
            if close > sma_200:
                score += 1
            if sma_150 > sma_200:
                score += 1
            if sma_200 > sma_200_20d_ago:
                score += 1
            if sma_50 > sma_150 > sma_200:
                score += 1
            if close > sma_50:
                score += 1
            if low_52w > 0 and close >= low_52w * 1.30:
                score += 1
            if high_52w > 0 and close >= high_52w * 0.75:
                score += 1

            if score < self.wf_config.min_template_score:
                continue

            # Bonus: has setup (base pattern or near breakout)
            has_setup = bool(
                row.get("breakout_ready")
                or row.get("vcp_candidate")
                or row.get("base_candidate")
                or row.get("flat_base_candidate")
                or row.get("cup_handle_candidate")
            )
            if has_setup:
                score += 2  # bonus for actionable setup

            roc_60 = float(row.get("roc_60", 0) or 0)
            candidates.append({
                "symbol": symbol,
                "template_score": score,
                "roc_60": roc_60,
                "roc_120": float(row.get("roc_120", 0) or 0),
                "adx_14": float(row.get("adx_14", 0) or 0),
                "has_setup": has_setup,
            })

        if not candidates:
            return []

        # Compute RS percentile among all candidates passing template
        if len(candidates) > 1:
            roc_values = [c["roc_60"] for c in candidates]
            roc_series = pd.Series(roc_values)
            percentiles = roc_series.rank(method="average", pct=True).mul(100.0)
            for c, pctl in zip(candidates, percentiles):
                c["rs_percentile"] = pctl
        else:
            candidates[0]["rs_percentile"] = 80.0

        # Filter by RS percentile
        qualified = [
            c for c in candidates
            if c["rs_percentile"] >= self.wf_config.min_rs_percentile
        ]

        # Industry/sector group rank: compute mean RS per group across all
        # template-passing candidates (not just RS-qualified), so that a stock
        # in a weak group can't sneak in even if its own RS is high.
        if self.sector_map and (
            self.wf_config.min_group_rank_pct > 0
            or True  # always annotate for diagnostics
        ):
            group_key = self.wf_config.group_by
            per_group_rs: Dict[str, List[float]] = {}
            for c in candidates:
                meta = self.sector_map.get(c["symbol"], {})
                group = meta.get(group_key) or "Unknown"
                per_group_rs.setdefault(group, []).append(c["rs_percentile"])

            group_means = {
                g: (sum(vals) / len(vals))
                for g, vals in per_group_rs.items()
                if len(vals) >= self.wf_config.group_min_members
            }
            if group_means:
                sorted_groups = sorted(group_means.items(), key=lambda x: x[1])
                group_rank = {
                    g: (i + 1) / len(sorted_groups) * 100.0
                    for i, (g, _) in enumerate(sorted_groups)
                }
            else:
                group_rank = {}

            for c in qualified:
                meta = self.sector_map.get(c["symbol"], {})
                group = meta.get(group_key) or "Unknown"
                c["group"] = group
                c["group_rank_pct"] = group_rank.get(group, 100.0)
                # 100.0 default: stocks in tiny groups pass by default

            apply_group_filter = (
                self.wf_config.min_group_rank_pct > 0
                and (
                    not self.wf_config.group_filter_regimes
                    or regime_label in self.wf_config.group_filter_regimes
                )
            )
            if apply_group_filter:
                qualified = [
                    c for c in qualified
                    if c["group_rank_pct"] >= self.wf_config.min_group_rank_pct
                ]

        # Rank: prefer setups, then template score, then momentum
        qualified.sort(
            key=lambda c: (c["has_setup"], c["template_score"], c["roc_60"]),
            reverse=True,
        )

        top_n = qualified[: self.wf_config.max_screen_candidates]
        return [c["symbol"] for c in top_n]
