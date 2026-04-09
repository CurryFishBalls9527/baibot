"""Track portfolio performance across time."""

import json
import logging
from datetime import date
from pathlib import Path
from typing import List, Dict, Optional

from tradingagents.broker.base_broker import BaseBroker
from tradingagents.storage.database import TradingDatabase

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """Captures daily snapshots and computes performance metrics."""

    def __init__(self, broker: BaseBroker, db: TradingDatabase):
        self.broker = broker
        self.db = db

    def take_daily_snapshot(self) -> Dict:
        account = self.broker.get_account()
        positions = self.broker.get_positions()

        pos_list = [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_entry_price": p.avg_entry_price,
                "current_price": p.current_price,
                "market_value": p.market_value,
                "unrealized_pl": p.unrealized_pl,
            }
            for p in positions
        ]

        self.db.take_snapshot(
            equity=account.equity,
            cash=account.cash,
            buying_power=account.buying_power,
            portfolio_value=account.portfolio_value,
            positions=pos_list,
            daily_pl=account.daily_pl,
            daily_pl_pct=account.daily_pl_pct,
        )

        logger.info(
            f"Snapshot: equity=${account.equity:,.2f}  "
            f"daily_pl=${account.daily_pl:,.2f} ({account.daily_pl_pct:.2%})  "
            f"positions={len(positions)}"
        )
        return {
            "equity": account.equity,
            "cash": account.cash,
            "daily_pl": account.daily_pl,
            "positions": len(positions),
        }

    def get_total_position_value(self) -> float:
        positions = self.broker.get_positions()
        return sum(p.market_value for p in positions)

    def build_daily_report(self, report_date: Optional[str] = None) -> Dict:
        report_day = report_date or date.today().isoformat()
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        trades = self.db.get_trades_on_date(report_day)
        trade_summary = self.db.get_trade_summary(report_day)
        setups = self.db.get_setup_candidates_on_date(report_day)
        screening_batch = self.db.get_latest_screening_batch()

        position_rows = [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "side": p.side,
                "entry": p.avg_entry_price,
                "current": p.current_price,
                "market_value": p.market_value,
                "unrealized_pl": p.unrealized_pl,
                "unrealized_pl_pct": p.unrealized_plpc,
            }
            for p in positions
        ]
        total_unrealized_pl = sum(p.unrealized_pl for p in positions)

        return {
            "date": report_day,
            "account": {
                "equity": account.equity,
                "cash": account.cash,
                "buying_power": account.buying_power,
                "portfolio_value": account.portfolio_value,
                "daily_pl": account.daily_pl,
                "daily_pl_pct": account.daily_pl_pct,
            },
            "trade_summary": trade_summary,
            "trades": trades,
            "setups": setups,
            "screening_batch": screening_batch,
            "positions": position_rows,
            "position_summary": {
                "open_positions": len(position_rows),
                "total_unrealized_pl": total_unrealized_pl,
            },
            "performance": self.get_performance_summary(),
        }

    def save_daily_report(self, report: Dict, output_dir: str) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"{report['date']}.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        return report_path

    def get_performance_summary(self) -> Dict:
        snapshots = self.db.get_snapshots(days=365)
        if not snapshots:
            return {"message": "No snapshots yet"}

        latest = snapshots[0]
        starting = self.db.get_starting_equity()
        equities = [s["equity"] for s in reversed(snapshots)]

        total_return = 0.0
        if starting and starting > 0:
            total_return = (latest["equity"] - starting) / starting

        max_equity = 0.0
        max_drawdown = 0.0
        for eq in equities:
            if eq > max_equity:
                max_equity = eq
            dd = (max_equity - eq) / max_equity if max_equity > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

        daily_returns = []
        for i in range(1, len(equities)):
            if equities[i - 1] > 0:
                daily_returns.append((equities[i] - equities[i - 1]) / equities[i - 1])

        win_days = sum(1 for r in daily_returns if r > 0)
        total_days = len(daily_returns)

        avg_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0
        std_return = (
            (sum((r - avg_return) ** 2 for r in daily_returns) / len(daily_returns)) ** 0.5
            if daily_returns else 0
        )
        sharpe = (avg_return / std_return * (252 ** 0.5)) if std_return > 0 else 0

        return {
            "current_equity": latest["equity"],
            "starting_equity": starting,
            "total_return": total_return,
            "total_return_pct": f"{total_return:.2%}",
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": f"{max_drawdown:.2%}",
            "sharpe_ratio": round(sharpe, 2),
            "win_day_rate": f"{win_days}/{total_days}" if total_days > 0 else "N/A",
            "total_snapshots": len(snapshots),
        }
