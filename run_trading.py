#!/usr/bin/env python3
"""Entry point for the TradingAgents automated trading platform.

Usage:
    # Run analysis immediately (manual trigger)
    python run_trading.py run

    # Start the scheduler (runs daily on autopilot)
    python run_trading.py schedule

    # Check current status (account, positions, performance)
    python run_trading.py status

    # Emergency: close all positions
    python run_trading.py close-all

    # Run with custom watchlist
    python run_trading.py run --symbols AAPL,NVDA,TSLA
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import dotenv_values


def _load_dotenv_skip_blanks() -> None:
    # A `KEY=` line with no value causes dotenv to set the env var to empty
    # string, which looks "set" to downstream consumers and silently produces
    # 401s (e.g. OpenAI rejects blank Authorization). Treat blank as absent:
    # read .env values, and only setdefault non-empty ones.
    for key, value in dotenv_values().items():
        if value:
            os.environ.setdefault(key, value)


_load_dotenv_skip_blanks()

from tradingagents.automation.config import build_config
from tradingagents.automation.launchd import (
    WATCHDOG_LABEL,
    install_launch_agent,
    install_watchdog_agent,
    uninstall_launch_agent,
    uninstall_watchdog_agent,
)
from tradingagents.automation.notifier import build_notifier
from tradingagents.automation.orchestrator import Orchestrator
from tradingagents.automation.scheduler import TradingScheduler
from tradingagents.automation.social_monitor import SocialFeedMonitor


def _resolve_variant_config(args) -> dict:
    """Build config for a specific variant from experiment YAML."""
    variant_name = getattr(args, "variant", None)
    if not variant_name:
        return build_config()

    experiment_path = getattr(args, "experiment", "experiments/paper_launch_v2.yaml")
    from tradingagents.testing.ab_config import load_experiment, build_variant_config
    experiment = load_experiment(experiment_path)

    for v in experiment.variants:
        if v.name == variant_name:
            base = build_config()
            return build_variant_config(base, v)

    available = [v.name for v in experiment.variants]
    print(f"Error: variant '{variant_name}' not found. Available: {', '.join(available)}")
    sys.exit(1)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("trading.log", mode="a"),
        ],
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("alpaca").setLevel(logging.WARNING)


def cmd_run(args):
    """Run analysis and trading immediately."""
    config = build_config()
    if args.symbols:
        config["watchlist"] = [s.strip().upper() for s in args.symbols.split(",")]
    if args.model:
        config["deep_think_llm"] = args.model
        config["quick_think_llm"] = args.model
    if args.disable_minervini:
        config["minervini_enabled"] = False
    if getattr(args, "experiment", None):
        config["experiment_config_path"] = args.experiment

    if config.get("experiment_config_path"):
        from tradingagents.testing.ab_config import load_experiment
        from tradingagents.testing.ab_runner import ABRunner

        experiment = load_experiment(config["experiment_config_path"])
        runner = ABRunner(experiment, config)
        print(f"\nExperiment: {experiment.experiment_id}")
        print(f"Variants: {', '.join(v.name for v in experiment.variants)}")
        print("-" * 50)
        all_results = runner.run_daily_analysis()
        print("\n" + "=" * 50)
        print("RESULTS")
        print("=" * 50)
        for variant_name, results in all_results.items():
            print(f"\n[{variant_name}]")
            if not isinstance(results, dict):
                print(f"  {results}")
                continue
            if "error" in results:
                print(f"  ERROR: {results['error']}")
                continue
            if results.get("status") == "market_closed":
                print("  Market closed — no analysis run.")
                continue
            for symbol, result in results.items():
                if not isinstance(result, dict):
                    print(f"  {symbol}: {result}")
                    continue
                if "error" in result:
                    print(f"  {symbol}: ERROR - {result['error']}")
                elif result.get("traded"):
                    print(f"  {symbol}: {result['side'].upper()} {result['qty']} shares ({result['status']})")
                elif result.get("screen_rejected"):
                    print(f"  {symbol}: SKIP - {result['screen_rejected']}")
                else:
                    print(f"  {symbol}: {result.get('action', '?')}")
        print()
        return

    orch = Orchestrator(config)
    print(f"\nRunning analysis for: {config['watchlist']}")
    print(f"Paper mode: {config['paper_trading']}")
    print(f"LLM: {config['deep_think_llm']}")
    print(f"Mechanical-only: {config.get('mechanical_only_mode', False)}")
    print("-" * 50)

    results = orch.run_daily_analysis()

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for symbol, result in results.items():
        if "error" in result:
            print(f"  {symbol}: ERROR - {result['error']}")
        elif result.get("traded"):
            print(f"  {symbol}: {result['side'].upper()} {result['qty']} shares ({result['status']})")
        elif result.get("screen_rejected"):
            print(f"  {symbol}: SKIP - {result['screen_rejected']}")
        elif result.get("risk_rejected"):
            print(f"  {symbol}: {result['action']} - BLOCKED by risk: {result['risk_rejected']}")
        else:
            print(f"  {symbol}: {result['action']} - no trade needed")
    print()


def cmd_schedule(args):
    """Start the automated scheduler."""
    config = build_config()
    if args.symbols:
        config["watchlist"] = [s.strip().upper() for s in args.symbols.split(",")]
    if args.mode:
        config["trading_mode"] = args.mode
    if args.disable_minervini:
        config["minervini_enabled"] = False
    if getattr(args, "experiment", None):
        config["experiment_config_path"] = args.experiment

    print(f"Starting TradingAgents Scheduler")
    print(f"  Mode: {config['trading_mode']}")
    print(f"  Paper: {config['paper_trading']}")
    if config.get("broad_market_enabled"):
        print("  Watchlist: dynamic broad-market universe")
    else:
        print(f"  Watchlist: {config['watchlist']}")
    print(f"  Swing time: {config['swing_analysis_time']} ET")
    print(f"  Minervini gate: {config['minervini_enabled']}")
    print(f"  Overlay: {config.get('overlay_enabled', False)}")
    print(f"  Press Ctrl+C to stop")
    print()

    scheduler = TradingScheduler(config)
    scheduler.start()


def cmd_status(args):
    """Show current status."""
    config = _resolve_variant_config(args)
    variant_name = getattr(args, "variant", None)
    strategy_type = config.get("strategy_type", "minervini")
    if strategy_type == "chan":
        from tradingagents.automation.chan_orchestrator import ChanOrchestrator
        orch = ChanOrchestrator(config)
    elif strategy_type == "chan_daily":
        from tradingagents.automation.chan_daily_orchestrator import ChanDailyOrchestrator
        orch = ChanDailyOrchestrator(config)
    else:
        orch = Orchestrator(config)
    status = orch.get_status()

    print("\n" + "=" * 50)
    print("TRADING SYSTEM STATUS")
    print("=" * 50)

    acct = status["account"]
    print(f"\n  Account:")
    print(f"    Equity:       ${acct['equity']:>12,.2f}")
    print(f"    Cash:         ${acct['cash']:>12,.2f}")
    print(f"    Buying Power: ${acct['buying_power']:>12,.2f}")
    print(f"    Daily P&L:    ${acct['daily_pl']:>12,.2f}  ({acct['daily_pl_pct']})")

    print(f"\n  Market: {'OPEN' if status['market']['is_open'] else 'CLOSED'}")
    print(f"    Next open:  {status['market']['next_open']}")
    print(f"    Next close: {status['market']['next_close']}")

    print(f"\n  Positions ({len(status['positions'])}):")
    if status["positions"]:
        for p in status["positions"]:
            print(f"    {p['symbol']:>6}: {p['qty']:>6.0f} shares @ ${p['entry']:.2f} "
                  f"-> ${p['current']:.2f}  P&L: ${p['pl']:>8,.2f} ({p['pl_pct']})")
    else:
        print("    (none)")

    perf = status["performance"]
    if perf.get("current_equity"):
        print(f"\n  Performance:")
        print(f"    Total return:  {perf.get('total_return_pct', 'N/A')}")
        print(f"    Max drawdown:  {perf.get('max_drawdown_pct', 'N/A')}")
        print(f"    Sharpe ratio:  {perf.get('sharpe_ratio', 'N/A')}")

    today = status["today"]
    trade_summary = today["trade_summary"]
    print(f"\n  Today:")
    print(f"    Orders:        {trade_summary['total_orders']}")
    print(f"    Filled:        {trade_summary['filled_orders']}")
    print(f"    Symbols:       {', '.join(trade_summary['symbols']) if trade_summary['symbols'] else '(none)'}")
    print(f"    Open P&L:      ${today['unrealized_pl']:>12,.2f}")

    screening = status.get("screening", {})
    if screening.get("screen_date"):
        print(f"\n  Screening:")
        print(f"    Screen date:   {screening['screen_date']}")
        print(f"    Regime:        {screening['market_regime']}")
        print(f"    Uptrend:       {screening['confirmed_uptrend']}")
        print(f"    Approved:      {', '.join(screening['approved_symbols']) if screening['approved_symbols'] else '(none)'}")
        print(f"    Setups:        {screening['setup_count']}")

    overlay = status.get("overlay", {})
    if overlay.get("enabled"):
        print(f"\n  Overlay:")
        print(f"    Symbol:        {overlay['symbol']}")
        print(f"    Trigger:       {overlay['trigger']}")
        print(f"    Regime:        {overlay['market_regime']}")
        print(f"    Score:         {overlay['market_score'] if overlay['market_score'] is not None else 'n/a'}")
        print(f"    Allowed:       {overlay['overlay_allowed']}")
        print(f"    Position Qty:  {overlay['position_qty']:.0f}")
        print(f"    Position Val:  ${overlay['position_value']:>12,.2f}")

    notifications = status.get("notifications", {})
    if notifications.get("enabled"):
        print(f"\n  Notifications:")
        print(f"    Provider:      {notifications['provider']}")
        print(f"    Topic:         {notifications['topic']}")
        print(f"    Server:        {notifications['server']}")

    if config.get("social_monitor_enabled"):
        print(f"\n  Social Monitor:")
        print(f"    Users:         {', '.join(config.get('social_monitor_usernames', [])) or '(none)'}")
        print(f"    Topic:         {config.get('social_ntfy_topic') or '(unset)'}")
        print(f"    Feed:          {config.get('social_feed_url_template')}")

    print(f"\n  Watchlist: {status['watchlist']}")
    if config.get("broad_market_enabled"):
        print("  Universe mode: dynamic broad-market coarse scan")
    print(f"  Paper mode: {status['paper_mode']}")
    print()


def cmd_close_all(args):
    """Emergency close all positions."""
    config = build_config()
    orch = Orchestrator(config)

    if not args.confirm:
        print("WARNING: This will close ALL open positions.")
        print("Run with --confirm to proceed.")
        return

    print("Closing all positions...")
    results = orch.emergency_close_all()
    print(f"Closed {len(results)} positions.")


def cmd_trades(args):
    """Show recent trades."""
    config = _resolve_variant_config(args)
    from tradingagents.storage.database import TradingDatabase
    db = TradingDatabase(config.get("db_path", "trading.db"))
    trades = db.get_recent_trades(limit=args.limit)

    print(f"\nRecent Trades (last {args.limit}):")
    print("-" * 80)
    for t in trades:
        print(f"  {t['timestamp']}  {t['side']:>4} {t['qty'] or 0:>6.0f} "
              f"{t['symbol']:>6} @ ${t['filled_price'] or 0:>8.2f}  [{t['status']}]")
    if not trades:
        print("  (no trades yet)")
    print()


def cmd_report(args):
    """Show and optionally save the current daily report."""
    config = _resolve_variant_config(args)
    strategy_type = config.get("strategy_type", "minervini")
    if strategy_type == "chan":
        from tradingagents.automation.chan_orchestrator import ChanOrchestrator
        orch = ChanOrchestrator(config)
    elif strategy_type == "chan_daily":
        from tradingagents.automation.chan_daily_orchestrator import ChanDailyOrchestrator
        orch = ChanDailyOrchestrator(config)
    else:
        orch = Orchestrator(config)
    report = orch.generate_daily_report(save=not args.no_save)

    acct = report["account"]
    trade_summary = report["trade_summary"]
    perf = report["performance"]
    position_summary = report["position_summary"]
    screening_batch = report.get("screening_batch")

    print("\n" + "=" * 50)
    print("DAILY REPORT")
    print("=" * 50)
    print(f"\n  Date: {report['date']}")
    print(f"  Paper mode: {report['paper_mode']}")
    print(f"  Watchlist: {report['watchlist']}")

    print(f"\n  Account:")
    print(f"    Equity:       ${acct['equity']:>12,.2f}")
    print(f"    Cash:         ${acct['cash']:>12,.2f}")
    print(f"    Buying Power: ${acct['buying_power']:>12,.2f}")
    print(f"    Daily P&L:    ${acct['daily_pl']:>12,.2f}  ({acct['daily_pl_pct']:.2%})")

    print(f"\n  Trades Today:")
    print(f"    Orders:        {trade_summary['total_orders']}")
    print(f"    Filled:        {trade_summary['filled_orders']}")
    print(f"    Buy orders:    {trade_summary['buy_orders']}")
    print(f"    Sell orders:   {trade_summary['sell_orders']}")
    print(f"    Gross notional:${trade_summary['gross_filled_notional']:>12,.2f}")
    print(f"    Symbols:       {', '.join(trade_summary['symbols']) if trade_summary['symbols'] else '(none)'}")

    print(f"\n  Open Positions:")
    print(f"    Count:         {position_summary['open_positions']}")
    print(f"    Unrealized P&L:${position_summary['total_unrealized_pl']:>12,.2f}")
    for pos in report["positions"]:
        print(
            f"    {pos['symbol']:>6}: {pos['qty']:>6.0f} shares @ ${pos['entry']:.2f} "
            f"-> ${pos['current']:.2f}  P&L: ${pos['unrealized_pl']:>8,.2f} "
            f"({pos['unrealized_pl_pct']:.2%})"
        )
    if not report["positions"]:
        print("    (none)")

    if perf.get("current_equity"):
        print(f"\n  Performance:")
        print(f"    Total return:  {perf.get('total_return_pct', 'N/A')}")
        print(f"    Max drawdown:  {perf.get('max_drawdown_pct', 'N/A')}")
        print(f"    Sharpe ratio:  {perf.get('sharpe_ratio', 'N/A')}")

    if screening_batch:
        print(f"\n  Screening Batch:")
        print(f"    Screen date:   {screening_batch['screen_date']}")
        print(f"    Regime:        {screening_batch['market_regime']}")
        print(f"    Uptrend:       {bool(screening_batch['market_confirmed_uptrend'])}")
        print(f"    Approved:      {', '.join(screening_batch['approved_symbols']) if screening_batch['approved_symbols'] else '(none)'}")
        print(f"    Rows:          {screening_batch['row_count']}")

    print(f"\n  Setups ({len(report['setups'])} rows):")
    for setup in report["setups"][: args.limit]:
        approved = "YES" if setup["selected_for_analysis"] else "NO"
        print(
            f"    {setup['symbol']:>6}  approved={approved:<3} "
            f"regime={setup['market_regime'] or 'n/a':<22} "
            f"rs={setup['rs_percentile'] or 0:>5.1f}  "
            f"base={setup['base_label'] or 'n/a':<12} "
            f"pivot={setup['pivot_price'] or 0:>8.2f}"
        )
    if not report["setups"]:
        print("    (no setups saved yet)")

    print(f"\n  Trade Log ({len(report['trades'])} entries):")
    for trade in report["trades"][: args.limit]:
        qty = trade["filled_qty"] or trade["qty"] or 0
        print(
            f"    {trade['timestamp']}  {trade['side']:>4} {qty:>6.0f} "
            f"{trade['symbol']:>6} @ ${trade['filled_price'] or 0:>8.2f}  [{trade['status']}]"
        )
    if not report["trades"]:
        print("    (no trades today)")

    if report.get("report_path"):
        print(f"\n  Saved: {report['report_path']}")
    print()


def cmd_setups(args):
    """Show the latest Minervini setup candidates saved by automation."""
    config = build_config()
    from tradingagents.storage.database import TradingDatabase

    db = TradingDatabase(config["db_path"])
    latest_batch = db.get_latest_screening_batch()
    if args.date:
        setups = db.get_setup_candidates_on_date(args.date)
    elif latest_batch:
        setups = db.get_setup_candidates_on_date(latest_batch["screen_date"])
    else:
        setups = db.get_latest_setup_candidates()

    print("\n" + "=" * 70)
    print("MINERVINI SETUPS")
    print("=" * 70)
    if latest_batch:
        print(f"\n  Screen date: {latest_batch['screen_date']}")
        print(f"  Market regime: {latest_batch['market_regime']}")
        print(f"  Confirmed uptrend: {bool(latest_batch['market_confirmed_uptrend'])}")
        print(f"  Approved symbols: {', '.join(latest_batch['approved_symbols']) if latest_batch['approved_symbols'] else '(none)'}")
        print(f"  Setup rows: {latest_batch['row_count']}")

    if not setups:
        print("\n  (no setup rows saved for this batch)\n")
        return

    print(f"\n  Rows ({min(len(setups), args.limit)} shown):")
    for setup in setups[: args.limit]:
        approved = "YES" if setup["selected_for_analysis"] else "NO"
        print(
            f"    {setup['symbol']:>6}  approved={approved:<3} "
            f"passed={bool(setup['passed_template'])!s:<5} "
            f"rs={setup['rs_percentile'] or 0:>5.1f}  "
            f"rev={setup['revenue_growth'] or 0:>6.2f}  "
            f"eps={setup['eps_growth'] or 0:>6.2f}  "
            f"base={setup['base_label'] or 'n/a':<12} "
            f"pivot={setup['pivot_price'] or 0:>8.2f}"
        )
    print()


def cmd_chart(args):
    """Generate an interactive Plotly chart for a symbol."""
    import webbrowser
    import tempfile
    from datetime import datetime, timedelta

    import duckdb
    import pandas as pd

    from tradingagents.dashboard.chart_builder import TradeChartBuilder

    symbol = args.symbol.upper()
    overlays = [o.strip() for o in args.overlays.split(",") if o.strip()]

    is_chan = getattr(args, "variant", None) == "chan"
    if is_chan:
        db_path = "research_data/intraday_30m_broad.duckdb"
        table, ts_col = "bars_30m", "ts"
    else:
        db_path = "research_data/market_data.duckdb"
        table, ts_col = "daily_bars", "trade_date"

    if not Path(db_path).exists():
        print(f"Error: OHLCV database not found: {db_path}")
        return

    conn_peek = duckdb.connect(db_path, read_only=True)
    max_date_row = conn_peek.execute(
        f"SELECT MAX({ts_col}) FROM {table} WHERE symbol = ?", [symbol]
    ).fetchone()
    conn_peek.close()
    if not max_date_row or not max_date_row[0]:
        print(f"No data found for {symbol} in {db_path}")
        return
    end_date = pd.Timestamp(max_date_row[0]).to_pydatetime()
    begin_date = end_date - timedelta(days=args.days)

    variant = getattr(args, "variant", None)

    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute(
        f"SELECT {ts_col} as ts, open, high, low, close, volume "
        f"FROM {table} WHERE symbol = ? AND {ts_col} >= ? AND {ts_col} <= ? "
        f"ORDER BY {ts_col}",
        [symbol, begin_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")],
    ).fetchdf()
    conn.close()

    if df.empty:
        print(f"No data found for {symbol} in {db_path}")
        return

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()

    builder = TradeChartBuilder(symbol, df).add_candlesticks()

    if "volume" in overlays:
        builder.add_volume()
    if "sma" in overlays:
        builder.add_sma([50, 150, 200])
    if "bollinger" in overlays:
        builder.add_bollinger()
    if "macd" in overlays:
        builder.add_macd()
    if "rsi" in overlays:
        builder.add_rsi()
    if "atr" in overlays:
        builder.add_atr()

    if "chan" in overlays and is_chan:
        try:
            from tradingagents.dashboard.chan_structures import extract_chan_structures
            structures = extract_chan_structures(
                symbol, begin_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"), db_path,
            )
            builder.add_chan_bi(structures["bi_list"])
            builder.add_chan_zs(structures["zs_list"])
            builder.add_chan_bsp(structures["bsp_list"])
        except Exception as e:
            print(f"Warning: Chan structure extraction failed: {e}")

    if variant:
        config = _resolve_variant_config(args)
        trade_db_path = config.get("db_path", "trading.db")
        if Path(trade_db_path).exists():
            from tradingagents.storage.database import TradingDatabase
            db = TradingDatabase(trade_db_path)
            trades = db.get_recent_trades(limit=200)
            symbol_trades = [t for t in trades if t.get("symbol") == symbol]
            if symbol_trades:
                builder.add_trade_markers(symbol_trades)
                print(f"  Overlaid {len(symbol_trades)} trades for {symbol}")

    fig = builder.build()

    out_dir = Path("results/charts")
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{symbol}_{args.days}d.html"
    fig.write_html(str(html_path))
    print(f"  Chart saved: {html_path}")
    webbrowser.open(f"file://{html_path.resolve()}")


def cmd_install_service(args):
    """Install the scheduler as a macOS launch agent."""
    repo_root = Path(__file__).resolve().parent
    project_python = repo_root / ".venv" / "bin" / "python"
    python_bin = project_python if project_python.exists() else Path(sys.executable)
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None

    plist_path = install_launch_agent(
        repo_root=str(repo_root),
        python_bin=str(python_bin),
        mode=args.mode,
        symbols=symbols,
        label=args.label,
        experiment=args.experiment,
    )

    print("Launch agent installed.")
    print(f"  Label: {args.label}")
    print(f"  Plist: {plist_path}")
    print(f"  Python: {python_bin}")
    print(f"  Mode: {args.mode}")
    print(f"  Symbols: {symbols or 'config watchlist'}")
    print(f"  Experiment: {args.experiment or '(none — single orchestrator)'}")
    print(f"  Logs: {repo_root / 'results' / 'service_logs'}")
    print()


def cmd_remove_service(args):
    """Remove the macOS launch agent."""
    plist_path = uninstall_launch_agent(label=args.label)
    print("Launch agent removed.")
    print(f"  Label: {args.label}")
    print(f"  Plist: {plist_path}")
    print()


def cmd_install_watchdog(args):
    """Install the watchdog as a separate macOS launch agent."""
    repo_root = Path(__file__).resolve().parent
    project_python = repo_root / ".venv" / "bin" / "python"
    python_bin = project_python if project_python.exists() else Path(sys.executable)

    plist_path = install_watchdog_agent(
        repo_root=str(repo_root),
        python_bin=str(python_bin),
        label=args.label,
    )

    print("Watchdog launch agent installed.")
    print(f"  Label: {args.label}")
    print(f"  Plist: {plist_path}")
    print(f"  Python: {python_bin}")
    print(f"  Logs: {repo_root / 'results' / 'service_logs'}/watchdog.{{out,err}}.log")
    print()
    print("Required env vars (set in .env):")
    print("  WATCHDOG_NTFY_TOPIC=<distinct from NTFY_TOPIC>")
    print("  WATCHDOG_TELEGRAM_CHAT_ID=<distinct from TELEGRAM_CHAT_ID>")
    print("  -- OR --")
    print("  WATCHDOG_REUSE_TRADE_CHANNEL=1   # reuse trade ntfy/telegram for now")
    print()


def cmd_remove_watchdog(args):
    """Remove the watchdog launch agent."""
    plist_path = uninstall_watchdog_agent(label=args.label)
    print("Watchdog launch agent removed.")
    print(f"  Label: {args.label}")
    print(f"  Plist: {plist_path}")
    print()


def cmd_dashboard(args):
    """Launch the local Streamlit dashboard."""
    repo_root = Path(__file__).resolve().parent
    project_python = repo_root / ".venv" / "bin" / "python"
    python_bin = project_python if project_python.exists() else Path(sys.executable)
    streamlit_app = repo_root / "tradingagents" / "dashboard" / "app.py"
    port = str(args.port)

    command = [
        str(python_bin),
        "-m",
        "streamlit",
        "run",
        str(streamlit_app),
        "--server.port",
        port,
        "--server.headless",
        "true",
    ]
    if args.address:
        command.extend(["--server.address", args.address])

    print("Starting dashboard...")
    print(f"  URL: http://{args.address}:{port}")
    print(f"  App: {streamlit_app}")
    print()
    subprocess.run(command, check=True, cwd=repo_root)


def cmd_notify_test(args):
    """Send a test ntfy notification."""
    config = build_config()
    notifier = build_notifier(config)
    if not notifier.enabled:
        print("No notifier enabled. Set NTFY_ENABLED=1 / TELEGRAM_ENABLED=1 first.")
        return

    ok = notifier.send(
        args.title,
        args.message,
        priority=args.priority,
        tags=["test_tube", "iphone"],
    )
    if ok:
        print("Test notification sent.")
        print(f"  Topic: {notifier.topic}")
        print(f"  Server: {notifier.server}")
    else:
        print("Test notification failed. Check your topic/server settings and logs.")


def cmd_experiment(args):
    """Manage A/B experiments."""
    from tradingagents.testing.ab_config import load_experiment, save_experiment
    from tradingagents.testing.ab_reporter import ABReporter
    from tradingagents.storage.database import TradingDatabase

    config = build_config()

    if args.exp_command == "create":
        exp = load_experiment(args.config)
        db = TradingDatabase(config["db_path"])
        import yaml
        with open(args.config) as f:
            config_yaml = f.read()
        db.save_experiment(exp.experiment_id, config_yaml, exp.start_date,
                           exp.primary_metric, exp.min_trades, exp.min_days)
        print(f"Experiment '{exp.experiment_id}' created with {len(exp.variants)} variants.")

    elif args.exp_command == "status":
        db = TradingDatabase(config["db_path"])
        exp_row = db.get_experiment(args.experiment_id)
        if not exp_row:
            print(f"Experiment '{args.experiment_id}' not found.")
            return
        print(f"\nExperiment: {exp_row['experiment_id']}")
        print(f"  Status:    {exp_row['status']}")
        print(f"  Started:   {exp_row['start_date']}")
        print(f"  Metric:    {exp_row['primary_metric']}")
        print(f"  Min trades:{exp_row['min_trades']}")
        print(f"  Min days:  {exp_row['min_days']}")

    elif args.exp_command == "report":
        db = TradingDatabase(config["db_path"])
        exp_row = db.get_experiment(args.experiment_id)
        if not exp_row:
            print(f"Experiment '{args.experiment_id}' not found.")
            return
        import yaml
        exp = load_experiment_from_yaml_str(exp_row["config_yaml"])
        reporter = ABReporter(exp)
        print(f"\n{reporter.summary_table()}")
        ready, reason = reporter.is_promotion_ready()
        print(f"\nPromotion ready: {ready}")
        print(f"  {reason}\n")

    elif args.exp_command == "promote":
        db = TradingDatabase(config["db_path"])
        exp_row = db.get_experiment(args.experiment_id)
        if not exp_row:
            print(f"Experiment '{args.experiment_id}' not found.")
            return
        db.update_experiment_status(args.experiment_id, "promoted")
        print(f"Experiment '{args.experiment_id}' status set to 'promoted'.")

    else:
        print("Unknown experiment command. Use: create, status, report, promote")


def load_experiment_from_yaml_str(yaml_str: str):
    """Load experiment from a YAML string (stored in DB)."""
    import yaml
    from tradingagents.testing.ab_models import Experiment, ExperimentVariant
    data = yaml.safe_load(yaml_str)
    variants = []
    for v in data.get("variants", []):
        variants.append(ExperimentVariant(
            name=v["name"], description=v.get("description", ""),
            config_overrides=v.get("config_overrides", {}),
            alpaca_api_key=v.get("alpaca_api_key", ""),
            alpaca_secret_key=v.get("alpaca_secret_key", ""),
            db_path=v.get("db_path", ""),
        ))
    return Experiment(
        experiment_id=data["experiment_id"], start_date=data.get("start_date", ""),
        variants=variants, min_trades=data.get("min_trades", 30),
        min_days=data.get("min_days", 20),
        primary_metric=data.get("primary_metric", "sharpe_ratio"),
        status=data.get("status", "running"),
    )


def cmd_social_check(args):
    """Poll configured X RSS feeds once."""
    config = build_config()
    monitor = SocialFeedMonitor(config)
    result = monitor.check_once()
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_social_test(args):
    """Send a test social notification."""
    config = build_config()
    monitor = SocialFeedMonitor(config)
    if not monitor.enabled:
        print("social monitor is not enabled. Set SOCIAL_MONITOR_ENABLED=1 and SOCIAL_NTFY_TOPIC first.")
        return
    ok = monitor.send_test_notification()
    if ok:
        print("Social test notification sent.")
        print(f"  Topic: {config.get('social_ntfy_topic')}")
        print(f"  Users: {', '.join(config.get('social_monitor_usernames', []))}")
    else:
        print("Social test notification failed.")


def main():
    parser = argparse.ArgumentParser(
        description="TradingAgents Automated Trading Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # run
    p_run = sub.add_parser("run", help="Run analysis and trading now")
    p_run.add_argument("--symbols", type=str, help="Comma-separated symbols (e.g. AAPL,NVDA)")
    p_run.add_argument("--model", type=str, help="LLM model to use")
    p_run.add_argument("--disable-minervini", action="store_true", help="Disable Minervini entry gating")
    p_run.add_argument("--experiment", type=str, help="Path to A/B experiment YAML (enables ABRunner)")

    # schedule
    p_sched = sub.add_parser("schedule", help="Start automated scheduler")
    p_sched.add_argument("--symbols", type=str, help="Comma-separated symbols")
    p_sched.add_argument("--mode", choices=["swing", "day", "both"], help="Trading mode")
    p_sched.add_argument("--disable-minervini", action="store_true", help="Disable Minervini entry gating")
    p_sched.add_argument("--experiment", type=str, help="Path to A/B experiment YAML (enables ABRunner)")

    # status
    p_status = sub.add_parser("status", help="Show current status")
    p_status.add_argument("--variant", type=str, help="Variant name (e.g. chan, mechanical, llm)")
    p_status.add_argument("--experiment", type=str, default="experiments/paper_launch_v2.yaml",
                          help="Path to experiment YAML")

    # close-all
    p_close = sub.add_parser("close-all", help="Emergency close all positions")
    p_close.add_argument("--confirm", action="store_true", help="Confirm close all")

    # trades
    p_trades = sub.add_parser("trades", help="Show recent trades")
    p_trades.add_argument("--limit", type=int, default=20, help="Number of trades")
    p_trades.add_argument("--variant", type=str, help="Variant name (e.g. chan, mechanical, llm)")
    p_trades.add_argument("--experiment", type=str, default="experiments/paper_launch_v2.yaml",
                          help="Path to experiment YAML")

    # report
    p_report = sub.add_parser("report", help="Show today's trade and P&L report")
    p_report.add_argument("--limit", type=int, default=20, help="Number of trade rows to show")
    p_report.add_argument("--no-save", action="store_true", help="Do not save the JSON report")
    p_report.add_argument("--variant", type=str, help="Variant name (e.g. chan, mechanical, llm)")
    p_report.add_argument("--experiment", type=str, default="experiments/paper_launch_v2.yaml",
                          help="Path to experiment YAML")

    # setups
    p_setups = sub.add_parser("setups", help="Show the latest Minervini setup candidates")
    p_setups.add_argument("--date", type=str, help="Specific screen date (YYYY-MM-DD)")
    p_setups.add_argument("--limit", type=int, default=20, help="Number of rows to show")

    # chart
    p_chart = sub.add_parser("chart", help="Generate interactive chart for a symbol")
    p_chart.add_argument("--symbol", type=str, required=True, help="Symbol to chart")
    p_chart.add_argument("--days", type=int, default=30, help="Days of data to show")
    p_chart.add_argument("--variant", type=str, help="Variant for trade overlay (chan, mechanical, llm)")
    p_chart.add_argument("--experiment", type=str, default="experiments/paper_launch_v2.yaml")
    p_chart.add_argument("--overlays", type=str, default="sma,volume",
                         help="Comma-separated: sma,macd,rsi,bollinger,atr,volume,chan")

    # install-service
    p_install = sub.add_parser(
        "install-service",
        help="Install the scheduler as a macOS launch agent",
    )
    p_install.add_argument(
        "--mode",
        choices=["swing", "day", "both"],
        default="both",
        help="Trading mode for the background scheduler",
    )
    p_install.add_argument("--symbols", type=str, help="Optional comma-separated symbols")
    p_install.add_argument(
        "--experiment",
        type=str,
        help="Path to A/B experiment YAML (enables ABRunner in the launch agent)",
    )
    p_install.add_argument(
        "--label",
        type=str,
        default="com.tradingagents.scheduler",
        help="launchd label to install",
    )

    # remove-service
    p_remove = sub.add_parser(
        "remove-service",
        help="Remove the macOS launch agent",
    )
    p_remove.add_argument(
        "--label",
        type=str,
        default="com.tradingagents.scheduler",
        help="launchd label to remove",
    )

    # install-watchdog
    p_install_wd = sub.add_parser(
        "install-watchdog",
        help="Install the external watchdog as a separate launch agent",
    )
    p_install_wd.add_argument(
        "--label",
        type=str,
        default=WATCHDOG_LABEL,
        help="launchd label for the watchdog",
    )

    # remove-watchdog
    p_remove_wd = sub.add_parser(
        "remove-watchdog",
        help="Remove the watchdog launch agent",
    )
    p_remove_wd.add_argument(
        "--label",
        type=str,
        default=WATCHDOG_LABEL,
        help="launchd label to remove",
    )

    # dashboard
    p_dashboard = sub.add_parser(
        "dashboard",
        help="Launch the local monitoring dashboard",
    )
    p_dashboard.add_argument("--port", type=int, default=8501, help="Dashboard port")
    p_dashboard.add_argument(
        "--address",
        type=str,
        default="0.0.0.0",
        help="Dashboard bind address",
    )

    # notify-test
    p_notify = sub.add_parser(
        "notify-test",
        help="Send a test phone notification through ntfy",
    )
    p_notify.add_argument("--title", type=str, default="TradingAgents Test")
    p_notify.add_argument(
        "--message",
        type=str,
        default="ntfy is connected. Future setup/order alerts will arrive here.",
    )
    p_notify.add_argument(
        "--priority",
        type=str,
        default="default",
        help="ntfy priority: min, low, default, high, max, urgent",
    )

    # experiment
    p_exp = sub.add_parser("experiment", help="Manage A/B experiments")
    exp_sub = p_exp.add_subparsers(dest="exp_command")
    p_exp_create = exp_sub.add_parser("create", help="Create experiment from YAML")
    p_exp_create.add_argument("--config", required=True, help="Path to experiment YAML")
    p_exp_status = exp_sub.add_parser("status", help="Show experiment status")
    p_exp_status.add_argument("experiment_id", help="Experiment ID")
    p_exp_report = exp_sub.add_parser("report", help="Compare variant performance")
    p_exp_report.add_argument("experiment_id", help="Experiment ID")
    p_exp_promote = exp_sub.add_parser("promote", help="Promote winning variant")
    p_exp_promote.add_argument("experiment_id", help="Experiment ID")

    p_social_check = sub.add_parser(
        "social-check",
        help="Poll the configured X RSS sources once",
    )

    p_social_test = sub.add_parser(
        "social-test",
        help="Send a test alert to the social ntfy topic",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    setup_logging(args.verbose)

    commands = {
        "run": cmd_run,
        "schedule": cmd_schedule,
        "status": cmd_status,
        "close-all": cmd_close_all,
        "trades": cmd_trades,
        "report": cmd_report,
        "setups": cmd_setups,
        "chart": cmd_chart,
        "install-service": cmd_install_service,
        "remove-service": cmd_remove_service,
        "install-watchdog": cmd_install_watchdog,
        "remove-watchdog": cmd_remove_watchdog,
        "dashboard": cmd_dashboard,
        "notify-test": cmd_notify_test,
        "experiment": cmd_experiment,
        "social-check": cmd_social_check,
        "social-test": cmd_social_test,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
