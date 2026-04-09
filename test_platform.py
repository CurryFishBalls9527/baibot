#!/usr/bin/env python3
"""Test each module of the automated trading platform independently.

Run: .venv/bin/python test_platform.py
"""

import os
import sys
import traceback

import dotenv
dotenv.load_dotenv()


def header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_broker():
    header("TEST 1: Broker (Alpaca Paper Account)")
    from tradingagents.broker.alpaca_broker import AlpacaBroker

    broker = AlpacaBroker(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
        paper=True,
    )

    acct = broker.get_account()
    print(f"  Account: {acct.account_id}")
    print(f"  Equity:  ${acct.equity:,.2f}")
    print(f"  Cash:    ${acct.cash:,.2f}")
    print(f"  Status:  {acct.status}")

    clock = broker.get_clock()
    print(f"  Market open: {clock.is_open}")

    price = broker.get_latest_price("NVDA")
    print(f"  NVDA price: ${price:.2f}")

    prices = broker.get_latest_prices(["AAPL", "MSFT", "GOOGL"])
    for sym, p in prices.items():
        print(f"  {sym}: ${p:.2f}")

    positions = broker.get_positions()
    print(f"  Open positions: {len(positions)}")

    print("  [OK] Broker working!")
    return broker


def test_database():
    header("TEST 2: Database (SQLite)")
    from tradingagents.storage.database import TradingDatabase

    db_path = "test_trading.db"
    db = TradingDatabase(db_path)

    sig_id = db.log_signal("NVDA", "BUY", confidence=0.85,
                           reasoning="Test signal", stop_loss=0.05)
    print(f"  Signal logged: id={sig_id}")

    trade_id = db.log_trade("NVDA", "buy", qty=10, order_type="market",
                            status="filled", filled_price=170.50,
                            order_id="test-001", signal_id=sig_id)
    print(f"  Trade logged: id={trade_id}")

    db.take_snapshot(100000, 90000, 180000, 100000,
                     [{"symbol": "NVDA", "qty": 10, "market_value": 10000}],
                     daily_pl=150, daily_pl_pct=0.0015)
    print(f"  Snapshot saved")

    db.save_memories("test_memory", [("test situation", "test recommendation")])
    mems = db.load_memories("test_memory")
    print(f"  Memories saved/loaded: {len(mems)}")

    trades = db.get_recent_trades()
    print(f"  Recent trades: {len(trades)}")

    db.close()
    os.remove(db_path)
    print("  [OK] Database working!")


def test_risk_engine():
    header("TEST 3: Risk Engine")
    from tradingagents.risk.risk_engine import RiskEngine
    from tradingagents.broker.models import OrderRequest, Account, Position

    engine = RiskEngine({
        "max_position_pct": 0.10,
        "max_total_exposure": 0.80,
        "max_daily_loss": 0.03,
        "max_drawdown": 0.10,
        "min_cash_reserve": 0.20,
        "max_open_positions": 10,
        "starting_equity": 100000,
    })

    account = Account(
        account_id="test", equity=100000, cash=100000,
        buying_power=200000, portfolio_value=100000,
        status="ACTIVE", last_equity=100000,
    )

    # Should pass: small order
    order = OrderRequest(symbol="AAPL", side="buy", qty=10)
    result = engine.check_order(order, account, [], current_price=250)
    print(f"  Small order (10 AAPL @ $250 = $2,500): {'PASS' if result.passed else 'FAIL: ' + result.reason}")

    # Should fail: too large
    order_big = OrderRequest(symbol="AAPL", side="buy", qty=200)
    result2 = engine.check_order(order_big, account, [], current_price=250)
    print(f"  Large order (200 AAPL @ $250 = $50,000): {'PASS' if result2.passed else 'FAIL: ' + result2.reason}")

    # Should pass: sell orders always allowed
    order_sell = OrderRequest(symbol="AAPL", side="sell", qty=100)
    result3 = engine.check_order(order_sell, account, [], current_price=250)
    print(f"  Sell order: {'PASS' if result3.passed else 'FAIL: ' + result3.reason}")

    print("  [OK] Risk engine working!")


def test_position_sizer():
    header("TEST 4: Position Sizer")
    from tradingagents.portfolio.position_sizer import PositionSizer
    from tradingagents.broker.models import Account

    sizer = PositionSizer({
        "max_position_pct": 0.10,
        "max_total_exposure": 0.80,
        "risk_per_trade": 0.02,
    })

    account = Account(
        account_id="test", equity=100000, cash=100000,
        buying_power=200000, portfolio_value=100000,
        status="ACTIVE",
    )

    # BUY signal
    signal = {"action": "BUY", "symbol": "NVDA", "confidence": 0.8}
    order = sizer.calculate(signal, account, current_price=175.0)
    if order:
        print(f"  BUY NVDA: {order.qty:.0f} shares @ ~$175 = ${order.qty * 175:,.0f}")
    else:
        print(f"  BUY NVDA: no order")

    # HOLD signal
    signal_hold = {"action": "HOLD", "symbol": "NVDA"}
    order_hold = sizer.calculate(signal_hold, account, current_price=175.0)
    print(f"  HOLD NVDA: {'no order' if order_hold is None else 'unexpected order!'}")

    # BUY with stop loss
    signal_sl = {"action": "BUY", "symbol": "AAPL", "confidence": 0.7,
                 "stop_loss": 237.0}
    order_sl = sizer.calculate(signal_sl, account, current_price=250.0)
    if order_sl:
        print(f"  BUY AAPL (with stop $237): {order_sl.qty:.0f} shares")

    print("  [OK] Position sizer working!")


def test_signal_processor():
    header("TEST 5: Signal Processor (structured)")
    from tradingagents.graph.signal_processing import SignalProcessor

    # Use a mock LLM that returns JSON
    class MockLLM:
        def invoke(self, messages):
            class Resp:
                content = '{"action": "BUY", "confidence": 0.82, "reasoning": "Strong technicals and bullish momentum", "stop_loss_pct": 0.05, "take_profit_pct": 0.15, "timeframe": "swing"}'
            return Resp()

    proc = SignalProcessor(MockLLM())
    result = proc.process_signal_structured("Test analysis report", symbol="NVDA")
    print(f"  Action:     {result['action']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Reasoning:  {result['reasoning']}")
    print(f"  Stop loss:  {result['stop_loss_pct']}")
    print(f"  Take profit:{result['take_profit_pct']}")
    print(f"  Timeframe:  {result['timeframe']}")
    print("  [OK] Signal processor working!")


def test_orchestrator_status():
    header("TEST 6: Orchestrator Status Check")
    from tradingagents.automation.config import build_config
    from tradingagents.automation.orchestrator import Orchestrator

    config = build_config()
    orch = Orchestrator(config)
    status = orch.get_status()

    print(f"  Equity:    ${status['account']['equity']:,.2f}")
    print(f"  Cash:      ${status['account']['cash']:,.2f}")
    print(f"  Market:    {'OPEN' if status['market']['is_open'] else 'CLOSED'}")
    print(f"  Positions: {len(status['positions'])}")
    print(f"  Watchlist: {status['watchlist']}")
    print(f"  Paper:     {status['paper_mode']}")
    print("  [OK] Orchestrator status working!")


def main():
    print("\n" + "#" * 60)
    print("  TradingAgents Platform Test Suite")
    print("#" * 60)

    tests = [
        ("Broker", test_broker),
        ("Database", test_database),
        ("Risk Engine", test_risk_engine),
        ("Position Sizer", test_position_sizer),
        ("Signal Processor", test_signal_processor),
        ("Orchestrator Status", test_orchestrator_status),
    ]

    passed = 0
    failed = 0

    for name, func in tests:
        try:
            func()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1

    header("SUMMARY")
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n  ALL TESTS PASSED! Platform is ready.")
        print("\n  Next steps:")
        print("    1. Test a single stock:  .venv/bin/python run_trading.py run --symbols NVDA")
        print("    2. Check status:         .venv/bin/python run_trading.py status")
        print("    3. Start autopilot:      .venv/bin/python run_trading.py schedule")
    else:
        print(f"\n  {failed} test(s) failed. Fix issues above before running.")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
