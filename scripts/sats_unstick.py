"""One-shot: inspect/cancel/close SATS on the chan paper account.

Usage:
    python scripts/sats_unstick.py inspect   # read-only dump
    python scripts/sats_unstick.py fix       # cancel open SATS orders + close position
"""
import os
import sys
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

load_dotenv()

API_KEY = os.environ["ALPACA_CHAN_API_KEY"]
SECRET = os.environ["ALPACA_CHAN_SECRET_KEY"]
SYMBOL = "SATS"

client = TradingClient(API_KEY, SECRET, paper=True)


def dump_orders(label):
    print(f"\n=== {label}: open orders for {SYMBOL} (nested) ===")
    req = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[SYMBOL],
        nested=True,
    )
    orders = client.get_orders(filter=req)
    if not orders:
        print("  (none)")
    for o in orders:
        print(f"  parent id={o.id} side={o.side} type={o.type} qty={o.qty} "
              f"status={o.status} order_class={o.order_class}")
        for leg in getattr(o, "legs", None) or []:
            print(f"    leg   id={leg.id} side={leg.side} type={leg.type} "
                  f"qty={leg.qty} status={leg.status} "
                  f"stop={leg.stop_price} limit={leg.limit_price}")
    return orders


def dump_position():
    try:
        p = client.get_open_position(SYMBOL)
        print(f"\n=== position ===\n  qty={p.qty} avg_entry={p.avg_entry_price} "
              f"current={p.current_price} market_value={p.market_value} "
              f"unrealized_pl={p.unrealized_pl}")
        return p
    except Exception as e:
        print(f"\n=== position ===\n  NONE ({e})")
        return None


def collect_open_order_ids():
    req = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[SYMBOL],
        nested=True,
    )
    ids = []
    for o in client.get_orders(filter=req):
        ids.append(str(o.id))
        for leg in getattr(o, "legs", None) or []:
            ids.append(str(leg.id))
    return ids


def cmd_inspect():
    dump_position()
    dump_orders("BEFORE")


def cmd_fix():
    dump_position()
    dump_orders("BEFORE")
    ids = collect_open_order_ids()
    print(f"\n=== canceling {len(ids)} order(s) ===")
    for oid in ids:
        try:
            client.cancel_order_by_id(oid)
            print(f"  cancel {oid} -> ok")
        except Exception as e:
            print(f"  cancel {oid} -> ERROR: {e}")

    dump_orders("AFTER CANCEL")

    print(f"\n=== closing position {SYMBOL} ===")
    try:
        resp = client.close_position(SYMBOL)
        print(f"  close_position response: id={resp.id} side={resp.side} "
              f"qty={resp.qty} status={resp.status}")
    except Exception as e:
        print(f"  close_position ERROR: {e}")

    dump_orders("AFTER CLOSE")
    dump_position()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    if mode == "inspect":
        cmd_inspect()
    elif mode == "fix":
        cmd_fix()
    else:
        print(f"unknown mode: {mode}; use 'inspect' or 'fix'")
        sys.exit(1)
