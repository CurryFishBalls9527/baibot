# Vendored third-party code

This directory contains a flat copy of external libraries we depend on,
committed directly to keep the repo self-contained (no submodule
gymnastics). Standard upstream code is unmodified except where noted.

## chan.py

* **Upstream**: https://github.com/Vespa314/chan.py
* **License**: MIT (see `chan.py/LICENSE`)
* **Vendored at**: commit `616b15f` ("doc: update README"), 2026-04-11
* **Used by** (live + research):
  - `tradingagents/research/chan_backtester.py`
  - `tradingagents/research/chan_daily_backtester.py`
  - `tradingagents/research/intraday_chan_signals.py`
  - `tradingagents/research/chan_adapter.py`

The `chan_v2` and `chan_daily` swing variants depend on this library at
runtime — fresh-clone setups will not be able to run those strategies
without it.

### Local additions (NOT in upstream)

We've added 3 DuckDB-backed DataAPI subclasses for chan.py's pluggable
data interface:

```
chan.py/DataAPI/DuckDBAPI.py
chan.py/DataAPI/DuckDBDailyAPI.py
chan.py/DataAPI/DuckDBWeeklyAPI.py
```

These bridge chan.py's `CCommonStockApi` interface to our
`research_data/intraday_30m_broad.duckdb` and `market_data.duckdb`
warehouses. They are the only files that should diverge from upstream;
keep them in sync if/when this directory is re-vendored from a newer
chan.py snapshot.

## Re-vendoring procedure

If you ever want to bump chan.py to a newer upstream commit:

```bash
# 1. Clone the upstream into a scratch dir
cd /tmp && git clone https://github.com/Vespa314/chan.py
cd chan.py && git log --oneline -1   # note the new commit SHA

# 2. Save your local DataAPI additions
cp /Users/myu/code/baibot/third_party/chan.py/DataAPI/DuckDB*.py /tmp/

# 3. Replace this dir's contents (preserving VENDORED.md)
cd /Users/myu/code/baibot/third_party
rm -rf chan.py
cp -r /tmp/chan.py ./
rm -rf chan.py/.git              # flat vendor, not submodule
find chan.py -name __pycache__ -type d -exec rm -rf {} +

# 4. Re-apply local additions
cp /tmp/DuckDB*.py chan.py/DataAPI/

# 5. Update the commit SHA in this file, run the chan_v2 + chan_daily
#    backtests, smoke-test the live scheduler, then commit.
```
