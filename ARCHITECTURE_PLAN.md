# TradingAgents 自动化交易平台架构方案

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Scheduler (APScheduler)                     │
│         每日定时触发 / 可配置多个时间点 (pre-market, intraday)         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Orchestrator (主控循环)                         │
│  1. 获取 watchlist                                                  │
│  2. 检查当前持仓                                                     │
│  3. 对每个标的调用 AI 分析引擎                                        │
│  4. 将信号转化为具体订单                                              │
│  5. 执行风控检查                                                     │
│  6. 提交订单                                                        │
│  7. 记录交易日志                                                     │
└────┬───────────┬────────────┬───────────┬──────────┬───────────────┘
     │           │            │           │          │
     ▼           ▼            ▼           ▼          ▼
┌─────────┐ ┌────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐
│ AI 分析  │ │ 仓位    │ │ 风控引擎  │ │ 订单    │ │ 数据存储  │
│ 引擎     │ │ 管理器  │ │          │ │ 执行器  │ │          │
│(现有的   │ │        │ │ 止损/止盈 │ │        │ │ SQLite   │
│ LangGraph│ │ 仓位大小│ │ 最大回撤  │ │ Alpaca │ │ + JSON   │
│ 多Agent) │ │ 持仓追踪│ │ 单笔限额  │ │ API    │ │ Memory   │
└─────────┘ └────────┘ └──────────┘ └─────────┘ └──────────┘
```

## 二、Alpaca 集成说明

### Alpaca 能提供什么

| 功能 | 说明 |
|------|------|
| **实时行情** | REST API (快照/bars) + WebSocket (实时推送) |
| **历史数据** | 日/分钟/秒级 OHLCV，最多可追溯到 2016 年 |
| **下单交易** | 市价单、限价单、止损单、止损限价单、Trailing Stop |
| **持仓管理** | 查询所有持仓、单个持仓、平仓、批量平仓 |
| **账户信息** | 余额、购买力、盈亏、保证金等 |
| **Paper Trading** | 完全相同的 API，换 base_url 即为模拟交易 |
| **订单事件** | WebSocket 推送订单状态变更 (filled, canceled 等) |

### Paper → Real 切换

```python
# Paper Trading (模拟)
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# Real Trading (真实) — 只改这一行
ALPACA_BASE_URL = "https://api.alpaca.markets"
```

API Key 和 Secret 分别对应 paper/live 账户，代码逻辑完全不变。

## 三、新增模块设计

### 3.1 目录结构

```
tradingagents/
├── automation/                    # 新增：自动化核心
│   ├── __init__.py
│   ├── scheduler.py               # 定时调度器
│   ├── orchestrator.py            # 主控逻辑
│   └── config.py                  # 自动化配置
│
├── broker/                        # 新增：券商集成
│   ├── __init__.py
│   ├── base_broker.py             # 抽象基类
│   ├── alpaca_broker.py           # Alpaca 实现
│   └── models.py                  # Order, Position, Account 数据模型
│
├── portfolio/                     # 新增：仓位管理
│   ├── __init__.py
│   ├── position_manager.py        # 持仓跟踪
│   ├── position_sizer.py          # 仓位大小计算
│   └── portfolio_tracker.py       # 组合级别 P&L
│
├── risk/                          # 新增：硬性风控
│   ├── __init__.py
│   ├── risk_engine.py             # 风控引擎
│   └── rules.py                   # 风控规则
│
├── storage/                       # 新增：持久化
│   ├── __init__.py
│   ├── database.py                # SQLite 交易记录
│   └── memory_store.py            # Agent Memory 持久化
│
├── dataflows/                     # 现有：增强
│   ├── alpaca_data.py             # 新增：Alpaca 数据源
│   └── ... (现有文件)
│
├── graph/                         # 现有：小改
│   ├── signal_processing.py       # 增强：输出更丰富的信号
│   └── ... (现有文件)
│
└── agents/                        # 现有：基本不动
    └── ...
```

### 3.2 Broker 模块 — `tradingagents/broker/`

```python
# base_broker.py — 抽象基类，方便以后接 IB 或其他券商
class BaseBroker(ABC):
    def get_account(self) -> Account: ...
    def get_positions(self) -> List[Position]: ...
    def get_position(self, symbol: str) -> Optional[Position]: ...
    def submit_order(self, order: OrderRequest) -> Order: ...
    def cancel_order(self, order_id: str) -> None: ...
    def close_position(self, symbol: str) -> Order: ...
    def close_all_positions(self) -> None: ...
    def get_order_status(self, order_id: str) -> Order: ...
    def is_market_open(self) -> bool: ...
    def get_clock(self) -> MarketClock: ...

# alpaca_broker.py — Alpaca 实现
class AlpacaBroker(BaseBroker):
    def __init__(self, api_key, secret_key, paper=True):
        base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.api = TradingClient(api_key, secret_key, url_override=base_url)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        ...

# models.py — 统一数据模型
@dataclass
class OrderRequest:
    symbol: str
    side: Literal["buy", "sell"]
    qty: Optional[float] = None        # 股数
    notional: Optional[float] = None   # 金额 (二选一)
    order_type: Literal["market", "limit", "stop", "stop_limit", "trailing_stop"] = "market"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_percent: Optional[float] = None
    time_in_force: Literal["day", "gtc", "ioc"] = "day"

@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_percent: float

@dataclass
class Account:
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
```

### 3.3 Position Sizer — `tradingagents/portfolio/position_sizer.py`

将 BUY/SELL/HOLD 转换为具体的股数和金额：

```python
class PositionSizer:
    """将 AI 信号转化为具体仓位大小"""

    def __init__(self, config):
        self.max_position_pct = config.get("max_position_pct", 0.10)  # 单只最多占总资产 10%
        self.max_total_exposure = config.get("max_total_exposure", 0.80)  # 总仓位不超过 80%
        self.default_risk_per_trade = config.get("risk_per_trade", 0.02)  # 每笔风险 2%

    def calculate(self, signal, account, current_price, atr=None):
        """
        signal: "BUY" / "SELL" / "HOLD"
        返回: OrderRequest 或 None
        """
        if signal == "HOLD":
            return None

        if signal == "BUY":
            available = account.equity * self.max_position_pct
            qty = int(available / current_price)
            return OrderRequest(symbol=..., side="buy", qty=qty, ...)

        if signal == "SELL":
            # 平掉现有仓位
            return OrderRequest(symbol=..., side="sell", qty=current_position.qty, ...)
```

### 3.4 Risk Engine — `tradingagents/risk/risk_engine.py`

在提交订单前的最后一道关卡：

```python
class RiskEngine:
    """硬性风控 — 不管 AI 怎么说，这些规则不可违反"""

    def __init__(self, config):
        self.max_position_pct = config.get("max_position_pct", 0.10)
        self.max_total_exposure = config.get("max_total_exposure", 0.80)
        self.max_daily_loss = config.get("max_daily_loss", 0.03)       # 日最大亏损 3%
        self.max_drawdown = config.get("max_drawdown", 0.10)           # 最大回撤 10%
        self.min_cash_reserve = config.get("min_cash_reserve", 0.20)   # 至少保留 20% 现金

    def check_order(self, order, account, positions) -> Tuple[bool, str]:
        """
        返回 (是否通过, 原因)
        """
        # 检查 1: 单只股票仓位不超限
        # 检查 2: 总仓位不超限
        # 检查 3: 今日亏损未超限
        # 检查 4: 总回撤未超限
        # 检查 5: 现金储备足够
        # 检查 6: 市场是否开盘
        ...
```

### 3.5 Storage — `tradingagents/storage/`

```python
# database.py — SQLite, 轻量不需要额外服务
class TradingDatabase:
    """交易记录、信号日志、每日快照"""

    def __init__(self, db_path="trading.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    # 表:
    # trades        — 每笔交易 (symbol, side, qty, price, timestamp, signal, reasoning)
    # signals       — AI 产生的所有信号 (包括被风控拒绝的)
    # daily_snapshots — 每日账户快照 (equity, cash, positions, P&L)
    # agent_memories  — 持久化的 Agent 记忆

# memory_store.py — Agent Memory 持久化
class PersistentMemory(FinancialSituationMemory):
    """继承现有 Memory，增加 save/load 到 SQLite"""

    def save(self): ...   # 序列化到数据库
    def load(self): ...   # 从数据库恢复
```

### 3.6 Scheduler — `tradingagents/automation/scheduler.py`

```python
class TradingScheduler:
    """每日自动运行分析和交易"""

    def __init__(self, config):
        self.scheduler = APScheduler()
        self.orchestrator = Orchestrator(config)

    def start(self):
        # Swing Trade 模式：每日收盘前分析
        self.scheduler.add_job(
            self.orchestrator.run_daily_analysis,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=15, minute=30,      # 3:30 PM ET (收盘前 30 分钟)
                timezone="US/Eastern"
            ),
            id="daily_swing_analysis"
        )

        # Day Trade 模式 (可选)：盘中多次分析
        self.scheduler.add_job(
            self.orchestrator.run_intraday_scan,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour="10-15",            # 每小时一次
                minute=0,
                timezone="US/Eastern"
            ),
            id="intraday_scan"
        )

        # 每日复盘：收盘后反思
        self.scheduler.add_job(
            self.orchestrator.run_daily_reflection,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=16, minute=30,      # 4:30 PM ET (收盘后 30 分钟)
                timezone="US/Eastern"
            ),
            id="daily_reflection"
        )

        self.scheduler.start()
```

### 3.7 Orchestrator — `tradingagents/automation/orchestrator.py`

这是系统的心脏，串联所有模块：

```python
class Orchestrator:
    """主控逻辑 — 串联 AI 分析 → 仓位计算 → 风控检查 → 订单执行"""

    def __init__(self, config):
        self.broker = AlpacaBroker(config)
        self.risk_engine = RiskEngine(config)
        self.position_sizer = PositionSizer(config)
        self.db = TradingDatabase(config)
        self.ta = TradingAgentsGraph(config=config)  # 现有的 AI 引擎

    def run_daily_analysis(self):
        """每日主分析流程"""

        # 0. 检查市场是否开盘
        if not self.broker.is_market_open():
            return

        account = self.broker.get_account()
        positions = self.broker.get_positions()

        # 1. 分析 watchlist 中的每个标的
        for symbol in self.watchlist:
            try:
                # 2. AI 分析（现有系统）
                today = date.today().isoformat()
                state, signal = self.ta.propagate(symbol, today)

                # 3. 记录信号
                self.db.log_signal(symbol, signal, state)

                # 4. 计算仓位大小
                current_price = self.broker.get_latest_price(symbol)
                order_request = self.position_sizer.calculate(
                    signal=signal,
                    account=account,
                    current_price=current_price,
                    current_position=self._get_position(positions, symbol)
                )

                if order_request is None:
                    continue

                # 5. 风控检查
                passed, reason = self.risk_engine.check_order(
                    order_request, account, positions
                )
                if not passed:
                    self.db.log_risk_rejection(symbol, order_request, reason)
                    continue

                # 6. 执行订单
                order = self.broker.submit_order(order_request)
                self.db.log_trade(order)

            except Exception as e:
                self.db.log_error(symbol, e)

        # 7. 拍摄每日快照
        self.db.take_daily_snapshot(account, positions)

    def run_daily_reflection(self):
        """每日收盘后复盘"""
        positions = self.broker.get_positions()
        for pos in positions:
            returns = pos.unrealized_pl
            self.ta.reflect_and_remember(returns)
        # 持久化 memory
        self.ta.save_memories()
```

### 3.8 增强 Signal Processing

现有系统只输出 BUY/SELL/HOLD。需要增强为更丰富的信号：

```python
@dataclass
class TradingSignal:
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float              # 0.0 - 1.0 置信度
    reasoning: str                 # 决策理由摘要
    suggested_entry: Optional[float]   # 建议入场价
    suggested_stop: Optional[float]    # 建议止损价
    suggested_target: Optional[float]  # 建议目标价
    timeframe: Literal["day", "swing", "position"]  # 交易周期
    risk_reward_ratio: Optional[float]
```

这需要修改 `SignalProcessor` 的 prompt，让 LLM 输出结构化的 JSON 而不只是一个词。

## 四、数据流增强

### 4.1 新增 Alpaca 作为数据源

```python
# tradingagents/dataflows/alpaca_data.py
class AlpacaDataSource:
    """通过 Alpaca Market Data API 获取实时和历史数据"""

    def get_stock_data(self, symbol, start, end):
        """替代 yfinance，获取 OHLCV"""
        ...

    def get_latest_price(self, symbol):
        """获取最新价格（实时）"""
        ...

    def get_intraday_bars(self, symbol, timeframe="1Min"):
        """获取分钟级数据（day trade 用）"""
        ...

    def get_snapshot(self, symbol):
        """获取实时快照：最新价、bid/ask、日内涨跌等"""
        ...
```

### 4.2 数据源优先级

| 用途 | 推荐数据源 | 原因 |
|------|-----------|------|
| 实时价格/快照 | **Alpaca** | 免费、低延迟、和交易同一 API |
| 历史日线 | **Alpaca** 或 yfinance | Alpaca 更稳定，yfinance 免费但有限频 |
| 分钟级数据 | **Alpaca** | yfinance 分钟数据不稳定 |
| 技术指标 | **本地计算 (stockstats)** | 基于 OHLCV 自己算，不依赖外部 |
| 基本面 | **yfinance** | Alpaca 不提供基本面数据 |
| 新闻 | **yfinance** 或 Alpha Vantage | Alpaca 有新闻 API (付费) |

## 五、配置文件设计

```python
AUTOMATION_CONFIG = {
    # === Alpaca ===
    "alpaca_api_key": "...",
    "alpaca_secret_key": "...",
    "paper_trading": True,          # True = paper, False = live

    # === 交易模式 ===
    "trading_mode": "swing",        # "swing" | "day" | "both"
    "watchlist": ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META"],

    # === 调度 ===
    "swing_analysis_time": "15:30", # ET
    "intraday_interval_minutes": 60,
    "reflection_time": "16:30",     # ET

    # === 仓位管理 ===
    "max_position_pct": 0.10,       # 单只股票最多占总资产 10%
    "max_total_exposure": 0.80,     # 总仓位不超过 80%
    "risk_per_trade": 0.02,         # 每笔交易风险 2%

    # === 风控 ===
    "max_daily_loss": 0.03,         # 日最大亏损 3%
    "max_drawdown": 0.10,           # 最大回撤 10%
    "min_cash_reserve": 0.20,       # 至少保留 20% 现金
    "default_stop_loss_pct": 0.05,  # 默认止损 5%
    "default_take_profit_pct": 0.15,# 默认止盈 15%

    # === 存储 ===
    "db_path": "trading.db",

    # === 通知 (Phase 4) ===
    "notify_on_trade": True,
    "notify_on_error": True,

    # === AI 引擎 (继承现有) ===
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5-mini",
    "quick_think_llm": "gpt-5-mini",
    "max_debate_rounds": 1,
    ...
}
```

## 六、实施路线图

### Phase 1: 基础设施 (1-2 周)

**目标：** 能连接 Alpaca Paper 账户，手动触发分析 + 下单

| 任务 | 说明 |
|------|------|
| 1.1 Broker 模块 | `BaseBroker` + `AlpacaBroker` 实现 |
| 1.2 数据模型 | `Order`, `Position`, `Account` 数据类 |
| 1.3 数据库 | SQLite 建表，基础 CRUD |
| 1.4 配置系统 | 统一 automation config |
| 1.5 Alpaca 数据源 | `alpaca_data.py` 替代/补充 yfinance |

**验证:** 能通过代码查询 Paper 账户余额、获取实时行情、下一笔市价单

### Phase 2: 交易逻辑 (1-2 周)

**目标：** AI 分析 → 自动下单，完整闭环

| 任务 | 说明 |
|------|------|
| 2.1 增强 SignalProcessor | 输出结构化信号 (含置信度、止损/止盈) |
| 2.2 Position Sizer | 根据风险参数计算仓位 |
| 2.3 Risk Engine | 硬性风控规则实现 |
| 2.4 Orchestrator | 串联 AI 分析 → 仓位 → 风控 → 下单 |
| 2.5 Memory 持久化 | Agent Memory 存入 SQLite |

**验证:** 对一只股票运行完整流程：分析 → BUY 信号 → 计算买入 X 股 → 风控通过 → Paper 账户下单成功

### Phase 3: 自动化 (1 周)

**目标：** 每天自动运行，无需人工干预

| 任务 | 说明 |
|------|------|
| 3.1 Scheduler | APScheduler 定时任务 |
| 3.2 Watchlist 管理 | 配置化 watchlist |
| 3.3 错误处理 | 重试、异常捕获、优雅降级 |
| 3.4 日志系统 | 结构化日志 (logging) |
| 3.5 每日复盘 | 自动 reflect_and_remember |

**验证:** 启动后连续 5 个交易日自动分析、交易、记录，无人工介入

### Phase 4: 监控与优化 (持续)

**目标：** 能看到表现，持续优化

| 任务 | 说明 |
|------|------|
| 4.1 Dashboard | 简单 Web UI 查看持仓/P&L/历史交易 |
| 4.2 通知 | Telegram/邮件 推送交易和异常 |
| 4.3 Performance 分析 | Sharpe ratio, Win rate, Max drawdown 等 |
| 4.4 A/B 测试 | 不同 LLM/prompt/参数的对比 |
| 4.5 切换 Live | Paper 验证后切换到真实账户 |

## 七、关键设计决策

### 为什么选 Alpaca 而不是 IB

| | Alpaca | Interactive Brokers |
|--|--------|-------------------|
| 注册 | 几分钟，在线完成 | 需要更多验证 |
| Paper Trading | 内置，同一套 API | 需要单独的 TWS |
| API 易用性 | REST + Python SDK，非常现代 | TWS API 较复杂 |
| 佣金 | 零佣金 | 低佣金但非零 |
| 数据 | 免费基础行情 | 需要订阅 |
| 适合 | 快速原型，swing/day trade | 更复杂策略，多资产类型 |

先用 Alpaca 快速验证，等策略成熟后可以加 IB 作为第二个 Broker 实现。

### 为什么用 SQLite 而不是 PostgreSQL

- 零部署成本，单文件数据库
- 对于单用户交易系统完全够用
- 不需要额外运行数据库服务
- 以后需要可以无缝迁移到 PostgreSQL

### LLM 成本控制

当前每次分析调用 4 个 analyst + 多轮辩论，token 消耗大。建议：

- Swing trade: 每日分析 5-10 只股票，用 `gpt-5-mini`，月成本约 $20-50
- Day trade: 每小时分析更频繁，考虑用更便宜的模型或减少辩论轮数
- 对 HOLD 的股票可以降低分析频率 (3 天一次)

## 八、安全注意事项

1. **API Key 绝不提交到 Git** — 只放 `.env` 文件
2. **Paper Trading 先行** — 至少跑 3 个月再考虑 live
3. **风控不可关闭** — Risk Engine 是最后防线，独立于 AI
4. **最大亏损熔断** — 日亏损超 3% 自动停止交易
5. **手动干预入口** — 永远保留手动平仓/停止系统的能力
