# TradingAgents-Astock 技术架构说明

> 版本：v0.2.16 · 本文基于源码梳理，聚焦「用了哪些技术、如何分层、数据怎么流动」。
> 面向想快速理解工程全貌或参与开发的读者。

## 1. 项目定位

基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) 的 **A 股深度特化 fork**。核心是一个**多 Agent 投研流水线**：多个 AI 分析师并行取数 → 数据质量门控 → 多空辩论 → 三方风险辩论 → 生成带 5 档评级的投资决策，全程可持久化、可断点续跑、跨次积累交易记忆。

- 语言：Python ≥ 3.10
- 协议：Apache 2.0
- 规模：约 1.25 万行 Python

---

## 2. 技术栈总览

| 层次 | 技术选型 | 用途 |
|------|---------|------|
| **Agent 编排** | [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` | 有状态多 Agent 工作流、条件边、断点 |
| **LLM 抽象** | LangChain（`langchain-openai` / `langchain-anthropic` / `langchain-google-genai`(可选)） | 统一 Chat 接口 + tool calling |
| **断点持久化** | `langgraph-checkpoint-sqlite`（SqliteSaver） | 逐节点保存状态，崩溃后续跑 |
| **数据获取** | 纯 `requests` + `mootdx`(TCP) 直连 HTTP/TCP | 9 个 A 股数据源，零第三方 DB 依赖 |
| **数据处理** | `pandas` / `stockstats` | K 线整形、技术指标计算 |
| **收益回测** | `yfinance`（拉沪深300基准算 alpha） | 延迟反思机制的真实收益核验 |
| **Web UI** | [Streamlit](https://streamlit.io/) | 可视化分析界面、历史、PDF 导出 |
| **CLI** | [Typer](https://typer.tiangolo.com/) + `questionary` + `rich` | 命令行交互入口 |
| **导出** | `fpdf2` | PDF 报告生成 |
| **配置** | `python-dotenv`（`.env`） | 密钥、路径、端点 |

**设计基调**：数据层零重型依赖（v0.2.5 移除 akshare，全部直连 HTTP）；LLM 层可插拔多供应商 + 第三方网关；编排层用 LangGraph 表达确定性的多 Agent 拓扑。

---

## 3. 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│  交互层    Web (Streamlit)          CLI (Typer)               │
│            web/                     cli/                       │
├─────────────────────────────────────────────────────────────┤
│  编排层    TradingAgentsGraph  (tradingagents/graph/)         │
│            LangGraph StateGraph · 条件逻辑 · 断点 · 反思 · 信号 │
├─────────────────────────────────────────────────────────────┤
│  Agent 层  7 分析师 + 质量门 + 多空辩论 + 风险辩论 + 经理      │
│            tradingagents/agents/                              │
├─────────────────────────────────────────────────────────────┤
│  LLM 层    create_llm_client 工厂  (llm_clients/)             │
│            openai/anthropic/google/azure + 8 家兼容供应商      │
├─────────────────────────────────────────────────────────────┤
│  数据层    route_to_vendor → a_stock / yfinance / alpha_vantage│
│            tradingagents/dataflows/                           │
├─────────────────────────────────────────────────────────────┤
│  数据源    mootdx·腾讯·东财·新浪·同花顺·财联社·百度 (直连)      │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 编排层（核心）

入口类 `TradingAgentsGraph`（`tradingagents/graph/trading_graph.py`）负责：装配 LLM、构建工具节点、编译 LangGraph、跑流水线、落盘、管理断点与记忆。

### 4.1 工作流拓扑（LangGraph StateGraph）

由 `graph/setup.py` 构建，`graph/conditional_logic.py` 控制条件边：

```
START
  │
  ▼  ┌──────────────────────────────────────────────┐
  │  │ 7 个分析师串行（每个：LLM ⇄ 工具节点 → 清消息）│
  │  │ Market→Social→News→Fundamentals→              │
  │  │ Policy→Hot_money→Lockup                        │
  │  └──────────────────────────────────────────────┘
  ▼
Quality Gate            ← 数据质量门控（A 股新增）
  ▼
Bull Researcher ⇄ Bear Researcher   ← 多空辩论（max_debate_rounds 轮）
  ▼
Research Manager        ← 深度模型裁决，出投资计划 + 评级
  ▼
Trader                  ← 交易计划
  ▼
Aggressive → Conservative → Neutral ← 三方风险辩论（max_risk_discuss_rounds 轮）
  ▼
Portfolio Manager       ← 深度模型出最终决策 + 5 档评级
  ▼
END
```

**每个分析师节点**是一个「LLM 决定是否调用工具」的小循环：
- `should_continue_<analyst>`（conditional_logic）判断上一条消息是否有 `tool_calls`；
- 有 → 进 `tools_<analyst>`（LangGraph `ToolNode`）执行取数，结果回灌 LLM；
- 没有 → 进 `Msg Clear <Analyst>`（`create_msg_delete`）清空该分析师的消息、写入报告字段，交棒下一个分析师。

**辩论轮次**用状态里的计数器控制：多空 `count >= 2*rounds` 结束；风险 `count >= 3*rounds` 结束（3 个角色轮流）。

### 4.2 双 LLM 策略

| 角色 | 模型 | 说明 |
|------|------|------|
| `quick_thinking_llm` | `quick_think_llm` | 分析师取数、辩论者、质量门复审、Trader |
| `deep_thinking_llm` | `deep_think_llm` | Research Manager、Portfolio Manager（需深度推理的裁决） |

### 4.3 其他编排组件（`graph/`）

- **`propagation.py`** — `Propagator`：构造初始 `AgentState`、注入 `past_context`（历史记忆）、组装 `graph.stream/invoke` 的参数（含 callbacks、递归上限）。
- **`reflection.py`** — `Reflector`：对已知真实收益的历史决策生成反思文本，回写记忆。
- **`signal_processing.py`** — `SignalProcessor`：从 PM 决策里用确定性解析（`rating.parse_rating`）提取 5 档评级，**无需额外 LLM 调用**（PM 用结构化输出保证 `**Rating**: X` 可解析）。
- **`checkpointer.py`** — 见 §7。

---

## 5. Agent 层（`tradingagents/agents/`）

### 5.1 角色清单

| 分类 | 角色 | 备注 |
|------|------|------|
| 原版分析师 | 市场(technical) / 情绪(social) / 新闻(news) / 基本面(fundamentals) | |
| **A 股特化分析师** | **政策分析师 / 游资追踪 / 解禁监控** | fork 新增 |
| 质量门 | Quality Gate | fork 新增，见 §5.3 |
| 研究团队 | Bull / Bear Researcher + Research Manager | 多空辩论 |
| 交易 | Trader | |
| 风险团队 | Aggressive / Conservative / Neutral + Portfolio Manager | 三方辩论 |

### 5.2 分析师节点范式

以 `analysts/market_analyst.py` 为代表：工厂函数 `create_xxx(llm)` 返回一个 `node(state)` 闭包。内部：
1. 用 `ChatPromptTemplate` 拼系统提示（含 A 股特殊规则：涨跌停/T+1/北向/换手率/量价，及「必采清单」）；
2. `get_language_instruction()` 追加输出语言指令（用户面向的报告用中文，内部辩论保持英文以保推理质量）；
3. `llm.bind_tools(tools)` 绑定该角色可用工具，`chain.invoke(state["messages"])`；
4. 无 `tool_calls` 时把 `result.content` 写入对应报告字段（如 `market_report`）。

各角色的工具集合在 `trading_graph._create_tool_nodes()` 里按角色装配（如 hot_money 拥有龙虎榜、资金流、北向、概念板块等 A 股专属工具）。

### 5.3 数据质量门控（`quality_gate.py`）

夹在最后一个分析师与多空辩论之间，两层校验：
- **硬检查（代码，A~F 评级）**：报告是否为空/过短/主要由失败信息构成、是否含汇总表格、`[数据缺失]` 次数。
- **LLM 复审**：≥4 份报告未过硬检查则跳过；否则一次 LLM 调用产出审核表。
- 结果写入 `data_quality_summary`，供辩论阶段参考数据可信度。

### 5.4 统一 5 档评级（`utils/rating.py`）

`Buy / Overweight / Hold / Underweight / Sell`，由 Research Manager、Portfolio Manager、信号处理、记忆日志**共用**同一词表与确定性解析器，避免各处评级漂移。结构化输出定义在 `agents/schemas.py`。

---

## 6. 数据层（`tradingagents/dataflows/`）

### 6.1 Vendor 路由

工具函数（如 `core_stock_tools.get_stock_data`）用 `@tool` 装饰后，函数体只调用 `route_to_vendor("get_stock_data", ...)`。路由逻辑在 `interface.py`：

```
route_to_vendor(method)
  → get_category_for_method → 找到 category
  → get_vendor(category, method)   # tool 级配置 > category 级配置
  → 按 VENDOR_METHODS[method] 里的 {a_stock, alpha_vantage, yfinance} 依次尝试
  → 仅 AlphaVantageRateLimitError 触发 fallback 到下一 vendor
```

配置在 `default_config.py` 的 `data_vendors`（category 级）与 `tool_vendors`（tool 级，优先）。A 股场景全部指向 `a_stock`；`signal_data` 类（题材归属/资金流/一致预期/龙虎榜/解禁/行业对比）**仅 a_stock** 实现。

### 6.2 A 股数据 vendor（`a_stock.py`，2140 行，核心）

v0.2.5 起零第三方 DB，全部直连 HTTP/TCP：

| 来源 | 协议 | 数据 |
|------|------|------|
| mootdx | TCP 7709 | OHLCV K线、财务快照、F10 |
| 腾讯财经 | HTTP | PE/PB/市值/换手率 |
| 东方财富 datacenter | HTTP | 龙虎榜、限售解禁、板块行情 |
| 东方财富 push2/push2his | HTTP | 实时行情、个股信息、资金流 |
| 东方财富 np-weblist | HTTP | 滚动新闻 |
| 新浪财经 | HTTP | K线历史、财报三表 |
| 同花顺 10jqka | HTTP | EPS 一致预期、热股题材 |
| 财联社 | HTTP | 全球财经快讯 |
| 百度股市通 | HTTP | 概念板块归属 |

**两个关键收口点**：
- **`_em_get()`** — 所有东财请求的统一节流入口：模块级串行限流（`EM_MIN_INTERVAL` 默认 1.0s）+ 随机抖动 + `requests.Session` 复用 + 默认 UA。防止批量多 Agent 触发东财封 IP。**新增东财端点必须走它**。
- **`resolve_ticker()` + `_build_name_code_map()`** — 中文股票名 → 6 位代码（mootdx 全市场映射，缓存）。

### 6.3 其他 vendor

`y_finance.py` / `yfinance_news.py`（美股/港股 + 反思阶段的收益核验）、`alpha_vantage_*.py`（备用海外源）。三套 vendor 通过 §6.1 的路由并存。

---

## 7. 状态、持久化与记忆

### 7.1 图状态 `AgentState`（`agents/utils/agent_states.py`）

继承 LangGraph `MessagesState`，累积字段：7 份分析师报告、`data_quality_summary`、`investment_debate_state`（多空辩论子状态）、`investment_plan`、`trader_investment_plan`、`risk_debate_state`（风险辩论子状态）、`final_trade_decision`、`past_context`（启动时注入的历史记忆）。

### 7.2 断点续跑（`graph/checkpointer.py`，可选）

`config["checkpoint_enabled"]=True` 时启用：
- **每个 ticker 一个 SQLite 库**（`checkpoints/<CODE>.db`），避免并发争用；
- `thread_id = sha256(ticker:date)[:16]`，同票同日恢复、不同日期从头；
- `checkpoint_step()` 读最新步数决定是否续跑；成功完成后 `clear_checkpoint`。
- Web UI 默认开启（`app.py` 里 `checkpoint_enabled=True`），配合「未完成任务」列表续跑。

### 7.3 交易记忆与延迟反思（`agents/utils/memory.py`）

- 每次决策先存为 **pending**（尚无真实收益）；
- 下次跑**同一票**时，`_resolve_pending_entries` 用 yfinance 拉持有期收益（对比沪深300 `000300.SS` 算 alpha），交给 `Reflector` 生成反思，批量写回记忆；
- 记忆落在 `analysis_data/memory/trading_memory.md`，启动时作为 `past_context` 注入。

### 7.4 产物落盘

- 完整状态 JSON：`analysis_data/logs/<代码>/TradingAgentsStrategy_logs/full_states_log_<日期>.json`（**同票同日重跑覆盖**）；
- 路径由 `.env` 的 `TRADINGAGENTS_RESULTS_DIR` / `TRADINGAGENTS_MEMORY_LOG_PATH` 指向（相对路径，**须从仓库根启动**）；
- ticker 经 `safe_ticker_component` 做路径安全校验，防目录穿越。

---

## 8. LLM 抽象层（`tradingagents/llm_clients/`）

### 8.1 工厂 + 适配器

`create_llm_client(provider, model, base_url, **kwargs)`（`factory.py`）懒加载对应 client：

- **OpenAI 兼容一族**（`OpenAIClient`）：`openai / xai / deepseek / qwen / glm / ollama / openrouter / minimax` —— 统一走 Chat Completions；OpenAI 官方直连时才启用 Responses API。
- **`AnthropicClient`** / **`GoogleClient`**（可选 `[google]` 依赖）/ **`AzureOpenAIClient`**。
- 每个 client 是 `BaseLLMClient` 子类，`get_llm()` 返回归一化后的 LangChain Chat 模型（`Normalized*` 子类统一 content 处理，兼容各家差异如 DeepSeek reasoning、结构化输出）。

### 8.2 端点解析（`endpoints.py`）

`resolve_base_url(provider, explicit)` 单一收口，优先级：
```
显式输入(Web/CLI) > 供应商专属 env(OPENAI_BASE_URL/ANTHROPIC_BASE_URL) > BACKEND_URL > 官方默认
```
支持第三方中转网关（v0.2.17）。`llm_api_key` 可由 Web/CLI 输入覆盖 env。

### 8.3 思考强度

按供应商透传：`google_thinking_level` / `openai_reasoning_effort` / `anthropic_effort`（`build_llm_kwargs`）。

---

## 9. 交互层

### 9.1 Web（`web/`，Streamlit）

- **`app.py`** — 页面主状态机，按 `viewing_history` / `tracker` 状态渲染：欢迎页 / 运行中进度 / 完成报告 / 历史报告 / 出错续跑。
- **`runner.py`** — `run_analysis_in_thread` 在**守护线程**跑流水线，`graph.stream` 逐 chunk 更新 `ProgressTracker`。
- **`progress.py`** — `ProgressTracker`：线程安全的进度/暂停/停止/统计容器（`threading.Lock` + `Event` 暂停闸门），12 个流水线阶段。
- **`history.py`** — 完成历史扫描 + 未完成任务索引（`incomplete_tasks.json`，原子写、线程安全）。
- **`components/`** — sidebar（输入/模型配置/历史/未完成任务）、progress_panel、report_viewer。
- **`pdf_export.py`** — 报告导出 Markdown/PDF。
- 启动：仓库根目录 `streamlit run web/app.py` 或 `tradingagents-web`。

### 9.2 CLI（`cli/`，Typer）

`cli/main.py`（1260 行）用 `questionary` 交互选择股票、日期、供应商/模型、分析师子集，`rich` 渲染进度；`stats_handler.py` 收集 LLM/工具调用统计（callback）。入口命令 `tradingagents`。

---

## 10. 端到端数据流（一次分析）

```
用户输入(代码/中文名) ─ resolve_ticker → 6位代码
        │
        ▼  TradingAgentsGraph.propagate(code, date)
   prepare_graph_run: 解析历史 pending 收益 → 反思 → 注入 past_context → 初始 AgentState
        │
        ▼  graph.stream(state)
   [7 分析师] 每人: LLM 决策 → @tool → route_to_vendor → a_stock._em_get/mootdx → 报告字段
        │
        ▼  Quality Gate（硬检查 + LLM 复审）→ data_quality_summary
        ▼  多空辩论 → Research Manager（投资计划+评级）
        ▼  Trader → 三方风险辩论 → Portfolio Manager（最终决策+5档评级）
        │
        ▼  finalize_graph_run
   落盘 full_states_log_<date>.json · 存决策入记忆(pending) · 清断点
        │
        ▼  process_signal → parse_rating → Buy/Overweight/Hold/Underweight/Sell
```

---

## 11. 关键设计决策与约束

| 决策 | 原因 |
|------|------|
| 数据层零重型依赖，全直连 HTTP | 避免 akshare/第三方 DB 的版本与稳定性拖累（v0.2.5） |
| 东财请求统一走 `_em_get` 节流 | 多 Agent 批量取数易触发东财封 IP（v0.2.11） |
| `safe_ticker_component` 路径安全边界 | ticker 用作文件路径组件，防目录穿越；改动须慎重 |
| 双 LLM（quick/deep）分工 | 平衡成本与推理质量 |
| 内部辩论英文、报告中文 | 保推理质量同时输出本地化报告 |
| 断点 + 延迟反思 | 长流水线抗崩溃；用真实收益闭环改进决策 |
| LLM 端点单点解析 `resolve_base_url` | 多入口共享同一优先级，支持第三方网关 |
| google 依赖设为可选 `[google]` | mootdx 锁 httpx==0.25.2 与 langchain-google-genai 冲突（v0.2.6） |

---

## 12. 目录结构速查

```
tradingagents/
├── graph/                    # 编排层
│   ├── trading_graph.py      # 主类 TradingAgentsGraph（入口）
│   ├── setup.py              # LangGraph 拓扑装配
│   ├── conditional_logic.py  # 条件边（分析师循环、辩论轮次）
│   ├── checkpointer.py       # 断点续跑（per-ticker SQLite）
│   ├── propagation.py        # 初始状态与运行参数
│   ├── reflection.py         # 反思生成
│   └── signal_processing.py  # 评级提取
├── agents/                   # Agent 层
│   ├── analysts/             # 7 个分析师
│   ├── researchers/          # 多空研究员
│   ├── risk_mgmt/            # 三方风险辩论
│   ├── managers/             # 研究经理 + 组合经理
│   ├── trader/               # 交易员
│   ├── quality_gate.py       # 数据质量门控
│   ├── schemas.py            # 结构化输出定义
│   └── utils/                # 工具封装、状态、记忆、评级
├── dataflows/                # 数据层
│   ├── interface.py          # vendor 路由（route_to_vendor）
│   ├── a_stock.py            # A 股数据 vendor（核心，2140行）
│   ├── utils.py              # safe_ticker_component + 中文解析
│   ├── y_finance.py / alpha_vantage_*.py  # 其他 vendor
│   └── config.py             # 运行时配置读写
├── llm_clients/              # LLM 抽象层
│   ├── factory.py            # create_llm_client
│   ├── endpoints.py          # resolve_base_url
│   ├── {openai,anthropic,google,azure}_client.py
│   └── model_catalog.py      # 各供应商模型选项
└── default_config.py         # 默认配置
web/                          # Streamlit Web UI
cli/                          # Typer CLI
analysis_data/                # 分析产物 + 记忆（纳入 git）
tests/                        # pytest 测试
```

---

## 13. 相关文档

- `README.md` — 安装与使用
- `CLAUDE.md` — 开发规范、已知问题、注意事项（权威）
- `CHANGELOG.md` / `DEV_LOG.md` / `CHANGES_FROM_UPSTREAM.md` — 变更历史与相对上游的差异
- `issues/` — GitHub Issue 归档（根因与修复记录）
- `docs/superpowers/` — 设计规格与实现计划
