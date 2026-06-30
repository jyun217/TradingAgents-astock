# 设计：自定义端点接入 Claude / GPT（非官方 API）

- **日期**: 2026-06-30
- **状态**: 已批准（待实现）
- **范围**: LLM 接入层 + Web / CLI / .env 三处配置入口
- **不在范围**: 图编排（graph）、数据层（dataflows）、Agent 角色逻辑

## 1. 背景与目标

用户使用**第三方中转网关**访问 Anthropic（Claude）与 OpenAI（GPT），而非官方 API。网关为
**原生双格式**：Claude 走 Anthropic 原生 `/v1/messages`，GPT 走 OpenAI `/v1/chat/completions`，
两者的 base_url 与 api_key 可能各不相同。

使用方式为**按次切换**：单次分析只用一家（全 Claude 或全 GPT），跨次运行可切换。配置入口
**Web UI 与 .env 都要用，并顺带补齐 CLI**。

目标：让用户能在 Web / CLI / .env 任一入口方便地为 openai / anthropic 两个 provider 填入
**自定义 base_url + 自定义 model ID + 自定义 api_key**，并使经网关访问 GPT 不再因 Responses API 报错。

## 2. 现状诊断（5 个缺口）

| # | 缺口 | 位置 | 影响 |
|---|------|------|------|
| 1 | `provider==openai` 强制 `use_responses_api=True`，走 `/v1/responses` | `llm_clients/openai_client.py:179-180` | 🔴 GPT 经网关直接报错（网关只有 `/v1/chat/completions`） |
| 2 | Web 上 openai/anthropic 只有官方模型名下拉，无自定义输入（自定义框仅在不在 `MODEL_OPTIONS` 的 provider 的 `else` 分支） | `web/components/sidebar.py:146-176` | 🔴 填不了网关自定义 model ID |
| 3 | Web 无 API Key 输入框，只能靠环境变量 | `web/components/sidebar.py` | 🟡 想直接填 key，目前做不到 |
| 4 | 只有单个 `BACKEND_URL`，两家网关地址不同 | `web/app.py:165`、`.env.example` | 🟡 .env 模式切换需手改 URL |
| 5 | config 无 `api_key` 通路，全靠 langchain 读固定 env | `default_config.py`、`trading_graph.py` | 🟡 #3 的底层前提 |
| C1 | CLI `select_llm_provider()` 把 base_url 硬编码为官方地址并覆盖 `.env`，无网关地址入口 | `cli/utils.py:241-278`、`cli/main.py:557,605,954` | 🔴 CLI 完全无法用网关 |
| C2 | CLI `get_model_options` 仅返回官方目录、无 `custom` 项，openai/anthropic 无自定义 model ID | `tradingagents/llm_clients/model_catalog.py:133`、`cli/utils.py:195-229` | 🔴 CLI 填不了自定义 model ID |
| C3 | CLI 无 API Key 提问，仅靠 `.env` | `cli/utils.py`、`cli/main.py` | 🟡 想在 CLI 填 key |

**已经可用、无需改动**：`backend_url` → `base_url` 主干；anthropic client 的 base_url + api_key 透传
（`anthropic_client.py:37-42`）；未知模型只 `warn_if_unknown_model` 警告不拦截。

## 3. 设计

### 3.1 核心数据通路（底层，三入口共用）

- `tradingagents/default_config.py`：新增 `"llm_api_key": None`。
- `tradingagents/graph/trading_graph.py`：创建 deep/quick client 时，若 `config["llm_api_key"]` 非空，
  注入 `llm_kwargs["api_key"]`。`api_key` 已在 anthropic 与 openai client 的 `_PASSTHROUGH_KWARGS`
  中，自动透传至 langchain client。留空则保持现状（langchain 读各自 env var）。
- **base_url 解析优先级**（统一规则，Web/CLI/.env 共用）：
  `显式输入(UI/CLI) > 供应商专属 env(OPENAI_BASE_URL / ANTHROPIC_BASE_URL) > 通用 BACKEND_URL > None(官方)`。
  在 Web `_build_config()` 与 CLI 配置组装处各自实现该优先级（共用一个小 helper，避免重复）。

### 3.2 修 OpenAI Responses API（关键 bug）

`tradingagents/llm_clients/openai_client.py`，将：

```python
if self.provider == "openai":
    llm_kwargs["use_responses_api"] = True
```

改为仅当**未配置自定义 base_url**（即官方直连）时开启：

```python
if self.provider == "openai" and not self.base_url:
    llm_kwargs["use_responses_api"] = True
```

填了网关 → 走标准 Chat Completions，兼容 one-api / new-api 等中转。官方直连行为不变。

### 3.3 Web 入口（`web/components/sidebar.py` + `web/app.py`）

`_render_llm_config()`：
- 对**所有** provider（含 openai/anthropic）的快速/深度模型选择，在下拉选项末尾追加
  「✏️ 自定义模型 ID」；选中后显示 `st.text_input` 填 model ID。
  实现方式：在已有 `MODEL_OPTIONS` 分支中给 options 末尾加一个 `("✏️ 自定义模型 ID", "__custom__")`
  哨兵项；选中哨兵时渲染文本框并以其值覆盖 `quick_think_llm` / `deep_think_llm`。
- base_url 输入框保留，help 文案更新（说明双格式网关与 provider 专属 env）。
- 新增「API Key（可选，留空用 .env）」`st.text_input(type="password")` → `st.session_state["llm_api_key"]`。

`_build_config()`：
- `config["llm_api_key"] = (st.session_state.get("llm_api_key") or "").strip() or None`
- `config["backend_url"]` 改用 §3.1 优先级解析（显式 > provider 专属 env > BACKEND_URL > None）。

### 3.4 CLI 入口（`cli/utils.py` + `cli/main.py`）

- `select_llm_provider()`：选定 provider 后追加一步「是否使用自定义/中转地址？」
  - 是 → `questionary.text` 填 base_url（返回该值）。
  - 否 → 返回 `None`，由配置组装处按 §3.1 回退到 provider 专属 env / `BACKEND_URL` / 官方默认。
  - 注意：移除当前「硬编码官方 url 并覆盖 .env」的行为（C1）。
- `_select_model()`：给目录型 provider（openai/anthropic 等）的下拉**统一追加** `Custom model ID` 选项，
  复用已有 `choice == "custom"` → `_prompt_custom_model_id()` 分支（C2，近零成本）。
- 新增可选 API Key 提问（`questionary.password`，留空回退 .env）→ `selections["llm_api_key"]`
  → `config["llm_api_key"]`（C3）。

### 3.5 .env 模板（`.env.example`）

新增并注释：

```
# 第三方网关分供应商地址（原生双格式网关常见：Claude 与 GPT 地址不同）
OPENAI_BASE_URL=
ANTHROPIC_BASE_URL=
# 通用兜底（上面两个留空时生效）
BACKEND_URL=
```

配合 §3.1 解析：两家网关地址可并存，切 provider 即自动切端点，无需手改。

## 4. 错误处理与兼容

- 所有新增配置项**可选**；留空 = 现有行为，不破坏其他 provider（deepseek/qwen/glm/minimax/openrouter/ollama
  仍走各自 `_PROVIDER_CONFIG` 默认）。
- 自定义 model ID 触发 `warn_if_unknown_model` 仅警告不拦截，符合预期。
- api_key 同时来自 UI/CLI 与 env 时，**显式值优先**。
- base_url 同时来自显式输入与 env 时，**显式输入优先**（§3.1）。

## 5. 测试（`tests/`）

新增单元测试，不发真实网络请求（构造 client、断言 kwargs，参照现有 `tests/` 中 llm_clients 风格）：

1. `provider=openai` + 设置 base_url 时，`use_responses_api` **不**出现在 kwargs；不设 base_url 时**出现**。
2. `provider=openai` / `anthropic` 透传 `api_key` 后，构造的 client kwargs 含该 `api_key`。
3. base_url 解析优先级：显式 > provider 专属 env > `BACKEND_URL` > None（用 monkeypatch 设置 env）。
4. （回归）其他 provider（如 deepseek）行为不变。

## 6. 改动文件清单

- `tradingagents/default_config.py` — 新增 `llm_api_key`
- `tradingagents/graph/trading_graph.py` — 注入 `api_key`
- `tradingagents/llm_clients/openai_client.py` — Responses API 条件化
- `web/components/sidebar.py` — 自定义 model ID + API Key 输入
- `web/app.py` — `_build_config` 注入 key、base_url 优先级
- `cli/utils.py` — 自定义 base_url 提问、custom model 选项、API Key 提问
- `cli/main.py` — 串联 CLI 新选项至 config
- `.env.example` — 新增 provider 专属 base_url 变量
- `tests/` — 新增单元测试

**不动**：图编排（`graph/`）、数据层（`dataflows/`）、Agent 角色。
