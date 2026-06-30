# 自定义 LLM 端点接入（Claude / GPT 非官方 API）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 openai / anthropic 两个 provider 能在 Web / CLI / .env 任一入口填入自定义 base_url + model ID + api_key，并修复 GPT 经网关访问时强制 Responses API 报错的问题。

**Architecture:** 在 LLM 客户端层做两件事——(1) 新增集中式 base_url 解析（显式 > provider 专属 env > 通用 BACKEND_URL > 官方），放在 `trading_graph` 单点调用，覆盖所有入口；(2) `openai_client` 仅在官方直连（无 base_url）时启用 Responses API。新增贯穿的 `llm_api_key` 配置项，Web 加自定义模型 ID + API Key 输入框，CLI 加自定义 base_url 提问 + 自定义模型选项 + API Key 提问。

**Tech Stack:** Python 3.10+，langchain-openai / langchain-anthropic，Streamlit，questionary，pytest（unittest 风格 + `@pytest.mark.unit`）。

## Global Constraints

- Python >= 3.10。
- 所有新增配置项**可选**；留空 = 现有行为，不得破坏其他 provider（deepseek/qwen/glm/minimax/openrouter/ollama/azure/google）。
- 不动图编排（`tradingagents/graph/setup.py` 节点拓扑）、数据层（`tradingagents/dataflows/`）、Agent 角色。
- 显式输入（UI/CLI）优先于环境变量；env 专属变量优先于通用 `BACKEND_URL`。
- 单元测试不得发真实网络请求：patch `NormalizedChat*` 类或 `create_llm_client`，断言传入 kwargs。
- 测试文件统一放 `tests/`，类上标 `@pytest.mark.unit`，参照 `tests/test_google_api_key.py` 风格。
- 跑测试命令：`python -m pytest tests/ -v`。

---

### Task 1: base_url 集中解析器

**Files:**
- Create: `tradingagents/llm_clients/endpoints.py`
- Test: `tests/test_custom_llm_endpoint.py`

**Interfaces:**
- Produces: `resolve_base_url(provider: str, explicit: str | None = None) -> str | None`
  优先级：`explicit(非空) > {OPENAI_BASE_URL|ANTHROPIC_BASE_URL}(按 provider) > BACKEND_URL > None`。

- [ ] **Step 1: 写失败测试**

在 `tests/test_custom_llm_endpoint.py` 创建：

```python
import os
import unittest
from unittest.mock import patch

import pytest

from tradingagents.llm_clients.endpoints import resolve_base_url


@pytest.mark.unit
class TestResolveBaseUrl(unittest.TestCase):
    def test_explicit_wins(self):
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://env/v1", "BACKEND_URL": "https://generic/v1"}, clear=False):
            self.assertEqual(resolve_base_url("openai", "https://explicit/v1"), "https://explicit/v1")

    def test_provider_env_over_backend(self):
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://oai/v1", "BACKEND_URL": "https://generic/v1"}, clear=False):
            self.assertEqual(resolve_base_url("openai", None), "https://oai/v1")
        with patch.dict(os.environ, {"ANTHROPIC_BASE_URL": "https://ant", "BACKEND_URL": "https://generic/v1"}, clear=False):
            self.assertEqual(resolve_base_url("anthropic", None), "https://ant")

    def test_backend_url_fallback(self):
        with patch.dict(os.environ, {"BACKEND_URL": "https://generic/v1"}, clear=True):
            self.assertEqual(resolve_base_url("openai", None), "https://generic/v1")

    def test_none_when_nothing_set(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_base_url("openai", None))
            self.assertIsNone(resolve_base_url("openai", "   "))

    def test_unknown_provider_uses_backend_only(self):
        with patch.dict(os.environ, {"BACKEND_URL": "https://generic/v1"}, clear=True):
            self.assertEqual(resolve_base_url("deepseek", None), "https://generic/v1")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_custom_llm_endpoint.py -v`
Expected: FAIL（`ModuleNotFoundError: tradingagents.llm_clients.endpoints`）

- [ ] **Step 3: 写实现**

创建 `tradingagents/llm_clients/endpoints.py`：

```python
"""Resolve the LLM endpoint base URL across explicit input and env vars.

A single place so every entry point (Web, CLI, direct `TradingAgentsGraph`
usage in main.py) shares the same precedence. Native dual-format gateways
expose Claude and GPT under different URLs, so each provider has its own
env var; a generic BACKEND_URL stays as a catch-all fallback.
"""

import os
from typing import Optional

# Provider-specific base URL env vars (checked before the generic fallback).
_PROVIDER_BASE_URL_ENV = {
    "openai": "OPENAI_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
}


def resolve_base_url(provider: str, explicit: Optional[str] = None) -> Optional[str]:
    """Return the base URL to use, or None to fall back to the official endpoint.

    Precedence: explicit (UI/CLI input) > provider-specific env
    (OPENAI_BASE_URL / ANTHROPIC_BASE_URL) > generic BACKEND_URL > None.
    """
    if explicit and explicit.strip():
        return explicit.strip()

    env_name = _PROVIDER_BASE_URL_ENV.get((provider or "").lower())
    if env_name:
        val = os.getenv(env_name)
        if val and val.strip():
            return val.strip()

    backend = os.getenv("BACKEND_URL")
    if backend and backend.strip():
        return backend.strip()

    return None
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_custom_llm_endpoint.py -v`
Expected: PASS（5 个用例）

- [ ] **Step 5: 提交**

```bash
git add tradingagents/llm_clients/endpoints.py tests/test_custom_llm_endpoint.py
git commit -m "feat(llm): add resolve_base_url with explicit > provider-env > BACKEND_URL precedence"
```

---

### Task 2: OpenAI Responses API 仅官方直连启用

**Files:**
- Modify: `tradingagents/llm_clients/openai_client.py:179-180`
- Test: `tests/test_custom_llm_endpoint.py`（追加类）

**Interfaces:**
- Consumes: `OpenAIClient(model, base_url=None, provider="openai", **kwargs).get_llm()`（现有）
- Produces: 行为变更——`use_responses_api` 仅当 `provider=="openai"` 且 `base_url` 为空时出现在 kwargs。

- [ ] **Step 1: 写失败测试**

在 `tests/test_custom_llm_endpoint.py` 追加：

```python
from tradingagents.llm_clients.openai_client import OpenAIClient


@pytest.mark.unit
class TestOpenAIResponsesApiGating(unittest.TestCase):
    @patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI")
    def test_custom_base_url_disables_responses_api(self, mock_chat):
        client = OpenAIClient("gpt-5.4", base_url="https://gw.example.com/v1",
                              provider="openai", api_key="k")
        client.get_llm()
        call_kwargs = mock_chat.call_args[1]
        self.assertNotIn("use_responses_api", call_kwargs)
        self.assertEqual(call_kwargs.get("base_url"), "https://gw.example.com/v1")
        self.assertEqual(call_kwargs.get("api_key"), "k")

    @patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI")
    def test_official_endpoint_keeps_responses_api(self, mock_chat):
        client = OpenAIClient("gpt-5.4", base_url=None, provider="openai", api_key="k")
        client.get_llm()
        call_kwargs = mock_chat.call_args[1]
        self.assertTrue(call_kwargs.get("use_responses_api"))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_custom_llm_endpoint.py::TestOpenAIResponsesApiGating -v`
Expected: FAIL（`test_custom_base_url_disables_responses_api`：`use_responses_api` 仍被无条件加入）

- [ ] **Step 3: 写实现**

`tradingagents/llm_clients/openai_client.py`，将现有：

```python
        # Native OpenAI: use Responses API for consistent behavior across
        # all model families. Third-party providers use Chat Completions.
        if self.provider == "openai":
            llm_kwargs["use_responses_api"] = True
```

改为：

```python
        # Native OpenAI (official endpoint): use Responses API for consistent
        # behavior across all model families. A custom base_url means a
        # third-party gateway that typically only speaks /v1/chat/completions,
        # so fall back to Chat Completions there.
        if self.provider == "openai" and not self.base_url:
            llm_kwargs["use_responses_api"] = True
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_custom_llm_endpoint.py -v`
Expected: PASS（全部）

- [ ] **Step 5: 提交**

```bash
git add tradingagents/llm_clients/openai_client.py tests/test_custom_llm_endpoint.py
git commit -m "fix(llm): only enable OpenAI Responses API on official endpoint, not custom gateways"
```

---

### Task 3: `llm_api_key` 配置 + trading_graph 接线（api_key 注入 + base_url 集中解析）

**Files:**
- Modify: `tradingagents/default_config.py:14-23`（LLM settings 区块）
- Modify: `tradingagents/graph/trading_graph.py:87-105`（client 创建）与 `141-161`（`_get_provider_kwargs`）
- Test: `tests/test_custom_llm_endpoint.py`（追加类）

**Interfaces:**
- Produces: 模块级 `build_llm_kwargs(config: dict) -> dict`，返回 provider 专属 thinking/effort kwargs，并在 `config["llm_api_key"]` 非空时含 `"api_key"`。
- Consumes: `resolve_base_url`（Task 1）、`OpenAIClient`/`AnthropicClient` 的 `_PASSTHROUGH_KWARGS`(`api_key`)（现有）。

- [ ] **Step 1: 写失败测试**

在 `tests/test_custom_llm_endpoint.py` 追加：

```python
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import build_llm_kwargs


@pytest.mark.unit
class TestBuildLlmKwargs(unittest.TestCase):
    def test_default_config_has_llm_api_key(self):
        self.assertIn("llm_api_key", DEFAULT_CONFIG)
        self.assertIsNone(DEFAULT_CONFIG["llm_api_key"])

    def test_api_key_injected_when_set(self):
        cfg = {"llm_provider": "openai", "llm_api_key": "sk-custom"}
        self.assertEqual(build_llm_kwargs(cfg).get("api_key"), "sk-custom")

    def test_api_key_absent_when_empty(self):
        cfg = {"llm_provider": "openai", "llm_api_key": None}
        self.assertNotIn("api_key", build_llm_kwargs(cfg))

    def test_provider_effort_preserved(self):
        cfg = {"llm_provider": "anthropic", "anthropic_effort": "high"}
        self.assertEqual(build_llm_kwargs(cfg).get("effort"), "high")
        cfg2 = {"llm_provider": "openai", "openai_reasoning_effort": "low"}
        self.assertEqual(build_llm_kwargs(cfg2).get("reasoning_effort"), "low")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_custom_llm_endpoint.py::TestBuildLlmKwargs -v`
Expected: FAIL（`ImportError: cannot import name 'build_llm_kwargs'` 及 `llm_api_key` 缺失）

- [ ] **Step 3: 写实现**

3a. `tradingagents/default_config.py`，在 `"anthropic_effort": None,` 一行之后加：

```python
    # Optional API key for the selected provider. When set, it is forwarded
    # to the LLM client (used with custom gateways). When None, the client
    # reads the provider's own env var (OPENAI_API_KEY / ANTHROPIC_API_KEY / ...).
    "llm_api_key": None,
```

3b. `tradingagents/graph/trading_graph.py`，在文件 import 区加（紧随 `from tradingagents.llm_clients import create_llm_client` 之后）：

```python
from tradingagents.llm_clients.endpoints import resolve_base_url
```

3c. 在 `TradingAgentsGraph` 类外、`class TradingAgentsGraph:` 定义之前，新增模块级函数：

```python
def build_llm_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build kwargs forwarded to create_llm_client from config.

    Includes provider-specific thinking/effort settings plus an optional
    unified api_key (forwarded to the langchain client for custom gateways).
    """
    kwargs: Dict[str, Any] = {}
    provider = (config.get("llm_provider") or "").lower()

    if provider == "google":
        if config.get("google_thinking_level"):
            kwargs["thinking_level"] = config["google_thinking_level"]
    elif provider == "openai":
        if config.get("openai_reasoning_effort"):
            kwargs["reasoning_effort"] = config["openai_reasoning_effort"]
    elif provider == "anthropic":
        if config.get("anthropic_effort"):
            kwargs["effort"] = config["anthropic_effort"]

    api_key = config.get("llm_api_key")
    if api_key:
        kwargs["api_key"] = api_key

    return kwargs
```

3d. 在 `__init__` 中，将现有：

```python
        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
```

替换为：

```python
        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = build_llm_kwargs(self.config)

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        # Resolve base URL once here (single source of truth across entry
        # points): explicit config > provider env > BACKEND_URL > official.
        base_url = resolve_base_url(
            self.config["llm_provider"], self.config.get("backend_url")
        )

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=base_url,
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=base_url,
            **llm_kwargs,
        )
```

3e. 删除现有 `_get_provider_kwargs` 方法（`def _get_provider_kwargs(self)...` 整段，约 `trading_graph.py:141-161`），其逻辑已并入 `build_llm_kwargs`。确认类内无其它调用点：

Run: `grep -n "_get_provider_kwargs" tradingagents/graph/trading_graph.py`
Expected: 无输出（已全部移除）

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_custom_llm_endpoint.py -v`
Expected: PASS（全部）

- [ ] **Step 5: 回归 + 提交**

Run: `python -m pytest tests/ -v`
Expected: 全绿（无新增失败）

```bash
git add tradingagents/default_config.py tradingagents/graph/trading_graph.py tests/test_custom_llm_endpoint.py
git commit -m "feat(llm): add llm_api_key config and centralize base_url resolution in trading_graph"
```

---

### Task 4: Web 入口 — 自定义模型 ID + API Key 输入框

**Files:**
- Modify: `web/components/sidebar.py:133-191`（`_render_llm_config`）
- Modify: `web/app.py:159-178`（`_build_config`）

**Interfaces:**
- Consumes: `st.session_state["llm_provider" | "quick_think_llm" | "deep_think_llm" | "llm_base_url" | "llm_api_key"]`
- Produces: `config["llm_api_key"]`（非空字符串或 None）；`config["backend_url"]` 仍为用户输入的 base_url 原值（解析在 trading_graph，Task 3）。

- [ ] **Step 1: 改 sidebar 模型选择，加自定义模型 ID 哨兵**

`web/components/sidebar.py`，将现有 `if provider_key in MODEL_OPTIONS:` 分支（行 146-176）替换为：

```python
    _CUSTOM = "__custom__"
    if provider_key in MODEL_OPTIONS:
        quick_options = list(MODEL_OPTIONS[provider_key]["quick"]) + [("✏️ 自定义模型 ID", _CUSTOM)]
        deep_options = list(MODEL_OPTIONS[provider_key]["deep"]) + [("✏️ 自定义模型 ID", _CUSTOM)]

        quick_labels = [label for label, _ in quick_options]
        quick_values = [value for _, value in quick_options]
        deep_labels = [label for label, _ in deep_options]
        deep_values = [value for _, value in deep_options]

        quick_idx = st.selectbox(
            "快速思考模型",
            range(len(quick_options)),
            format_func=lambda i: quick_labels[i],
            key="quick_model_idx",
            help="用于常规分析任务，速度优先；选「自定义模型 ID」可填写网关上的模型名",
        )
        if quick_values[quick_idx] == _CUSTOM:
            st.session_state["quick_think_llm"] = st.text_input(
                "快速思考模型 ID", key="custom_quick_model",
                placeholder="例: claude-3-5-sonnet-20241022 或 gpt-4o",
            ).strip()
        else:
            st.session_state["quick_think_llm"] = quick_values[quick_idx]

        deep_idx = st.selectbox(
            "深度思考模型",
            range(len(deep_options)),
            format_func=lambda i: deep_labels[i],
            key="deep_model_idx",
            help="用于辩论/决策等需要深度推理的任务；选「自定义模型 ID」可填写网关上的模型名",
        )
        if deep_values[deep_idx] == _CUSTOM:
            st.session_state["deep_think_llm"] = st.text_input(
                "深度思考模型 ID", key="custom_deep_model",
                placeholder="例: claude-3-5-sonnet-20241022 或 gpt-4o",
            ).strip()
        else:
            st.session_state["deep_think_llm"] = deep_values[deep_idx]
    else:
        custom_quick = st.text_input("快速思考模型 ID", key="custom_quick_model")
        custom_deep = st.text_input("深度思考模型 ID", key="custom_deep_model")
        st.session_state["quick_think_llm"] = custom_quick
        st.session_state["deep_think_llm"] = custom_deep
```

- [ ] **Step 2: 加 API Key 密码框**

`web/components/sidebar.py`，在 `_render_llm_config` 末尾的 base_url `st.text_input(...)` 之后追加：

```python
    st.text_input(
        "API Key（可选，留空用 .env）",
        key="llm_api_key",
        type="password",
        placeholder="留空则用环境变量 OPENAI_API_KEY / ANTHROPIC_API_KEY 等",
        help=(
            "经第三方网关访问时可直接填入该网关的 Key；留空则从 .env 按供应商读取。"
            "双格式网关（Claude 与 GPT 地址不同）可在 .env 设 OPENAI_BASE_URL / "
            "ANTHROPIC_BASE_URL，切换供应商即自动切端点。"
        ),
    )
```

- [ ] **Step 3: app.py 写入 config**

`web/app.py`，在 `_build_config()` 中 `config["backend_url"] = backend_url or None` 之后追加：

```python
    config["llm_api_key"] = (st.session_state.get("llm_api_key") or "").strip() or None
```

（`backend_url` 行保持不变——其值作为「显式输入」交给 trading_graph 的 `resolve_base_url` 处理。）

- [ ] **Step 4: 手动验证**

Run: `streamlit run web/launch.py`
验证步骤（在浏览器侧边栏「⚙️ 模型配置」）：
1. 选 Anthropic → 深/快速模型下拉末尾出现「✏️ 自定义模型 ID」，选中后出现文本框，可填 `claude-3-5-sonnet-20241022`。
2. 出现「API Key（可选）」密码框，输入字符显示为圆点。
3. 填 base_url + 自定义模型 + key 后点开始分析，不报 `/v1/responses` 相关错误（确认走 Chat Completions / Messages）。
Expected: 三项均符合；留空 key 时仍能用 .env 跑通。

- [ ] **Step 5: 提交**

```bash
git add web/components/sidebar.py web/app.py
git commit -m "feat(web): custom model ID option + API key field for openai/anthropic gateways"
```

---

### Task 5: CLI 入口 — 自定义 base_url + 自定义模型 + API Key

**Files:**
- Modify: `cli/utils.py:195-229`（`_select_model`）、`:241-278`（`select_llm_provider`），新增 `ask_llm_api_key()`
- Modify: `cli/main.py:557`、`:599-612`（selections）、`:954`（config 组装），新增 `config["llm_api_key"]`

**Interfaces:**
- Produces: `select_llm_provider() -> tuple[str, str | None]`（第二项改为「用户输入的自定义 base_url 或 None」，不再回传官方硬编码 URL）；`ask_llm_api_key() -> str | None`。
- Consumes: 现有 `_prompt_custom_model_id()`、`choice == "custom"` 分支。

- [ ] **Step 1: `_select_model` 给目录型 provider 追加自定义项**

`cli/utils.py`，在 `_select_model` 中构造 `choice = questionary.select(...)` 处，将 choices 由：

```python
        choices=[
            questionary.Choice(display, value=value)
            for display, value in get_model_options(provider, mode)
        ],
```

改为（追加 Custom 项；已有 `if choice == "custom": return _prompt_custom_model_id()` 分支会接住）：

```python
        choices=[
            questionary.Choice(display, value=value)
            for display, value in get_model_options(provider, mode)
        ] + [questionary.Choice("Custom model ID", value="custom")],
```

- [ ] **Step 2: `select_llm_provider` 改为询问自定义 base_url**

`cli/utils.py`，将 `select_llm_provider` 的 `PROVIDERS` 列表与返回逻辑替换为：

```python
def select_llm_provider() -> tuple[str, str | None]:
    """Select the LLM provider, optionally with a custom/gateway base URL."""
    PROVIDERS = [
        ("OpenAI", "openai"),
        ("Google", "google"),
        ("Anthropic", "anthropic"),
        ("xAI", "xai"),
        ("DeepSeek", "deepseek"),
        ("Qwen", "qwen"),
        ("GLM", "glm"),
        ("OpenRouter", "openrouter"),
        ("Azure OpenAI", "azure"),
        ("Ollama", "ollama"),
    ]

    choice = questionary.select(
        "Select your LLM Provider:",
        choices=[
            questionary.Choice(display, value=provider_key)
            for display, provider_key in PROVIDERS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No LLM provider selected. Exiting...[/red]")
        exit(1)

    use_custom = questionary.confirm(
        "Use a custom / proxy base URL (third-party gateway)?",
        default=False,
    ).ask()

    base_url = None
    if use_custom:
        base_url = questionary.text(
            "Enter base URL (e.g. https://your-gateway.com/v1):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a URL.",
        ).ask().strip()

    return choice, base_url
```

注意：移除了原先「硬编码官方 URL」。`base_url=None` 时由 `resolve_base_url`（Task 3）回退到 provider env / `BACKEND_URL` / 官方默认；保留官方时 openai 仍走 Responses API（base_url 为 None）。

- [ ] **Step 3: 新增 `ask_llm_api_key`**

`cli/utils.py`，在 `ask_anthropic_effort` 附近新增：

```python
def ask_llm_api_key() -> str | None:
    """Optionally enter an API key for the selected provider (custom gateway).

    Empty input keeps the .env / environment-variable behaviour.
    """
    val = questionary.password(
        "Enter API Key (optional; leave blank to use .env):",
    ).ask()
    val = (val or "").strip()
    return val or None
```

- [ ] **Step 4: `cli/main.py` 串联**

4a. 在 `cli/main.py` 顶部 import 处确认 `ask_llm_api_key` 已随 `from cli.utils import *` 或显式导入引入；若为显式 import 列表，加入 `ask_llm_api_key`。

Run: `grep -n "from cli.utils import\|ask_anthropic_effort" cli/main.py | head`
据结果：若是 `from cli.utils import (...)` 列表，则在其中加 `ask_llm_api_key`。

4b. 在 `get_user_selections` 中，Step 8 provider-specific 配置之后、`return {` 之前，加入：

```python
    # Step 9: Optional API key (for custom gateways)
    console.print(
        create_question_box(
            "Step 9: API Key", "Optionally provide an API key (blank = use .env)"
        )
    )
    llm_api_key = ask_llm_api_key()
```

4c. 在 `get_user_selections` 的返回 dict 中加一行：

```python
        "llm_api_key": llm_api_key,
```

4d. 在 config 组装处（约 `cli/main.py:955` `config["llm_provider"] = ...` 之后）加：

```python
    config["llm_api_key"] = selections.get("llm_api_key")
```

- [ ] **Step 5: 写测试（非交互逻辑）**

在 `tests/test_custom_llm_endpoint.py` 追加，验证 `_select_model` 自定义分支与 `ask_llm_api_key` 归一化：

```python
from unittest.mock import MagicMock


@pytest.mark.unit
class TestCliCustomModelAndKey(unittest.TestCase):
    @patch("cli.utils._prompt_custom_model_id", return_value="claude-custom")
    @patch("cli.utils.questionary")
    def test_select_model_custom_branch(self, mock_q, mock_prompt):
        mock_q.select.return_value.ask.return_value = "custom"
        from cli.utils import _select_model
        self.assertEqual(_select_model("anthropic", "deep"), "claude-custom")

    @patch("cli.utils.questionary")
    def test_ask_llm_api_key_blank_returns_none(self, mock_q):
        mock_q.password.return_value.ask.return_value = "   "
        from cli.utils import ask_llm_api_key
        self.assertIsNone(ask_llm_api_key())

    @patch("cli.utils.questionary")
    def test_ask_llm_api_key_value(self, mock_q):
        mock_q.password.return_value.ask.return_value = "sk-x"
        from cli.utils import ask_llm_api_key
        self.assertEqual(ask_llm_api_key(), "sk-x")
```

- [ ] **Step 6: 跑测试确认通过**

Run: `python -m pytest tests/test_custom_llm_endpoint.py -v`
Expected: PASS（含 CLI 三个用例）

- [ ] **Step 7: 手动冒烟 + 提交**

Run: `tradingagents`（走到 Step 6-9，选 Anthropic → 确认可填自定义 base_url、选 Custom model ID 填模型名、可填/留空 API Key）
Expected: 流程顺畅，留空 key 时用 .env 跑通。

```bash
git add cli/utils.py cli/main.py tests/test_custom_llm_endpoint.py
git commit -m "feat(cli): prompt custom base URL + custom model ID + optional API key"
```

---

### Task 6: .env 模板与文档

**Files:**
- Modify: `.env.example`
- Modify: `CLAUDE.md`（架构/注意事项处加一行说明）

**Interfaces:** 无代码接口；纯文档。

- [ ] **Step 1: 更新 `.env.example`**

将现有 `BACKEND_URL` 段（文件末尾）替换为：

```
# 第三方中转 / 代理网关地址。
# 原生双格式网关（Claude 走 /v1/messages、GPT 走 /v1/chat/completions，
# 地址常不同）可分供应商设置，切换供应商即自动切端点：
OPENAI_BASE_URL=
ANTHROPIC_BASE_URL=
# 通用兜底（上面两个留空时对所有供应商生效）。Web 侧边栏 / CLI 也可直接填写并覆盖此处。
# 例: BACKEND_URL=https://your-proxy.com/v1
BACKEND_URL=
```

- [ ] **Step 2: 更新 `CLAUDE.md`**

在 `## 已知问题与注意事项` 区新增一小节：

```markdown
### 自定义 LLM 端点（v0.2.17）
openai / anthropic 可经第三方网关接入：base_url 优先级为 显式输入(Web/CLI) > `OPENAI_BASE_URL`/`ANTHROPIC_BASE_URL` > `BACKEND_URL` > 官方，集中在 `trading_graph.py` 的 `resolve_base_url` 解析。`openai` 仅在官方直连（无 base_url）时启用 Responses API，经网关自动改走 Chat Completions。可选 `config["llm_api_key"]`（Web/CLI 输入框）覆盖环境变量 Key。
```

- [ ] **Step 3: 回归全量测试**

Run: `python -m pytest tests/ -v`
Expected: 全绿。

- [ ] **Step 4: 提交**

```bash
git add .env.example CLAUDE.md
git commit -m "docs: document per-provider base URL env vars and custom LLM endpoint flow"
```

---

## Self-Review

**1. Spec coverage**（对照 `2026-06-30-custom-llm-endpoint-design.md`）：
- §3.1 核心通路（`llm_api_key` + base_url 优先级）→ Task 1 + Task 3 ✔（解析集中于 trading_graph，较 spec 更 DRY，效果一致）
- §3.2 OpenAI Responses API 条件化 → Task 2 ✔
- §3.3 Web 自定义模型 + API Key → Task 4 ✔
- §3.4 CLI 自定义 base_url + 模型 + Key（含移除硬编码官方 URL，C1/C2/C3）→ Task 5 ✔
- §3.5 .env 模板 → Task 6 ✔
- §4 兼容（可选、显式优先）→ 各 Task 实现 + Task 1/3 测试覆盖 ✔
- §5 测试（responses gating / api_key 注入 / base_url 优先级 / 回归）→ Task 1/2/3/5 ✔

**2. Placeholder scan:** 无 TBD/TODO；所有代码步骤含完整代码。

**3. Type consistency:** `resolve_base_url(provider, explicit)`、`build_llm_kwargs(config)`、`select_llm_provider() -> (str, str|None)`、`ask_llm_api_key() -> str|None` 在定义与调用处一致；Web 哨兵 `__custom__`、CLI 哨兵 `"custom"` 各自闭环。

无遗留问题。
