# AI Interview Agent

基于 **ReAct**（Reasoning + Acting）的面试场景 AI Agent：可阅读简历 PDF、按约定格式进行推理与工具调用，并输出对公司有价值的内容摘要。项目使用 LangChain 对接 OpenAI 兼容 API，支持自定义角色、工具与提示模板。

---

## 功能概览

- **ReAct 循环**：模型按 `Thought → Action → Action Input` 输出，解析后调用对应工具，将结果写入 Observation 并进入下一轮，直至使用 `answer` 或达到轮数上限。
- **简历阅读与筛选**：通过 `read_pdf` 工具解析 PDF 简历，由 Agent 筛选教育背景、项目经验、技能等有价值信息并输出。
- **可配置**：角色定义、工具清单（描述）、格式制约、动态上下文均在 `Prompts` 中配置；工具实现可在 `ReAct` 中注册与扩展；循环轮数可调。
- **中间过程输出**：运行时可打印每轮的模型输出、解析结果、工具调用与 Observation，便于调试。

---

## 项目结构

```
AIagent/
├── README.md           # 本说明
├── CONFIG_README.md    # 工具 / Prompts / 循环参数配置说明
├── AIagent.env         # 环境变量（API Key、Base URL、模型名等），勿提交敏感信息
├── demo.py             # 演示：互联网面试官阅读简历并筛选有价值内容
├── AgentLLM.py         # 大模型调用封装（加载 env、invoke、兼容多种 API 返回格式）
├── Prompts.py          # 提示模板：角色、工具清单、格式制约、动态上下文
├── ReAct.py            # ReAct 解析（Thought/Action/Action Input）、工具执行、主循环
├── PDFReader.py        # PDF 解析（pdfplumber），供 read_pdf 工具使用
└── help.py             # 其他辅助
```

---

## 环境与依赖

### 1. Python

建议 Python 3.10+。

### 2. 依赖安装

```bash
pip install langchain-openai openai pdfplumber
```

### 3. 环境变量（`AIagent.env`）

在项目根目录创建或修改 `AIagent.env`，例如：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
```

- **OPENAI_API_KEY**：必填，大模型 API 密钥。
- **OPENAI_BASE_URL**：可选，默认为 OpenAI 官方；若使用第三方兼容接口（如代理、Gemini 等）请填写对应 base URL。
- **MODEL_NAME**：可选，模型名称，默认 `gpt-4.1-mini`。

请勿将包含真实密钥的 `AIagent.env` 提交到版本库。

---

## 快速开始

### 运行简历筛选演示

1. 在同目录下放置至少一个 PDF 简历（如 `Resume.pdf`），或通过参数指定路径。
2. 执行：

```bash
python demo.py
# 或指定 PDF 路径
python demo.py D:/path/to/resume.pdf
```

3. 程序会以「互联网面试官」角色运行 ReAct：先调用 `read_pdf` 读取简历，再根据内容筛选有价值信息，最后用 `answer` 输出结果。控制台会打印每轮的模型输出、解析结果、工具调用与 Observation。

### 在代码中调用 ReAct

```python
from AgentLLM import AgentLLM
from Prompts import Prompts
from ReAct import get_default_tools, run_react_loop

agent = AgentLLM("AIagent.env")
prompts = Prompts()
prompts.set_context(task="阅读简历并筛选", resume_path="D:/Resume.pdf")

result = run_react_loop(
    user_input="请读取简历并筛选对公司有价值的内容，用 answer 输出。",
    agent=agent,
    prompts=prompts,
    tools=get_default_tools(),
    max_steps=3,
    verbose=True,
)
print(result)
```

---

## 模块说明

| 模块 | 说明 |
|------|------|
| **AgentLLM** | 从 `AIagent.env` 加载配置，创建 ChatOpenAI 实例；`invoke(prompt)` 返回纯文本。支持 API 返回消息块列表（如含 `type='text'` 的块）时自动提取可解析文本。 |
| **Prompts** | 封装系统提示：角色定义、工具清单（名称+描述）、格式制约（Thought/Action/Action Input）、动态上下文。`build_system_prompt()` 输出完整系统提示；`set_context(**kwargs)` 更新动态上下文。 |
| **ReAct** | 从模型输出中解析 **Thought**、**Action**、**Action Input**（纯字符串查找，不依赖复杂正则）；根据 Action 调用注册工具，得到 **Observation**；将 Action/Observation 写入 Prompts 动态上下文并循环。提供 `run_react_loop()`、`get_default_tools()`（含 `answer`、`read_pdf`）。 |
| **PDFReader** | 使用 pdfplumber 解析 PDF，提供 `read(path)`（结构化结果）、`read_text_only(path)`、`read_as_tool_result(path)`（供 ReAct 的 read_pdf 工具使用）。 |

---

## 配置与扩展

- **角色、工具描述、格式、动态上下文**：通过 `Prompts(role=..., tools_spec=..., format_constraints=...)` 与 `prompts.set_context(...)` 配置。
- **工具实现**：`get_default_tools()` 返回 `answer` 与 `read_pdf`；可在此字典上增加或覆盖工具（如 `get_current_question`、`get_interview_context`），再传入 `run_react_loop(..., tools=tools)`。工具名需与 `Prompts` 中 `tools_spec` 的 `name` 一致。
- **循环轮数**：`run_react_loop(..., max_steps=N)`，默认见 `ReAct.MAX_LOOP_STEPS`。
- 更详细的配置说明见 **[CONFIG_README.md](CONFIG_README.md)**。

---

## 输出格式约定

模型需按以下格式输出，以便 ReAct 正确解析并调用工具：

```
**Thought:** （推理过程）
**Action:** （工具名，如 read_pdf / answer）
**Action Input:** （该工具的输入，如 PDF 路径或最终回复内容）
```

当 Agent 要直接回复用户时，应使用 **Action:** `answer`，并在 **Action Input** 中填写完整回复内容。

---

## 许可证与免责声明

本项目仅供学习与内部使用。使用第三方 API 时请遵守其服务条款；请勿将 API Key 等敏感信息提交到公开仓库。
