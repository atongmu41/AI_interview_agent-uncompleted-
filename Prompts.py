"""
Prompts: 封装 ReAct 智能体的提示模板，包含角色定义、工具清单、格式制约与动态上下文。
"""

from typing import Any


# ---------------------------------------------------------------------------
# 角色定义
# ---------------------------------------------------------------------------

DEFAULT_ROLE = """你是一名专业的面试助手 Agent。你的任务是根据当前面试阶段与题目，进行推理(Thought)、选择工具(Action)，并根据观察结果(Observation)给出回复或下一步行动。
你需要严格按「格式制约」中规定的格式输出，确保可被解析。"""


# ---------------------------------------------------------------------------
# 格式制约（ReAct 输出格式）
# ---------------------------------------------------------------------------

FORMAT_CONSTRAINTS = """## 输出格式
你必须按以下格式逐轮输出，且每段必须以对应标签开头：

**Thought:** （你的推理过程，分析当前情况与下一步该做什么）
**Action:** （从工具清单中选一个工具名，仅输出工具名）
**Action Input:** （该工具所需的输入，一般为 JSON 或简短文本）
**Observation:** （本段由系统在调用工具后自动填充，你不需要写）

当你要直接回复用户、结束本轮或给出最终答案时，使用工具：
- **Action:** answer
- **Action Input:** 你的回复内容

每轮只输出一个 Thought / Action / Action Input 组合；等收到 Observation 后再继续下一轮。"""


# ---------------------------------------------------------------------------
# 默认工具清单（名称与描述，供模型选择）
# ---------------------------------------------------------------------------

DEFAULT_TOOLS_SPEC = [
    {
        "name": "answer",
        "description": "直接向用户输出回复或最终答案。当无需再调用其他工具、或要结束当前轮次时使用。",
    },
    {
        "name": "get_current_question",
        "description": "获取当前面试题目内容。当需要查看本轮的完整题目或题干时调用。",
    },
    {
        "name": "get_interview_context",
        "description": "获取当前面试的上下文：轮次、已问题目、岗位等。用于把握整体进度与考察重点。",
    },
]


def _format_tool_list(tools: list[dict[str, str]]) -> str:
    """将工具清单格式化为可放入 prompt 的文本。"""
    lines = ["## 可用工具"]
    for t in tools:
        lines.append(f"- **{t['name']}**: {t['description']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts 类：角色 + 工具 + 格式 + 动态上下文
# ---------------------------------------------------------------------------

class Prompts:
    """ReAct 智能体提示模板：角色定义、工具清单、格式制约、动态上下文。"""

    def __init__(
        self,
        role: str | None = None,
        tools_spec: list[dict[str, str]] | None = None,
        format_constraints: str | None = None,
    ) -> None:
        self._role = role or DEFAULT_ROLE
        self._tools_spec = tools_spec or DEFAULT_TOOLS_SPEC
        self._format_constraints = format_constraints or FORMAT_CONSTRAINTS
        # 动态上下文：可在运行时更新
        self._dynamic_context: dict[str, Any] = {}

    # ------------------------- 动态上下文 -------------------------

    def set_context(self, **kwargs: Any) -> None:
        """设置或更新动态上下文字段。"""
        self._dynamic_context.update(kwargs)

    def get_context(self, key: str, default: Any = None) -> Any:
        """获取动态上下文字段。"""
        return self._dynamic_context.get(key, default)

    def clear_context(self) -> None:
        """清空动态上下文。"""
        self._dynamic_context.clear()

    def _format_dynamic_context(self) -> str:
        """将动态上下文格式化为一段可插入 prompt 的文本。"""
        if not self._dynamic_context:
            return ""
        lines = ["## 当前上下文"]
        for k, v in self._dynamic_context.items():
            lines.append(f"- **{k}**: {v}")
        return "\n".join(lines)

    # ------------------------- 组件访问 -------------------------

    @property
    def role(self) -> str:
        """Agent 角色定义。"""
        return self._role

    @role.setter
    def role(self, value: str) -> None:
        self._role = value

    @property
    def tools_spec(self) -> list[dict[str, str]]:
        """工具清单（name + description）。"""
        return self._tools_spec

    def set_tools(self, tools_spec: list[dict[str, str]]) -> None:
        """替换工具清单。"""
        self._tools_spec = tools_spec

    @property
    def format_constraints(self) -> str:
        """格式制约说明。"""
        return self._format_constraints

    # ------------------------- 组装完整 prompt -------------------------

    def build_system_prompt(self) -> str:
        """组装完整的系统提示：角色 + 工具清单 + 格式制约 + 动态上下文。"""
        parts = [
            self._role,
            _format_tool_list(self._tools_spec),
            self._format_constraints,
        ]
        dynamic = self._format_dynamic_context()
        if dynamic:
            parts.append(dynamic)
        return "\n\n".join(parts)

    def build_user_prompt(self, user_input: str) -> str:
        """组装用户侧提示（当前轮的用户输入）。可在此扩展历史等。"""
        return user_input
