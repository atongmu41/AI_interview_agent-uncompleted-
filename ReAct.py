"""
ReAct: 从 LLM 返回的纯文本中解析 Thought、Action、Action Input；
并调用工具，将执行结果写入 Observation。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Match

from PDFReader import PDFReader

if TYPE_CHECKING:
    from AgentLLM import AgentLLM
    from Prompts import Prompts

# 工具类型：接收 action_input 字符串，返回 Observation 字符串
ToolFunc = Callable[[str], str]


# 标签字面量，用于纯字符串解析（不依赖正则，避免编码/空白导致匹配失败）
_TAG_THOUGHT = "**Thought:**"
_TAG_ACTION = "**Action:**"
_TAG_ACTION_INPUT = "**Action Input:**"


def _find_tag(text: str, tag: str) -> int:
    """在 text 中查找 tag，先按原样再按小写，返回起始下标，未找到返回 -1。"""
    i = text.find(tag)
    if i >= 0:
        return i
    return text.lower().find(tag.lower())


@dataclass
class ReActBlock:
    """单轮 ReAct 解析结果：Thought + Action + Action Input。"""

    thought: str
    action: str
    action_input: str

    @property
    def has_action(self) -> bool:
        """是否包含有效 Action（非空且已 strip）。"""
        return bool(self.action.strip())


def _extract_block(match: Match[str], label: str) -> str:
    """从正则匹配中取出指定标签对应的内容并 strip。"""
    return "" if match.lastgroup != label else (match.group(2) or "").strip()


def _ensure_str(raw: str | list) -> str:
    """将 LLM 返回值规范为字符串（invoke 有时返回 list 如 AIMessage 列表）。"""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = []
        for x in raw:
            if hasattr(x, "content"):
                parts.append(str(x.content))
            else:
                parts.append(str(x))
        return "\n".join(parts)
    return str(raw)


def parse_react(raw_text: str | list) -> ReActBlock | None:
    """
    从 LLM 返回的纯文本中提取 Thought、Action、Action Input。
    支持 raw_text 为 str 或 list（如 AIMessage 列表），会先规范为字符串再解析。

    支持格式：
        **Thought:** 推理内容...
        **Action:** 工具名
        **Action Input:** 输入内容...

    若未找到任何 **Thought:** / **Action:** / **Action Input:** 块，返回 None。
    否则返回 ReActBlock，缺失的字段为空字符串。
    """
    raw_text = _ensure_str(raw_text)
    if not raw_text or not raw_text.strip():
        return None

    # 用纯字符串查找截取，不依赖正则，避免编码/空白导致匹配失败
    thought = ""
    action = ""
    action_input = ""

    i_t = _find_tag(raw_text, _TAG_THOUGHT)
    i_a = _find_tag(raw_text, _TAG_ACTION)
    i_ai = _find_tag(raw_text, _TAG_ACTION_INPUT)

    if i_t >= 0:
        start = i_t + len(_TAG_THOUGHT)
        next_tags = [x for x in (i_a, i_ai) if x > i_t]
        end = min(next_tags) if next_tags else len(raw_text)
        thought = raw_text[start:end].strip()
    if i_a >= 0:
        start = i_a + len(_TAG_ACTION)
        end = i_ai if i_ai > i_a else len(raw_text)
        action = raw_text[start:end].strip()
    if i_ai >= 0:
        start = i_ai + len(_TAG_ACTION_INPUT)
        action_input = raw_text[start:].strip()

    # 若一个块都没匹配到，尝试兼容无 ** 的写法（如 Thought: / Action:）
    if not thought and not action and not action_input:
        fallback = _parse_react_fallback(raw_text)
        if fallback:
            return fallback

    return ReActBlock(thought=thought, action=action, action_input=action_input)


def _parse_react_fallback(raw_text: str) -> ReActBlock | None:
    """兼容无加粗写法及中文标签：Thought/思考/推理、Action/动作/行动、Action Input/输入。"""
    # 英文：Thought / Action / Action Input（可能带空格）
    en_pattern = re.compile(
        r"(?:^|\n)\s*(Thought|Action|Action\s+Input)\s*[：:]\s*(.*?)"
        r"(?=(?:^|\n)\s*(?:Thought|Action|Action\s+Input)\s*[：:]|\s*$)",
        re.DOTALL | re.IGNORECASE,
    )
    thought = ""
    action = ""
    action_input = ""

    for m in en_pattern.finditer(raw_text):
        key = m.group(1).strip().lower().replace(" ", "")
        value = (m.group(2) or "").strip()
        if key == "thought":
            thought = value
        elif key == "action":
            action = value
        elif key == "actioninput":
            action_input = value

    # 若英文未匹配到，尝试中文标签
    if not thought and not action and not action_input:
        cn_pattern = re.compile(
            r"(?:^|\n)\s*(思考|推理|想法|Thought)\s*[：:]\s*(.*?)"
            r"(?=(?:^|\n)\s*(?:思考|推理|想法|Thought|动作|行动|Action|输入|Action\s+Input)\s*[：:]|\s*$)",
            re.DOTALL | re.IGNORECASE,
        )
        action_cn_pattern = re.compile(
            r"(?:^|\n)\s*(动作|行动|Action)\s*[：:]\s*(.*?)"
            r"(?=(?:^|\n)\s*(?:输入|Action\s+Input|思考|推理|Thought|动作|行动|Action)\s*[：:]|\s*$)",
            re.DOTALL | re.IGNORECASE,
        )
        input_pattern = re.compile(
            r"(?:^|\n)\s*(输入|Action\s+Input)\s*[：:]\s*(.*?)"
            r"(?=(?:^|\n)\s*(?:思考|推理|Thought|动作|行动|Action|输入|Action\s+Input)\s*[：:]|\s*$)",
            re.DOTALL | re.IGNORECASE,
        )
        for m in cn_pattern.finditer(raw_text):
            thought = (m.group(2) or "").strip()
            break
        for m in action_cn_pattern.finditer(raw_text):
            action = (m.group(2) or "").strip()
            break
        for m in input_pattern.finditer(raw_text):
            action_input = (m.group(2) or "").strip()
            break

    if not thought and not action and not action_input:
        return None
    return ReActBlock(thought=thought, action=action, action_input=action_input)


# ---------------------------------------------------------------------------
# 工具调用：执行 Action，将结果写入 Observation
# ---------------------------------------------------------------------------


@dataclass
class ReActStep:
    """单轮完整步骤：解析结果 + 工具执行后的 Observation。"""

    thought: str
    action: str
    action_input: str
    observation: str


def get_default_tools() -> dict[str, ToolFunc]:
    """返回默认工具注册表：answer（内置）、read_pdf（PDFReader）。"""
    pdf_reader = PDFReader(include_tables=True)

    def read_pdf(action_input: str) -> str:
        path = action_input.strip().strip('"').strip("'")
        return pdf_reader.read_as_tool_result(path)

    return {
        "answer": lambda x: x.strip(),  # 直接回复：Observation 即用户看到的答案内容
        "read_pdf": read_pdf,
    }


def run_tool(
    action: str,
    action_input: str,
    tools: dict[str, ToolFunc] | None = None,
) -> str:
    """
    根据 Action 与 Action Input 调用对应工具，返回 Observation 字符串。

    Args:
        action: 工具名（如 answer, read_pdf, get_current_question）。
        action_input: 该工具所需的输入（如 PDF 路径、回复内容等）。
        tools: 工具注册表；为 None 时使用 get_default_tools()。

    Returns:
        执行结果字符串，即本轮的 Observation。
    """
    registry = tools if tools is not None else get_default_tools()
    action = (action or "").strip()
    if not action:
        return "[Observation] 未指定 Action，无法调用工具。"

    # 模型可能输出 "read_pdf" 或 "read_pdf 工具" 等，取首词作为工具名
    action_key = action.split()[0] if action else ""
    if action_key not in registry:
        return f"[Observation] 未知工具: {action}。可用工具: {', '.join(registry.keys())}"

    try:
        return registry[action_key](action_input or "")
    except Exception as e:
        return f"[Observation] 工具执行异常: {e!s}"


def execute_step(
    block: ReActBlock,
    tools: dict[str, ToolFunc] | None = None,
) -> ReActStep:
    """
    执行 ReActBlock：若存在 Action 则调用工具，将结果写入 Observation。
    """
    observation = ""
    if block.has_action:
        observation = run_tool(block.action, block.action_input, tools)
    return ReActStep(
        thought=block.thought,
        action=block.action,
        action_input=block.action_input,
        observation=observation,
    )


# ---------------------------------------------------------------------------
# ReAct 主循环：将 Action / Observation 写入动态上下文，循环次数上限 2
# ---------------------------------------------------------------------------

MAX_LOOP_STEPS = 3


def _truncate(s: str, max_len: int = 600) -> str:
    """过长时截断并加省略号。"""
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[:max_len] + "\n... [已截断]"


def run_react_loop(
    user_input: str,
    agent: AgentLLM,
    prompts: Prompts,
    tools: dict[str, ToolFunc] | None = None,
    max_steps: int = MAX_LOOP_STEPS,
    verbose: bool = True,
) -> str:
    """
    执行 ReAct 循环：LLM 输出 -> 解析 -> 执行工具 -> 将 Action 与 Observation 写入动态上下文 -> 下一轮。
    循环次数上限为 max_steps（默认 2）；若某轮 Action 为 answer 则提前结束并返回回复内容。

    Args:
        user_input: 用户首轮输入（如问题或指令）。
        agent: 用于调用的 LLM 封装（AgentLLM）。
        prompts: 提示模板，其动态上下文会在每轮后更新为「历史 Action / Observation」。
        tools: 工具注册表；为 None 时使用 get_default_tools()。
        max_steps: 最大循环轮数。
        verbose: 是否打印每轮 Thought/Action/工具调用/Observation，默认 True。

    Returns:
        若某轮使用了 answer，返回其 Action Input（最终回复）；否则返回最后一轮的 Observation 或空字符串。
    """
    steps_history: list[str] = []
    last_observation = ""
    prev_observation = ""  # 上一轮 Observation，用于 answer 未填内容时的回退

    for step_index in range(max_steps):
        if verbose:
            print(f"\n{'='*60}\n【第 {step_index + 1} 轮】\n{'='*60}")
        system_prompt = prompts.build_system_prompt()
        if step_index == 0:
            user_message = user_input
        else:
            user_message = (
                "请根据「当前上下文」中的上一轮 Action 与 Observation 继续推理，"
                "输出本轮的 **Thought:**、**Action:**、**Action Input:**。"
                "若已可给出最终回复，请使用 Action: answer，并在 Action Input 中填写完整回复内容。"
            )
        full_prompt = f"{system_prompt}\n\n--- 当前轮用户消息 ---\n{user_message}"

        if verbose:
            print("[调用模型] 请求中...")
        llm_output = agent.invoke(full_prompt)
        raw_str = _ensure_str(llm_output)
        if verbose:
            print("[模型原始输出]")
            print(_truncate(raw_str, 1000))
        block = parse_react(llm_output)

        # 解析失败或解析结果全空（模型未按约定格式输出）时，用原始输出并退出
        if block is None or (not (block.thought or block.action or block.action_input)):
            last_observation = raw_str
            if verbose:
                print("[解析] 未识别到 Thought/Action/Action Input 格式，使用模型原始输出并结束。")
            break

        if verbose:
            print("[解析结果]")
            print("  Thought:", _truncate(block.thought) or "(无)")
            print("  Action:", block.action or "(无)")
            print("  Action Input:", _truncate(block.action_input, 200) or "(无)")
            act_name = (block.action or "").strip().split()[0] if (block.action or "").strip() else "(无)"
            print("[调用工具]", act_name, "| 输入:", _truncate(block.action_input, 200) or "(无)")

        step = execute_step(block, tools)
        last_observation = step.observation

        if verbose:
            print("[Observation]")
            print(_truncate(step.observation))

        # 将本轮的 Action 与 Observation 加入动态上下文
        steps_history.append(
            f"第{step_index + 1}轮\nAction: {step.action}\nAction Input: {step.action_input or '(无)'}\nObservation: {step.observation}"
        )
        prompts.set_context(recent_steps="\n\n".join(steps_history))

        action_first = (step.action or "").strip().lower().split()[0] if (step.action or "").strip() else ""
        if action_first == "answer":
            if verbose:
                print("\n[结束] 模型使用 answer，输出最终回复。")
            # 若 answer 未填 Action Input，用上一轮 Observation（如 PDF 内容）作为输出
            return (step.action_input or step.observation or prev_observation or "").strip()

        prev_observation = last_observation

    if verbose and steps_history:
        print("\n[结束] 已达最大轮数或解析中断。")
    return last_observation
