"""
AgentLLM: 封装大模型 API 调用，便于在面试 AI Agent 中复用。
"""
import os
from pathlib import Path

from langchain_openai import ChatOpenAI


def _load_env(env_path: str = "AIagent.env") -> None:
    """从指定文件加载环境变量。"""
    path = Path(env_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到环境文件: {path.resolve()}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


class AgentLLM:
    """封装 OpenAI 兼容 API 的调用，统一管理配置与 invoke。"""

    def __init__(self, env_path: str = "AIagent.env") -> None:
        _load_env(env_path)
        self._llm = ChatOpenAI(
            model=os.environ.get("MODEL_NAME", "gpt-4.1-mini"),
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL"),
            temperature=0,
        )

    def invoke(self, prompt: str) -> str:
        """调用大模型 API，返回模型回复的文本内容。"""
        message = self._llm.invoke(prompt)
        return message.content if hasattr(message, "content") else str(message)

    @property
    def llm(self):
        """暴露底层 LLM 实例，便于与 PromptTemplate 等组合使用。"""
        return self._llm
