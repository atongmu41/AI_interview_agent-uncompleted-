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

    def _extract_text(self, out) -> str:
        """从 API 返回值中提取可解析的文本。支持消息块列表（含 type='text' 的 text 字段）。"""
        if isinstance(out, list):
            parts = []
            for x in out:
                if isinstance(x, dict):
                    # 兼容 type='text' 的块（如部分 API 返回 reasoning + text 分块）
                    if x.get("type") == "text" and "text" in x:
                        parts.append(str(x["text"]))
                    elif "text" in x:
                        parts.append(str(x["text"]))
                    elif "content" in x:
                        parts.append(str(x["content"]))
                    else:
                        continue  # reasoning/encrypted 等块跳过，不拼进可解析文本
                elif hasattr(x, "content"):
                    parts.append(str(x.content))
                else:
                    parts.append(str(x))
            return "\n".join(parts) if parts else str(out)
        if hasattr(out, "content"):
            # AIMessage.content 也可能是 list（多模态）
            c = out.content
            if isinstance(c, list):
                return self._extract_text(c)
            return str(c)
        return str(out)

    def invoke(self, prompt: str) -> str:
        """调用大模型 API，返回模型回复的文本内容（保证为 str）。"""
        out = self._llm.invoke(prompt)
        return self._extract_text(out)

    @property
    def llm(self):
        """暴露底层 LLM 实例，便于与 PromptTemplate 等组合使用。"""
        return self._llm
