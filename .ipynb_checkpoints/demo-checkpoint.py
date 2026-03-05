import os
from pathlib import Path

from langchain_openai import ChatOpenAI


def load_env(env_path: str = "AIagent.env") -> None:
    """
    简单加载 AIagent.env 中的环境变量到 os.environ。
    只支持 KEY="VALUE" 或 KEY=VALUE 这种格式。
    """
    path = Path(env_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到环境文件: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value


def main() -> None:
    # 加载 AIagent.env 中的配置
    load_env("AIagent.env")

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model_name = os.environ.get("MODEL_NAME", "gpt-4.1-mini")

    if not api_key or not base_url:
        raise RuntimeError("请确认 AIagent.env 中配置了 OPENAI_API_KEY 和 OPENAI_BASE_URL。")

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
    )

    # 让大模型扮演互联网公司 HR，对应届本科生进行面试
    prompt = (
        "你现在是一家大型互联网公司的 HR 面试官，"
        "正在用中文对应届计算机相关专业本科毕业生进行一轮技术+综合面试。\n"
        "请一次性给出 8~10 个循序渐进的面试问题，"
        "既包含计算机基础（数据结构、算法、计算机网络、操作系统、数据库等），"
        "也包含项目经历、实习经历、学习能力、沟通协作、职业规划等方面。\n"
        "只输出问题清单，用有序编号列出，不要给出参考答案。"
    )

    result = llm.invoke(prompt)

    content = result.content
    if isinstance(content, str):
        text = content
    else:
        # 期望是一个包含若干 {"type": "text", "text": "..."} 的列表
        try:
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            text = "".join(parts) or str(content)
        except Exception:
            text = str(content)

    print("=== HR 面试官对应届本科生的面试问题 ===")
    print(text)


if __name__ == "__main__":
    main()

import os
from pathlib import Path

from datetime import datetime, timedelta, timezone

from langchain_openai import ChatOpenAI


def load_env(env_path: str = "AIagent.env") -> None:
    """
    简单加载 AIagent.env 中的环境变量到 os.environ。
    只支持 KEY="VALUE" 或 KEY=VALUE 这种格式。
    """
    path = Path(env_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到环境文件: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value


def main() -> None:
    # 加载 AIagent.env 中的配置
    load_env("AIagent.env")

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model_name = os.environ.get("MODEL_NAME", "gpt-4.1-mini")

    if not api_key or not base_url:
        raise RuntimeError("请确认 AIagent.env 中配置了 OPENAI_API_KEY 和 OPENAI_BASE_URL。")

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
    )

    # 本地计算当前北京时间（UTC+8）
    beijing_now = datetime.now(timezone(timedelta(hours=8)))
    beijing_str = beijing_now.strftime("%Y-%m-%d %H:%M:%S")
    print("本地计算的当前北京时间：", beijing_str)

    # 通过大模型做一次简单回显测试，确认 API 可用
    prompt = f"这是我本地计算的北京时间：{beijing_str}。请原样返回这串字符，不要添加任何其它内容。"

    result = llm.invoke(prompt)

    content = result.content
    # 兼容不同内容格式（有的返回字符串，有的返回分块列表）
    if isinstance(content, str):
        text = content
    else:
        # 期望是一个包含若干 {"type": "text", "text": "..."} 的列表
        try:
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            text = "".join(parts) or str(content)
        except Exception:
            text = str(content)

    print("模型返回的当前北京时间：", text)


if __name__ == "__main__":
    main()

