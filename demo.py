"""
demo.py: 使用 ReAct 模型验证「互联网面试官」阅读简历并筛选有价值内容。
运行前请在同目录下放置至少一个 .pdf 简历文件（或通过命令行传入路径）。
"""
import sys
from pathlib import Path

from AgentLLM import AgentLLM
from Prompts import Prompts
from ReAct import get_default_tools, run_react_loop


def main() -> None:
    # 确定简历 PDF 路径：命令行参数 > 同目录下第一个 .pdf
    script_dir = Path(__file__).resolve().parent
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        pdfs = list(script_dir.glob("*.pdf"))
        if not pdfs:
            print("当前目录下未找到 .pdf 文件，请放置简历 PDF 或通过参数传入路径。")
            return
        pdf_path = pdfs[0]
    pdf_path = pdf_path.resolve()
    print(f"使用简历: {pdf_path}\n")

    # 互联网面试官角色 + 工具描述（与 get_default_tools 一致：answer, read_pdf）
    role = """你是一名互联网公司的面试官助手。你的任务是根据简历内容，筛选出对公司有价值的信息（如教育背景、项目经验、技能、亮点等），并严格按「格式制约」输出 Thought / Action / Action Input。"""
    tools_spec = [
        {"name": "answer", "description": "向用户输出最终筛选结果或总结，仅在此步骤使用。"},
        {"name": "read_pdf", "description": "读取 PDF 文件内容。Action Input 中填写 PDF 的完整路径。"},
    ]

    prompts = Prompts(role=role, tools_spec=tools_spec)
    prompts.set_context(task="阅读简历并筛选对公司有价值的内容", resume_path=str(pdf_path))

    agent = AgentLLM("AIagent.env")
    tools = get_default_tools()

    user_input = (
        "请严格按以下三行格式回复，每行以对应标签开头，不要省略：\n"
        "**Thought:** （你的推理）\n**Action:** （工具名）\n**Action Input:** （输入内容）\n\n"
        f"任务：先使用 read_pdf 工具读取简历 {pdf_path}，再根据内容筛选对公司有价值的部分，最后用 answer 输出筛选结果。"
    )

    print("正在运行 ReAct 模型（阅读简历 → 筛选有价值内容）...\n")
    result = run_react_loop(
        user_input=user_input,
        agent=agent,
        prompts=prompts,
        tools=tools,
        max_steps=3,
    )
    print("--- 最终输出（对公司有价值的内容） ---")
    print(result)


if __name__ == "__main__":
    main()
