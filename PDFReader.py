"""
PDFReader: 使用 pdfplumber 解析 PDF 文件，提取文本与表格等相关信息。
可作为 ReAct 智能体的工具，用于读取面试题、简历等 PDF 内容。
"""

from pathlib import Path
from typing import Any

import pdfplumber


class PDFReader:
    """PDF 解析器：读取 .pdf 文件并提取文本、表格等信息。"""

    def __init__(self, include_tables: bool = True) -> None:
        """
        Args:
            include_tables: 是否提取表格内容（默认 True，适合含题目/选项的 PDF）。
        """
        self._include_tables = include_tables

    def read(self, path: str | Path) -> dict[str, Any]:
        """
        解析 PDF 文件，提取文本与（可选）表格。

        Args:
            path: PDF 文件路径（字符串或 Path）。

        Returns:
            包含以下字段的字典：
            - "success": 是否解析成功
            - "path": 原始路径
            - "num_pages": 页数
            - "text": 全文（按页合并，页与页之间用换行分隔）
            - "pages": 每页的文本列表，pages[i] 为第 i+1 页的文本
            - "tables": 若 include_tables 为 True，则为所有页的表格列表，每页一个元素（该页表格的列表）
            - "error": 若失败，为错误信息；否则不存在或为空
        """
        path = Path(path)
        result: dict[str, Any] = {
            "success": False,
            "path": str(path.resolve()),
            "num_pages": 0,
            "text": "",
            "pages": [],
            "error": "",
        }
        if self._include_tables:
            result["tables"] = []

        if not path.exists():
            result["error"] = f"文件不存在: {path}"
            return result

        if path.suffix.lower() != ".pdf":
            result["error"] = "仅支持 .pdf 文件"
            return result

        try:
            with pdfplumber.open(path) as pdf:
                result["num_pages"] = len(pdf.pages)
                page_texts: list[str] = []
                all_tables: list[list[list[list[str | None]]]] = []

                for page in pdf.pages:
                    # 提取当前页文本
                    raw = page.extract_text()
                    text = (raw or "").strip()
                    page_texts.append(text)

                    # 可选：提取当前页表格
                    if self._include_tables:
                        tables = page.extract_tables()
                        all_tables.append(tables or [])

                result["pages"] = page_texts
                result["text"] = "\n\n".join(page_texts)
                result["success"] = True
                if self._include_tables:
                    result["tables"] = all_tables

        except Exception as e:
            result["error"] = str(e)

        return result

    def read_text_only(self, path: str | Path) -> str:
        """
        仅提取全文，失败时返回空字符串。
        便于需要「一段连续文本」的场景（如交给 LLM）。
        """
        data = self.read(path)
        if data["success"]:
            return data["text"]
        return ""

    def read_as_tool_result(self, path: str | Path) -> str:
        """
        以「工具返回字符串」的形式返回，供 ReAct 的 Observation 使用。
        成功时返回摘要 + 全文；失败时返回错误信息。
        """
        data = self.read(path)
        if not data["success"]:
            return f"[PDF 读取失败] {data.get('error', '未知错误')}"

        lines = [
            f"已解析 PDF: {data['path']}",
            f"页数: {data['num_pages']}",
            "--- 正文 ---",
            data["text"],
        ]
        if self._include_tables and data.get("tables"):
            lines.append("--- 表格（按页） ---")
            for i, page_tables in enumerate(data["tables"]):
                if page_tables:
                    lines.append(f"第 {i + 1} 页:")
                    for j, table in enumerate(page_tables):
                        lines.append(f"  表格 {j + 1}: {table}")
        return "\n".join(lines)
