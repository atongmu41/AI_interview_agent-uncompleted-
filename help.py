from langchain_openai import OpenAI
import sys


def main() -> None:
    # 将 help(OpenAI) 的输出重定向到 help.txt 中
    with open("help.txt", "w", encoding="utf-8") as f:
        # 备份原始 stdout
        original_stdout = sys.stdout
        try:
            sys.stdout = f
            help(OpenAI)
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()