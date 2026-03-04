"""
单独测试 RAG 检索是否正常
直接在终端打印检索结果，不经过 Agent
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.rag_service import RagSummarizeService


def main():
    print("\n{'═' * 50}")
    print("  RAG 检索测试")
    print("{'═' * 50}")
    print("  输入 exit 退出\n")

    rag = RagSummarizeService()

    while True:
        try:
            query = input("[查询] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue

        if query.lower() == "exit":
            break

        print("\n[检索结果]")
        print("─" * 40)
        result = rag.rag_summarize(query)
        print(result)
        print("─" * 40 + "\n")


if __name__ == "__main__":
    main()
