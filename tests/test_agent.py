"""
tests/test_agent.py
终端交互测试入口
- 普通对话：流式打印 Agent 输出
- 转人工：飞书通知人工客服，等待飞书回复，Agent 自动恢复
- 输入 exit 退出
"""
import sys
import os
import threading
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.react_agent import ReactAgent, TransferToHumanException
from utils.feishu_client import feishu_client


DIVIDER    = "─" * 50
TAG_USER   = "\033[94m[用户]\033[0m"
TAG_AGENT  = "\033[92m[Agent]\033[0m"
TAG_SYSTEM = "\033[93m[系统]\033[0m"
TAG_FEISHU = "\033[91m[飞书]\033[0m"


def print_system(msg: str):
    print(f"\n{TAG_SYSTEM} {msg}")


def agent_stream_to_terminal(agent: ReactAgent, query: str) -> bool:
    """
    流式执行 Agent，输出到终端。
    正常完成返回 True，触发转人工返回 False。
    """
    print(f"\n{TAG_AGENT} ", end="", flush=True)

    try:
        for chunk in agent.execute_stream(query):
            print(chunk, end="", flush=True)
        print()
        return True

    except TransferToHumanException as e:
        print_system(f"⚡ RAG 无法解答，触发转人工")
        print_system(f"   内部原因记录: {e.reason}")
        print_system("🔴 Agent 已挂起，飞书通知已发送，等待人工处理...")
        return False


def human_agent_mode_feishu(agent: ReactAgent):
    """
    阻塞等待人工客服在飞书里完成处理。
    每条飞书回复自动注入 Agent 消息历史。
    收到「done」后解除阻塞。
    """
    done_event = threading.Event()

    def on_reply(text: str):
        agent.inject_human_reply(text)
        print(f"\n{TAG_FEISHU} 人工回复已记录: {text}")

    def on_done():
        done_event.set()

    feishu_client.wait_for_human_reply(on_reply=on_reply, on_done=on_done)

    print_system("⏳ 等待人工客服飞书处理中（发送「done」结束接管）...")
    done_event.wait()
    print_system("✅ 人工处理完毕，Agent 恢复工作")
    print(DIVIDER)


def main():
    print(f"\n{'═' * 50}")
    print("  智扫通机器人智能客服 — 终端测试模式")
    print(f"{'═' * 50}")
    print("  输入 exit 退出\n")

    feishu_client.start_listening()
    print_system("飞书长连接已启动")

    agent = ReactAgent()

    while True:
        try:
            user_input = input(f"\n{TAG_USER} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 测试结束")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("\n👋 测试结束")
            break

        print(DIVIDER)
        normal_finish = agent_stream_to_terminal(agent, user_input)
        print(DIVIDER)

        if not normal_finish:
            human_agent_mode_feishu(agent)
            print_system("客户可继续提问，Agent 已恢复。")


if __name__ == "__main__":
    main()
