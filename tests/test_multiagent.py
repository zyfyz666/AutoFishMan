"""
tests/test_multiagent.py

多 Agent 架构终端测试 —— 你扮演客户，与主客服 Agent 对话。

支持三种输入方式：
  ① 直接输入文字     → 普通文字对话
  ② /img <URL或路径> → 携带图片提问（VisionAgent 分析后主 Agent 回复）
  ③ /audio <路径>    → 发送本地音频（AudioAgent 转文字后主 Agent 回复）

特殊指令：
  /img  <图片URL或本地路径> [问题]   图片消息（问题可省略，默认描述图片）
  /audio <音频路径> [wav|mp3|...]   语音消息
  /history                          查看当前对话历史
  /clear                            清空对话历史，开始新会话
  exit / quit                       退出

示例：
  你好，扫拖机器人有哪些型号？
  /img https://img.alicdn.com/xxx.jpg 这个商品有问题吗？
  /img D:/photos/my product image.jpg
  /audio D:/voice/question.wav
"""
import sys
import os
import re
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.react_agent import ReactAgent, TransferToHumanException

# ── 飞书（可选，转人工时才需要）──────────────────────────────────────────────
_feishu_enabled = False
try:
    import threading
    from utils.feishu_client import feishu_client
    _feishu_enabled = True
except Exception:
    pass

# ── 终端颜色 ──────────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
DIM     = "\033[2m"

TAG_USER   = f"{CYAN}{BOLD}[你]{RESET}"
TAG_AGENT  = f"{GREEN}{BOLD}[客服Agent]{RESET}"
TAG_VISION = f"{MAGENTA}{BOLD}[视觉Agent]{RESET}"
TAG_AUDIO  = f"{BLUE}{BOLD}[语音Agent]{RESET}"
TAG_SYSTEM = f"{YELLOW}[系统]{RESET}"
TAG_FEISHU = f"{RED}[飞书]{RESET}"

DIVIDER       = f"{DIM}{'─' * 55}{RESET}"
THICK_DIVIDER = f"{BOLD}{'═' * 55}{RESET}"

# 支持的图片扩展名
_IMG_EXTS = ("jpg", "jpeg", "png", "gif", "webp", "bmp")
# 匹配「路径/URL（以图片扩展名结尾）」后可选跟随「空格+问题文字」
# 使用非贪婪 .+? 配合扩展名锚定，正确处理路径中含空格的情况
_IMG_PATTERN = re.compile(
    r'(.+?\.(?:' + '|'.join(_IMG_EXTS) + r'))(?:\s+(.+))?$',
    re.IGNORECASE | re.DOTALL,
)


def ps(msg: str):
    """打印系统提示"""
    print(f"\n{TAG_SYSTEM} {msg}")


# ── 输入解析 ──────────────────────────────────────────────────────────────────

def parse_input(raw: str):
    """
    解析用户输入，返回 (mode, args)

    mode:
      'text'    → args = query str
      'image'   → args = (image_input, question)
      'audio'   → args = (audio_path, fmt)
      'history' → args = None
      'clear'   → args = None
      'exit'    → args = None
      'unknown' → args = 错误提示 str
    """
    stripped = raw.strip()

    if stripped.lower() in ("exit", "quit"):
        return "exit", None

    if stripped.lower() == "/history":
        return "history", None

    if stripped.lower() == "/clear":
        return "clear", None

    if stripped.lower().startswith("/img"):
        rest = stripped[4:].strip()  # 去掉 "/img"

        if not rest:
            return "unknown", "/img 需要提供图片路径或 URL，例如：/img https://example.com/a.jpg"

        # ── 修复：按图片扩展名位置切割，正确处理路径中含空格的情况 ──────────
        m = _IMG_PATTERN.match(rest)
        if m:
            image_input = m.group(1).strip()
            question    = (m.group(2) or "").strip() or "请详细描述这张图片的内容。"
        else:
            # fallback：原逻辑（适用于无扩展名的特殊 URL）
            parts = rest.split(" ", 1)
            image_input = parts[0].strip()
            question    = parts[1].strip() if len(parts) > 1 else "请详细描述这张图片的内容。"

        if not image_input:
            return "unknown", "/img 需要提供图片路径或 URL，例如：/img https://example.com/a.jpg"

        return "image", (image_input, question)

    if stripped.lower().startswith("/audio"):
        rest  = stripped[6:].strip()  # 去掉 "/audio"
        parts = rest.split(" ", 1)
        audio_path = parts[0].strip()
        fmt        = parts[1].strip() if len(parts) > 1 else "wav"

        if not audio_path:
            return "unknown", "/audio 需要提供音频文件路径，例如：/audio D:/voice/q.wav"

        return "audio", (audio_path, fmt)

    return "text", stripped


# ── 流式输出 ──────────────────────────────────────────────────────────────────

def stream_to_terminal(generator) -> tuple[bool, str]:
    """
    消费 generator，实时打印到终端。
    返回 (正常完成?, 完整回复文字)
    """
    collected = []
    print(f"\n{TAG_AGENT} ", end="", flush=True)
    try:
        for chunk in generator:
            text = chunk.rstrip("\n")
            if text:
                print(text, end=" ", flush=True)
                collected.append(text)
        print()
        return True, " ".join(collected)
    except TransferToHumanException as e:
        print()
        ps(f"⚡ RAG 无法解答，触发转人工")
        ps(f"   内部记录原因: {e.reason}")
        ps("🔴 Agent 已挂起，飞书通知已发送，等待人工处理...")
        return False, ""


# ── 飞书转人工等待 ────────────────────────────────────────────────────────────

def wait_for_human_agent(agent: ReactAgent):
    if not _feishu_enabled:
        ps("⚠️  飞书未启用，无法等待人工处理。按 Enter 让 Agent 恢复。")
        input()
        return

    done_event = __import__("threading").Event()

    def on_reply(text: str):
        agent.inject_human_reply(text)
        print(f"\n{TAG_FEISHU} 人工回复已注入: {text}")

    def on_done():
        done_event.set()

    feishu_client.wait_for_human_reply(on_reply=on_reply, on_done=on_done)
    ps("⏳ 等待人工客服处理中（飞书发送「done」结束）...")
    done_event.wait()
    ps("✅ 人工处理完毕，Agent 恢复")
    print(DIVIDER)


# ── 历史记录展示 ──────────────────────────────────────────────────────────────

def show_history(agent: ReactAgent):
    history = agent.message_history
    if not history:
        ps("当前无对话历史。")
        return

    print(f"\n{BOLD}  对话历史（共 {len(history)} 条）{RESET}")
    print(DIVIDER)
    for i, msg in enumerate(history, 1):
        role    = msg["role"]
        content = str(msg["content"])
        preview = content[:120] + "..." if len(content) > 120 else content

        if role == "user":
            label = f"{CYAN}用户{RESET}"
        elif role == "assistant":
            label = f"{GREEN}Agent{RESET}"
        else:
            label = f"{RED}人工客服{RESET}"

        print(f"  {DIM}{i:02d}.{RESET} {label}  {preview}")
    print(DIVIDER)


# ── 帮助信息 ──────────────────────────────────────────────────────────────────

def show_help():
    print(f"""
{BOLD}  使用说明{RESET}
{DIVIDER}
  {CYAN}直接输入文字{RESET}              → 普通对话
  {MAGENTA}/img <URL或路径> [问题]{RESET}   → 图片消息（问题可省略；路径含空格也支持）
  {BLUE}/audio <路径> [格式]{RESET}      → 语音消息（格式默认 wav）
  {YELLOW}/history{RESET}                   → 查看对话历史
  {YELLOW}/clear{RESET}                     → 清空历史，开始新会话
  {YELLOW}exit / quit{RESET}               → 退出
{DIVIDER}
  {DIM}图片示例: /img https://example.com/a.jpg 这个商品有划痕吗？{RESET}
  {DIM}图片示例: /img D:/my photos/product.webp 有问题吗？{RESET}
  {DIM}音频示例: /audio D:/voice/question.wav{RESET}
""")


# ── 主循环 ────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{THICK_DIVIDER}")
    print(f"  {BOLD}闲鱼客服 Agent — 终端测试{RESET}（你扮演客户）")
    print(THICK_DIVIDER)
    show_help()

    # 启动飞书（转人工才需要）
    if _feishu_enabled:
        try:
            feishu_client.start_listening()
            ps("飞书长连接已启动（转人工时会自动通知客服）")
        except Exception as e:
            ps(f"飞书启动失败（不影响普通对话）: {e}")
    else:
        ps("飞书未配置，转人工功能不可用（普通对话不受影响）")

    agent = ReactAgent()
    ps("Agent 已就绪，可以开始对话 👇\n")

    while True:
        # ── 读取输入 ──────────────────────────────────────────────────────────
        try:
            raw = input(f"{TAG_USER} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n👋 测试结束")
            break

        if not raw:
            continue

        mode, args = parse_input(raw)
        print(DIVIDER)

        # ── 分发处理 ──────────────────────────────────────────────────────────

        if mode == "exit":
            print(f"\n👋 测试结束")
            break

        elif mode == "history":
            show_history(agent)
            continue

        elif mode == "clear":
            agent.message_history.clear()
            ps("对话历史已清空，开始新会话。")
            continue

        elif mode == "unknown":
            ps(f"⚠️  {args}")
            continue

        elif mode == "text":
            ps(f"模式: 文字对话")
            ok, _ = stream_to_terminal(agent.execute_stream(args))

        elif mode == "image":
            image_input, question = args
            ps(f"模式: 图片消息 → {TAG_VISION} 分析中...")
            ps(f"图片: {image_input}")
            ps(f"问题: {question}")
            ok, _ = stream_to_terminal(
                agent.execute_stream_with_image(query=question, image_input=image_input)
            )

        elif mode == "audio":
            audio_path, fmt = args
            if not os.path.exists(audio_path):
                ps(f"⚠️  音频文件不存在: {audio_path}")
                print(DIVIDER)
                continue
            ps(f"模式: 语音消息 → {TAG_AUDIO} 识别中...")
            ps(f"文件: {audio_path}  格式: {fmt}")
            ok, _ = stream_to_terminal(
                agent.execute_stream_with_audio(audio_input=audio_path, audio_format=fmt)
            )

        print(DIVIDER)

        # ── 转人工处理 ────────────────────────────────────────────────────────
        if mode in ("text", "image", "audio") and not ok:
            wait_for_human_agent(agent)
            ps("Agent 已恢复，你可以继续提问。")


if __name__ == "__main__":
    main()