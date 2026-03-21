from typing import Callable
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    tool_name = request.tool_call['name']

    logger.info(f"[tool monitor] 执行工具：{tool_name}")
    logger.info(f"[tool monitor] 传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor] 工具 {tool_name} 调用成功")

        if tool_name == "fill_context_for_report":
            request.runtime.context["report"] = True

        if tool_name == "transfer_to_human":
            reason = request.tool_call['args'].get('reason', '未知原因')
            request.runtime.context["transfer"] = True
            request.runtime.context["transfer_reason"] = reason
            logger.info(f"[monitor_tool] 转人工标记已设置 | 原因: {reason}")

            messages = request.state.get("messages", [])
            user_query = ""
            for msg in reversed(messages):
                if hasattr(msg, "type") and msg.type == "human":
                    user_query = msg.content
                    break

            from utils.session_manager import session_manager
            from utils.feishu_client import feishu_client

            chat_id = request.runtime.context.get("chat_id", "")
            user_id = request.runtime.context.get("user_id", "")
            user_name = request.runtime.context.get("user_name", "")

            if chat_id:
                session = session_manager.get_or_create_session(
                    chat_id=chat_id,
                    user_id=user_id,
                    user_name=user_name,
                )
                session_id = session_manager.set_pending_human(
                    chat_id=chat_id,
                    question=user_query,
                    reason=reason,
                )
                if session_id:
                    feishu_client.send_to_human_agent(
                        user_query=user_query,
                        reason=reason,
                        session_id=session_id,
                        extra_info=f"chat_id={chat_id} user_id={user_id}"
                    )
            else:
                feishu_client.send_to_human_agent(
                    user_query=user_query,
                    reason=reason,
                )

        return result

    except Exception as e:
        logger.error(f"工具 {tool_name} 调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(
        state: AgentState,
        runtime: Runtime,
):
    logger.info(f"[log_before_model] 即将调用模型，带有 {len(state['messages'])} 条消息。")
    last_msg = state["messages"][-1]
    raw = last_msg.content
    if isinstance(raw, list):
        preview = str(raw)[:120]
    else:
        preview = str(raw).strip()[:120]
    logger.debug(f"[log_before_model] {type(last_msg).__name__} | {preview}")
    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return load_report_prompts()
    return load_system_prompts()
