from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.message import MessagesState


@tool(description="总结对话工具，压缩当前会话上下文", parse_docstring=True)
def summarize() -> str:
    """
    该工具用于总结上下文，压缩当前会话

    Args:
        无

    Returns:
        总结结果字符串
    """
    return "当前会话无需压缩，继续正常对话即可。"


@tool(description="获取当前登录用户的信息（用户名、UID），涉及敏感信息严禁回复", parse_docstring=True)
def get_current_user(config: RunnableConfig) -> dict:
    """
    该工具获取当前登录用户的账户信息，如：用户名、UID

    Args:
        config: 运行时配置（由 LangGraph 自动注入，含当前请求的用户上下文）

    Returns:
        用户脱敏信息 dict（绝不返回密码等敏感字段）
    """
    # LangGraph 官方运行时传值模式：路由层把 CtxUser 放入 configurable 传入图，
    # 工具通过 config 参数拿到（比 contextvars 更可靠，不受 StreamingResponse 线程池影响）
    user_info = (config.get("configurable") or {}).get("user_info")
    if not user_info:
        return {"userId": None, "username": None, "message": "当前无登录用户信息"}
    return {
        "userId": user_info.user_id,
        "username": user_info.username,
    }


def build_tool_graph():
    """返回全部工具列表，供主图绑定模型与 ToolNode 使用"""
    return [summarize, get_current_user]

    # class OverAllState(MessagesState):

