from langchain_core.tools import tool, BaseTool
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.message import MessagesState


def build_tool_graph(mcp_tools: list[BaseTool] | None = None) -> list[BaseTool]:
    """本地内置工具（未来扩展）+ MCP 远程工具合并，供主图 bind_tools 与 ToolNode 使用"""
    local_tools: list[BaseTool] = []
    return local_tools + (mcp_tools or [])
