from typing import TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool, BaseTool
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.message import MessagesState
from langgraph.constants import START, END
from langgraph.graph.state import StateGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.types import Send

from init import model
from utils.tools_util import rule_based_filter


def build_tool_graph(query: str, mcp_tools: list[BaseTool] | None = None) -> list[BaseTool]:
    """本地内置工具（未来扩展）+ MCP 远程工具合并，供主图 bind_tools 与 ToolNode 使用"""

    class ToolSubgraphState(TypedDict):
        messages: list[BaseMessage]
        available_tools: list[BaseTool]
        iterations: int
        max_iterations: int
        # 注意：不要放主图特有的字段（如 query、reranked_docs 等）

    class OutPut(TypedDict):
        output: list[BaseMessage]

    def llm_node(state: ToolSubgraphState) -> dict:
        model_with_tools = model.bind_tools(state["available_tools"])
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response], "iterations": state["iterations"] + 1}

    def should_continue(state) -> list[Send]:
        payload = {
            "messages": state["messages"],
            "available_tools": state["available_tools"]
        }
        if state["iterations"] >= state["max_iterations"]:
            return [Send(END, payload)]
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return [Send("tools", payload)]
        return [Send(END, payload)]

    def output_node(state: ToolSubgraphState) -> dict:
        return {"output": state["messages"]}

    builder = StateGraph(state_schema=ToolSubgraphState,output_schema= OutPut)
    builder.add_node("llm_node", llm_node)
    builder.add_node("should_continue", should_continue)
    builder.add_node("tools", ToolNode(tools=rule_based_filter(query, mcp_tools)))
    builder.add_node("output_node", output_node)

    builder.add_edge(START, "tools")
    builder.add_edge("tools", "llm_node")
    builder.add_edge("llm_node", "should_continue")
    builder.add_conditional_edges(
        "should_continue",
        should_continue,
        ["tools", END]
    )

