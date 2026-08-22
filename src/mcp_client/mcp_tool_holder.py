import asyncio
from dataclasses import dataclass
from typing import Any

from langchain_core.tools.base import BaseTool
from mcp import ClientSession
from pydantic import Field

@dataclass
class McpToolHolder:
    """单个 MCP 工具的封装：统一调用接口（按工具名调用远端工具）。

    注意：不要在此类内手写 __init__，否则会覆盖 dataclass 自动生成的
    字段构造函数，导致 name=/session= 等关键字实参全部报"意外实参"。
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    session: ClientSession          # 持有会话，用于调用工具
    server_name: str = ""           # 所属服务器标识

    async def call(self, arguments: dict[str, Any]) -> str:
        """调用工具并返回文本结果"""
        result = await self.session.call_tool(self.name, arguments)
        return "\n".join(
            content.text for content in result.content if content.type == "text"
        )

class McpToolAdapter(BaseTool):
    holder: McpToolHolder = Field(exclude=True)
    name: str = ""
    description: str = ""

    def __init__(self, holder: McpToolHolder, **kwargs):
        super().__init__(
            name=holder.name,
            description=holder.description,
            args_schema=create_schema_from_input(holder.input_schema),
            holder=holder,
            **kwargs
        )

    def _run(self, **kwargs):
        return asyncio.run(self.holder.call(kwargs))