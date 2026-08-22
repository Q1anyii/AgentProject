import asyncio
from typing import List

from langchain_core.tools import BaseTool

from config import load_mcp_server_configs
from mcp_client.client import init_mcp_holders
from loguru import logger
from rich import print as rprint


async def list_mcp_tools():
    mcp_holders = await init_mcp_holders(load_mcp_server_configs())
    tools_list = List
    for h in mcp_holders:
        for t in h.tools:
            logger.success(f"已加载 MCP 工具：{t.name}")
            rprint(t)

    mcp_tools = [t for h in mcp_holders for t in h.tools]

def rule_based_filter(query ,tools: list[BaseTool]):
    query_lower = query.lower()
    selected = []
    for t in tools:
        # 检查 tags 或 keywords
        if any(tag in query_lower for tag in t.tags):
            selected.append(t)
    return selected or tools # 若没有匹配则返回全部，避免漏选


def safety_filter(tools: list[BaseTool], user_role: str = "user") -> list[BaseTool]:
    """安全硬过滤：破坏性工具需管理员权限，写操作需场景允许"""
    result = []
    for tool in tools:
        meta = tool.metadata or {}
        # 破坏性操作（如 git_reset、delete_file）：仅管理员可用
        if meta.get("destructiveHint") and user_role != "admin":
            continue
        # 写操作（如 git_add、write_file）：只读模式下禁用
        # if not meta.get("readOnlyHint") and read_only_mode:
        #     continue
        result.append(tool)
    return result

if __name__ == "__main__":
    asyncio.run(list_mcp_tools())