import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters, stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

async def connect_mcp(server_params: StdioServerParameters):
    # 初始化stack,
    stack = AsyncExitStack()
    # stdio_client 打开传输，拿到 read、write 流
    read, write = await stack.enter_async_context(stdio_client(server_params))
    # 包装成 ClientSession
    session = await stack.enter_async_context(ClientSession(read, write))
    # 建立握手
    await session.initialize()
    # 使用langchain获取 mcp tools
    tools = await load_mcp_tools(session)
    return stack, session, tools
