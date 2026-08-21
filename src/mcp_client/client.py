"""
MCP客户端管理层
目标：对接MCP服务，输出LangChain BaseTool供给LangGraph Agent使用。

参考MCP官方简单demo无法直接工程使用，会存在：
1. stdio异步生成器GC回收导致连接静默断开；
2. Windows平台子进程失败CancelledError污染事件循环；
3. 多MCP服务需要单服务故障降级；
4. 子进程资源泄漏问题。

本模块在MCP、langchain_mcp_adapters官方API基础上，借助AI完成工程健壮性封装；
已经完成调试验证，理解各个防御逻辑对应的故障场景。

核心分层：
- McpToolHolder：MCP工具简易封装，屏蔽协议细节
- McpServerConnection：单个MCP服务完整生命周期管理
- init_mcp_holders：批量初始化、故障降级
"""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from loguru import logger
from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.client.sse import sse_client


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


class McpServerConnection:
    """连接单个 MCP 服务器：生命周期管理 + 工具加载（每个连接独立 AsyncExitStack）"""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.tools: list[BaseTool] = []          # LangChain 工具（供图 bind_tools / ToolNode）
        self.holders: list[McpToolHolder] = []   # 按工具的统一调用封装（可选）

    async def open(self) -> None:
        """建立连接、初始化会话并加载全部工具。

        连接失败统一转为 ConnectionError 抛出（含 Windows 下 mcp 库抛出的内部
        CancelledError），由调用方降级处理，不阻塞主流程。
        """
        server_type = self.cfg.get("type", "stdio")
        if server_type == "stdio":
            params = StdioServerParameters(
                command=self.cfg["command"],
                args=self.cfg.get("args", []),
                cwd=self.cfg.get("cwd"),
                env=self.cfg.get("env"),
            )
            # 预检：服务器脚本必须存在。避免子进程启动失败触发 mcp 库在
            # Windows 上的取消作用域泄漏 bug（anyio cancel scope 跨任务退出），
            # 该 bug 会污染同一事件循环中后续服务器的连接
            if params.args:
                script = Path(params.cwd or ".") / params.args[0]
                if not script.exists():
                    raise ConnectionError(f"MCP 服务器脚本不存在：{script}")
            cm = stdio_client(params)
        elif server_type == "sse":
            cm = sse_client(self.cfg["url"])
        else:
            raise ValueError(f"不支持的 MCP 服务器类型: {server_type}")

        try:
            read, write = await self._enter_context(cm)
            session = await self._enter_context(ClientSession(read, write))
            await session.initialize()
        except (Exception, asyncio.CancelledError) as e:
            raise ConnectionError(
                f"MCP 服务器 [{self.cfg.get('name', server_type)}] 连接失败：{e}"
            ) from e
        self.session = session

        # LangChain 工具：适配器自动完成 JSON Schema → pydantic 转换，闭包捕获 session
        self.tools = await load_mcp_tools(session)
        # 同时保留按工具名的统一调用封装
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            self.holders.append(McpToolHolder(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
                session=session,
                server_name=self.cfg.get("name", server_type),
            ))
        logger.info(f"MCP 服务器 [{self.cfg.get('name', server_type)}] 已连接，工具：{[t.name for t in self.tools]}")

    async def _enter_context(self, cm) -> Any:
        """进入 async 上下文管理器并登记到退出栈；进入失败时显式关闭底层生成器。

        成功路径必须交给 self._stack.enter_async_context 登记：stdio_client 是
        async generator，若只有局部变量持有，open() 返回后会被 GC 提前 aclose，
        导致会话连接静默关闭（call_tool 报 Connection closed）。

        失败路径（Windows 下子进程启动失败时 mcp 库抛 CancelledError）下
        enter_async_context 不会登记退出回调，这里手动 aclose() 让内部任务
        正常退出，并吞掉其清理阶段的异常（mcp/anyio 已知跨任务 cancel scope 问题）。
        """
        try:
            return await self._stack.enter_async_context(cm)
        except BaseException:
            closer = getattr(cm, "aclose", None)
            if closer is not None:
                try:
                    await closer()
                except BaseException:
                    pass
            raise

    async def close(self) -> None:
        """按启动逆序释放连接（先关会话再关传输）"""
        await self._stack.aclose()


async def init_mcp_holders(servers: list[dict[str, Any]]) -> list[McpServerConnection]:
    """按配置连接全部 MCP 服务器，返回连接列表。

    - connections[i].tools：LangChain 工具列表（供图使用）
    - connections[i].holders：按工具的统一调用封装（可选）
    - 关闭：for conn in connections: await conn.close()

    单个服务器连接失败只跳过该服务器并告警，不影响其他服务器与主流程
    （图内工具列表缺少远程工具时自动降级为纯 LLM 回答）。
    """
    connections: list[McpServerConnection] = []
    for cfg in servers:
        try:
            conn = McpServerConnection(cfg)
            await conn.open()
            connections.append(conn)
        except Exception as e:
            logger.warning(
                f"MCP 服务器 [{cfg.get('name', cfg.get('type', 'unknown'))}] 连接失败，已跳过：{e}"
            )
    return connections


async def demo_call(servers: list[dict[str, Any]]) -> None:
    """调试入口：连接后调用每个服务器的第一个工具，验证链路后关闭"""
    connections = await init_mcp_holders(servers)
    for conn in connections:
        for holder in conn.holders:
            # 仅演示：用空参数调用第一个工具（工具实际入参需按 input_schema 提供）
            logger.info(f"调用 {holder.server_name}.{holder.name} ...")
            await holder.call({})
    for conn in connections:
        await conn.close()
