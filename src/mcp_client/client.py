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
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from loguru import logger
from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.client.sse import sse_client

from mcp_client.mcp_tool_holder import McpToolHolder


class McpServerConnection:
    """连接单个 MCP 服务器：生命周期管理 + 工具加载（每个连接独立 AsyncExitStack）"""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.tools: list[BaseTool] = []          # LangChain 工具（供图 bind_tools / ToolNode）
        self.holders: list[McpToolHolder] = []   # 按工具的统一调用封装（可选）

    @staticmethod
    def _build_env(extra: dict[str, str]) -> dict[str, str]:
        """构造子进程环境：继承父进程全部环境变量，并把解释器同目录的 Scripts
        （conda/venv 下的 uvx.exe、pip.exe 等）注入 PATH 最前。

        注意：该注入不影响 CreateProcess 的可执行文件搜索（它按父进程 PATH
        搜），只影响子进程内部再启动命令时的解析；命令本身能否启动由
        _resolve_command 用绝对路径保证。
        """
        env = dict(os.environ)
        env.update(extra)
        scripts = str(Path(sys.executable).resolve().parent / "Scripts")
        if scripts not in env.get("PATH", "").split(os.pathsep):
            env["PATH"] = scripts + os.pathsep + env.get("PATH", "")
        return env

    @staticmethod
    def _resolve_command(command: str) -> str:
        """解析启动命令：Windows 下 CreateProcess 按父进程 PATH 搜索可执行文件
        （env 里的 PATH 不影响搜索），PyCharm/未激活终端里 conda Scripts 不在
        PATH，裸命令 uvx 会 WinError 2。若命令在解释器 Scripts 目录存在，
        补全为绝对路径。
        """
        if os.name != "nt" or os.path.isabs(command):
            return command
        scripts = Path(sys.executable).resolve().parent / "Scripts"
        for candidate in (scripts / command, scripts / f"{command}.exe"):
            if candidate.is_file():
                return str(candidate)
        return command

    async def open(self) -> None:
        """建立连接、初始化会话并加载全部工具。

        连接失败统一转为 ConnectionError 抛出（含 Windows 下 mcp 库抛出的内部
        CancelledError），由调用方降级处理，不阻塞主流程。
        """
        server_type = self.cfg.get("type", "stdio")
        if server_type == "stdio":
            params = StdioServerParameters(
                command=self._resolve_command(self.cfg["command"]),
                args=self.cfg.get("args", []),
                cwd=self.cfg.get("cwd"),
                env=self._build_env(self.cfg.get("env") or {}),
            )
            # 预检：服务器脚本必须存在。避免子进程启动失败触发 mcp 库在
            # Windows 上的取消作用域泄漏 bug（anyio cancel scope 跨任务退出），
            # 该 bug 会污染同一事件循环中后续服务器的连接
            # 注意：只对 python/python3/py 等直接运行脚本的命令做检查；
            # uvx/npx/pipx 等包管理器的 args[0] 是包名，不是本地脚本路径
            _SCRIPT_RUNNERS = {"python", "python3", "py"}
            if params.command in _SCRIPT_RUNNERS and params.args and not params.args[0].startswith("-"):
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
        """按启动逆序释放连接（先关会话再关传输）。

        mcp SDK 在 Windows asyncio 下关闭 stdio 传输时可能抛 cancel scope
        相关 RuntimeError（库已知问题），关闭失败连接已无可用性，吞掉即可。
        """
        try:
            await self._stack.aclose()
        except BaseException:
            pass


async def init_mcp_holders(servers: list[dict[str, Any]], timeout: int = 15) -> list[McpServerConnection]:
    """按配置连接全部 MCP 服务器，返回连接列表。

    - connections[i].tools：LangChain 工具列表（供图使用）
    - connections[i].holders：按工具的统一调用封装（可选）
    - 关闭：for conn in connections: await conn.close()

    单个服务器连接失败只跳过该服务器并告警，不影响其他服务器与主流程
    （图内工具列表缺少远程工具时自动降级为纯 LLM 回答）。

    Args:
        servers: MCP 服务器配置列表
        timeout: 单个服务器连接超时时间（秒），默认 15 秒。
                 防止 MCP 服务器启动后 stdio 通信无响应时阻塞整个后端启动。
    """
    async def _connect_one(cfg: dict[str, Any]) -> McpServerConnection | None:
        """连接单个服务器（每个连接在独立 task 中执行）。

        必须在独立 task 里调用 open()：mcp 库 stdio_client 的 anyio cancel
        scope 归属创建它的 task，Windows 下子进程快速失败时 scope 跨任务退出
        泄漏的取消只作用于本任务（已失败，无影响），不会像顺序版那样注入到
        后续连接的 await 点甚至 lifespan 协程。
        """
        server_name = cfg.get('name', cfg.get('type', 'unknown'))
        conn = McpServerConnection(cfg)
        try:
            # 超时保护：单个 MCP 服务器连接超时后跳过，不阻塞主流程
            await asyncio.wait_for(conn.open(), timeout=timeout)
            return conn
        except asyncio.TimeoutError:
            logger.warning(
                f"MCP 服务器 [{server_name}] 连接超时（{timeout}s），已跳过。"
                f"请检查该服务器是否能正常响应 stdio 通信。"
            )
        except (Exception, asyncio.CancelledError) as e:
            # 必须捕获 CancelledError：mcp 库泄漏的取消会异步注入到 await 点，
            # 捕获后降级为告警跳过，避免污染主流程（本任务隔离后即使漏网也
            # 只影响本任务）。
            logger.warning(
                f"MCP 服务器 [{server_name}] 连接失败，已跳过：{e}"
            )
        await conn.close()   # 兜底释放 open() 已登记的部分资源
        return None

    # 并发连接全部服务器：启动更快，且单个失败被任务隔离，互不污染
    results = await asyncio.gather(*(_connect_one(cfg) for cfg in servers))
    return [c for c in results if c is not None]


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
