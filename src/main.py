import asyncio
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse as FastAPIJSONResponse
from loguru import logger

from config import validate_config, load_mcp_server_configs
from mcp_client.client import init_mcp_holders
from mcp_client.mcp_server.agent_server import mcp
from routers.auth_router import router as auth_router
from routers.chat_router import router as chat_router
from routers.mcp_router import router as mcp_router
from routers.system_router import router as system_router
from routers.user_router import router as user_router
from service.cache_service import cache_service
from service.chat_service import chat_service
from service.file_upload_service import file_upload_service
from service.login_service import login_service
from service.user_profile_service import user_profile_service
from utils.tools_util import safety_filter, tools_embedding

# 注意：.env 加载由 config.py 统一处理，无需重复 load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化资源，关闭时释放资源。"""

    # ===== 启动阶段：yield 之前 =====
    # 第一步：校验必填环境变量，缺失时直接报错（快速失败，不拖到首个请求才 500）
    validate_config()
    # 工具常驻事件循环：MCP session 创建与工具调用必须同一循环——同步图在
    # StreamingResponse 线程池执行工具时会临时新建 loop，跨循环调用 session 会
    # 失败/挂起（Windows 下 mcp 库 cancel scope 泄漏还会注入 CancelledError 中断整图），
    # 故 MCP 连接与工具调用全部提交到本循环（见 mcp_client.make_sync_tool）
    tool_loop = asyncio.new_event_loop()
    threading.Thread(target=tool_loop.run_forever, daemon=True, name="mcp-tool-loop").start()
    mcp_holders = asyncio.run_coroutine_threadsafe(
        init_mcp_holders(load_mcp_server_configs()), tool_loop
    ).result(timeout=35)
    mcp_tools = [t for h in mcp_holders for t in h.tools]
    filtered_tools = []
    if mcp_tools:
        filtered_tools = safety_filter(mcp_tools)
        tools_embedding(filtered_tools)
        logger.success(f"已加载{len(filtered_tools)}个MCP 工具，共{len(mcp_holders)}类")
    logger.info("正在初始化 LangGraph 资源...")
    chat_service.open(filtered_tools)
    login_service.open()
    cache_service.open()
    user_profile_service.open()
    file_upload_service.open()
    logger.success("资源初始化完成")
    yield                                # ===== 应用运行期间（yield 挂起）=====
    # ===== 关闭阶段：yield 之后 =====
    logger.info("正在释放资源...")
    # MCP 子进程连接需在服务关闭前释放（工具闭包依赖 session）；连接创建在工具
    # 常驻循环上，关闭必须提交到该循环，否则跨循环 await 报错
    if mcp_holders:
        asyncio.run_coroutine_threadsafe(_close_mcp_holders(mcp_holders), tool_loop).result(timeout=10)
    tool_loop.call_soon_threadsafe(tool_loop.stop)
    chat_service.close(timeout=10)
    login_service.close(timeout=10)
    cache_service.close()
    user_profile_service.close()
    file_upload_service.close()
    logger.info("资源已释放")


async def _close_mcp_holders(holders: list) -> None:
    """在工具常驻循环内按序关闭全部 MCP 连接（连接/工具调用同循环，关闭也必须同循环）。"""
    for holder in holders:
        await holder.close()


app = FastAPI(title="Mitta AI", lifespan=lifespan)

# 挂载 MCP 服务器端点（fastmcp 3.x：http_app 返回 Starlette app，2.x 的 streamable_http_app 已改名）
# 外部 MCP 客户端（Claude Desktop 等）通过 http://localhost:8000/mcp 调用 agent 能力
app.mount("/mcp", mcp.http_app())

# 注册请求限流中间件（对 /api/chat/ 等消耗 LLM 配额的接口限流）
from middleware.rate_limit_middleware import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware)


# ============================================================
# 全局异常处理：统一返回结构化错误，避免暴露堆栈信息
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常捕获：所有未处理的异常统一返回 500 + 结构化错误。

    - 记录完整异常信息到日志（含堆栈），便于排查
    - 返回给客户端的信息不包含堆栈，只返回通用错误提示
    - HTTPException 由 FastAPI 默认处理，不会进入此处理器
    """
    logger.exception(f"未处理的异常 | path={request.url.path} | method={request.method}")
    return FastAPIJSONResponse(
        status_code=500,
        content={
            "ok": False,
            "detail": "服务器内部错误，请稍后重试或联系管理员",
            "error_type": type(exc).__name__,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTPException 统一包装为 {ok, detail} 格式，与业务接口响应风格一致。"""
    return FastAPIJSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "detail": exc.detail},
        headers=exc.headers,
    )


# ============================================================
# 路由注册：按模块拆分到 routers/ 目录
# 注意：system_router 必须最后注册（SPA 兜底路由 /{full_path:path} 会匹配所有未捕获路径）
# ============================================================
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(user_router)
app.include_router(mcp_router)
app.include_router(system_router)  # 必须最后注册


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
