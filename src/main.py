import os
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from loguru import logger

from config import validate_config, get_env_int, load_mcp_server_configs
from context.user_context import CtxUser
from constant.cache_constant import USER_TOKEN_KEY, USER_REFRESH_TOKEN_KEY
from mcp_client.client import init_mcp_holders
from schemas.request_schemas.chat_schema import ChatRequest
from schemas.request_schemas.login_schema import *
from service.cache_service import cache_service
from service.chat_service import chat_service
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from mcp_client.mcp_server.agent_server import mcp
from service.login_service import login_service
from utils.response_util import Response
from utils.jwt_utils import get_current_user, create_access_token, create_refresh_token, TokenData, \
    REFRESH_TOKEN_EXPIRE_DAYS

# 注意：.env 加载由 config.py 统一处理，无需重复 load_dotenv()


"""

添加多模态功能，可读取用户上传的文件并自动解析成文档，细化拆分graph，完善tool_graph

"""



@asynccontextmanager
async def lifespan(app: FastAPI):

    # ===== 启动阶段：yield 之前 =====
    # 第一步：校验必填环境变量，缺失时直接报错（快速失败，不拖到首个请求才 500）
    validate_config()
    mcp_holders = await init_mcp_holders(load_mcp_server_configs())
    mcp_tools = [t for h in mcp_holders for t in h.tools]
    if mcp_tools:
        logger.success(f"已加载 MCP 工具：{[t.name for t in mcp_tools]}")
    logger.info("正在初始化 LangGraph 资源...")
    chat_service.open(mcp_tools)
    login_service.open()
    cache_service.open()
    logger.success("资源初始化完成")
    yield                                # ===== 应用运行期间（yield 挂起）=====
    # ===== 关闭阶段：yield 之后 =====
    logger.info("正在释放资源...")
    # MCP 子进程连接需在服务关闭前释放（工具闭包依赖 session）
    for holder in mcp_holders:
        await holder.close()
    chat_service.close(timeout=10)
    login_service.close(timeout=10)
    cache_service.close()
    logger.info("资源已释放")

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
from fastapi import Request
from fastapi.responses import JSONResponse as FastAPIJSONResponse


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


# 资源归属校验：只允许本人访问自己的记忆/会话，管理员角色放行
# FastAPI 会自动把路径参数 user_id 注入本依赖（必须定义在使用它的路由之前）
def require_self_or_admin(user_id: str, current_user: TokenData = Depends(get_current_user)):
    if str(current_user.user_id) != user_id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权访问该用户资源")
    return current_user

@app.post("/api/chat/")
def chat(request_body: ChatRequest, current_user: TokenData = Depends(get_current_user)):
    query = request_body.query
    thread_id = request_body.thread_id
    # 会话归属校验（与 history/delete 一致）：会话已存在但非本人所有时拒绝，
    # 否则任意用户可用他人 thread_id 发消息，LangGraph 会用当前用户覆盖该会话归属 metadata 造成劫持
    owner = chat_service.get_thread_user_id(thread_id)
    if owner and owner != str(current_user.user_id) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权使用该会话")
    # 认证在路由层完成：JWT 解析出 user_id/username 后查库，构造请求级用户上下文（供图内工具读取）
    user_row = login_service.get_user_by_id(str(current_user.user_id))
    user_info = (
        CtxUser(
            uid=user_row["id"],
            user_id=user_row["user_id"],
            password=None,  # 敏感字段不注入，工具无法访问
            username=current_user.username,  # 直接取 JWT 解析出的 username（token → 解析 → 上下文）
            create_time=user_row["create_time"],
            update_time=user_row["update_time"],
        )
        if user_row
        else None
    )
    event_stream = chat_service.stream(current_user.user_id, thread_id, query, user_info=user_info)
    return StreamingResponse(event_stream, media_type="text/event-stream")

@app.get("/api/chat/{thread_id}/history")
def get_history_session(thread_id: str, current_user: TokenData = Depends(get_current_user)):
    # 会话归属校验：会话存在但非本人所有时拒绝（管理员放行）
    owner = chat_service.get_thread_user_id(thread_id)
    if owner and owner != str(current_user.user_id) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权访问该会话")
    history_session = chat_service.get_history_session(thread_id)
    return history_session

@app.get("/api/users/{user_id}/memory")
def get_memory(user_id: str, current_user: TokenData = Depends(require_self_or_admin)):
    memory = chat_service.get_memory(user_id)
    return memory

@app.delete("/api/chat/{thread_id}")
def delete_session_by_id(thread_id: str, current_user: TokenData = Depends(get_current_user)):
    # 会话归属校验：会话存在但非本人所有时拒绝（管理员放行）
    owner = chat_service.get_thread_user_id(thread_id)
    if owner and owner != str(current_user.user_id) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权删除该会话")
    flag, response = chat_service.delete_session_by_id(thread_id)
    if flag:
        return Response.success(response)
    else:
        return Response.failed(response)


@app.get("/api/users/{user_id}/sessions")
def get_sessions_by_user_id(user_id: str, current_user: TokenData = Depends(require_self_or_admin)):
    sessions = chat_service.get_user_sessions(user_id)
    return sessions

@app.get("/health")
def check_db_health():
    status = chat_service.check_db_health()
    return status

# ===== 认证接口：MySQL 用户表校验 =====

# 修复：使用 get_env_int 提供默认值 15，避免环境变量缺失时 int(None) 报错
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = get_env_int("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 15)

@app.post("/api/login")
def login(request_body: LoginRequest):
    user_id = request_body.userId
    password = request_body.password
    user_info = login_service.login(user_id, password)
    # login 返回 dict 才是成功：密码错误/用户不存在时返回的是字符串提示
    if not isinstance(user_info, dict):
        return Response.failed(user_info or "用户 ID 或密码错误")
    token = create_access_token(
        data={
            "sub": str(user_info["user_id"] + ":" + user_info["username"]),
            "role": user_info.get("role", "学员"),  # 管理员角色用于资源越权放行
        },
        expires_delta=timedelta(minutes=int(JWT_ACCESS_TOKEN_EXPIRE_MINUTES)),
    )
    # 隐式 refresh token：只存 Redis 不下发前端，access 过期时由后端（jwt_utils）自动续签
    refresh_token = create_refresh_token(
        data={"sub": str(user_info["user_id"] + ":" + user_info["username"])}
    )
    r = cache_service.redis
    # setex 第二参数单位是「秒」：access 配置为分钟需 ×60
    r.setex(USER_TOKEN_KEY.format(user_id=user_id), int(JWT_ACCESS_TOKEN_EXPIRE_MINUTES) * 60, token)
    r.setex(
        USER_REFRESH_TOKEN_KEY.format(user_id=user_id),
        REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        refresh_token,
    )
    return {"ok": True, "token": token, "user_info": user_info}


@app.post("/api/register")
def register(request_body: RegisterRequest):
    # 显式传递参数，替代原 *request_body 隐式展开
    flag, response = login_service.register(
        username=request_body.userName,
        user_id=request_body.userId,
        password=request_body.password
    )
    if flag:
        return Response.success(response)
    else:
        return Response.failed(response)


@app.post("/api/recover")
def recover(request_body: RecoverRequest):
    user_id = request_body.userId
    new_password = request_body.newPassword
    response = login_service.recover(user_id, new_password)
    if not response:
        return Response.failed("注册失败")
    elif response == 1:
        return Response.success()
    else:
        return Response.failed(response)


# ===== 前端静态托管（开发模式：localhost:8000 直达页面，免 Nginx）=====
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "resources" / "frontend"

# 认证页面（SPA 路由）：Vue Router history 模式无 #，直链/刷新 /api/login 等路径时
# 由后端直接返回 index.html，交给前端路由渲染对应认证组件（nginx 走 /api/ 代理同样生效）
@app.get("/api/login")
@app.get("/api/register")
@app.get("/api/recover")
def auth_page():
    return FileResponse(FRONTEND_DIR / "index.html")

# SPA 兜底（替代 mount 静态托管，需注册在所有 API 路由之后）：
#   - 静态资源（favicon.png 等真实文件）直接返回文件
#   - /api/* 未知路径返回 404 JSON（API 路由已在上面优先匹配）
#   - 其余路径（/chat 等前端 history 路由）返回 index.html，直链/刷新不再 404 空白
@app.get("/{full_path:path}")
def spa_or_static(full_path: str):
    file = FRONTEND_DIR / full_path
    if full_path and file.is_file():
        return FileResponse(file)
    if full_path.startswith("api/"):
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)