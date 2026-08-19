import os
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from loguru import logger

from context.user_context import CtxUser
from schemas.request_schemas.chat_schema import ChatRequest
from schemas.request_schemas.login_schema import *
from service.chat_service import ChatService
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse

from dotenv import load_dotenv

from service.login_service import LoginService
from temp.response_temp import Response
from utils.jwt_utils import get_current_user, create_access_token, TokenData

load_dotenv()

chat_service = ChatService()
login_service = LoginService()

@asynccontextmanager
async def lifespan(app: FastAPI):

    # ===== 启动阶段：yield 之前 =====
    logger.info("正在初始化 LangGraph 资源...")
    chat_service.open()
    login_service.open()
    logger.info("资源初始化完成")
    yield                                # ===== 应用运行期间（yield 挂起）=====
    # ===== 关闭阶段：yield 之后 =====
    logger.info("正在释放资源...")
    chat_service.close(timeout=10)
    login_service.close(timeout=10)
    logger.info("资源已释放")

app = FastAPI(title="Mitta AI", lifespan=lifespan)


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
    # 认证在路由层完成：JWT 解析出 user_id 后查库，构造请求级用户上下文（供图内工具读取）
    user_row = login_service.get_user_by_id(str(current_user.user_id))
    user_info = (
        CtxUser(
            uid=user_row["id"],
            user_id=user_row["user_id"],
            password=None,  # 敏感字段不注入，工具无法访问
            username=user_row["username"],
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

JWT_ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES")


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
            "sub": str(user_info["user_id"]),
            "role": user_info.get("role", "学员"),  # 管理员角色用于资源越权放行
        },
        expires_delta=timedelta(minutes=int(JWT_ACCESS_TOKEN_EXPIRE_MINUTES)),
    )
    return {"ok": True, "token": token, "user_info": user_info}


@app.post("/api/register")
def register(request_body: RegisterRequest):

    flag, response = login_service.register(*request_body)
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
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "recourses" / "frontend"

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