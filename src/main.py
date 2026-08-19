from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from loguru import logger

from schemas.request_schemas.chat_schema import ChatRequest
from schemas.request_schemas.login_schema import *
from service.chat_service import ChatService
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse

from dotenv import load_dotenv

from service.login_service import LoginService

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

@app.post("/api/chat/")
async def chat(request_body: ChatRequest):
    query = request_body.query
    thread_id = request_body.thread_id
    # answer = await chat_service.a_invoke("user_001", thread_id, query)
    # return {"thread_id": thread_id, "answer": answer}
    event_stream = chat_service.stream("user_001", thread_id, query)
    return StreamingResponse(event_stream, media_type="text/event-stream")

@app.get("/api/chat/{thread_id}/history")
def get_history_session(thread_id: str):
    history_session = chat_service.get_history_session(thread_id)
    return history_session

@app.get("/api/users/{user_id}/memory")
def get_memory(user_id: str):
    memory = chat_service.get_memory(user_id)
    return memory

@app.delete("/api/chat/{thread_id}")
def delete_session_by_id(thread_id: str):
    success = chat_service.delete_session_by_id(thread_id)
    return success

@app.get("/api/users/{user_id}/sessions")
def get_sessions_by_user_id(user_id: str):
    sessions = chat_service.get_user_sessions(user_id)
    return sessions

@app.get("/health")
def check_db_health():
    status = chat_service.check_db_health()
    return status

# ===== 认证：静态测试账号 user / 1234（演示环境）=====
# 静态账号：登录成功签发临时 token（演示用），注册/找回为占位提示
STATIC_ACCOUNT = {"id": "user", "password": "1234", "name": "测试用户", "role": "学员"}


@app.post("/api/login")
def login(request_body: LoginRequest):
    if request_body :
        user_id = request_body.userId
        password = request_body.password
        user_info = login_service.login(user_id, password)
        return {"ok": True, "token": uuid4().hex, "user_info": user_info}
    return JSONResponse({"ok": False, "message": "用户 ID 或密码错误"}, status_code=401)


@app.post("/api/register")
def register(request_body: RegisterRequest):
    return JSONResponse(
        {"ok": False, "message": "演示环境暂不支持注册，请使用测试账号 user / 1234"},
        status_code=400,
    )


@app.post("/api/recover")
def recover(request_body: RecoverRequest):
    return JSONResponse(
        {"ok": False, "message": "演示环境暂不支持找回密码，请联系管理员"},
        status_code=400,
    )

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