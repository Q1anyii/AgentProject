"""
系统路由：健康检查 / 认证页面 / SPA 静态托管兜底

对应原 main.py 中的 /health、认证页面 GET 路由、以及 SPA 兜底 /{full_path:path}。
注意：本 router 必须最后注册（SPA 兜底路由会匹配所有未被其他路由捕获的路径）。
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from service.chat_service import chat_service

router = APIRouter(tags=["系统"])

# 前端静态资源目录
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "resources" / "frontend"


@router.get("/health")
def check_db_health():
    """健康检查：返回数据库连接状态。"""
    status = chat_service.check_db_health()
    return status


# 认证页面（SPA 路由）：Vue Router history 模式无 #，直链/刷新 /api/login 等路径时
# 由后端直接返回 index.html，交给前端路由渲染对应认证组件（nginx 走 /api/ 代理同样生效）
@router.get("/api/login")
@router.get("/api/register")
@router.get("/api/recover")
def auth_page():
    """返回前端 index.html（认证页面由前端路由渲染）。"""
    return FileResponse(FRONTEND_DIR / "index.html")


# SPA 兜底（替代 mount 静态托管，需注册在所有 API 路由之后）：
#   - 静态资源（favicon.png 等真实文件）直接返回文件
#   - /api/* 未知路径返回 404 JSON（API 路由已在上面优先匹配）
#   - 其余路径（/chat 等前端 history 路由）返回 index.html，直链/刷新不再 404 空白
@router.get("/{full_path:path}")
def spa_or_static(full_path: str):
    """SPA 静态托管兜底。"""
    file = FRONTEND_DIR / full_path
    if full_path and file.is_file():
        return FileResponse(file)
    if full_path.startswith("api/"):
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    return FileResponse(FRONTEND_DIR / "index.html")
