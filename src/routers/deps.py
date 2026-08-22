"""
公共依赖注入（FastAPI Depends）

存放跨路由复用的依赖函数，避免在 main.py 和各 router 中重复定义。
"""

from fastapi import Depends, HTTPException

from utils.jwt_utils import get_current_user, TokenData


def require_self_or_admin(user_id: str, current_user: TokenData = Depends(get_current_user)):
    """资源归属校验：只允许本人访问自己的资源，管理员角色放行。

    FastAPI 会自动把路径参数 user_id 注入本依赖（必须定义在使用它的路由之前）。
    """
    if str(current_user.user_id) != user_id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权访问该用户资源")
    return current_user
