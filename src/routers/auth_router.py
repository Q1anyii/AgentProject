"""
认证路由：登录 / 注册 / 密码找回

对应原 main.py 中的 /api/login、/api/register、/api/recover 接口。
"""

from datetime import timedelta

from fastapi import APIRouter

from config import get_env_int
from constant.cache_constant import USER_TOKEN_KEY, USER_REFRESH_TOKEN_KEY
from schemas.request_schemas.login_schema import LoginRequest, RegisterRequest, RecoverRequest
from service.cache_service import cache_service
from service.login_service import login_service
from utils.response_util import Response
from utils.jwt_utils import create_access_token, create_refresh_token, REFRESH_TOKEN_EXPIRE_DAYS

router = APIRouter(tags=["认证"])

# access token 过期时间（分钟），默认 15
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = get_env_int("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 15)


@router.post("/api/login")
def login(request_body: LoginRequest):
    """用户登录：校验 MySQL 用户表，返回 JWT token + 用户信息。"""
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


@router.post("/api/register")
def register(request_body: RegisterRequest):
    """用户注册：创建新用户。"""
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


@router.post("/api/recover")
def recover(request_body: RecoverRequest):
    """密码找回/重置：根据 user_id 设置新密码。"""
    user_id = request_body.userId
    new_password = request_body.newPassword
    response = login_service.recover(user_id, new_password)
    if not response:
        return Response.failed("注册失败")
    elif response == 1:
        return Response.success()
    else:
        return Response.failed(response)
