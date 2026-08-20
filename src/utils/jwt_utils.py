import os
from datetime import datetime, timedelta, UTC
from typing import Optional

import bcrypt
import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Response, status
from fastapi.security import OAuth2PasswordBearer
from loguru import logger
from pydantic import BaseModel

# 加载环境变量
load_dotenv()

# ---------------------- JWT配置 ----------------------
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 120))
# 隐式 refresh token 有效期（天）：只存 Redis 不下发前端，access 过期时由后端自动续签
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", 30))


# 从Header拿token： Authorization: Bearer <token>
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")


class TokenData(BaseModel):
    user_id: Optional[str] = None
    username: Optional[str] = None
    role: Optional[str] = None


# ---------------------- 密码哈希（bcrypt最大72字节，先截断）---------------------
def get_password_hash(plain_password: str) -> str:
    """生成密码哈希"""
    pw_bytes = plain_password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pw_bytes, salt)
    return hashed.decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """校验密码"""
    pw_bytes = plain_password.encode("utf-8")[:72]
    hash_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(pw_bytes, hash_bytes)


# ---------------------- JWT Token工具 ----------------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    # exp 必须是 datetime/数字时间戳，转成字符串会导致所有 token 验签失败
    expire = datetime.now(UTC) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"type": "access", "exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None):
    """签发隐式 refresh token：仅存 Redis，不下发前端，access 过期时由后端自动续签。"""
    to_encode = data.copy()
    expire = datetime.now(UTC) + (expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
    to_encode.update({"type": "refresh", "exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ---------------------- 依赖：解析token，获取当前登录用户 ----------------------
def get_current_user(token: str = Depends(oauth2_scheme), response: Response = None):
    """路由层鉴权：JWT 验签 + Redis 登录态校验，access 过期时用隐式 refresh 自动续签。

    - Redis 无该用户 token：未登录/已登出 → 401（登出即时生效）
    - 请求 token 与 Redis 不一致：已被轮换（并发续签/他端登录），以 Redis 中最新 token 为准
      并回传 X-New-Access-Token 让前端自动同步，避免并发下误判 401
    - access 过期但 refresh（仅存 Redis）有效：签发新 access+refresh 覆盖 Redis，回传新 access
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # 第一步：先不验过期解析（过期 token 也要能定位用户，用于后续续签判定）
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
        sub = payload.get("sub")
        if sub is None:
            raise credentials_exception
        user_id = str(sub).split(":")[0]
        username = str(sub).split(":")[-1]
        role = payload.get("role")
    except jwt.PyJWTError:
        raise credentials_exception

    # 第二步：Redis 登录态校验（惰性导入避免循环依赖）
    from constant.cache_constant import USER_TOKEN_KEY, USER_REFRESH_TOKEN_KEY
    from service.cache_service import cache_service

    r = cache_service.redis
    token_key = USER_TOKEN_KEY.format(user_id=user_id)
    try:
        stored = r.get(token_key)
        if isinstance(stored, bytes):
            stored = stored.decode('utf-8')
    except Exception as e:
        # Redis 不可用：无法确认登录态，保守按未登录拒绝（401），避免 500 泄漏
        logger.warning(f"Redis 不可用，鉴权拒绝：{e}")
        raise credentials_exception
    if not stored:
        # 登录态不存在：未登录或已登出（登出即删除该 key，实现主动失效）
        raise credentials_exception

    effective_token = token
    if stored != token:
        # 请求 token 已被轮换：以 Redis 中最新 token 为准，用其 payload 重建身份
        try:
            stored_payload = jwt.decode(stored, SECRET_KEY, algorithms=[ALGORITHM])
        except jwt.PyJWTError:
            raise credentials_exception
        sub = stored_payload.get("sub")
        if sub is None:
            raise credentials_exception
        user_id = str(sub).split(":")[0]
        username = str(sub).split(":")[-1]
        role = stored_payload.get("role")
        effective_token = stored
    else:
        try:
            # 严格验签+过期校验：token 仍有效则直接放行
            jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        except jwt.ExpiredSignatureError:
            # 隐式续签：refresh token 只在 Redis 里，前端全程无感知
            refresh_token = r.get(USER_REFRESH_TOKEN_KEY.format(user_id=user_id))
            if isinstance(refresh_token, bytes):
                refresh_token = refresh_token.decode('utf-8')
            if not refresh_token:
                raise credentials_exception
            try:
                jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
            except jwt.PyJWTError:
                raise credentials_exception
            effective_token = create_access_token({"sub": f"{user_id}:{username}", "role": role})
            # refresh 轮换：续签时生成新 refresh 覆盖旧值，防重放
            new_refresh = create_refresh_token({"sub": f"{user_id}:{username}"})
            try:
                # setex 第二参数单位是「秒」：access 配置为分钟需 ×60
                r.setex(token_key, ACCESS_TOKEN_EXPIRE_MINUTES * 60, effective_token)
                r.setex(USER_REFRESH_TOKEN_KEY.format(user_id=user_id), REFRESH_TOKEN_EXPIRE_DAYS * 86400, new_refresh)
            except Exception as e:
                # 续签写 Redis 失败：无法落定新登录态，保守拒绝
                logger.warning(f"续签写 Redis 失败，鉴权拒绝：{e}")
                raise credentials_exception
        except jwt.PyJWTError:
            raise credentials_exception

    # 新 token 通过响应头回传，前端拦截器同步本地登录态，实现无感续签
    if response is not None and effective_token != token:
        response.headers["X-New-Access-Token"] = effective_token

    return TokenData(user_id=user_id, username=username, role=role)


def get_username_from_token(token: str) -> Optional[str]:
    """解析 token 提取 username（供图内节点等场景：先按 Redis key 取 token，再解析）。

    不校验过期：登录态是否有效由 Redis key 是否存在决定，这里只做身份解析。
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
        sub = payload.get("sub")
        return str(sub).split(":")[-1] if sub else None
    except jwt.PyJWTError:
        return None

if __name__ == "__main__":
    print(get_password_hash("1234"))
