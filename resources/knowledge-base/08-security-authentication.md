# 安全与认证实践

> 基于 AgentProject 项目总结，涵盖 JWT 认证、密码加密、资源归属校验、会话安全、敏感信息保护、限流等安全实践。

## 一、认证架构

```
┌─────────────────────────────────────────────────────┐
│  前端 (Vue 3 SPA)                                     │
│  - 登录获取 token，存储在 localStorage                │
│  - 每次请求带 Authorization: Bearer {token}          │
│  - 401 时自动跳转登录页                               │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP Header
┌──────────────────────▼──────────────────────────────┐
│  API 层 (FastAPI)                                     │
│  - Depends(get_current_user) 解析 JWT                │
│  - require_self_or_admin 资源归属校验                  │
│  - 会话归属校验（thread_id 属于当前用户）              │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  服务层                                                │
│  - login_service：密码验证、用户查询                   │
│  - cache_service：Redis 存储 token（支持主动失效）     │
│  - jwt_utils：token 生成、解析、验证、自动续签         │
└─────────────────────────────────────────────────────┘
```

## 二、JWT 认证

### 2.1 Token 生成

```python
from datetime import timedelta
from jose import jwt

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def create_refresh_token(data: dict):
    expire = datetime.utcnow() + timedelta(days=30)
    to_encode = data.copy()
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
```

**使用**：
```python
token = create_access_token(
    data={
        "sub": str(user_info["user_id"] + ":" + user_info["username"]),
        "role": user_info.get("role", "学员"),
    },
    expires_delta=timedelta(minutes=15),
)
```

### 2.2 Token 解析与验证

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        sub: str = payload.get("sub")
        if sub is None:
            raise HTTPException(status_code=401, detail="无效的认证凭证")
        user_id, username = sub.split(":", 1)
        role = payload.get("role", "学员")
        return TokenData(user_id=user_id, username=username, role=role)
    except JWTError:
        raise HTTPException(status_code=401, detail="认证已过期，请重新登录")
```

### 2.3 Redis 存储 Token（支持主动失效）

```python
# 登录时存储
r.setex(USER_TOKEN_KEY.format(user_id=user_id), 15 * 60, token)
r.setex(USER_REFRESH_TOKEN_KEY.format(user_id=user_id), 30 * 86400, refresh_token)

# 验证时检查 Redis（可选，支持主动登出/改密码后失效）
def verify_token_in_redis(user_id, token):
    stored_token = r.get(USER_TOKEN_KEY.format(user_id=user_id))
    return stored_token == token
```

**为什么用 Redis 存 token**：
- JWT 本身是无状态的，签发后无法主动失效
- Redis 存储可以支持：主动登出、改密码后旧 token 失效、限流
- 隐式 refresh token：只存 Redis 不下发前端，access 过期时后端自动续签

### 2.4 双 Token 机制

| Token 类型 | 有效期 | 存储位置 | 用途 |
|-----------|--------|----------|------|
| Access Token | 15 分钟 | 前端 localStorage + Redis | API 认证 |
| Refresh Token | 30 天 | 仅 Redis（不下发前端） | 自动续签 access token |

**设计特点**：
- Access token 短期有效，降低泄露风险
- Refresh token 不下发前端，更安全
- 后端自动续签，用户无感知

## 三、密码安全

### 3.1 密码加密存储

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
```

**bcrypt 特点**：
- 自动加盐，无需手动管理 salt
- 计算成本可调（默认 12 轮），抗暴力破解
- 哈希结果包含算法、成本、salt，验证时自动解析

### 3.2 密码修改流程

```python
@app.put("/api/users/{user_id}/password")
def update_password(user_id, request_body, current_user = Depends(require_self_or_admin)):
    # 1. 验证原密码
    user_info = login_service.login(user_id, request_body.old_password)
    if not isinstance(user_info, dict):
        return Response.failed("原密码错误")
    # 2. 修改密码（复用 recover 逻辑）
    result = login_service.recover(user_id, request_body.new_password)
    if result == 1:
        return Response.success("密码修改成功")
    return Response.failed(result or "密码修改失败")
```

**关键点**：
- 修改密码必须验证原密码
- 新密码要有强度要求（最少 6 位）
- 修改成功后旧 token 应失效（清除 Redis 中的 token）

## 四、资源归属校验

### 4.1 用户级资源校验

```python
def require_self_or_admin(user_id: str, current_user: TokenData = Depends(get_current_user)):
    """只允许本人访问自己的资源，管理员角色放行。"""
    if str(current_user.user_id) != user_id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权访问该用户资源")
    return current_user

# 使用
@app.get("/api/users/{user_id}/profile")
def get_profile(user_id: str, current_user: TokenData = Depends(require_self_or_admin)):
    ...
```

### 4.2 会话级资源校验

会话（thread_id）不是用户 ID，需要在业务层校验归属。

```python
@app.post("/api/chat/")
def chat(request_body, current_user = Depends(get_current_user)):
    thread_id = request_body.thread_id
    # 会话归属校验：会话已存在但非本人所有时拒绝
    owner = chat_service.get_thread_user_id(thread_id)
    if owner and owner != str(current_user.user_id) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权使用该会话")
    ...
```

**为什么需要会话归属校验**：
- 如果不校验，任意用户可用他人 thread_id 发消息
- LangGraph 会用当前用户覆盖该会话归属 metadata，造成会话劫持
- 历史记录、删除会话等接口也需要同样校验

### 4.3 文件级资源校验

```python
@app.delete("/api/files/{file_id}")
def delete_file(file_id: int, current_user = Depends(get_current_user)):
    # 删除时校验文件属于当前用户
    success = file_upload_service.delete_file(file_id, str(current_user.user_id))
    if success:
        return Response.success("文件删除成功")
    return Response.failed("文件不存在或无权删除")
```

## 五、敏感信息保护

### 5.1 不注入敏感字段到上下文

```python
@dataclass
class CtxUser:
    uid: int
    user_id: str
    password: str | None  # 敏感字段不注入，传 None
    username: str
    ...

# 使用时
user_info = CtxUser(
    uid=user_row["id"],
    user_id=user_row["user_id"],
    password=None,  # 敏感字段不注入，工具无法访问
    username=current_user.username,
    ...
)
```

### 5.2 日志脱敏

```python
def print_config_summary():
    for key, desc in REQUIRED_ENV_VARS:
        value = os.getenv(key)
        if value and ("KEY" in key or "SECRET" in key or "PASSWORD" in key):
            # 敏感信息只显示前4后4位
            masked = value[:4] + "*" * (len(value) - 8) + value[-4:]
            logger.info(f"  {key}: {masked} (已配置)")
```

### 5.3 响应中不返回敏感字段

```python
@app.get("/api/users/{user_id}/profile")
def get_profile(user_id, current_user = Depends(require_self_or_admin)):
    profile = user_profile_service.get_profile(user_id)
    if profile:
        return {
            "ok": True,
            "data": {
                "user_id": profile.get("user_id"),
                "username": profile.get("username"),
                "avatar": profile.get("avatar"),
                # 不返回 password、system_prompt 等敏感/大字段
                "assistant_style": profile.get("assistant_style"),
                "theme": profile.get("theme", "default"),
            }
        }
```

### 5.4 .env 不提交版本控制

```gitignore
# .gitignore
.env
.env.local
*.key
*.pem
```

## 六、请求限流

### 6.1 限流中间件

```python
from middleware.rate_limit_middleware import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware)
```

**限流策略**：
- 对 `/api/chat/` 等消耗 LLM 配额的接口限流
- 按 user_id 或 IP 计数
- 时间窗口内超过限制返回 429

### 6.2 Redis 限流实现

```python
def is_rate_limited(self, key: str, limit: int = 10, window: int = 60) -> bool:
    """滑动窗口限流。"""
    count = self.redis.incr(key)
    if count == 1:
        self.redis.expire(key, window)
    return count > limit
```

## 七、常见安全陷阱

### 7.1 SQL 注入

```python
# 错误：字符串拼接
cur.execute(f"SELECT * FROM users WHERE id = '{user_id}'")

# 正确：参数化查询
cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
```

### 7.2 路径遍历

文件上传/下载时，用户可能传入 `../../etc/passwd` 等路径。

```python
# 错误：直接使用用户输入的路径
file_path = f"/data/{user_input}"

# 正确：校验路径，限制在指定目录内
base_dir = Path("/data").resolve()
file_path = (base_dir / user_input).resolve()
if not str(file_path).startswith(str(base_dir)):
    raise HTTPException(status_code=400, detail="非法路径")
```

### 7.3 JWT 密钥硬编码

```python
# 错误：硬编码密钥
JWT_SECRET_KEY = "my-secret-key-123"

# 正确：从环境变量读取
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
# 启动时校验
if not JWT_SECRET_KEY:
    raise ConfigError("JWT_SECRET_KEY 未配置")
```

### 7.4 CORS 配置过宽

```python
# 错误：允许所有来源
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True)

# 正确：只允许指定来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 7.5 未验证文件类型

文件上传时只检查扩展名，不检查实际内容。

```python
# 错误：只检查扩展名
if not file.filename.endswith(('.png', '.jpg')):
    raise HTTPException(400, "不支持的文件类型")

# 正确：检查 MIME 类型 + 扩展名 + 文件头（魔数）
ALLOWED_TYPES = {"image/png", "image/jpeg"}
if file.content_type not in ALLOWED_TYPES:
    raise HTTPException(400, "不支持的文件类型")
```

## 八、个人见解

1. **安全是分层的，不是单点的**：JWT 认证、密码加密、资源归属校验、限流、日志脱敏，每一层都不能少。攻击者只需要找到一个漏洞，防守者需要堵住所有漏洞。

2. **资源归属校验是最容易被忽视的安全漏洞**：很多项目做了登录认证，但没有做资源归属校验，导致用户 A 可以访问用户 B 的数据。本项目在用户级、会话级、文件级都做了归属校验，是正确的做法。

3. **JWT 不是银弹**：JWT 无状态的优点也是缺点——签发后无法主动失效。本项目用 Redis 存储 token 来支持主动失效，是很好的实践。但要注意 Redis 故障时的降级策略。

4. **密码安全怎么强调都不过分**：bcrypt 是最低要求，还要考虑密码强度策略、登录失败锁定、改密码后旧 token 失效。很多项目的密码安全只做了加密存储，其他都没做。

5. **安全要在设计时考虑，不是上线前补**：本项目的 CtxUser 不注入 password、响应不返回敏感字段、日志脱敏，都是在设计时就考虑了安全。如果等上线前再补，很容易遗漏，而且改造成本高。
