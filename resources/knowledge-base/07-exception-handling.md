# 异常处理机制实践

> 基于 AgentProject 项目总结，涵盖全局异常处理、业务异常、降级策略、资源清理、错误码设计等异常处理最佳实践。

## 一、异常处理分层架构

```
┌─────────────────────────────────────────┐
│  API 层 (FastAPI)                        │
│  - 全局异常处理器 (@exception_handler)   │
│  - HTTPException 统一包装                 │
│  - 响应格式统一 {ok, detail}             │
├─────────────────────────────────────────┤
│  服务层 (Service)                         │
│  - 业务异常抛出 (ValueError, 自定义异常)  │
│  - 降级策略 (非核心功能失败不中断主流程)   │
│  - 事务回滚                               │
├─────────────────────────────────────────┤
│  数据层 (Database)                        │
│  - 连接异常 (重连机制)                    │
│  - SQL 异常 (参数化查询防注入)            │
│  - 超时处理                               │
└─────────────────────────────────────────┘
```

## 二、全局异常处理

### 2.1 捕获所有未处理异常

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常捕获：所有未处理的异常统一返回 500 + 结构化错误。

    - 记录完整异常信息到日志（含堆栈），便于排查
    - 返回给客户端的信息不包含堆栈，只返回通用错误提示
    - HTTPException 由 FastAPI 默认处理，不会进入此处理器
    """
    logger.exception(f"未处理的异常 | path={request.url.path} | method={request.method}")
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "detail": "服务器内部错误，请稍后重试或联系管理员",
            "error_type": type(exc).__name__,
        },
    )
```

### 2.2 HTTPException 统一包装

```python
@app.exception_handler(HTTPException)
async def http_exception_handler(exc: HTTPException):
    """HTTPException 统一包装为 {ok, detail} 格式，与业务接口响应风格一致。"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "detail": exc.detail},
        headers=exc.headers,
    )
```

### 2.3 为什么需要全局异常处理

- **统一响应格式**：所有错误返回 `{ok: false, detail: "..."}`，前端处理一致
- **安全**：不向客户端暴露堆栈信息和内部实现细节
- **可观测性**：集中记录异常日志，便于监控和排查
- **用户体验**：返回友好的错误提示，而非原始异常信息

## 三、业务异常处理

### 3.1 业务错误返回（不抛异常）

对于预期内的业务错误（如密码错误、用户不存在），用返回值而非异常。

```python
# service/login_service.py
def login(self, user_id, password):
    user = self._get_user(user_id)
    if not user:
        return "用户不存在"  # 返回字符串错误提示
    if not verify_password(password, user["password"]):
        return "用户 ID 或密码错误"
    return user  # 返回 dict 表示成功

# API 层
@app.post("/api/login")
def login(request_body: LoginRequest):
    user_info = login_service.login(user_id, password)
    if not isinstance(user_info, dict):
        return Response.failed(user_info or "用户 ID 或密码错误")
    # 成功逻辑...
    return {"ok": True, "token": token, "user_info": user_info}
```

**原则**：
- 预期内的业务错误用返回值，不抛异常
- 未预期的系统错误抛异常，由全局处理器处理
- 用返回类型区分成功/失败（如 dict 成功，str 失败）

### 3.2 自定义异常类

对于需要跨层传递的业务错误，定义自定义异常。

```python
class ConfigError(Exception):
    """配置错误异常：缺少必填环境变量或值格式错误时抛出"""
    pass

class BusinessError(Exception):
    """业务异常基类"""
    def __init__(self, message: str, code: int = 400):
        self.message = message
        self.code = code
        super().__init__(message)
```

### 3.3 参数校验异常

FastAPI + Pydantic 自动处理参数校验，返回 422 错误。

```python
from pydantic import BaseModel, Field

class PasswordUpdateRequest(BaseModel):
    old_password: str = Field(..., min_length=6, description="原密码")
    new_password: str = Field(..., min_length=6, description="新密码")

@app.put("/api/users/{user_id}/password")
def update_password(user_id: str, request_body: PasswordUpdateRequest):
    # 如果 old_password 少于 6 位，FastAPI 自动返回 422
    ...
```

## 四、降级策略（Graceful Degradation）

### 4.1 非核心功能降级

非核心功能失败时，记录日志并降级，不中断主流程。

```python
def get_user_system_prompt(user_id: str, base_prompt: str = None) -> str:
    """组装用户级 system prompt：基础默认 + 用户自定义内容。

    获取用户自定义内容失败时，降级使用基础 prompt，不影响对话。
    """
    if base_prompt is None:
        base_prompt = system_prompt

    try:
        from service.user_profile_service import user_profile_service
        custom_prompt = user_profile_service.get_system_prompt(user_id)
        if custom_prompt and custom_prompt.strip():
            return f"{base_prompt}\n\n【用户自定义设定】\n{custom_prompt.strip()}"
    except Exception as e:
        # 降级：获取失败时使用基础 prompt
        logger.warning(f"获取用户自定义 system prompt 失败 user_id={user_id}: {e}，使用基础 prompt")

    return base_prompt
```

### 4.2 可选配置降级

MCP 配置错误时不阻塞应用启动。

```python
def load_mcp_server_configs() -> list[dict]:
    """加载 MCP 服务器配置。

    MCP 是可选项，配置错误不阻塞应用启动，单条无效只跳过该条。
    """
    raw = os.getenv("MCP_SERVERS")
    if not raw:
        logger.info("未配置 MCP_SERVERS，跳过 MCP 工具加载")
        return []
    try:
        servers = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"MCP_SERVERS 不是合法 JSON，跳过 MCP 工具加载：{e}")
        return []
    # 单条校验失败只跳过该条，不影响其他
    for cfg in servers:
        if not isinstance(cfg, dict):
            logger.warning(f"忽略无效的 MCP 服务器配置项：{cfg}")
            continue
        ...
    return validated
```

### 4.3 降级策略分类

| 场景 | 降级方式 | 示例 |
|------|----------|------|
| 非核心功能失败 | 记录日志，使用默认值 | 用户自定义 prompt 获取失败 → 用基础 prompt |
| 可选配置错误 | 跳过该配置，不阻塞启动 | MCP 配置错误 → 不加载 MCP 工具 |
| 外部服务不可用 | 使用缓存或默认值 | 重排序 API 超时 → 用检索原始排序 |
| 单条数据异常 | 跳过该条，处理其余 | MCP 配置某条无效 → 跳过该条 |
| 数据库连接失败 | 重试 + 快速失败 | MySQL 连接断开 → 自动重连 |

## 五、资源清理

### 5.1 Lifespan 上下文管理器

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动阶段
    chat_service.open()
    cache_service.open()
    try:
        yield  # 运行期间
    finally:
        # 关闭阶段：确保资源释放，即使运行中出错
        chat_service.close(timeout=10)
        cache_service.close()
```

### 5.2 数据库连接关闭

```python
class UserProfileService:
    def close(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception as e:
                logger.warning(f"关闭 MySQL 连接失败：{e}")
            finally:
                self._conn = None
```

### 5.3 MCP 子进程关闭

```python
# lifespan 关闭阶段
for holder in mcp_holders:
    try:
        await holder.close()
    except Exception as e:
        logger.warning(f"关闭 MCP holder 失败：{e}")
```

**原则**：
- 资源释放要带超时，避免无限等待
- 释放失败要记录日志但不中断其他资源释放
- 用 try/finally 确保即使出错也执行清理

## 六、异常处理常见陷阱

### 6.1 吞掉异常

```python
# 错误：静默吞掉异常，难以排查
try:
    do_something()
except:
    pass

# 正确：至少记录日志
try:
    do_something()
except Exception as e:
    logger.error(f"操作失败：{e}", exc_info=True)
    raise  # 或降级处理
```

### 6.2 过度宽泛的异常捕获

```python
# 错误：捕获所有异常，可能掩盖编程错误
try:
    result = 1 / 0  # ZeroDivisionError 是 bug，不是预期异常
except Exception:
    return "操作失败"

# 正确：只捕获预期异常
try:
    result = int(user_input)
except ValueError:
    return "输入不是有效数字"
```

### 6.3 在循环中捕获异常不继续

```python
# 错误：一条失败导致整个批次中断
for item in items:
    process(item)  # 如果抛异常，后续 item 不处理

# 正确：单条失败不影响其他
for item in items:
    try:
        process(item)
    except Exception as e:
        logger.warning(f"处理 {item} 失败：{e}")
        continue
```

### 6.4 异常信息泄露

```python
# 错误：向客户端返回原始异常信息（可能包含敏感信息）
return JSONResponse(status_code=500, content={"error": str(exc)})

# 正确：返回通用错误提示，详细信息只记日志
logger.exception(f"内部错误：{exc}")
return JSONResponse(status_code=500, content={"detail": "服务器内部错误"})
```

### 6.5  finally 中 return 覆盖异常

```python
# 错误：finally 中的 return 会覆盖 try 中抛出的异常
def func():
    try:
        raise ValueError("错误")
    finally:
        return "ok"  # 异常被吞掉！

# 正确：finally 中只做清理，不 return
def func():
    try:
        raise ValueError("错误")
    finally:
        cleanup()
```

## 七、错误码设计

### 7.1 HTTP 状态码规范

| 状态码 | 含义 | 使用场景 |
|--------|------|----------|
| 200 | 成功 | 请求成功处理 |
| 400 | 请求错误 | 参数校验失败、业务逻辑错误 |
| 401 | 未认证 | 未登录、token 无效 |
| 403 | 无权限 | 登录但无权访问该资源 |
| 404 | 不存在 | 资源不存在 |
| 422 | 参数校验失败 | Pydantic 校验失败 |
| 429 | 请求过多 | 限流触发 |
| 500 | 服务器错误 | 未处理的异常 |

### 7.2 业务错误码

```python
class ErrorCode:
    # 通用错误 1000-1999
    SUCCESS = 0
    INVALID_PARAMS = 1001
    UNAUTHORIZED = 1002
    FORBIDDEN = 1003
    NOT_FOUND = 1004

    # 用户相关 2000-2999
    USER_NOT_EXIST = 2001
    WRONG_PASSWORD = 2002
    USER_ALREADY_EXIST = 2003

    # 对话相关 3000-3999
    SESSION_NOT_EXIST = 3001
    SESSION_FORBIDDEN = 3002
```

## 八、个人见解

1. **异常处理的核心是边界清晰**：哪些异常是预期内的（用返回值），哪些是未预期的（抛异常），要在设计时就明确。模糊的边界会导致代码里到处 try-except，既不优雅也不可靠。

2. **全局异常处理器是最后一道防线，不是唯一的防线**：不要把所有错误处理都推给全局处理器。业务层应该处理预期内的错误，全局处理器只处理未预期的系统错误。

3. **降级策略要设计在架构里，不是出问题后临时加**：本项目的用户自定义 prompt 降级、MCP 配置降级，都是在设计时就考虑了失败场景。好的架构应该假设每个外部依赖都可能失败，并设计好降级路径。

4. **日志是异常处理的另一半**：捕获异常但不记日志，等于没处理。日志要包含足够的上下文（path、method、user_id、参数），便于排查。但也要注意不要记录敏感信息（密码、token）。

5. **异常处理不要过度设计**：小项目不需要复杂的自定义异常体系和错误码系统。用 FastAPI 的 HTTPException + 全局处理器 + 日志，就能覆盖 90% 的场景。等项目变大了再逐步完善。
