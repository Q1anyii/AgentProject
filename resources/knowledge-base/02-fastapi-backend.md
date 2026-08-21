# FastAPI 后端开发实践

> 基于 AgentProject 项目总结，涵盖依赖注入、中间件、全局异常、SSE 流式、文件上传、SPA 托管等。

## 一、应用生命周期管理（Lifespan）

FastAPI 推荐用 `lifespan` 上下文管理器替代 `startup` / `shutdown` 事件，统一管理资源初始化和释放。

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ===== 启动阶段 =====
    validate_config()                    # 1. 校验配置（快速失败）
    chat_service.open()                  # 2. 初始化服务连接
    cache_service.open()
    logger.success("资源初始化完成")
    yield                                # ===== 运行期间挂起 =====
    # ===== 关闭阶段 =====
    logger.info("正在释放资源...")
    chat_service.close(timeout=10)      # 3. 释放资源（带超时）
    cache_service.close()

app = FastAPI(title="Mitta AI", lifespan=lifespan)
```

**最佳实践**：
- 启动时先校验配置，缺失则直接报错（快速失败原则）
- 关闭时按依赖逆序释放，带超时避免无限等待
- 数据库连接、Redis 连接、MCP 子进程都应在这里管理

## 二、依赖注入（Depends）

### 2.1 JWT 认证依赖

```python
from fastapi import Depends
from utils.jwt_utils import get_current_user, TokenData

@app.post("/api/chat/")
def chat(request_body: ChatRequest, current_user: TokenData = Depends(get_current_user)):
    # current_user 已包含 user_id, username, role
    ...
```

### 2.2 资源归属校验依赖

定义可复用的依赖函数，校验用户只能访问自己的资源。

```python
def require_self_or_admin(user_id: str, current_user: TokenData = Depends(get_current_user)):
    if str(current_user.user_id) != user_id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权访问该用户资源")
    return current_user

# 使用
@app.get("/api/users/{user_id}/profile")
def get_profile(user_id: str, current_user: TokenData = Depends(require_self_or_admin)):
    ...
```

**关键点**：
- FastAPI 自动把路径参数 `user_id` 注入依赖函数
- 依赖函数可以嵌套依赖（`require_self_or_admin` 依赖 `get_current_user`）
- 管理员角色放行，普通用户只能访问自己的资源

### 2.3 会话归属校验（业务层）

对于会话级资源，需要在业务层校验归属，因为 `thread_id` 不是用户 ID。

```python
@app.post("/api/chat/")
def chat(request_body: ChatRequest, current_user: TokenData = Depends(get_current_user)):
    # 会话归属校验：会话已存在但非本人所有时拒绝
    owner = chat_service.get_thread_user_id(thread_id)
    if owner and owner != str(current_user.user_id) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权使用该会话")
    ...
```

## 三、中间件（Middleware）

### 3.1 请求限流中间件

```python
from middleware.rate_limit_middleware import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware)
```

中间件适用于横切关注点：限流、日志、CORS、请求 ID 等。

**中间件 vs 依赖**：
- 中间件：全局生效，处理请求/响应的通用逻辑
- 依赖：按路由注入，处理业务相关的认证/校验

## 四、全局异常处理

### 4.1 捕获所有未处理异常

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常捕获：记录完整堆栈，返回通用错误，不暴露内部信息"""
    logger.exception(f"未处理的异常 | path={request.url.path}")
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "detail": "服务器内部错误，请稍后重试或联系管理员",
            "error_type": type(exc).__name__,  # 只返回异常类型，不返回堆栈
        },
    )
```

### 4.2 HTTPException 统一包装

```python
@app.exception_handler(HTTPException)
async def http_exception_handler(exc: HTTPException):
    """HTTPException 统一包装为 {ok, detail} 格式"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "detail": exc.detail},
        headers=exc.headers,
    )
```

**原则**：
- 日志记录完整堆栈（便于排查）
- 响应不暴露堆栈（安全考虑）
- 统一响应格式（`{ok, detail}`）

## 五、SSE 流式响应

### 5.1 StreamingResponse

```python
from fastapi.responses import StreamingResponse

@app.post("/api/chat/")
def chat(request_body: ChatRequest, current_user: TokenData = Depends(get_current_user)):
    event_stream = chat_service.stream(user_id, thread_id, query)
    return StreamingResponse(event_stream, media_type="text/event-stream")
```

### 5.2 前端接收（fetch + ReadableStream）

```javascript
const response = await fetch('/api/chat/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
    body: JSON.stringify({ query, thread_id }),
    signal: abortController.signal,  // 支持中断
});

const reader = response.body.getReader();
const decoder = new TextDecoder('utf-8');
let buffer = '';

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop();
    for (const event of events) {
        const line = event.trim();
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        // 处理 SSE 数据...
    }
}
```

### 5.3 中断 SSE（AbortController）

```javascript
// 发起请求时
const abortController = new AbortController();
const response = await fetch(url, { signal: abortController.signal, ... });

// 用户点击停止时
abortController.abort();
// fetch 会抛出 AbortError，在 catch 中处理
```

**注意**：LangGraph 同步 `stream` 无法在服务端主动中断，实际停止由前端关闭连接实现。

## 六、文件上传

### 6.1 multipart/form-data 上传

```python
from fastapi import UploadFile, File, Form
from typing import Optional

@app.post("/api/chat/upload")
async def upload_file(
    file: UploadFile = File(...),
    thread_id: Optional[str] = Form(None),
    current_user: TokenData = Depends(get_current_user),
):
    content = await file.read()
    result = file_upload_service.save_file(
        user_id=str(current_user.user_id),
        file_name=file.filename,
        file_content_bytes=content,
        file_type=file.content_type,
        thread_id=thread_id,
    )
    return {"ok": True, "data": result}
```

### 6.2 nginx 配置

```nginx
http {
    client_max_body_size 20m;  # 支持文件上传，比单文件限制大一些留余量
}
```

## 七、SPA 静态托管

### 7.1 兜底路由（替代 mount）

```python
from pathlib import Path
from fastapi.responses import FileResponse, JSONResponse

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "resources" / "frontend"

# 认证页面直链（Vue Router history 模式）
@app.get("/api/login")
@app.get("/api/register")
def auth_page():
    return FileResponse(FRONTEND_DIR / "index.html")

# SPA 兜底（必须注册在所有 API 路由之后）
@app.get("/{full_path:path}")
def spa_or_static(full_path: str):
    file = FRONTEND_DIR / full_path
    if full_path and file.is_file():
        return FileResponse(file)           # 静态资源直接返回
    if full_path.startswith("api/"):
        return JSONResponse({"detail": "Not Found"}, status_code=404)  # API 404
    return FileResponse(FRONTEND_DIR / "index.html")  # 前端路由返回 index.html
```

**关键点**：
- 兜底路由必须注册在所有具体路由之后
- 静态资源（favicon, css, js）优先返回文件
- `/api/*` 未知路径返回 404 JSON
- 其余路径返回 index.html，支持 Vue Router history 模式直链/刷新

## 八、响应格式统一

### 8.1 通用响应工具

```python
# utils/response_util.py
class Response:
    @staticmethod
    def success(data=None, msg="success"):
        return {"ok": True, "data": data, "msg": msg}

    @staticmethod
    def failed(msg="error", code=400):
        return {"ok": False, "detail": msg}
```

**使用**：
```python
if flag:
    return Response.success(response)
else:
    return Response.failed(response)
```

## 九、个人见解

1. **lifespan 是 FastAPI 最被低估的特性**：很多人还用 `@app.on_event("startup")`，但官方已标记为 deprecated。lifespan 能保证启动和关闭逻辑在同一个上下文，资源管理更可靠。

2. **依赖注入是 FastAPI 的灵魂**：善用 `Depends` 可以把认证、校验、分页等逻辑抽成可复用组件，路由函数只关注业务逻辑。不要在每个路由里重复写 JWT 解析代码。

3. **SSE 比 WebSocket 更适合 AI 对话**：SSE 是单向流式，协议简单，天然支持 HTTP 缓存和代理，不需要额外的连接管理。只有需要双向实时通信时才用 WebSocket。

4. **全局异常处理要分层**：`HTTPException` 处理业务错误（4xx），`Exception` 处理未预期错误（5xx）。不要在业务代码里直接 `return JSONResponse(status_code=500)`，应该抛异常让全局处理器统一处理。
