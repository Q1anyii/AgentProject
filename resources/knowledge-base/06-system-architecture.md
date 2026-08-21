# 系统架构设计实践

> 基于 AgentProject 项目总结，涵盖分层架构、服务层设计、上下文管理、配置管理、常量管理、MCP 集成等架构层面的实践。

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      前端 (Vue 3 SPA)                         │
│  index.html 单文件应用，SSE 流式接收，AbortController 中断    │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / SSE
┌──────────────────────────▼──────────────────────────────────┐
│                    API 层 (FastAPI main.py)                   │
│  路由定义 / 依赖注入 / 全局异常 / 中间件 / SSE 流式响应        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   服务层 (src/service/)                       │
│  chat_service / login_service / cache_service /              │
│  user_profile_service / file_upload_service                   │
└──────┬───────────────┬───────────────┬─────────────────────┘
       │               │               │
┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────────┐
│  Graph 层    │ │  数据层      │ │  外部集成        │
│ LangGraph    │ │ MySQL/PG/Redis│ │ MCP / LLM API   │
│ main_graph   │ │              │ │                  │
│ retrieve_graph│ │              │ │                  │
└─────────────┘ └─────────────┘ └──────────────────┘
```

## 二、分层架构

### 2.1 分层职责

| 层级 | 职责 | 不做什么 |
|------|------|----------|
| API 层 | 路由定义、参数校验、依赖注入、响应封装 | 不写业务逻辑 |
| 服务层 | 业务逻辑编排、事务管理、多数据源协作 | 不直接处理 HTTP 请求/响应 |
| Graph 层 | LangGraph 状态图定义、节点逻辑 | 不关心 HTTP 和数据库连接细节 |
| 数据层 | 数据库连接、CRUD 操作 | 不写业务逻辑 |
| 工具层 | 通用工具函数（JWT、响应、文档处理） | 不依赖业务层 |

### 2.2 依赖方向

```
API 层 → 服务层 → Graph 层 → 数据层
  ↓          ↓          ↓          ↓
  依赖注入   单例服务   状态图     连接管理
```

**原则**：
- 上层依赖下层，下层不依赖上层
- 同层之间尽量不互相依赖
- 依赖通过构造函数或参数注入，不硬编码

## 三、服务层设计

### 3.1 单例服务模式

每个服务在模块顶层创建单例，全局复用。

```python
# service/chat_service.py
class ChatService:
    def __init__(self):
        self._graph = None
        self._conn = None

    def open(self, mcp_tools=None):
        """初始化：构建图、建立数据库连接"""
        ...

    def close(self, timeout=10):
        """释放资源"""
        ...

    def stream(self, user_id, thread_id, query, user_info=None):
        """流式对话"""
        ...

# 模块级单例
chat_service = ChatService()
```

### 3.2 生命周期管理

在 FastAPI lifespan 中统一 open/close。

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：按依赖顺序 open
    validate_config()
    chat_service.open(mcp_tools)
    login_service.open()
    cache_service.open()
    user_profile_service.open()
    file_upload_service.open()
    yield
    # 关闭：按依赖逆序 close
    chat_service.close(timeout=10)
    login_service.close(timeout=10)
    cache_service.close()
    user_profile_service.close()
    file_upload_service.close()
```

### 3.3 服务列表

| 服务 | 职责 | 数据源 |
|------|------|--------|
| chat_service | 对话编排、LangGraph 调用、历史管理 | PostgreSQL (Checkpointer) |
| login_service | 登录、注册、密码找回 | MySQL (userInfo) |
| cache_service | 检索缓存、JWT 存储、限流 | Redis |
| user_profile_service | 用户扩展信息 CRUD | MySQL (user_profile) |
| file_upload_service | 文件上传、存储、查询 | MySQL (user_files) |

## 四、上下文管理

### 4.1 CtxUser（请求级用户上下文）

在 API 层构造用户上下文，传入 Graph 供工具节点读取。

```python
# context/user_context.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CtxUser:
    uid: int
    user_id: str
    password: str | None  # 敏感字段不注入，传 None
    username: str
    create_time: datetime
    update_time: datetime
```

**使用**：
```python
@app.post("/api/chat/")
def chat(request_body, current_user = Depends(get_current_user)):
    user_row = login_service.get_user_by_id(str(current_user.user_id))
    user_info = CtxUser(
        uid=user_row["id"],
        user_id=user_row["user_id"],
        password=None,  # 敏感字段不注入
        username=current_user.username,
        create_time=user_row["create_time"],
        update_time=user_row["update_time"],
    ) if user_row else None
    event_stream = chat_service.stream(..., user_info=user_info)
```

**设计要点**：
- 敏感字段（password）不注入上下文，工具无法访问
- username 直接从 JWT 解析，不查库（减少一次查询）
- 上下文是请求级的，每次请求新建，不复用

## 五、配置管理

### 5.1 环境变量分层

```python
# config.py
REQUIRED_ENV_VARS = [
    ("DEEPSEEK_API_KEY", "DeepSeek API 密钥"),
    ("SILICONFLOW_API_KEY", "SiliconFlow API 密钥"),
    ("POSTGRESQL_DB_URL", "PostgreSQL 连接串"),
    ("MYSQL_DB_URL", "MySQL 连接串"),
    ("REDIS_DB_URL", "Redis 连接串"),
    ("JWT_SECRET_KEY", "JWT 签名密钥"),
]

OPTIONAL_ENV_VARS = [
    ("MODEL_NAME", "deepseek:deepseek-v4-flash", "模型名称"),
    ("JWT_ALGORITHM", "HS256", "JWT 签名算法"),
    ("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "15", "access token 有效期"),
]
```

### 5.2 配置校验

```python
def validate_config() -> None:
    """启动时校验所有必填环境变量，缺失则抛出 ConfigError（快速失败）"""
    missing = []
    for key, desc in REQUIRED_ENV_VARS:
        value = os.getenv(key)
        if not value or value.strip() == "":
            missing.append(f"  - {key}: {desc}")
    if missing:
        raise ConfigError("缺少必填环境变量：\n" + "\n".join(missing))
```

### 5.3 类型安全的配置访问

```python
def get_env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(f"环境变量 {key} 值 '{value}' 不是有效整数，使用默认值 {default}")
        return default

def get_env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes", "on")
```

### 5.4 敏感信息脱敏

```python
def print_config_summary() -> None:
    """打印配置摘要，敏感信息只显示前4后4位"""
    for key, desc in REQUIRED_ENV_VARS:
        value = os.getenv(key)
        if value and ("KEY" in key or "SECRET" in key or "PASSWORD" in key):
            masked = value[:4] + "*" * (len(value) - 8) + value[-4:]
            logger.info(f"  {key}: {masked} (已配置)")
```

## 六、常量管理

### 6.1 目录结构

```
src/constant/
├── __init__.py
├── embedding_constants.py    # 嵌入相关常量
├── retrieval_constants.py    # 检索相关常量
├── cache_constant.py         # 缓存相关常量
└── ...
```

### 6.2 常量定义

```python
# constant/retrieval_constants.py
TOP_K = 10
DISTANCE_THRESHOLD = 0.5
RRF_K = 60
REWRITE_PROMPT = """..."""

# constant/embedding_constants.py
COLLECTION_NAME = "mitta_ai_knowledge"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# constant/cache_constant.py
USER_TOKEN_KEY = "user:token:{user_id}"
USER_REFRESH_TOKEN_KEY = "user:refresh_token:{user_id}"
```

**原则**：
- 按模块/领域分类，不堆在一个文件里
- 常量名全大写，单词间用下划线
- 魔法数字（如 TOP_K=10）必须提取为常量
- 固定字符串（如 Redis key 模板）提取为常量

## 七、MCP 集成

### 7.1 MCP 服务器配置

```python
# .env
MCP_SERVERS='[{"name":"filesystem","type":"stdio","command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","/path/to/files"]}]'
```

### 7.2 配置加载

```python
def load_mcp_server_configs() -> list[dict]:
    """从环境变量加载 MCP 服务器配置，校验格式，失败返回空列表（不阻塞启动）"""
    raw = os.getenv("MCP_SERVERS")
    if not raw:
        return []
    try:
        servers = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("MCP_SERVERS 不是合法 JSON，跳过 MCP 加载")
        return []
    # 校验每项配置...
    return validated
```

### 7.3 MCP 工具加载

```python
# lifespan 中
mcp_holders = await init_mcp_holders(load_mcp_server_configs())
mcp_tools = [t for h in mcp_holders for t in h.tools]
chat_service.open(mcp_tools)  # 传入图中

# 关闭时释放
for holder in mcp_holders:
    await holder.close()
```

### 7.4 MCP 服务器端点

```python
# 挂载 MCP 服务器，供外部客户端（如 Claude Desktop）调用
app.mount("/mcp", mcp.http_app())
```

## 八、常见架构陷阱

### 8.1 循环依赖

模块 A 导入模块 B，模块 B 导入模块 A，导致 ImportError。

**解决**：
- 延迟导入（在函数内部 import）
- 重新划分模块，提取公共部分到第三方模块
- 用依赖注入反转依赖方向

### 8.2 上帝服务（God Service）

一个服务类承担太多职责，代码膨胀，难以维护。

**解决**：
- 按业务领域拆分服务（chat_service, user_service, file_service）
- 每个服务单一职责
- 服务之间通过明确的接口协作

### 8.3 硬编码配置

数据库连接串、API Key、阈值等硬编码在代码中。

**解决**：
- 所有配置走环境变量
- 常量提取到 constant 目录
- 提供 .env.example 模板

### 8.4 跨层调用

API 层直接操作数据库，跳过服务层。

**解决**：
- 严格分层，API 层只调用服务层
- 数据库操作封装在服务层或数据层
- Code Review 检查分层违规

## 九、个人见解

1. **分层架构的价值在约束，不在形式**：很多项目名义上有分层，但代码里到处跨层调用。分层的真正价值是建立约束，让代码职责清晰。如果不遵守约束，分再多层也没用。

2. **单例服务是双刃剑**：单例方便、性能好，但不利于测试和扩展。如果项目需要支持多租户或动态配置，考虑用工厂模式或依赖注入容器。本项目是单租户应用，单例是合理选择。

3. **配置管理要从第一天做起**：很多项目初期硬编码，后期想抽出来很痛苦。建议项目初始化时就建立 config.py + .env.example 的模式，所有配置走环境变量。

4. **MCP 是 AI 应用的插件化未来**：MCP（Model Context Protocol）让 AI 应用可以动态加载外部工具，不需要硬编码集成。本项目已经支持 MCP，这是很好的架构决策，未来扩展工具能力不需要改核心代码。

5. **上下文管理是安全的关键**：把用户信息传入 Graph 时，要严格控制哪些字段可见。password 等敏感字段绝对不能注入，否则工具节点可能泄露。本项目用 CtxUser  dataclass 明确字段，password 传 None，是正确的做法。
