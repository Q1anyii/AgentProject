# 数据库设计与多存储协作实践

> 基于 AgentProject 项目总结，涵盖 MySQL、PostgreSQL、Redis 三种数据库的职责划分、连接管理、表结构设计、最佳实践。

## 一、多存储架构总览

```
┌─────────────────────────────────────────────────────────┐
│                      应用层 (FastAPI)                      │
├──────────┬──────────────────┬───────────────────────────┤
│  MySQL   │   PostgreSQL     │         Redis             │
│          │                  │                           │
│ 用户信息  │ LangGraph        │ 检索缓存                  │
│ 登录认证  │ Checkpointer     │ JWT 登录态                │
│ 文件上传  │ Store            │ 限流计数                  │
│ 扩展配置  │ (会话状态/长期记忆)│ 会话临时数据              │
└──────────┴──────────────────┴───────────────────────────┘
```

**职责划分原则**：
- **MySQL**：结构化业务数据，需要事务和复杂查询
- **PostgreSQL**：LangGraph 生态原生支持，JSONB 适合存储图状态
- **Redis**：高速缓存、临时状态、计数器

## 二、MySQL 设计

### 2.1 用户表（userInfo）

```sql
CREATE TABLE userInfo (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL UNIQUE COMMENT '用户登录ID',
    username VARCHAR(64) NOT NULL COMMENT '显示用户名',
    password VARCHAR(255) NOT NULL COMMENT '加密后的密码',
    role VARCHAR(32) DEFAULT '学员' COMMENT '角色：学员/管理员',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id)
);
```

### 2.2 用户扩展信息表（user_profile）

```sql
CREATE TABLE user_profile (
    user_id VARCHAR(64) PRIMARY KEY COMMENT '关联 userInfo.user_id',
    username VARCHAR(64) COMMENT '显示用户名（冗余，避免联表）',
    avatar TEXT COMMENT '头像（base64 data URL）',
    assistant_style TEXT COMMENT '助手风格设定',
    system_prompt TEXT COMMENT '用户自定义 system prompt（全局）',
    theme VARCHAR(32) DEFAULT 'default' COMMENT '主题名称',
    mcp_config JSON COMMENT 'MCP 服务器配置（JSON数组）',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

**设计要点**：
- 与 userInfo 1:1 关联，user_id 作为主键
- 大字段（avatar, system_prompt, mcp_config）单独存扩展表，避免用户表过宽
- mcp_config 用 JSON 类型，支持灵活结构
- username 冗余存储，避免查询个人信息时联表

### 2.3 用户文件表（user_files）

```sql
CREATE TABLE user_files (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL COMMENT '上传用户ID',
    thread_id VARCHAR(128) COMMENT '关联会话ID（NULL=全局文件）',
    file_name VARCHAR(255) NOT NULL COMMENT '原始文件名',
    file_type VARCHAR(128) COMMENT 'MIME类型',
    file_ext VARCHAR(32) COMMENT '扩展名',
    file_size INT COMMENT '文件大小（字节）',
    file_content LONGTEXT COMMENT 'base64编码的文件内容',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_thread_id (thread_id)
);
```

**设计要点**：
- 文件内容用 LONGTEXT 存储 base64，单文件最大约 4GB（实际限制 10MB）
- thread_id 可空，支持会话级和全局级文件
- 按 user_id 和 thread_id 建索引，支持快速查询

### 2.4 MySQL 连接管理

```python
import pymysql
from pymysql.cursors import DictCursor

class UserProfileService:
    def __init__(self):
        self._conn = None

    def open(self):
        """在 lifespan 启动时调用，建立连接"""
        self._conn = pymysql.connect(
            host=..., port=..., user=..., password=..., database=...,
            cursorclass=DictCursor, charset='utf8mb4'
        )

    def close(self):
        """在 lifespan 关闭时调用"""
        if self._conn:
            self._conn.close()

    def _get_conn(self):
        """获取连接，自动重连"""
        if not self._conn or not self._conn.open:
            self.open()
        return self._conn
```

**最佳实践**：
- 用 `DictCursor`，返回字典而非元组
- `charset='utf8mb4'`，支持 emoji 和生僻字
- 连接在 lifespan 中统一管理，不每次请求新建
- 自动重连机制，应对 MySQL 超时断开

### 2.5 自动建表

```python
def _ensure_table(self):
    """启动时自动建表，不存在则创建"""
    conn = self._get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                user_id VARCHAR(64) PRIMARY KEY,
                ...
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
    conn.commit()
```

## 三、PostgreSQL 设计

### 3.1 LangGraph Checkpointer

PostgresSaver 自动管理表结构，无需手动创建。

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(os.getenv("POSTGRESQL_DB_URL"))
checkpointer.setup()  # 自动创建 checkpoint 相关表
```

**Checkpointer 表结构**（自动创建）：
- `checkpoints`：图执行状态快照
- `checkpoint_blobs`：大状态数据（blob 存储）
- `checkpoint_writes`：中间写入记录

### 3.2 LangGraph Store

```python
from langgraph.store.postgres import PostgresStore

store = PostgresStore.from_conn_string(os.getenv("POSTGRESQL_DB_URL"))
store.setup()  # 自动创建 store 表
```

**Store 用途**：
- 用户全局配置（如自定义 prompt）
- 跨会话长期记忆
- 知识库元数据

### 3.3 为什么用 PostgreSQL 而不是 MySQL

- LangGraph 官方原生支持 PostgresSaver / PostgresStore
- PostgreSQL 的 JSONB 类型适合存储图状态（半结构化数据）
- PostgreSQL 支持更复杂的查询和索引（GIN 索引 JSONB）

## 四、Redis 设计

### 4.1 检索缓存

```python
# Key 格式：chat:cache:{thread_id}:{question_hash}
# Value：序列化后的文档列表
# TTL：15 秒

def store_cache(self, thread_id, question, docs):
    key = f"chat:cache:{thread_id}:{hash(question)}"
    self.redis.setex(key, 15, json.dumps([doc_to_dict(d) for d in docs]))

def query_cache(self, thread_id, question, n=3):
    # 模糊匹配最近 n 条相似问题
    ...
```

### 4.2 JWT 登录态

```python
# Key 格式：user:token:{user_id}
# Value：access token
# TTL：15 分钟（与 access token 有效期一致）

r.setex(USER_TOKEN_KEY.format(user_id=user_id), 15 * 60, token)

# refresh token：30 天
r.setex(USER_REFRESH_TOKEN_KEY.format(user_id=user_id), 30 * 86400, refresh_token)
```

### 4.3 限流计数

```python
# Key 格式：rate_limit:{user_id}:{api_path}
# Value：请求计数
# TTL：时间窗口（如 60 秒）

def is_rate_limited(self, user_id, path, limit=10, window=60):
    key = f"rate_limit:{user_id}:{path}"
    count = self.redis.incr(key)
    if count == 1:
        self.redis.expire(key, window)
    return count > limit
```

### 4.4 Redis 连接管理

```python
import redis

class CacheService:
    def __init__(self):
        self._redis = None

    def open(self):
        self._redis = redis.from_url(os.getenv("REDIS_DB_URL"), decode_responses=True)

    def close(self):
        if self._redis:
            self._redis.close()

    @property
    def redis(self):
        return self._redis
```

## 五、多存储协作模式

### 5.1 登录流程

```
用户提交账号密码
    ↓
MySQL 查询用户信息（userInfo 表）
    ↓ 验证密码
生成 JWT access token + refresh token
    ↓
Redis 存储 token（setex，带 TTL）
    ↓
返回 token 给前端
```

### 5.2 对话流程

```
用户发送消息（带 JWT）
    ↓
JWT 验证（Redis 检查 token 是否存在）
    ↓
LangGraph 执行（PostgreSQL Checkpointer 存会话状态）
    ↓
RAG 检索
    ├── Redis 检查缓存（命中则直接返回）
    ├── ChromaDB 向量检索
    └── Redis 存储检索结果（15s TTL）
    ↓
LLM 生成回答
    ↓
PostgreSQL Store 存储长期记忆（如需要）
    ↓
返回 SSE 流式响应
```

### 5.3 用户配置流程

```
用户修改个人信息/主题/MCP配置
    ↓
MySQL user_profile 表更新
    ↓
返回成功
    ↓
下次对话时
    ├── init.py 从 MySQL 读取用户自定义 system prompt
    └── main_graph.py 组装到 system_prompt 中
```

## 六、常见陷阱

### 6.1 MySQL 8 小时超时

MySQL 默认 `wait_timeout=28800`（8小时），长时间空闲后连接会断开。

**解决**：
- 使用连接池（如 DBUtils、SQLAlchemy pool）
- 每次查询前检查连接状态（`conn.open`）
- 配置 `pool_recycle` 小于 wait_timeout

### 6.2 Redis 连接耗尽

Redis 默认最大连接数 10000，但如果不释放连接会耗尽。

**解决**：
- 使用连接池（`redis.ConnectionPool`）
- 确保每次操作后连接归还池
- 监控连接数

### 6.3 PostgreSQL 连接数限制

PostgreSQL 默认 `max_connections=100`，LangGraph Checkpointer 可能占用较多连接。

**解决**：
- 使用连接池（如 psycopg2 pool、SQLAlchemy）
- 合理配置 pool_size 和 max_overflow
- 监控连接数，必要时调大 max_connections

### 6.4 事务未提交

pymysql 默认不自动提交，DML 操作后必须 `conn.commit()`，否则数据不写入。

```python
with conn.cursor() as cur:
    cur.execute("INSERT INTO ...")
conn.commit()  # 必须提交
```

### 6.5 SQL 注入

永远不要用字符串拼接构造 SQL，用参数化查询。

```python
# 错误
cur.execute(f"SELECT * FROM users WHERE id = '{user_id}'")

# 正确
cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
```

## 七、个人见解

1. **不要用一种数据库解决所有问题**：MySQL 做业务数据，PostgreSQL 做图状态，Redis 做缓存，各司其职。强行用 MySQL 做缓存或用 Redis 做持久化都是技术债。

2. **连接管理是后端稳定性的基石**：90% 的数据库相关线上问题都是连接管理不当导致的（连接泄漏、超时断开、连接耗尽）。 invest 在连接池和健康检查上。

3. **自动建表适合小项目，大项目用迁移工具**：本项目用 `CREATE TABLE IF NOT EXISTS` 自动建表，适合快速开发。但如果表结构频繁变更，应该用 Alembic 等迁移工具管理版本。

4. **Redis 缓存要考虑穿透和雪崩**：本项目的检索缓存 TTL 15 秒比较短，雪崩风险低。但如果缓存大量数据且 TTL 一致，要加随机偏移避免同时失效。

5. **文件存数据库要谨慎**：本项目把文件 base64 存 MySQL LONGTEXT，优点是简单、备份方便，缺点是数据库体积膨胀、备份慢。如果文件量大或大文件多，应该用对象存储（S3/OSS），数据库只存元数据和 URL。
