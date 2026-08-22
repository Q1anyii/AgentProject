# Mitta AI 智能助理（米塔）

基于 **LangGraph + RAG + MCP + 流式 SSE** 的企业级智能助理系统。系统内置完整的知识库检索链路（查询改写 → 多路召回 → RRF 融合 → 在线重排），支持短期记忆（多轮对话恢复）与长期记忆（用户档案），通过 MCP 协议接入外部工具（文件系统、Git、数据库等），并通过 SSE 流式输出实现打字机效果。

## 功能特性

- **意图路由**：LLM 分类器判断问题是否需要检索知识库，`Send` 条件路由按需走检索链路，避免无谓延迟
- **RAG 增强检索**：查询改写（主查询 + 子查询 + 关键词）→ 多路向量召回 → RRF 融合 → SiliconFlow 在线重排
- **MCP 工具集成**：通过 Model Context Protocol 接入 filesystem、git、fetch、sqlite、sequential-thinking、memory 等外部工具；工具常驻事件循环，支持故障降级
- **智能工具筛选**：规则层（tags 关键词命中）+ 语义层（向量检索）并集，每轮只暴露相关工具给 LLM，避免工具过多导致注意力稀释
- **双通道记忆**：
  - 短期记忆：PostgresSaver 按 `thread_id` 恢复多轮对话
  - 长期记忆：PostgresStore 按 `user_id` 保存用户档案（跨会话生效）
- **用户自定义 System Prompt**：支持用户在个人信息界面上传自定义设定文件，与默认 Prompt 合并后作用于全局
- **文件上传与解析**：支持上传多种格式文件，上传后立即解析文本内容，发送消息时与用户输入一并送入 LLM
- **流式输出**：`stream_mode="messages"` 逐 token 输出，前端打字机效果；工具调用时实时显示加载状态
- **多主题前端**：Vue 3 SPA（CDN 单文件），支持多种配色主题、个人信息管理、MCP 配置、文件上传
- **安全认证**：JWT（access token + 隐式 refresh token 自动续签）+ bcrypt 密码哈希 + 请求限流
- **节点级缓存**：LangGraph CachePolicy + Redis，检索/工具/记忆节点结果按 TTL 缓存，降低 API 消耗

## 技术栈

| 层次        | 技术                                                                                               |
| --------- | ------------------------------------------------------------------------------------------------ |
| 语言/环境     | Python 3.13                                                                                      |
| Agent 编排  | LangGraph 1.x（StateGraph / Send 条件路由 / CachePolicy / Checkpointer / Store）                       |
| LLM 框架    | LangChain 1.x / langchain-openai / langchain-mcp-adapters                                        |
| 大模型       | 腾讯混元（deepseek-v4-flash 主模型 + hy-mt2-plus 工具筛选），OpenAI 兼容协议                                       |
| Embedding | SiliconFlow `BAAI/bge-m3`（1024 维）                                                                |
| 重排        | SiliconFlow `BAAI/bge-reranker-v2-m3` 在线重排                                                       |
| 向量库       | Milvus（默认）/ ChromaDB（可插拔，Protocol 抽象，零业务改动切换）                                                    |
| 关系数据库     | PostgreSQL 16（LangGraph Checkpointer/Store）+ MySQL 8.0（用户表 userInfo / user_profile / user_files） |
| 缓存        | Redis 7（节点级缓存 + 检索缓存 LSH + JWT 登录态 + 限流计数）                                                       |
| MCP       | MCP Python SDK + FastMCP（内置 agent_server + 外部 stdio/sse 服务器连接）                                   |
| Web 框架    | FastAPI + Uvicorn（SSE 流式响应）                                                                      |
| 前端        | Vue 3（CDN 单文件 SPA）+ 原生 CSS 多主题                                                                   |
| 反向代理      | Nginx（静态托管 + API 代理 + SSE 缓冲关闭）                                                                  |
| 认证        | JWT（PyJWT）+ bcrypt 密码哈希                                                                          |
| 可观测性      | LangSmith 链路追踪（可选）+ Loguru 结构化日志                                                                 |

## 系统架构

### # Mitta AI 流程图

## 1. 主对话图（main_graph）

```mermaid
flowchart TD
    START([START]) --> CLASSIFY[classify_node]

    CLASSIFY -->|LLM 判断是否需要检索| ROUTE{needs_retrieval?}

    ROUTE -->|Yes| RETRIEVE[retrieve_node]
    ROUTE -->|No| LLM

    RETRIEVE -->|检索结果转dict存入state| LLM[llm_node]

    LLM -->|组装提示词+tools过滤| ROUTE_LLM{route_after_llm 
    tool_calls?}

    ROUTE_LLM -->|Yes| TOOL[tool_nodeToolNode 执行 MCP 工具]
    ROUTE_LLM -->|No| MEMORY[memory_node]

    TOOL -->|工具执行结果 ToolMessage| LLM
    TOOL -.->|CachePolicy TTL=10s<br/>同工具同参数复用结果| TOOL

    MEMORY -->|idle 闲聊轮快速跳过executed/unavailable 轮LLM 提取记忆写入 Store| END_NODE([END])

    RETRIEVE -.->|CachePolicy TTL=10s
    key=input_str| RETRIEVE

    classDef llmNode fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b
    classDef toolNode fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100
    classDef cacheNode fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
    classDef decision fill:#fff9c4,stroke:#f9a825,stroke-width:2px,color:#f57f17
    classDef terminal fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20

    class CLASSIFY,LLM,MEMORY llmNode
    class TOOL toolNode
    class RETRIEVE cacheNode
    class ROUTE,ROUTE_LLM decision
    class START,END_NODE terminal
```

### 节点说明

| 节点                | 职责                | 关键实现                                                                                                                |
| ----------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------- |
| **classify_node** | LLM 判断问题是否需要知识库检索 | `model.invoke([CLASSIFIER_PROMPT, user_input])`，返回 yes/no                                                           |
| **retrieve_node** | 调用 RAG 子图检索知识库    | `retrieve_graph.invoke()`，Document 转 dict 存入 state（checkpoint 反序列化兼容）                                               |
| **llm_node**      | 核心生成节点            | 组装 System Prompt（默认+用户自定义+长期记忆）→ ToolFilter 筛选工具 → `model.bind_tools()` → `model.stream()` → 合并 chunk 提取 tool_calls |
| **tool_node**     | 执行 MCP 工具         | LangGraph `ToolNode`，按工具名路由；CachePolicy 缓存同参数结果                                                                     |
| **memory_node**   | 提取长期记忆            | LLM 从对话中提取用户档案写入 PostgresStore；idle 闲聊轮快速跳过避免阻塞 SSE                                                                 |

### 条件路由

- **classify_node → route**：`needs_retrieval=True` 走检索链路，否则直接到 llm_node
- **llm_node → route_after_llm**：`tool_calls` 非空走 tool_node，否则走 memory_node
- **tool_node → llm_node**：工具执行结果回到 LLM 生成最终回答（可多轮循环）

---

## 2. RAG 检索子图（retrieve_graph）

```mermaid
flowchart TD
    START([START]) --> CHECK_CACHE[check_cacheRedis 缓存检查]

    CHECK_CACHE --> CACHE_HIT{缓存命中?}

    CACHE_HIT -->|命中| RETURN_CACHE[直接返回缓存文档reranked_docs]
    CACHE_HIT -->|未命中| REWRITE[rewrite_nodeLLM 查询改写]

    REWRITE -->|主查询 + 子查询 +关键词| RETRIEVE[retrieve_node多路向量召回]

    RETRIEVE -->|每个查询独立向量检索Top-K=5 / 距离阈值=0.3| RRF[RRF 融合Reciprocal Rank Fusion]

    RRF -->|多查询结果去重融合k=60 排名权重衰减| RERANK[rerank_node在线重排]

    RERANK -->|SiliconFlow bge-reranker-v2-m3按相关性降序| CACHE_WRITE[写入 Redis 缓存]

    CACHE_WRITE --> RETURN[返回 Top-K 文档output: List Document]
    RETURN_CACHE --> RETURN
    RETURN --> END_NODE([END])

    classDef llmNode fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b
    classDef vectorNode fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20
    classDef cacheNode fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
    classDef decision fill:#fff9c4,stroke:#f9a825,stroke-width:2px,color:#f57f17
    classDef terminal fill:#fce4ec,stroke:#c62828,stroke-width:2px,color:#b71c1c

    class REWRITE,RERANK llmNode
    class RETRIEVE,RRF vectorNode
    class CHECK_CACHE,CACHE_WRITE,RETURN_CACHE cacheNode
    class CACHE_HIT decision
    class START,END_NODE,RETURN terminal
```

### 节点说明

| 节点                | 职责           | 关键实现                                                               |
| ----------------- | ------------ | ------------------------------------------------------------------ |
| **check_cache**   | Redis 检索缓存检查 | `cache_service.query_cache(thread_id, question)`，LSH 快速过滤 + 向量重排验证 |
| **rewrite_node**  | LLM 查询改写     | 输出 JSON：`{主查询, 子查询[], 关键词[]}`，解决多轮指代问题                             |
| **retrieve_node** | 多路向量召回       | 每个改写查询独立检索 Milvus/ChromaDB，Top-K=5，cosine distance < 0.3           |
| **RRF 融合**        | 多查询结果融合      | Reciprocal Rank Fusion（k=60），按排名融合去重，避免单查询偏差                       |
| **rerank_node**   | 在线重排         | SiliconFlow `BAAI/bge-reranker-v2-m3`，按 relevance_score 降序取 Top-N  |
| **cache_write**   | 写入 Redis     | 缓存键 `retrieve_cache:{thread_id}:{bucket_id}`，TTL=900 秒             |

### 关键参数

| 参数                 | 值                       | 位置                                |
| ------------------ | ----------------------- | --------------------------------- |
| TOP_K              | 5                       | `constant/retrieval_constants.py` |
| DISTANCE_THRESHOLD | 0.3（cosine distance）    | `constant/retrieval_constants.py` |
| RRF_K              | 60                      | `constant/retrieval_constants.py` |
| 缓存 TTL             | 900 秒                   | `constant/cache_constant.py`      |
| Embedding 模型       | BAAI/bge-m3（1024 维）     | `constant/embedding_constants.py` |
| 重排模型               | BAAI/bge-reranker-v2-m3 | `init.py`                         |

### 工具筛选机制

每轮对话时，`ToolFilter.select_tools(query, tools)` 执行两层筛选：

1. **规则层**：检查工具 `tags`（如 filesystem 工具含 `["文件","目录","读写"]`），query 中包含关键词即命中
2. **语义层**：将工具描述向量化存入 Milvus `MCP_TOOLS` 集合，用 query 做语义检索，top_k=12
3. 两层结果按工具名去重并集，只把候选工具 `bind_tools` 给 LLM；无命中时注入"无工具可用"提示

### 记忆体系

| 类型   | 存储                      | 隔离维度              | 生命周期                        |
| ---- | ----------------------- | ----------------- | --------------------------- |
| 短期记忆 | PostgreSQL Checkpointer | thread_id         | 会话级，可恢复                     |
| 长期记忆 | PostgreSQL Store        | user_id           | 跨会话持久                       |
| 节点缓存 | Redis                   | 输入哈希              | TTL 10~900 秒                |
| 检索缓存 | Redis + LSH             | thread_id + query | TTL 900 秒                   |
| 登录态  | Redis                   | user_id           | access 15 分钟 / refresh 30 天 |

## 目录结构

```
AgentProject/
├── src/                                  # 后端源码
│   ├── main.py                           # FastAPI 入口：lifespan 资源管理 + 路由注册 + 全局异常
│   ├── init.py                           # 模型/Embedding/重排/System Prompt 初始化
│   ├── config.py                         # 环境变量加载/校验 + MCP/向量库配置文件管理
│   ├── constant/                         # 常量定义（按模块分类）
│   │   ├── cache_constant.py             # Redis 缓存/向量索引/Token key/节点 TTL
│   │   ├── embedding_constants.py        # 集合名/切分参数/模型名
│   │   ├── prompt_constants.py           # 记忆提取/意图分类提示词
│   │   ├── retrieval_constants.py        # TOP_K/距离阈值/RRF 参数/改写提示词
│   │   └── tool_constant.py              # 工具集合名/筛选 top_k/距离阈值
│   ├── context/
│   │   └── user_context.py               # CtxUser 请求级用户上下文
│   ├── graphs/                           # LangGraph 图定义
│   │   ├── main_graph.py                 # 主对话图：classify→retrieve/llm→tool→memory
│   │   ├── retrieve_graph.py             # RAG 子图：cache→rewrite→retrieve→rerank
│   │   └── tool_filter.py                # 工具筛选：规则层 + 语义层
│   ├── mcp_client/                       # MCP 客户端
│   │   ├── client.py                     # MCP 连接管理/工具同步包装/故障降级/tags 注入
│   │   ├── mcp_tool_holder.py            # MCP 工具封装
│   │   ├── demo.py                       # MCP 调试示例
│   │   └── mcp_server/
│   │       └── agent_server.py           # 内置 FastMCP 服务器（chat/get_user/summarize）
│   ├── middleware/
│   │   └── rate_limit_middleware.py      # 基于 Redis 的请求限流中间件
│   ├── routers/                          # FastAPI 路由（按模块拆分）
│   │   ├── deps.py                       # 公共依赖（require_self_or_admin）
│   │   ├── auth_router.py                # 登录/注册/密码找回
│   │   ├── chat_router.py                # 对话(SSE)/历史/删除/停止/文件上传
│   │   ├── user_router.py                # 个人资料/密码/Prompt/主题/记忆/文件/MCP
│   │   ├── mcp_router.py                 # 全局 MCP 配置读写
│   │   └── system_router.py              # 健康检查/认证页面/SPA 兜底（必须最后注册）
│   ├── schemas/                          # Pydantic 请求/响应模型
│   │   ├── request_schemas/
│   │   │   ├── chat_schema.py            # ChatRequest（含 file_ids）
│   │   │   ├── login_schema.py
│   │   │   └── user_schema.py
│   │   └── response_schemas/
│   │       └── login_schema.py
│   ├── service/                          # 业务服务层
│   │   ├── chat_service.py               # 对话编排：图构建/流式输出/文件解析缓存
│   │   ├── login_service.py              # 用户登录/注册（MySQL 连接池）
│   │   ├── user_profile_service.py       # 用户扩展信息（头像/风格/Prompt/主题/MCP）
│   │   ├── file_upload_service.py        # 文件上传（base64 存 MySQL）/文本解析
│   │   └── cache_service.py              # Redis 缓存/LSH 向量检索/重排验证
│   ├── utils/                            # 工具函数
│   │   ├── jwt_utils.py                  # JWT 签发/验证/密码哈希
│   │   ├── response_util.py              # 统一响应格式
│   │   ├── doc_util.py                   # Document ↔ dict 转换
│   │   ├── lsh_util.py                   # 局部敏感哈希（缓存快速过滤）
│   │   ├── rand_id_util.py               # 随机 ID 生成
│   │   └── tools_util.py                 # 工具安全过滤/向量化/格式化
│   └── vector/                           # 向量库抽象层
│       ├── vector_store.py               # VectorStore Protocol + Chroma/Milvus 实现
│       ├── embedding.py                  # EmbeddingProcessor：文档加载→切分→入库
│       └── retrieve_doc.py               # RetrievedDoc 数据结构
├── resources/
│   ├── config/
│   │   ├── vector_db.json                # 向量库配置（type/uri/collection）
│   │   ├── mcp_servers.json              # MCP 服务器配置（JSON 数组）
│   │   ├── .mcp_config_path              # MCP 配置文件路径记录
│   │   └── .vector_config_path           # 向量库配置文件路径记录
│   ├── frontend/
│   │   ├── index.html                    # Vue 3 SPA 单文件前端
│   │   ├── nginx.conf                    # Nginx 配置（静态托管+API代理+SSE）
│   │   └── favicon.png
│   ├── system_prompt/
│   │   └── default_system_prompt.txt     # 默认 System Prompt（Mitta 角色设定）
│   ├── knowledge-base/                   # 编程知识库（Markdown）
│   │   ├── ingest_knowledge.py           # 知识库入库脚本
│   │   ├── 01~10-*.md                    # 分类知识文档
│   │   └── test-qa/                      # 测试 QA 集
│   ├── FAQ/                              # 在线学习平台 FAQ 知识库
│   └── chroma_db/                        # ChromaDB 持久化目录（Milvus 模式下不用）
├── tests/                                # 单元测试
├── docs/                                 # 项目文档
├── .env.example                          # 环境变量模板
├── requirements.txt                      # Python 依赖
├── Dockerfile                            # 后端容器镜像
├── docker-compose.yml                    # 一键部署（PostgreSQL+MySQL+Redis+Milvus+API+Nginx）
└── README.md
```

## 快速开始

### 环境要求

- Python 3.13+
- PostgreSQL 16+
- MySQL 8.0+
- Redis 7+
- Milvus 2.x（或使用 ChromaDB 免部署）
- Node.js（MCP stdio 服务器需要 npx/uvx）

### 1. 克隆项目并安装依赖

```bash
git clone <repo-url> AgentProject
cd AgentProject
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填写以下必填项：

| 变量                    | 说明                                 |
| --------------------- | ---------------------------------- |
| `HUNYUAN_API_KEY`     | 腾讯混元 API 密钥（主模型 + 工具筛选模型）          |
| `SILICONFLOW_API_KEY` | 硅基流动 API 密钥（Embedding + 重排）        |
| `POSTGRESQL_DB_URL`   | PostgreSQL 连接串（Checkpointer/Store） |
| `MYSQL_DB_URL`        | MySQL 连接串（用户表）                     |
| `REDIS_DB_URL`        | Redis 连接串                          |
| `JWT_SECRET_KEY`      | JWT 签名密钥（随机强密钥）                    |

### 3. 启动基础设施

```bash
# 使用 Docker Compose 启动 PostgreSQL + MySQL + Redis + Milvus
docker-compose up -d postgres mysql redis etcd minio milvus
```

或手动启动各服务。MySQL 需创建数据库 `mitta`，PostgreSQL 需创建数据库 `agentproject`（表由服务启动时自动创建）。

### 4. 配置向量库

编辑 `resources/config/vector_db.json`：

```json
{
  "type": "milvus",
  "uri": "http://localhost:19530",
  "collection": "FAQ_KNOWLEDGE_BASE"
}
```

如无 Milvus，可改为 ChromaDB（免部署）：

```json
{
  "type": "chroma",
  "persist_path": "../resources/chroma_db",
  "collection": "FAQ_KNOWLEDGE_BASE"
}
```

### 5. 配置 MCP 服务器（可选）

编辑 `resources/config/mcp_servers.json`，添加需要的 MCP 服务器。示例配置：

```json
[
  {
    "name": "filesystem",
    "type": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "E:/工作文件/AgentProject"]
  },
  {
    "name": "git",
    "type": "stdio",
    "command": "uvx",
    "args": ["mcp-server-git", "--repository", "E:/工作文件/AgentProject"]
  }
]
```

不配置 MCP 不影响核心对话功能。

### 6. 知识库入库（可选）

```bash
cd src
python ../resources/knowledge-base/ingest_knowledge.py
```

### 7. 启动后端

```bash
cd src
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 8. 启动前端（Nginx）

将 `resources/frontend/nginx.conf` 复制到 Nginx 配置目录，修改 `root` 路径指向 `resources/frontend/`，然后：

```bash
nginx
# 或 nginx -s reload
```

访问 `http://localhost` 即可使用。开发阶段也可直接访问 `http://localhost:8000`（后端托管 SPA）。

## API 接口一览

### 认证

| 方法   | 路径              | 说明   |
| ---- | --------------- | ---- |
| POST | `/api/login`    | 用户登录 |
| POST | `/api/register` | 用户注册 |
| POST | `/api/recover`  | 密码找回 |

### 对话

| 方法     | 路径                              | 说明                         |
| ------ | ------------------------------- | -------------------------- |
| POST   | `/api/chat/`                    | 发送消息（SSE 流式响应，支持 file_ids） |
| GET    | `/api/chat/{thread_id}/history` | 获取会话历史                     |
| DELETE | `/api/chat/{thread_id}`         | 删除会话                       |
| POST   | `/api/chat/stop`                | 停止回复                       |
| POST   | `/api/chat/upload`              | 上传文件（保存后立即解析文本）            |
| DELETE | `/api/files/{file_id}`          | 删除已上传文件                    |

### 用户

| 方法      | 路径                                   | 说明                     |
| ------- | ------------------------------------ | ---------------------- |
| GET     | `/api/users/{user_id}/profile`       | 获取个人信息                 |
| PUT     | `/api/users/{user_id}/profile`       | 更新个人信息（用户名/头像/风格）      |
| PUT     | `/api/users/{user_id}/password`      | 修改密码                   |
| GET/PUT | `/api/users/{user_id}/system-prompt` | 获取/更新自定义 System Prompt |
| GET/PUT | `/api/users/{user_id}/theme`         | 获取/更新前端主题              |
| GET     | `/api/users/{user_id}/memory`        | 获取长期记忆                 |
| GET     | `/api/users/{user_id}/sessions`      | 获取会话列表                 |
| GET     | `/api/users/{user_id}/files`         | 获取已上传文件列表              |
| GET/PUT | `/api/users/{user_id}/mcp`           | 获取/更新用户级 MCP 配置        |

### MCP / 系统

| 方法      | 路径                | 说明                    |
| ------- | ----------------- | --------------------- |
| GET/PUT | `/api/mcp/config` | 全局 MCP 配置读写           |
| GET     | `/health`         | 健康检查                  |
| GET     | `/mcp`            | 内置 MCP 服务器端点（FastMCP） |

## 核心设计说明

### MCP 工具常驻事件循环

MCP 工具通过 `langchain_mcp_adapters` 加载为 async 工具，闭包捕获绑定创建时事件循环的 `ClientSession`。同步图（ToolNode）在线程池执行时会临时新建事件循环，跨循环调用 session 会失败/挂起（Windows 下 mcp 库 cancel scope 泄漏还会注入 CancelledError 中断整图）。解决方案：

- 启动时创建专用守护线程运行独立事件循环（`mcp-tool-loop`），MCP 连接建立与工具调用全部提交到该循环（`asyncio.run_coroutine_threadsafe`）
- `make_sync_tool` 将 async 工具包装为同步 StructuredTool，含 30 秒调用超时，超时由 ToolNode 转错误消息，不中断对话链路
- 单个 MCP 服务器连接失败不影响其他服务器（15 秒连接超时 + 故障降级跳过）
- MCP 工具按服务器名注入 tags（`SERVER_TAGS` 映射），供工具筛选规则层命中
- 关闭时按序在工具循环内释放 MCP 子进程连接，避免资源泄漏

### 智能工具筛选

每轮对话时，`ToolFilter.select_tools(query, tools)` 执行两层筛选并集，只把候选工具暴露给 LLM：

1. **规则层**：检查工具 `tags`（如 filesystem 工具含 `["文件","目录","读写","file"]`），query 中包含关键词即命中，零延迟
2. **语义层**：工具描述向量化存入 Milvus `MCP_TOOLS` 集合，用 query 做语义检索（top_k=12，距离阈值 0.6），失败自动熔断降级为纯规则层
3. 两层结果按工具名去重并集；无命中时不 bind 空列表（OpenAI 兼容 API 会 400），改用裸模型并注入"无工具可用"提示
4. 多轮指代增强：输入含"继续/刚才/那个"等指代词时，拼接最近一轮 AI 回复前 200 字符辅助筛选

### 节点级缓存（LangGraph CachePolicy + Redis）

LangGraph `CachePolicy` 配合 `RedisCache`，在图编译时注入，节点结果按 TTL 缓存到 Redis：

| 节点            | 缓存键                                    | TTL | 策略                |
| ------------- | -------------------------------------- | --- | ----------------- |
| retrieve_node | 用户输入 `input_str`                       | 10s | 短窗口去重重复检索，不随历史变化  |
| tool_node     | 工具名+参数（排除调用 ID）+消息轮次                   | 10s | 同工具同参数复用结果，含失败结果  |
| memory_node   | 消息轮次+输入+AI回复（仅 executed/unavailable 轮） | 10s | idle 闲聊轮返回随机键永不命中 |

缓存键自定义设计：默认 key_func 对节点输入整体 pickle 哈希，而 Send payload 含每轮变化的 messages，会导致缓存键每轮都变、永不命中。自定义 key_func 只取稳定部分（用户输入/工具参数），确保缓存可命中。

### 检索结果缓存（CacheService + Redis Search + LSH）

除 LangGraph 节点级缓存外，`CacheService` 基于 Redis Search 构建了独立的检索结果缓存层，在 `retrieve_graph` 的 `check_cache` 节点使用：

- **缓存键**：`retrieve_cache:{thread_id}:{bucket_id}`，其中 bucket_id 通过 LSH（局部敏感哈希）对 query 向量做嵌套分桶，解决 Redis 无法直接做向量相似度检索的问题
- **两级验证**：LSH 快速过滤候选 → 用 bge-reranker-v2-m3 验证候选问题与当前问题是否语义等价（阈值 0.5，实测同义改写 0.89+，无关问题 0.0）
- **动态 TTL**：默认 900 秒，每命中一次自动刷新过期时间，兼顾热点问题长缓存与冷门问题快速淘汰
- **降级策略**：Redis 不可用时静默降级为不缓存，不阻塞检索主链路

### 流式输出与工具调用状态

- 使用 `stream_mode="messages"` 捕获图中所有 LLM token 事件，按 `meta["langgraph_node"]` 过滤只输出 llm_node 的增量
- SSE 事件类型：`content`（文本 token）、`tool_call_start`（工具名+参数）、`tool_call_end`（工具名+结果摘要）、`error`（异常）、`[DONE]`（结束）
- 前端监听 `tool_call_start/end` 事件，在 AI 消息下方显示"正在调用工具：xxx"加载条
- 流式模式下 tool_calls 分块传输，通过 `AIMessageChunk.__add__` 合并所有 chunk 提取完整工具调用，避免取最后一个 chunk 导致 tool_calls 为空

### 文件上传与解析

1. 前端上传文件 → `POST /api/chat/upload` → 保存到 MySQL `user_files` 表（base64 编码，单文件上限 10MB）
2. 保存后立即调用 `chat_service.parse_and_cache_file()` 解析文本（阻塞执行，接口返回即解析完成）
3. 解析结果缓存到内存 `_file_content_cache`（key=`{user_id}:{file_id}`），避免重复解析
4. 发送消息时前端传 `file_ids` → 后端从缓存读取文件内容 → 以"【文件名】+内容"格式拼接到 `input_str` → 传入 LLM
5. 支持 txt/md/csv/json/py/js 等纯文本格式（UTF-8/GBK 编码兼容）；PDF 使用 PyPDFLoader 解析；不支持的格式返回 `parsed=false`
6. 删除文件时同步清除解析缓存

### 安全设计

- JWT access token 15 分钟过期，Redis 存 refresh token 30 天，后端在 token 过期时自动续签（对前端透明）
- 密码使用 bcrypt 哈希（截断 72 字节，bcrypt 上限）
- `/api/chat/` 接口限流：每 IP 60 秒 30 次（Redis 计数器，Redis 不可用时降级内存限流）
- MCP 配置文件路径白名单校验（仅允许项目 resources/、config/ 和用户主目录），防止写入系统敏感目录
- 会话归属校验：非本人 thread_id 返回 403，防止会话劫持
- 全局异常处理器：记录完整堆栈到日志，返回给客户端的信息不含堆栈细节
- MCP 文件系统工具通过 allowed directories 限制访问范围

## Docker 部署

### 一键启动全部服务

```bash
docker-compose up -d
```

服务端口：

| 服务         | 端口        | 说明                 |
| ---------- | --------- | ------------------ |
| Nginx      | 80        | 前端 + API 统一入口      |
| FastAPI    | 8000      | 后端 API（直接访问）       |
| PostgreSQL | 5432      | Checkpointer/Store |
| MySQL      | 3306      | 用户数据               |
| Redis      | 6379      | 缓存                 |
| Milvus     | 19530     | 向量库                |
| etcd       | 2379      | Milvus 依赖          |
| MinIO      | 9000/9001 | Milvus 依赖          |

### 仅启动后端

```bash
docker build -t mitta-ai .
docker run -p 8000:8000 --env-file .env mitta-ai
```

## 开发说明

### 新增 MCP 工具

1. 在 `resources/config/mcp_servers.json` 添加服务器配置
2. 如需规则层命中，在 `src/mcp_client/client.py` 的 `SERVER_TAGS` 中添加关键词
3. 重启后端，日志会显示加载的工具数量

### 切换向量库

修改 `resources/config/vector_db.json` 的 `type` 字段（`milvus` 或 `chroma`），业务代码零改动。

### 添加新的 API 路由

1. 在 `src/routers/` 下新建或编辑路由文件
2. 在 `src/main.py` 中 `app.include_router()` 注册
3. 注意 `system_router` 必须最后注册（SPA 兜底路由）



***<u>后续可持续性内容优化</u>*：**

    1.完善向量存储多模态功能
    2.完善mcp_server/自定义MCP模块,原生支持某些工具而非外部依赖
    3.引入skills相关功能
    4.引入interrupt功能，在涉及敏感操作时，由用户确认是否继续
    5.目前只在源码层面支持自定义模型，后续需在设置界面添加接口
    6.重排序评分系统

## 许可证

见 [LICENSE](LICENSE) 文件。
