# 米塔 AI 智能助理（Mitta）

基于**LangGraph + RAG + 流式 SSE**的多智能体智能AI系统。系统内置知识库检索链路（查询改写 → 多路召回 → RRF 融合 → 在线重排），支持短期记忆（多轮对话恢复）与长期记忆（用户档案），并通过 SSE 流式输出实现打字机效果。

## 功能特性

- 🤖 **意图路由**：LLM 分类器判断问题是否需要检索知识库，`Send` 条件路由按需走检索链路，避免无谓延迟
- 🔍 **RAG 增强检索**：查询改写（主查询 + 子查询 + 关键词）→ 多路向量召回 → RRF 融合 → SiliconFlow 在线重排
- 💬 **双通道记忆**：
  - 短期记忆：PostgresSaver 按 `thread_id` 恢复多轮对话
  - 长期记忆：PostgresStore 按 `user_id` 保存用户档案（跨会话生效）
- ⚡ **流式输出**：`stream_mode="messages"` 逐 token 输出，前端打字机效果
- 🎨 **原生前端**：无框架单页应用，手写 SSE 解析，支持会话管理、健康状态展示、移动端适配

## 技术栈

| 层次        | 技术                                                                           |
| --------- | ---------------------------------------------------------------------------- |
| 语言/环境     | Python 3.13（conda 环境 `langchain1.2`）                                         |
| Agent 编排  | LangGraph 1.1.2（StateGraph / Send 条件路由 / CachePolicy / Checkpointer / Store） |
| 大模型       | DeepSeek / 腾讯混元（OpenAI 兼容协议，`init_chat_model` 初始化）                           |
| 向量库       | ChromaDB（持久化于 `resources/chroma_db`，cosine 距离）                               |
| Embedding | SiliconFlow `BAAI/bge-m3`                                                    |
| 重排        | SiliconFlow `BAAI/bge-reranker-v2-m3` 在线重排                                   |
| 数据库       | PostgreSQL（LangGraph Checkpointer/Store）+ MySQL（用户表 userInfo）                |
| 缓存        | Redis（检索缓存 LSH + 向量重排验证 / JWT 登录态存储）                                         |
| 认证        | JWT（access token + 隐式 refresh token 自动续签）+ bcrypt 密码哈希                       |
| Web 服务    | FastAPI + Uvicorn（SSE 流式响应）                                                  |
| 前端/部署     | 原生 HTML/JS/CSS + Nginx（静态托管 + API 反向代理）                                      |
| 可观测性      | LangSmith 链路追踪（可选）                                                           |

## 目录结构

```
AgentProject/
├── src/                              # 后端源码
│   ├── main.py                       # FastAPI 入口：lifespan 资源管理 + REST 端点
│   ├── init.py                       # 模型 / Embedding / 重排 / 系统提示词初始化
│   ├── embedding.py                  # EmbeddingProcessor：文档解析 → 切分 → 向量入库
│   ├── constant/                     # 常量定义（按模块分类）
│   │   ├── cache_constant.py         # Redis 缓存 / 向量索引 / Token key 常量
│   │   ├── retrieval_constants.py    # 检索参数（TOP_K / 距离阈值）/ 查询改写提示词
│   │   ├── prompt_constants.py       # 主图提示词（记忆提取 / 意图分类 / 无信息标记）
│   │   └── embedding_constants.py    # 向量库集合名 / 切分参数 / 模型名
│   ├── context/                      # 请求级上下文
│   │   └── user_context.py           # CtxUser 用户上下文类 + contextvars
│   ├── graphs/                       # LangGraph 图定义（按图拆分）
│   │   ├── main_graph.py             # 主对话图：classify → route → (retrieve | llm) → memory
│   │   ├── retrieve_graph.py         # 检索增强图：rewrite → retrieve(RRF) → rerank
│   │   └── tool_graph.py             # 工具图（summarize / get_current_user）
│   ├── schemas/                      # Pydantic 请求/响应模型
│   │   ├── request_schemas/          # 请求模型
│   │   │   ├── chat_schema.py        # 聊天请求（query / thread_id）
│   │   │   └── login_schema.py       # 登录/注册/找回密码请求
│   │   └── response_schemas/         # 响应模型
│   │       └── login_schema.py       # 登录/注册/找回密码响应
│   ├── service/                      # 业务服务层
│   │   ├── chat_service.py           # ChatService：资源生命周期 + 对话/记忆/会话业务
│   │   ├── cache_service.py          # CacheService：Redis 检索缓存（LSH + 向量重排验证）
│   │   └── login_service.py          # LoginService：MySQL 用户登录/注册/找回密码
│   ├── middleware/                   # 中间件
│   │   └── rate_limit_middleware.py # 请求限流（基于 Redis，对 /api/chat/ 限流）
│   ├── test/                         # 集成测试
│   │   └── test_graph.py             # 图测试（独立实现，含 InMemoryCache）
│   └── utils/                        # 工具函数
│       ├── doc_util.py               # 文档转换（unpack_query_results / documents_to_dicts）
│       ├── jwt_utils.py              # JWT 签发/验证 + bcrypt 密码哈希 + 隐式自动续签
│       ├── lsh_util.py               # 随机投影 LSH（缓存桶映射）
│       ├── rand_id_util.py           # 随机 ID 生成（基于 uuid4，MySQL int 范围内）
│       └── response_util.py          # 统一响应封装（Response.success / failed）
├── tests/                            # 单元测试（pytest）
│   ├── test_config.py                # 配置管理模块测试
│   ├── test_jwt_utils.py             # JWT 工具测试（密码哈希、Token 签发验证）
│   └── test_rand_id_util.py          # ID 生成工具测试
├── resources/                        # 资源文件
│   ├── FAQ/                          # 知识库源文档（在线学习平台 FAQ，RAG 专用）
│   ├── system_prompt/                # 助理系统提示词（人设与回答规则）
│   ├── chroma_db/                    # ChromaDB 向量库持久化数据（运行时生成）
│   └── frontend/                     # 前端静态资源
│       ├── index.html                # 聊天界面（原生 JS 单页应用，含登录/注册）
│       ├── favicon.png               # 站点图标
│       └── nginx.conf                # Nginx 配置（前端托管 + /api 反向代理 + SSE）
├── .env                              # 环境变量（含 API 密钥，勿提交版本库）
├── .env.example                      # 环境变量模板（含注释，复制为 .env 后填写）
├── requirements.txt                  # 完整依赖清单（按用途分组，含注释）
├── .gitignore                        # Git 忽略规则
├── .dockerignore                     # Docker 构建忽略规则
├── Dockerfile                        # 后端服务 Docker 镜像构建
├── docker-compose.yml                # 一键部署（postgres + mysql + redis + api）
├── LICENSE                           # 开源协议
├── docs/                             # 项目文档
│   ├── guide.txt                     # 开发指南（架构设计、改造步骤）
│   ├── TODO.md                       # 项目待办（功能规划、改进点）
│   └── README.md                     # 文档索引
└── README.md                         # 项目说明文档
```

## 快速开始

### 1. 环境准备

```bash
# 创建并激活 conda 环境（Python 3.13）
conda create -n langchain1.2 python=3.13 -y
conda activate langchain1.2

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写（`.env.example` 中每个变量均有详细注释）：

| 变量分组             | 变量名                                                              | 说明                                           | 必填  |
| ---------------- | ---------------------------------------------------------------- | -------------------------------------------- | --- |
| **大模型**          | `DEEPSEEK_API_KEY`                                               | DeepSeek 平台密钥                                | 是   |
|                  | `MODEL_NAME`                                                     | 模型名，如 `deepseek:deepseek-v4-flash`           | 是   |
|                  | `BASE_URL`                                                       | DeepSeek OpenAI 兼容接口地址                       | 是   |
|                  | `HUNYUAN_API_KEY`                                                | 腾讯混元 API 密钥（`init.py` 当前使用）                  | 是   |
|                  | `KIMI_API_KEY` / `KIMI_BASE_URL` / `KIMI_MODEL_NAME`             | Kimi (Moonshot) 配置（预留）                       | 否   |
| **Embedding/重排** | `SILICONFLOW_API_KEY`                                            | SiliconFlow 密钥（Embedding + 在线重排）             | 是   |
|                  | `SILICONFLOW_BASE_URL`                                           | SiliconFlow 接口地址                             | 是   |
| **数据库**          | `POSTGRESQL_DB_URL`                                              | PostgreSQL 连接串（LangGraph Checkpointer/Store） | 是   |
|                  | `MYSQL_DB_URL`                                                   | MySQL 连接串（用户表 userInfo，登录/注册）                | 是   |
| **缓存**           | `REDIS_DB_URL`                                                   | Redis 连接串（检索缓存 + JWT 登录态）                    | 是   |
| **JWT 认证**       | `JWT_SECRET_KEY`                                                 | JWT 签名密钥（生产环境务必使用随机强密钥）                      | 是   |
|                  | `JWT_ALGORITHM`                                                  | JWT 签名算法，默认 `HS256`                          | 否   |
|                  | `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`                                | access token 有效期（分钟），默认 15                   | 否   |
|                  | `JWT_REFRESH_TOKEN_EXPIRE_DAYS`                                  | refresh token 有效期（天），默认 30                   | 否   |
| **可观测性**         | `LANGSMITH_TRACING`                                              | 是否开启 LangSmith 链路追踪，`true`/`false`           | 否   |
|                  | `LANGSMITH_API_KEY` / `LANGSMITH_ENDPOINT` / `LANGSMITH_PROJECT` | LangSmith 配置                                 | 否   |

### 3. 准备基础设施

- **PostgreSQL**：创建数据库（如 `agentproject`）。表结构由 `PostgresSaver.setup()` / `PostgresStore.setup()` 在服务启动时自动创建，无需手动建表。
- **MySQL**：创建数据库（如 `mitta`）和用户表 `userInfo`（字段：id, user_id, password, username, create_time, update_time）。登录/注册/找回密码功能依赖此表。
- **Redis**：启动 Redis 服务（默认端口 6380，可在 `.env` 中通过 `REDIS_DB_URL` 配置）。用于检索缓存（LSH + 向量重排验证）和 JWT 登录态存储。
- **知识库入库**：将 FAQ 文档写入向量库（交互式输入文档路径与 `source`/`category`）：

```bash
cd src
python embedding.py
# 示例输入：
# 文档路径: ../resources/FAQ/在线学习平台FAQ知识库（智能助理RAG专用）.md
# source 与 category: FAQ 平台业务
```

支持 `.md`（按标题切分）、`.txt`、`.pdf`，切分参数 chunk=300 / overlap=50。

### 4. 启动后端

```bash
cd src
python main.py        # 等价于 uvicorn main:app --host localhost --port 8000 --reload
```

启动时 `lifespan` 会依次初始化 ChromaDB 连接、PostgreSQL 连接池、Checkpointer/Store、两个 LangGraph 图；数据库不可用会在启动阶段直接报错（快速失败）。

### 5. 启动前端

将 `resources/frontend/nginx.conf` 替换到 Nginx 的 `conf/` 目录（按注释调整 `root` 路径），然后：

```bash
nginx            # 或 nginx -s reload
```

浏览器访问 `http://localhost` 即可使用。Nginx 同时负责将 `/api/*` 反向代理到 `localhost:8000`（已配置 `proxy_buffering off` 保证 SSE 流式）。

> 开发调试时也可直接打开 `resources/frontend/index.html`（前后端同域时 `API_BASE` 为空字符串使用相对路径）。

## 系统架构

### 整体流程

```
用户输入
   │
   ▼
POST /api/chat/ (SSE)
   │
   ▼
ChatService.stream() ── stream_mode="messages"（按 langgraph_node 过滤，只转发 llm_node 的 token）
   │
   ▼
┌────────────────── 主对话图 (main_graph.py) ──────────────────┐
│  START → classify_node（是否需检索？yes/no）                   │
│              │                                                │
│              ├─ yes → retrieve_node ──────────────┐           │
│              │        （CachePolicy ttl=10）       │           │
│              └─ no ────────────────────────────────┼→ llm_node │
│                                                    │           │
│                                    llm_node（组装提示词 + 流式生成）│
│                                                    │           │
│                                                    ▼           │
│                                            memory_node（提取长期记忆）→ END
└─────────────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────── 检索增强图 (rerank_graph.py) ────────────────┐
│  rewrite（LLM 改写：主查询+子查询+关键词）                        │
│      → retrieve（多路召回 + RRF 融合，top_k=8）                  │
│      → rerank（SiliconFlow 在线重排，top_n=10）                  │
└─────────────────────────────────────────────────────────────────┘
```

### 记忆机制

- **短期记忆**：Checkpointer（PostgresSaver）按 `thread_id` 保存每轮消息，多轮对话自动恢复；`llm_node` 把用户本轮输入与完整回答写入消息历史。
- **长期记忆**：Store（PostgresStore）以 `("rag_chat", user_id)` 命名空间保存 `user_profile` 档案。每轮对话结束后 `memory_node` 用 LLM 提取/合并用户长期信息（姓名、职业、偏好等），仅在内容有实质变化时写入。

### 流式输出要点

- 图内 `llm_node` 使用 `model.stream()` 收集增量后返回**完整消息**（LangGraph 1.x 节点禁止返回生成器，否则 `add_messages` 报 `Unsupported message type: generator`）。
- `stream_mode="messages"` 会捕获图中**所有** LLM 调用的 token（含分类器的 `yes/no`、记忆提取输出），后端 `stream()` 必须按 `meta["langgraph_node"] == "llm_node"` 过滤，避免杂音混入回答。
- SSE 格式：每块 `data: {"content": "..."}\n\n`，结尾 `data: [DONE]\n\n`；前端手写 `ReadableStream` 解析（`EventSource` 不支持 POST）。

## API 接口

所有接口（除 `/health` 和认证接口外）均需在请求头携带 `Authorization: Bearer <access_token>`。

### 认证接口

| 方法   | 路径              | 说明                         |
| ---- | --------------- | -------------------------- |
| POST | `/api/login`    | 用户登录，返回 access token 和用户信息 |
| POST | `/api/register` | 用户注册                       |
| POST | `/api/recover`  | 找回密码（重置密码）                 |
| GET  | `/api/login`    | 登录页面（SPA，返回 index.html）    |
| GET  | `/api/register` | 注册页面（SPA）                  |
| GET  | `/api/recover`  | 找回密码页面（SPA）                |

### 对话与记忆接口

| 方法     | 路径                              | 说明                                                       |
| ------ | ------------------------------- | -------------------------------------------------------- |
| POST   | `/api/chat/`                    | 发起对话，返回 SSE 流。请求体 `{"query": "...", "thread_id": "..."}` |
| GET    | `/api/chat/{thread_id}/history` | 获取会话历史消息（需会话归属校验）                                        |
| DELETE | `/api/chat/{thread_id}`         | 删除会话（需会话归属校验）                                            |
| GET    | `/api/users/{user_id}/sessions` | 获取用户的会话列表（按更新时间倒序，需本人或管理员）                               |
| GET    | `/api/users/{user_id}/memory`   | 获取用户的长期记忆档案（需本人或管理员）                                     |
| GET    | `/health`                       | 健康检查（含数据库连通性探测，返回 `ok` / `degraded`）                     |

> `user_id` 从 JWT access token 中解析（`sub` 字段格式为 `user_id:username`），支持 `admin` 角色越权访问。access token 过期时由后端通过隐式 refresh token 自动续签，并通过响应头 `X-New-Access-Token` 返回新 token。

## 知识库维护

- 源文档位于 `resources/FAQ/`（Markdown，含账号、课程、付费、证书、故障等七个分类）。
- 入库使用 `src/embedding.py`，元数据 `source`（来源）与 `category`（分类）会随向量一并存储，供检索侧过滤。
- 向量库与 RAG 侧共用同一 `embedding_function`（SiliconFlow bge-m3），保证查询向量与入库向量空间一致。

## 常见问题排查

| 现象                             | 原因 / 处理                                            |
| ------------------------------ | -------------------------------------------------- |
| 启动报数据库连接失败                     | 检查 `POSTGRESQL_DB_URL` 与 PostgreSQL 服务；连接池 5 秒快速失败 |
| 流式输出混入 `no`/`yes`              | `stream()` 未按 `langgraph_node` 过滤（见上"流式输出要点"）      |
| 前端打字机失效、内容成块出现                 | Nginx 未开启 `proxy_buffering off`（SSE 被缓冲）           |
| 节点返回生成器报 `NotImplementedError` | LangGraph 1.x 节点必须返回完整消息，流式用 `model.stream()` 收集   |
| 检索结果为空                         | 确认向量库已入库（`GET /health` 正常但无结果时检查 ChromaDB 集合）      |

## 路线图

- [x] 鉴权：`user_id` 从 JWT/会话解析，杜绝客户端伪造
- [ ] 工具图 `tool_graph.py`：接入订单查询、转人工等业务工具
- [ ] 评测集与在线指标（问题解决率、平均响应时长）
- [ ] Supervisor 多 Agent 模式演进（意图识别已具备雏形：`classify_node` + `route`）
- [ ] Docker / docker-compose 一键部署
