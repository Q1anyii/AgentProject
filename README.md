# 米塔 AI 智能客服（MITA）

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
| 大模型       | DeepSeek（OpenAI 兼容协议，`init_chat_model` 初始化）                                  |
| 向量库       | ChromaDB（持久化于 `recourses/chroma_db`，cosine 距离）                               |
| Embedding | SiliconFlow `BAAI/bge-m3`                                                    |
| 重排        | SiliconFlow `BAAI/bge-reranker-v2-m3` 在线重排                                   |
| 数据库       | PostgreSQL（psycopg3 + psycopg_pool 连接池）                                      |
| Web 服务    | FastAPI + Uvicorn（SSE 流式响应）                                                  |
| 前端/部署     | 原生 HTML/JS/CSS + Nginx（静态托管 + API 反向代理）                                      |
| 可观测性      | LangSmith 链路追踪（可选）                                                           |

## 目录结构

```
AgentProject/
├── src/                          # 后端源码
│   ├── main.py                   # FastAPI 入口：lifespan 资源管理 + REST 端点
│   ├── chat_service.py           # ChatService 核心类：资源生命周期 + 全部业务方法
│   ├── graphs/                   # LangGraph 图定义（按图拆分）
│   │   ├── main_graph.py         # 主对话图：classify → route → (retrieve | llm) → memory
│   │   ├── rerank_graph.py       # 检索增强图：rewrite → retrieve(RRF) → rerank
│   │   └── tool_graph.py         # 工具图（预留占位）
│   ├── init.py                   # 模型 / Embedding / 重排 / 系统提示词初始化
│   ├── embedding.py              # EmbeddingProcessor：文档解析 → 切分 → 向量入库
│   ├── schemas.py                # Pydantic 请求模型
│   └── graph.py                  # 旧版单文件实现（重构前的类封装版本，当前未被引用）
├── recourses/
│   ├── FAQ/                      # 知识库源文档（在线学习平台 FAQ，RAG 专用）
│   ├── system_prompt/            # 客服系统提示词（小亦人设与回答规则）
│   ├── chroma_db/                # ChromaDB 向量库持久化数据
│   └── frontend/                 # 前端静态资源
│       ├── index.html            # 聊天界面（原生 JS 单页应用）
│       ├── favicon.png           # 站点图标
│       └── nginx.conf            # Nginx 配置（前端托管 + /api 反向代理）
├── .env                          # 环境变量（含 API 密钥，勿提交版本库）
├── env_template.txt              # 环境变量模板
├── requirements_full.txt         # 完整依赖清单（按用途分组）
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
# 创建并激活 conda 环境（Python 3.13）
conda create -n langchain1.2 python=3.13 -y
conda activate langchain1.2

# 安装依赖
pip install -r requirements_full.txt
```

### 2. 配置环境变量

复制 `env_template.txt` 为 `.env` 并填写：

| 变量                                             | 说明                                                                                |
| ---------------------------------------------- | --------------------------------------------------------------------------------- |
| `DEEPSEEK_API_KEY`                             | DeepSeek 平台密钥（主模型）                                                                |
| `MODEL_NAME`                                   | 模型名，如 `deepseek:deepseek-v4-flash`                                                |
| `BASE_URL`                                     | DeepSeek OpenAI 兼容接口地址，如 `https://api.deepseek.com`                               |
| `SILICONFLOW_API_KEY` / `SILICONFLOW_BASE_URL` | SiliconFlow 密钥与地址（Embedding + 重排）                                                 |
| `POSTGRESQL_DB_URL`                            | PostgreSQL 连接串，如 `postgresql://user:pass@localhost:5432/rag_chat?sslmode=disable` |
| `LANGSMITH_*`                                  | LangSmith 可观测性（可选，`LANGSMITH_TRACING=false` 可关闭）                                  |

### 3. 准备基础设施

- **PostgreSQL**：创建数据库（如 `rag_chat`）。表结构由 `PostgresSaver.setup()` / `PostgresStore.setup()` 在服务启动时自动创建，无需手动建表。
- **知识库入库**：将 FAQ 文档写入向量库（交互式输入文档路径与 `source`/`category`）：

```bash
cd src
python embedding.py
# 示例输入：
# 文档路径: ../recourses/FAQ/在线学习平台FAQ知识库（智能客服RAG专用）.md
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

将 `recourses/frontend/nginx.conf` 替换到 Nginx 的 `conf/` 目录（按注释调整 `root` 路径），然后：

```bash
nginx            # 或 nginx -s reload
```

浏览器访问 `http://localhost` 即可使用。Nginx 同时负责将 `/api/*` 反向代理到 `localhost:8000`（已配置 `proxy_buffering off` 保证 SSE 流式）。

> 开发调试时也可直接打开 `recourses/frontend/index.html`（前后端同域时 `API_BASE` 为空字符串使用相对路径）。

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

| 方法     | 路径                              | 说明                                                       |
| ------ | ------------------------------- | -------------------------------------------------------- |
| POST   | `/api/chat/`                    | 发起对话，返回 SSE 流。请求体 `{"query": "...", "thread_id": "..."}` |
| GET    | `/api/chat/{thread_id}/history` | 获取会话历史消息                                                 |
| DELETE | `/api/chat/{thread_id}`         | 删除会话                                                     |
| GET    | `/api/users/{user_id}/sessions` | 获取用户的会话列表（按更新时间倒序）                                       |
| GET    | `/api/users/{user_id}/memory`   | 获取用户的长期记忆档案                                              |
| GET    | `/health`                       | 健康检查（含数据库连通性探测，返回 `ok` / `degraded`）                     |

> 当前 `user_id` 为演示固定值 `user_001`（前端常量），后续可接入 JWT/会话鉴权。

## 知识库维护

- 源文档位于 `recourses/FAQ/`（Markdown，含账号、课程、付费、证书、故障等七个分类）。
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

- [ ] 鉴权：`user_id` 从 JWT/会话解析，杜绝客户端伪造
- [ ] 工具图 `tool_graph.py`：接入订单查询、转人工等业务工具
- [ ] 评测集与在线指标（问题解决率、平均响应时长）
- [ ] Supervisor 多 Agent 模式演进（意图识别已具备雏形：`classify_node` + `route`）
- [ ] Docker / docker-compose 一键部署
