# 工程化实践总结

> 基于 AgentProject 项目总结，涵盖项目结构、代码规范、日志管理、环境配置、依赖管理、测试、部署等工程化实践。

## 一、项目结构

### 1.1 目录结构

```
AgentProject/
├── src/                          # 后端源码
│   ├── main.py                   # FastAPI 入口，路由定义
│   ├── init.py                   # 全局初始化（模型、嵌入、重排、system prompt）
│   ├── config.py                 # 配置管理（环境变量、校验、类型安全访问）
│   ├── embedding.py              # 文档嵌入与向量库操作
│   ├── constant/                 # 常量管理（按模块分类）
│   │   ├── embedding_constants.py
│   │   ├── retrieval_constants.py
│   │   └── cache_constant.py
│   ├── context/                  # 上下文管理（用户上下文）
│   ├── graphs/                   # LangGraph 图定义
│   │   ├── main_graph.py         # 主对话图
│   │   └── retrieve_graph.py     # RAG 检索图
│   ├── middleware/               # 中间件（限流等）
│   ├── mcp_client/               # MCP 客户端
│   ├── schemas/                  # Pydantic 请求/响应模型
│   │   ├── request_schemas/
│   │   └── response_schemas/
│   ├── service/                  # 业务服务层
│   │   ├── chat_service.py
│   │   ├── login_service.py
│   │   ├── cache_service.py
│   │   ├── user_profile_service.py
│   │   └── file_upload_service.py
│   ├── utils/                    # 工具函数
│   │   ├── jwt_utils.py
│   │   ├── response_util.py
│   │   ├── doc_util.py
│   │   └── lsh_util.py
│   └── test/                     # 测试代码
├── resources/                    # 资源文件
│   ├── frontend/                 # 前端单文件应用
│   │   ├── index.html
│   │   └── nginx.conf
│   ├── FAQ/                      # FAQ 知识库
│   ├── system_prompt/            # 系统提示词
│   ├── chroma_db/                # ChromaDB 持久化数据
│   └── knowledge-base/           # 编程知识库（本目录）
├── docs/                         # 项目文档
├── tests/                        # 单元测试
├── .env.example                  # 环境变量模板
├── .gitignore                    # Git 忽略配置
├── requirements.txt              # Python 依赖
└── README.md                     # 项目说明
```

### 1.2 结构设计原则

- **按职责分层**：API 层 → 服务层 → 数据层，每层职责清晰
- **按领域分模块**：constant、context、graphs、service、utils 等目录按功能划分
- **资源与代码分离**：resources/ 存放前端、知识库、向量库等非代码资源
- **测试独立**：tests/ 和 src/test/ 分离，单元测试和集成测试分开

## 二、代码规范

### 2.1 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 模块/文件 | 小写 + 下划线 | `user_profile_service.py` |
| 类 | 大驼峰 | `UserProfileService`, `CtxUser` |
| 函数/方法 | 小写 + 下划线 | `get_user_profile()`, `hash_password()` |
| 常量 | 全大写 + 下划线 | `TOP_K`, `COLLECTION_NAME`, `JWT_SECRET_KEY` |
| 变量 | 小写 + 下划线 | `user_id`, `thread_id`, `is_loading` |
| 私有方法 | 下划线前缀 | `_get_conn()`, `_ensure_table()` |

### 2.2 注释规范

```python
def get_user_system_prompt(user_id: str, base_prompt: str = None) -> str:
    """组装用户级 system prompt：基础默认 + 用户自定义内容。

    Args:
        user_id: 用户 ID
        base_prompt: 基础 system prompt，默认使用模块级 system_prompt

    Returns:
        组装后的 system prompt 字符串

    Note:
        获取用户自定义内容失败时降级使用基础 prompt，不影响对话。
    """
```

**注释原则**：
- 公共函数必须有 docstring，说明功能、参数、返回值
- 复杂逻辑加行内注释，解释"为什么"而非"做什么"
- 注释与代码同步更新，过时的注释比没有注释更糟
- 中文注释（项目团队语言）

### 2.3 类型提示

```python
# 函数签名完整标注
def online_rerank(query: str, documents: list[str], top_n: int = 10) -> list[dict]:
    ...

# TypedDict 用于结构化字典
class RAGState(TypedDict):
    question: str
    history: List[Dict[str, str]]
    rewritten_queries: List[str]

# Pydantic BaseModel 用于数据校验
class QueryRewriteResult(BaseModel):
    main_query: str = Field(..., alias="主查询")
    sub_queries: List[str] = Field(default_factory=list)
```

## 三、日志管理

### 3.1 Loguru 配置

```python
from loguru import logger

# 基本使用
logger.info("信息")
logger.warning("警告")
logger.error("错误")
logger.success("成功")
logger.exception("异常（自动包含堆栈）")

# 带上下文的日志
logger.info(f"用户登录 user_id={user_id}, ip={ip}")
logger.warning(f"获取用户自定义 prompt 失败 user_id={user_id}: {e}")
```

### 3.2 日志级别使用

| 级别 | 使用场景 | 示例 |
|------|----------|------|
| DEBUG | 调试信息，开发时用 | `logger.debug("查询参数: {}", query)` |
| INFO | 正常业务流程 | `logger.info("用户登录成功 user_id={}", user_id)` |
| SUCCESS | 操作成功 | `logger.success("资源初始化完成")` |
| WARNING | 非致命问题，可降级 | `logger.warning("MCP 配置错误，跳过加载")` |
| ERROR | 错误，需要关注 | `logger.error("数据库连接失败")` |
| EXCEPTION | 异常，含堆栈 | `logger.exception("未处理的异常")` |

### 3.3 日志最佳实践

- **异常用 logger.exception**：自动包含堆栈信息，便于排查
- **关键操作记日志**：登录、登出、数据修改、外部 API 调用
- **日志包含上下文**：user_id、thread_id、path 等，便于追踪
- **不记录敏感信息**：密码、token、密钥等不记日志
- **生产环境控制日志量**：避免 DEBUG 级别日志过多影响性能

## 四、环境配置

### 4.1 .env.example 模板

```env
# ===== 必填配置 =====
DEEPSEEK_API_KEY=your_deepseek_api_key
SILICONFLOW_API_KEY=your_siliconflow_api_key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
POSTGRESQL_DB_URL=postgresql://user:password@localhost:5432/mitta
MYSQL_DB_URL=mysql+pymysql://user:password@localhost:3306/mitta
REDIS_DB_URL=redis://localhost:6379/0
JWT_SECRET_KEY=your_jwt_secret_key_change_in_production

# ===== 可选配置 =====
MODEL_NAME=deepseek:deepseek-v4-flash
BASE_URL=https://api.deepseek.com
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30

# ===== MCP 配置（可选）=====
MCP_SERVERS=[{"name":"filesystem","type":"stdio","command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","/data"]}]
```

### 4.2 配置校验

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

### 4.3 配置加载顺序

1. `.env` 文件（`load_dotenv(override=True)`）
2. 系统环境变量
3. 代码中的默认值

**原则**：`.env` 优先于系统环境变量（override=True），便于本地开发覆盖。

## 五、依赖管理

### 5.1 requirements.txt

```txt
# ===== Web 框架 =====
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9

# ===== AI / LLM =====
langchain==0.3.0
langchain-openai==0.2.0
langgraph==0.2.0
langgraph-checkpoint-postgres==2.0.0
langgraph-store-postgres==0.1.0

# ===== 向量库 / 嵌入 =====
chromadb==0.5.0
FlagEmbedding==1.2.10

# ===== 数据库 =====
pymysql==1.1.0
psycopg2-binary==2.9.9
redis==5.0.0

# ===== 安全 / 认证 =====
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0

# ===== 工具 =====
loguru==0.7.2
pydantic==2.9.0
```

### 5.2 依赖管理原则

- **版本锁定**：生产环境锁定版本，避免意外升级导致不兼容
- **分类注释**：按功能分组，加注释说明用途
- **定期更新**：关注安全漏洞，定期升级依赖
- **区分生产/开发**：开发依赖（pytest, black, mypy）单独放 requirements-dev.txt

## 六、测试

### 6.1 测试结构

```
tests/
├── conftest.py              # pytest 配置和 fixture
├── test_api/                # API 接口测试
├── test_service/             # 服务层测试
├── test_graph/               # LangGraph 测试
└── test_utils/               # 工具函数测试
```

### 6.2 测试类型

| 类型 | 测试对象 | 工具 | 示例 |
|------|----------|------|------|
| 单元测试 | 单个函数/类 | pytest | `test_hash_password()` |
| 集成测试 | 多模块协作 | pytest + TestClient | `test_login_flow()` |
| API 测试 | HTTP 接口 | FastAPI TestClient | `test_chat_endpoint()` |
| 端到端测试 | 完整流程 | 自动化测试框架 | `test_full_conversation()` |

### 6.3 测试最佳实践

- **测试命名**：`test_<功能>_<场景>_<预期结果>`
- **Arrange-Act-Assert**：每个测试按 准备→执行→断言 结构
- **Mock 外部依赖**：数据库、LLM API 等用 mock，测试只关注逻辑
- **测试边界情况**：空值、最大值、异常输入
- **测试失败场景**：不仅测试成功路径，也要测试错误处理

## 七、Git 工作流

### 7.1 .gitignore

```gitignore
# ===== Python =====
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
env/

# ===== 环境配置 =====
.env
.env.local
*.key
*.pem

# ===== 数据库 =====
*.db
*.sqlite
resources/chroma_db/

# ===== IDE =====
.idea/
.vscode/
*.swp

# ===== 系统 =====
.DS_Store
Thumbs.db

# ===== 日志 =====
*.log
logs/
```

### 7.2 提交规范

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type 类型**：
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具/依赖

**示例**：
```
feat(auth): 添加 JWT refresh token 自动续签

- 新增 refresh token 存储到 Redis
- access token 过期时自动续签
- 续签失败返回 401 跳转登录

Closes #123
```

## 八、部署

### 8.1 nginx 配置

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 文件上传大小限制
    client_max_body_size 20m;

    # 前端静态文件
    location / {
        root /path/to/frontend;
        try_files $uri $uri/ /index.html;  # SPA history 模式
    }

    # API 代理
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # SSE 流式响应配置
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }

    # MCP 端点
    location /mcp/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 8.2 启动命令

```bash
# 开发模式
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 生产模式（多 worker）
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# 或用 gunicorn + uvicorn worker
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 8.3 部署检查清单

- [ ] 环境变量配置正确（.env 不提交 Git）
- [ ] 数据库连接正常（MySQL/PostgreSQL/Redis）
- [ ] API Key 有效（DeepSeek/SiliconFlow）
- [ ] nginx 配置正确（client_max_body_size、SSE 代理）
- [ ] 防火墙开放端口
- [ ] 日志目录可写
- [ ] 备份策略配置

## 九、常见工程化陷阱

### 9.1 硬编码配置

```python
# 错误：API Key 硬编码
api_key = "sk-abc123..."

# 正确：从环境变量读取
api_key = os.getenv("DEEPSEEK_API_KEY")
```

### 9.2 魔法数字

```python
# 错误：魔法数字散落在代码中
if dist < 0.5: ...
results = collection.query(n_results=10)

# 正确：提取为常量
DISTANCE_THRESHOLD = 0.5
TOP_K = 10
```

### 9.3 重复代码

```python
# 错误：每个接口都写一遍 JWT 解析
@app.get("/api/users/{user_id}/profile")
def get_profile(user_id, authorization: str = Header(...)):
    token = authorization.replace("Bearer ", "")
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    ...

# 正确：封装为依赖
@app.get("/api/users/{user_id}/profile")
def get_profile(user_id, current_user = Depends(get_current_user)):
    ...
```

### 9.4 缺少错误处理

```python
# 错误：假设外部 API 一定成功
resp = requests.post(url, json=data)
result = resp.json()  # 如果 resp 不是 JSON 会报错

# 正确：检查状态码，处理异常
try:
    resp = requests.post(url, json=data, timeout=30)
    resp.raise_for_status()
    return resp.json()
except requests.RequestException as e:
    logger.error(f"API 调用失败: {e}")
    raise
```

### 9.5 不清理资源

```python
# 错误：数据库连接不关闭
conn = pymysql.connect(...)
cur = conn.cursor()
cur.execute("SELECT ...")
# 忘记 close

# 正确：用上下文管理器或 finally
try:
    conn = pymysql.connect(...)
    with conn.cursor() as cur:
        cur.execute("SELECT ...")
finally:
    conn.close()
```

## 十、个人见解

1. **工程化是项目可维护性的基石**：很多项目初期追求快速开发，忽略了工程化（规范、测试、文档、配置管理）。等项目变大后，技术债累积，改一个功能要翻半天代码。本项目在工程化方面做得比较好：分层清晰、常量集中、配置统一、有 .env.example，这些都是可维护性的保障。

2. **代码规范要从第一天执行**：命名规范、注释规范、类型提示，这些看起来是小事，但累积起来影响巨大。等代码写了几万行再想统一规范，成本极高。建议项目初始化时就配置好 ruff/black/mypy 等工具，CI 自动检查。

3. **日志是线上排查的唯一手段**：本地开发可以打断点，但线上出问题只能靠日志。本项目用 loguru，关键操作都有日志，异常用 logger.exception 记录堆栈，这是很好的实践。但要注意日志量，生产环境不要开 DEBUG，否则日志爆炸。

4. **环境变量管理容易被忽视**：很多项目的 .env 格式不统一、缺少注释、没有 .env.example，新人接手要到处问配置。本项目有 config.py 统一管理、必填校验、类型安全访问，还有 .env.example 模板，这是很好的实践。

5. **测试是信心的来源**：没有测试的项目，改代码像拆弹——不知道会不会炸。本项目目前测试覆盖不足（src/test/ 下只有 TODO），建议优先补充核心逻辑的单元测试（密码加密、JWT 解析、RAG 检索、异常处理），然后逐步扩展到集成测试和 API 测试。

6. **Git 提交规范是团队协作的基础**：清晰的提交历史能让 code review 更高效、问题回溯更容易。建议配置 commitlint + husky，强制提交规范。本项目的 Git 历史中有一些 "在变基之前未提交的更改" 这类不规范的提交，建议后续改进。
