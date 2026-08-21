# 测试 QA - 代码调试题（10条）

> 涵盖 Python、FastAPI、LangGraph、前端等领域的常见代码 bug 和调试技巧。

---

## Q1: 以下代码有什么问题？如何修复？

```python
def get_user_system_prompt(user_id: str, base_prompt: str = None) -> str:
    from service.user_profile_service import user_profile_service
    custom_prompt = user_profile_service.get_system_prompt(user_id)
    if custom_prompt and custom_prompt.strip():
        return f"{base_prompt}\n\n【用户自定义设定】\n{custom_prompt.strip()}"
    return base_prompt
```

**A:**

**问题**：
1. **没有异常处理**：如果 `user_profile_service` 未初始化、数据库连接失败、或 `get_system_prompt` 抛异常，整个对话会中断
2. **base_prompt 可能为 None**：如果调用时不传 base_prompt，`f"{base_prompt}"` 会输出 "None" 字符串

**修复**：
```python
def get_user_system_prompt(user_id: str, base_prompt: str = None) -> str:
    if base_prompt is None:
        base_prompt = system_prompt  # 使用模块级默认值
    try:
        from service.user_profile_service import user_profile_service
        custom_prompt = user_profile_service.get_system_prompt(user_id)
        if custom_prompt and custom_prompt.strip():
            return f"{base_prompt}\n\n【用户自定义设定】\n{custom_prompt.strip()}"
    except Exception as e:
        # 降级：获取失败时使用基础 prompt，不影响对话
        logger.warning(f"获取用户自定义 system prompt 失败 user_id={user_id}: {e}")
    return base_prompt
```

**核心原则**：非核心功能失败时应降级，不应中断主流程。

---

## Q2: 以下 FastAPI 代码有什么问题？

```python
@app.post("/api/chat/upload")
async def upload_file(file: UploadFile = File(...), thread_id: str = Form(None)):
    content = await file.read()
    result = file_upload_service.save_file(
        user_id="123",  # 硬编码
        file_name=file.filename,
        file_content_bytes=content,
        file_type=file.content_type,
        thread_id=thread_id,
    )
    return {"ok": True, "data": result}
```

**A:**

**问题**：
1. **缺少认证**：没有 `Depends(get_current_user)`，任何人都能上传文件
2. **硬编码 user_id**：`user_id="123"` 应该从 JWT 解析
3. **缺少文件大小校验**：大文件可能撑爆内存或数据库
4. **缺少文件类型校验**：可能上传恶意文件
5. **thread_id 归属未校验**：用户可能上传到他人的会话

**修复**：
```python
@app.post("/api/chat/upload")
async def upload_file(
    file: UploadFile = File(...),
    thread_id: Optional[str] = Form(None),
    current_user: TokenData = Depends(get_current_user),
):
    content = await file.read()
    # 大小校验
    if len(content) > 10 * 1024 * 1024:
        return Response.failed("文件不能超过 10MB")
    # 类型校验
    ALLOWED_EXT = {'.txt', '.md', '.pdf', '.docx', '.png', '.jpg', ...}
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return Response.failed(f"不支持的文件类型: {ext}")
    # 会话归属校验（如果 thread_id 不为空）
    if thread_id:
        owner = chat_service.get_thread_user_id(thread_id)
        if owner and owner != str(current_user.user_id) and current_user.role != "admin":
            raise HTTPException(status_code=403, detail="无权上传到该会话")
    try:
        result = file_upload_service.save_file(
            user_id=str(current_user.user_id),
            file_name=file.filename,
            file_content_bytes=content,
            file_type=file.content_type,
            thread_id=thread_id,
        )
        return {"ok": True, "data": result}
    except ValueError as e:
        return Response.failed(str(e))
```

---

## Q3: 以下 LangGraph 节点代码有什么问题？

```python
def retrieve(state: RAGState) -> dict:
    queries = state["rewritten_queries"]
    results = []
    for q in queries:
        res = collection.query(query_texts=[q], n_results=TOP_K)
        results.append(res)
    docs = []
    for res in results:
        for doc in res["documents"]:
            docs.append(doc)
    return {"merged_docs": docs}
```

**A:**

**问题**：
1. **串行检索**：for 循环逐个查询，没有利用并行性
2. **ChromaDB 返回嵌套列表处理错误**：`res["documents"]` 是 `[["doc1", "doc2"]]`，直接 append 会把整个列表作为一个元素
3. **没有距离阈值过滤**：可能引入低相关性噪声
4. **没有去重**：不同查询可能返回相同文档
5. **没有 RRF 融合**：简单合并可能导致排名靠前的文档被淹没

**修复**：
```python
from concurrent.futures import ThreadPoolExecutor

def retrieve(state: RAGState) -> dict:
    queries = state["rewritten_queries"]
    # 并行检索
    with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as ex:
        raw_results = list(ex.map(
            lambda q: collection.query(query_texts=[q], n_results=TOP_K),
            queries
        ))
    # 距离阈值过滤 + 正确处理嵌套列表
    filtered_results = []
    for res in raw_results:
        docs_list = res["documents"][0]  # 取 [0]
        dists_list = res["distances"][0]
        meta_list = res["metadatas"][0]
        id_list = res["ids"][0]
        keep = []
        for doc, dist, meta, did in zip(docs_list, dists_list, meta_list, id_list):
            if dist < DISTANCE_THRESHOLD:
                meta["_distance"] = dist
                keep.append(Document(page_content=doc, metadata=meta, id=did))
        filtered_results.append(keep)
    # RRF 融合（去重 + 排名加权）
    merged_docs = rrf_fusion(filtered_results)
    return {"merged_docs": merged_docs}
```

---

## Q4: 以下前端 SSE 接收代码有什么问题？

```javascript
async function apiChat(query, threadId) {
    const response = await fetch('/api/chat/', {
        method: 'POST',
        body: JSON.stringify({ query, thread_id: threadId })
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let answer = '';
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value);
        const data = JSON.parse(text);
        answer += data.content;
        console.log(answer);
    }
    return answer;
}
```

**A:**

**问题**：
1. **缺少认证头**：没有 `Authorization: Bearer {token}`
2. **缺少 Content-Type**：POST JSON 需要 `Content-Type: application/json`
3. **SSE 事件解析错误**：SSE 格式是 `data: {json}\n\n`，不是直接的 JSON
4. **缓冲区处理缺失**：一次 read 可能包含多个事件或不完整事件，需要缓冲区累积
5. **没有错误处理**：401、403、500 等状态码没有处理
6. **不支持中断**：没有 AbortController，无法停止回复
7. **decoder.decode 没有 stream:true**：可能导致多字节字符被截断

**修复要点**：
```javascript
async function apiChat(query, threadId, onStream, signal) {
    const response = await fetch('/api/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ query, thread_id: threadId }),
        signal: signal,  // AbortController 信号
    });
    // 错误处理
    if (response.status === 401) { /* 跳转登录 */ }
    if (!response.ok) throw new Error(`请求失败: ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let answer = '';
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });  // 注意 stream:true
        const events = buffer.split('\n\n');
        buffer = events.pop();  // 保留不完整事件
        for (const event of events) {
            const line = event.trim();
            if (!line.startsWith('data:')) continue;
            const payload = line.slice(5).trim();
            if (!payload || payload === '[DONE]') continue;
            try {
                const chunk = JSON.parse(payload);
                const text = extractContentText(chunk.content);
                if (text) { answer += text; if (onStream) onStream(answer); }
            } catch { continue; }
        }
    }
    return answer;
}
```

---

## Q5: 以下 Python 代码有什么问题？

```python
def compute_doc_hash(docs):
    ids = []
    for doc in docs:
        raw = doc.page_content + str(doc.metadata)
        ids.append(hashlib.sha256(raw.encode("utf-8")).hexdigest())
    return ids
```

**A:**

**问题**：
1. **metadata 包含不稳定字段**：`doc.metadata` 可能包含时间戳、随机 ID、距离值等，导致同一文档每次哈希不同
2. **换行符不统一**：`\r\n`、`\r`、`\n` 混用会导致哈希不同
3. **没有 strip**：内容前后的空白会影响哈希
4. **str(metadata) 顺序不确定**：Python 字典的 str() 输出顺序在某些情况下可能不确定

**修复**：
```python
def compute_doc_hash_with_meta(docs):
    ids = []
    for doc in docs:
        content = doc.page_content.strip().replace("\r\n", "\n").replace("\r", "\n")
        # 只选择稳定的元数据字段，不要全部 metadata
        meta_keys = ["source", "file_name", "url"]
        meta_part = [f"{k}:{doc.metadata.get(k, '')}" for k in meta_keys]
        raw = content + "\n" + "\n".join(meta_part)
        ids.append(hashlib.sha256(raw.encode("utf-8")).hexdigest())
    return ids
```

---

## Q6: 以下 nginx 配置有什么问题？

```nginx
server {
    listen 80;
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
    }
    location / {
        root /path/to/frontend;
    }
}
```

**A:**

**问题**：
1. **缺少 client_max_body_size**：文件上传会被 nginx 限制（默认 1MB）
2. **SSE 代理配置缺失**：没有关闭 proxy_buffering，流式响应可能被缓冲
3. **SPA history 模式缺失**：`try_files $uri $uri/ /index.html` 没有配置，刷新页面会 404
4. **缺少代理头**：`proxy_set_header Host`、`X-Real-IP` 等没有设置
5. **MCP 端点缺失**：`/mcp/` 需要 WebSocket 升级支持

**修复**：
```nginx
server {
    listen 80;
    client_max_body_size 20m;  # 文件上传大小限制

    location / {
        root /path/to/frontend;
        try_files $uri $uri/ /index.html;  # SPA history 模式
    }

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

    location /mcp/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## Q7: 以下代码有什么问题？

```python
class UserProfileService:
    def __init__(self):
        self._conn = pymysql.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DB"),
        )

    def get_profile(self, user_id):
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT * FROM user_profile WHERE user_id = '{user_id}'")
            return cur.fetchone()
```

**A:**

**问题**：
1. **SQL 注入**：`f"... WHERE user_id = '{user_id}'"` 字符串拼接，用户可注入恶意 SQL
2. **连接在 __init__ 中创建**：模块导入时就建立连接，不利于测试和延迟初始化
3. **没有自动重连**：MySQL 8小时超时后连接断开，不会自动重连
4. **没有 DictCursor**：返回元组而非字典，使用不便
5. **没有 charset**：默认字符集可能不支持 emoji
6. **环境变量可能为 None**：os.getenv 返回 None 时 pymysql 可能报错

**修复**：
```python
class UserProfileService:
    def __init__(self):
        self._conn = None

    def open(self):
        """在 lifespan 启动时调用"""
        self._conn = pymysql.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DB", "mitta"),
            cursorclass=DictCursor,
            charset='utf8mb4',
        )

    def _get_conn(self):
        """自动重连"""
        if not self._conn or not self._conn.open:
            self.open()
        return self._conn

    def get_profile(self, user_id):
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM user_profile WHERE user_id = %s", (user_id,))  # 参数化
            return cur.fetchone()
```

---

## Q8: 以下 Vue 3 代码有什么问题？

```javascript
const messages = ref([]);

async function sendMessage() {
    const userMsg = { id: Date.now(), role: 'user', content: inputText.value };
    messages.value.push(userMsg);
    const aiMsg = { id: Date.now(), role: 'assistant', content: '' };
    messages.value.push(aiMsg);

    const answer = await apiChat(inputText.value);
    aiMsg.content = answer;  // 直接修改对象属性
}
```

**A:**

**问题**：
1. **Vue 3 响应式丢失**：`aiMsg` 是普通对象，push 到 ref 数组后，直接修改 `aiMsg.content` 可能不触发更新（取决于 Vue 版本和代理方式）
2. **ID 可能重复**：`Date.now()` 在快速连续调用时可能重复
3. **没有错误处理**：apiChat 失败时 aiMsg.content 保持空字符串
4. **没有 isLoading 状态**：用户可能重复发送
5. **流式输出不支持**：直接赋值 answer，没有流式更新

**修复**：
```javascript
import { ref, nextTick } from 'vue';

const messages = ref([]);
const isLoading = ref(false);

function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

async function sendMessage() {
    if (isLoading.value || !inputText.value.trim()) return;
    isLoading.value = true;

    const userMsg = { id: generateId(), role: 'user', content: inputText.value };
    messages.value.push(userMsg);

    const aiMsg = { id: generateId(), role: 'assistant', content: '' };
    messages.value.push(aiMsg);
    inputText.value = '';

    try {
        // 流式输出：通过回调更新
        const answer = await apiChat(query, threadId, (text) => {
            // 通过索引更新，确保触发响应式
            const idx = messages.value.findIndex(m => m.id === aiMsg.id);
            if (idx !== -1) {
                messages.value[idx].content = text;
            }
        });
    } catch (err) {
        const idx = messages.value.findIndex(m => m.id === aiMsg.id);
        if (idx !== -1) {
            messages.value[idx].content = '抱歉，发生错误：' + err.message;
        }
    } finally {
        isLoading.value = false;
    }
}
```

---

## Q9: 以下代码有什么问题？

```python
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"ok": False, "detail": str(exc), "traceback": traceback.format_exc()}
    )
```

**A:**

**问题**：
1. **泄露敏感信息**：`str(exc)` 和 `traceback.format_exc()` 可能包含数据库连接串、API Key、文件路径等敏感信息
2. **没有日志记录**：异常没有记录到日志，无法排查
3. **traceback 暴露给客户端**：安全风险，攻击者可以利用堆栈信息了解系统内部结构

**修复**：
```python
import traceback
from loguru import logger

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # 完整异常信息只记日志，不返回给客户端
    logger.exception(
        f"未处理的异常 | path={request.url.path} | "
        f"method={request.method} | type={type(exc).__name__}"
    )
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "detail": "服务器内部错误，请稍后重试或联系管理员",
            "error_type": type(exc).__name__,  # 只返回异常类型，便于前端分类处理
        },
    )
```

---

## Q10: 以下代码有什么问题？

```python
def load_mcp_server_configs():
    raw = os.getenv("MCP_SERVERS")
    servers = json.loads(raw)
    validated = []
    for cfg in servers:
        if cfg.get("type") == "stdio":
            validated.append({**cfg, "type": "stdio"})
        elif cfg.get("type") == "sse":
            validated.append({**cfg, "type": "sse"})
    return validated
```

**A:**

**问题**：
1. **raw 可能为 None**：环境变量未配置时 `json.loads(None)` 会抛 TypeError
2. **JSON 解析失败未处理**：格式错误时直接抛异常，可能阻塞应用启动
3. **servers 可能不是列表**：如果配置是对象或字符串，for 循环会出错
4. **cfg 可能不是字典**：列表中可能包含非对象元素
5. **stdio 缺少 command 未校验**：无效配置应该跳过而非加入
6. **sse 缺少 url 未校验**：同上
7. **cwd 相对路径未解析**：相对路径受工作目录影响，应该解析为绝对路径
8. **不支持的 type 静默忽略**：应该记录警告

**修复**：
```python
def load_mcp_server_configs() -> list[dict]:
    raw = os.getenv("MCP_SERVERS")
    if not raw or not raw.strip():
        logger.info("未配置 MCP_SERVERS，跳过 MCP 工具加载")
        return []
    try:
        servers = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"MCP_SERVERS 不是合法 JSON，跳过 MCP 工具加载：{e}")
        return []
    if not isinstance(servers, list):
        logger.warning("MCP_SERVERS 应为 JSON 数组，跳过 MCP 工具加载")
        return []

    project_root = Path(__file__).resolve().parent.parent
    validated = []
    for cfg in servers:
        if not isinstance(cfg, dict):
            logger.warning(f"忽略无效的 MCP 服务器配置项（非对象）：{cfg}")
            continue
        server_type = cfg.get("type", "stdio")
        if server_type == "stdio":
            if not cfg.get("command"):
                logger.warning(f"忽略 MCP 服务器配置项（stdio 缺少 command）：{cfg}")
                continue
            cwd = cfg.get("cwd")
            if cwd and not Path(cwd).is_absolute():
                cwd = str(project_root / cwd)  # 相对路径解析为绝对路径
            validated.append({**cfg, "type": "stdio", "cwd": cwd})
        elif server_type == "sse":
            if not cfg.get("url"):
                logger.warning(f"忽略 MCP 服务器配置项（sse 缺少 url）：{cfg}")
                continue
            validated.append({**cfg, "type": "sse"})
        else:
            logger.warning(f"忽略 MCP 服务器配置项（不支持的 type={server_type}）：{cfg}")
    logger.info(f"MCP 服务器配置加载完成，共 {len(validated)} 个")
    return validated
```
