# 测试 QA - 基础概念题（10条）

> 涵盖 Python、FastAPI、LangGraph、RAG、数据库等领域的基础概念。

---

## Q1: TypedDict 和 BaseModel 的区别是什么？各自适用于什么场景？

**A:**

| 特性 | TypedDict | BaseModel (Pydantic) |
|------|-----------|----------------------|
| 运行时校验 | 否（仅类型提示） | 是（自动校验类型和必填） |
| 序列化/反序列化 | 否 | 是（.dict(), .json(), parse_obj()） |
| 别名支持 | 否 | 是（Field(alias=...)） |
| 默认值 | 支持 | 支持（Field(default=...)） |
| 性能 | 高（无运行时开销） | 较低（有校验开销） |

**适用场景**：
- **TypedDict**：LangGraph 状态定义、内部函数参数、不需要校验的结构化字典
- **BaseModel**：API 请求/响应模型、外部数据输入、需要校验和序列化的场景

**本项目示例**：
- `RAGState` 用 TypedDict（LangGraph 状态，内部传递）
- `QueryRewriteResult` 用 BaseModel（LLM 输出解析，需要校验和别名）

---

## Q2: FastAPI 的 Depends 依赖注入有什么优势？

**A:**

1. **代码复用**：认证、校验、分页等逻辑封装为依赖，多个路由复用
2. **自动参数解析**：FastAPI 自动解析路径参数、查询参数、请求体注入依赖
3. **依赖嵌套**：依赖可以依赖其他依赖（如 `require_self_or_admin` 依赖 `get_current_user`）
4. **生命周期管理**：支持 `yield` 依赖，在请求结束后自动清理资源
5. **可测试性**：测试时可以用 `app.dependency_overrides` 替换依赖

**本项目示例**：
```python
def require_self_or_admin(user_id: str, current_user: TokenData = Depends(get_current_user)):
    if str(current_user.user_id) != user_id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权访问该用户资源")
    return current_user
```

---

## Q3: LangGraph 的 Checkpointer 和 Store 有什么区别？

**A:**

| 特性 | Checkpointer | Store |
|------|-------------|-------|
| 用途 | 会话执行状态、对话历史 | 跨会话长期记忆、用户配置 |
| 粒度 | thread_id（会话级） | namespace + key（任意层级） |
| 自动管理 | LangGraph 自动写入 | 手动 put/get |
| 数据结构 | 图状态快照 | 任意 key-value |
| 典型场景 | 多轮对话、中断恢复、人机交互 | 用户偏好、知识库、长期记忆 |

**本项目示例**：
- Checkpointer（PostgresSaver）：存储对话历史和图执行状态
- Store（PostgresStore）：存储用户全局自定义 prompt（`("user_global", user_id), "custom_prompt"`）

---

## Q4: RAG 检索中为什么需要 Query 改写？直接用用户问题检索不行吗？

**A:**

直接用用户问题检索的问题：
1. **口语化冗余**：用户问题通常包含"请问"、"我想知道"等无意义词汇
2. **指代不明**：多轮对话中"它"、"这个"等指代需要结合历史解析
3. **单一视角**：一个查询只能覆盖一个检索角度，可能漏检相关文档
4. **关键词缺失**：用户可能用同义词而非文档中的术语

Query 改写的价值：
- **生成主查询**：最能代表用户意图的检索词
- **生成子查询**：2-3个相关补充检索词，覆盖多视角
- **提取关键词**：用于后续过滤或高亮
- **提升召回率**：多查询并行检索 + RRF 融合，显著提升召回

**本项目实现**：用 LLM 生成 JSON 格式的 `{主查询, 子查询, 关键词}`，然后多查询并行检索。

---

## Q5: RRF（Reciprocal Rank Fusion）融合算法的原理是什么？

**A:**

**公式**：`score(d) = Σ 1 / (k + rank_i(d) + 1)`

- `k`：平滑常数，通常取 60
- `rank_i(d)`：文档 d 在第 i 个查询结果中的排名（从 0 开始）
- 对每个查询的结果，按排名计算分数并累加
- 最终按总分降序排列

**特点**：
1. **不需要归一化**：不同查询的相似度分数尺度可能不同，RRF 只关心排名
2. **对排名敏感**：排名越靠前权重越高，但不是线性衰减
3. **实现简单**：时间复杂度 O(n*m)，n 为查询数，m 为结果数
4. **效果稳定**：在多查询融合场景下表现稳定，是工业界常用方案

**本项目实现**：`retrieve_graph.py` 中的 `rrf_fusion()` 函数，`RRF_K=60`。

---

## Q6: 为什么 RAG 检索后还需要重排序（Rerank）？

**A:**

向量检索（双编码器）和重排序（交叉编码器）的区别：

| 阶段 | 模型 | 目标 | 速度 | 数据量 |
|------|------|------|------|--------|
| 检索（召回） | 双编码器（bge-m3） | 高召回率，从全库筛 TOP_K | 快（向量索引） | 全库 |
| 重排序（精排） | 交叉编码器（bge-reranker） | 高精确率，从候选中筛 top_n | 慢（逐对计算） | 候选集 |

为什么需要重排序：
1. **向量检索是近似匹配**：为了速度用 HNSW 等近似索引，可能有误差
2. **双编码器独立编码**：query 和 document 分别编码，交互信息有限
3. **交叉编码器深度交互**：query 和 document 拼接后一起编码，能捕捉细粒度相关性
4. **TOP_K 中有噪声**：检索返回的 TOP_K（如 10 条）中可能有不相关的，重排序能把真正相关的排到前面

**本项目实现**：用 `BAAI/bge-reranker-v2-m3` 在线 API 重排序，top_n=10。

---

## Q7: JWT 认证中为什么还要用 Redis 存储 token？JWT 本身不是无状态的吗？

**A:**

JWT 确实是无状态的——签发后包含所有信息，服务端不需要存储。但无状态也意味着**无法主动失效**。

用 Redis 存储 token 的目的：
1. **主动登出**：用户登出时删除 Redis 中的 token，使旧 token 立即失效
2. **改密码后失效**：用户改密码后，清除 Redis 中所有旧 token
3. **限流**：基于 user_id 统计请求次数
4. **双 Token 机制**：refresh token 只存 Redis 不下发前端，access 过期时后端自动续签
5. **强制下线**：管理员可以强制某个用户下线

**代价**：
- 引入 Redis 依赖，增加系统复杂度
- 每次请求需要查 Redis（可优化为本地缓存 + Redis）
- 失去了 JWT 的纯无状态优势

**本项目实现**：access token 存 Redis（15分钟），refresh token 存 Redis（30天，不下发前端）。

---

## Q8: ChromaDB 的 collection.query() 返回格式是什么？为什么是嵌套列表？

**A:**

**返回格式**：
```python
{
    "documents": [["doc1", "doc2", ...]],   # 嵌套列表
    "distances": [[0.1, 0.2, ...]],
    "metadatas": [[{"source": "..."}, ...]],
    "ids": [["id1", "id2", ...]]
}
```

**为什么是嵌套列表**：
- ChromaDB 支持一次传入多个查询（`query_texts=["q1", "q2"]`）
- 外层列表对应每个查询，内层列表对应该查询的结果
- 即使只传一个查询，也要取 `[0]` 才能拿到结果

**常见错误**：
```python
# 错误：直接取 documents，得到的是列表的列表
docs = result["documents"]  # [["doc1", "doc2"]]，不是 ["doc1", "doc2"]

# 正确：取 [0]
docs = result["documents"][0]  # ["doc1", "doc2"]
```

**本项目处理**：`retrieve_graph.py` 中明确取 `res["documents"][0]`、`res["distances"][0]` 等。

---

## Q9: FastAPI 中全局异常处理器和 HTTPException 处理器的执行顺序是什么？

**A:**

**执行顺序**：
1. 如果抛出 `HTTPException`，FastAPI 优先用 `@app.exception_handler(HTTPException)` 处理
2. 如果抛出其他异常，用 `@app.exception_handler(Exception)` 处理
3. 如果没有注册对应处理器，用 FastAPI 默认处理器

**关键细节**：
- `HTTPException` 是 FastAPI 的特殊异常，不会被 `Exception` 处理器捕获（除非 HTTPException 处理器未注册）
- 异常处理器按注册顺序匹配，但 FastAPI 会优先匹配更具体的异常类型
- 自定义异常类如果继承自 HTTPException，会被 HTTPException 处理器捕获

**本项目实现**：
```python
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    # 处理所有未处理异常，返回 500
    logger.exception(...)
    return JSONResponse(status_code=500, content={"ok": False, "detail": "服务器内部错误"})

@app.exception_handler(HTTPException)
async def http_exception_handler(exc):
    # 统一包装 HTTPException 为 {ok, detail} 格式
    return JSONResponse(status_code=exc.status_code, content={"ok": False, "detail": exc.detail})
```

---

## Q10: 什么是 SSE（Server-Sent Events）？和 WebSocket 有什么区别？

**A:**

**SSE**：服务器向客户端单向推送事件的 HTTP 协议，基于 HTTP 长连接。

| 特性 | SSE | WebSocket |
|------|-----|-----------|
| 通信方向 | 单向（服务器→客户端） | 双向 |
| 协议 | HTTP（文本） | 独立协议（ws://） |
| 数据格式 | 文本（默认 UTF-8） | 文本或二进制 |
| 自动重连 | 支持（内置） | 需手动实现 |
| 事件类型 | 支持自定义事件名 | 无内置事件类型 |
| 代理兼容 | 好（HTTP 协议） | 差（需代理支持升级） |
| 浏览器支持 | EventSource API | WebSocket API |
| 适用场景 | 通知、流式输出、实时更新 | 聊天、游戏、协作编辑 |

**为什么 AI 对话用 SSE**：
1. 单向流式输出足够（用户发消息，AI 流式回复）
2. 协议简单，不需要额外的连接管理
3. 天然支持 HTTP 缓存和代理
4. 自动重连机制
5. FastAPI 原生支持 `StreamingResponse`

**本项目实现**：`/api/chat/` 接口返回 `StreamingResponse(event_stream, media_type="text/event-stream")`，前端用 `fetch + ReadableStream` 接收。
