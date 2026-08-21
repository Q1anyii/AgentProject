# 项目待添加功能与改进点分析

## 一、`.TODO` 待添加功能逐项分析

| #   | 待添加功能                        | 现状                                                                                                             | 实现思路                                                                         | 优先级                |
| --- | ---------------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------ |
| 1   | **鉴权**（JWT 解析 user_id）       | `user_id` 硬编码 `user_001`（[main.py](file:///E:/工作文件/AgentProject/src/main.py) 与前端 `CURRENT_USER_ID`），任何人可伪造、串记忆 | FastAPI 依赖注入解析 JWT → 注入 `ChatService` 方法；前端登录后携带 token                       | 🔴 P0（安全）          |
| 2   | **工具图** `tool_graph.py` + 重试 | 仅占位 `return None`；无任何外部工具（订单查询、转人工）                                                                            | `build_tool_graph(self)` 构建工具调用节点，用 `tenacity`（已在依赖中）封装 LLM/API 重试           | 🟡 P1              |
| 3   | **评测集与在线指标**                 | 无评测；仅有 LangSmith 追踪（可观察，不可量化）                                                                                  | FAQ 构建问答对离线评测（召回率/MRR/端到端准确率）；在线记录问题解决率、平均响应时长                               | 🟡 P1              |
| 4   | **Supervisor 多 Agent**       | `classify_node` + `route` 已是意图路由雏形                                                                             | 拆 Supervisor/IntentAgent/RAGAgent/MemoryAgent；当前架构（Service + 按图拆分）已为它提供干净挂载点 | 🟢 P2（架构演进）        |
| 5   | **Docker 一键部署**              | 手动 conda + nginx + postgres                                                                                    | Dockerfile（后端）+ docker-compose（postgres + chroma + nginx + api）              | 🟡 P1（交付）          |
| 6   | **超步限制**                     | 无 `recursion_limit` 配置，图死循环时无限执行                                                                               | 调用时传 `config={"recursion_limit": 25}`（LangGraph 默认 25，需显式设保险值）               | 🟡 P1（稳定性）         |
| 7   | **异常处理**                     | 仅启动期 DB 检查；运行时 LLM/检索异常直接 500 裸错误                                                                              | 统一 `ExceptionHandler` 返回结构化 `{"detail": ...}`；区分可重试（超时/限流）与不可重试错误            | 🔴 P0              |
| 8   | **Redis 缓存**                 | `InMemoryCache`（单进程、重启即失）；`retrieve_node` 有 `CachePolicy(ttl=10)`                                              | 换 `langgraph.cache.redis.RedisCache`（需 redis 依赖）；或仅对改写结果/热点 FAQ 加 Redis      | 🟡 P1（多 worker 必需） |
| 9   | **多模态**                      | 仅文本输入；FAQ 纯 Markdown                                                                                           | 前端文件上传 → 后端解析（`unstructured` 已在依赖中）→ 走 `embedding.py` 入库或临时 RAG              | 🟢 P2              |

## 二、代码审查发现的补充改进点（TODO 未覆盖）

### 🔴 正确性缺陷（建议优先修）

1. **异常日志是字面省略号**：[chat_service.py:76](file:///E:/工作文件/AgentProject/src/chat_service.py) 与旧版 graph.py 中 `logger.error(...)` 的 `...` 是字面量——`open()` 里 DB 连接失败的**真实异常被吞掉**，排查问题只能靠猜。应改为 `logger.error(f"数据库连接失败：{e}")`。

2. **知识库入库会覆盖旧文档**：[embedding.py:117](file:///E:/工作文件/AgentProject/src/embedding.py) 的 `ids = [f"C{i}" ...]` 每次从 `C0` 重新编号，`upsert` 会导致**新文档 chunk 与旧文档 id 冲突互相覆盖**（目前只入了一份 FAQ 所以未暴露）。应改为基于内容哈希或 `source+序号` 的稳定 id，且同源文档重入时先 `delete(where=source)`。

3. **无相似度阈值过滤**：检索只按 top_k 截断，无关文档也会进入重排和上下文。应在 `retrieve` 节点对 chroma 返回的 `distance` 设阈值（如 cosine < 0.3 过滤），防止噪声污染回答（此前有相关经验记录）。

### 🟡 健壮性/性能改进

4. **无请求限流与并发保护**：`/api/chat/` 无速率限制，恶意/异常调用可打爆 DeepSeek API 配额。建议 slowapi 或中间件限流。

5. **`get_user_sessions` 全表扫描**：file:///E:/工作文件/AgentProject/src/chat_service.py `checkpointer.list(None)` 遍历所有 checkpoint 再按 user_id 过滤，会话量大时性能差。应为 PostgresSaver 的 checkpoint 表加 `user_id` 查询条件（metadata 已有该字段）。

6. **每次对话 3 次 LLM 调用**：classify + 生成 + 记忆提取，记忆提取对**寒暄类**对话也白跑一次。可对 `needs_retrieval=False` 且无实质内容的对话跳过 `memory_node`，或记忆提取改为异步。

7. **`make_serializable` 死代码**：[chat_service.py:106](file:///E:/工作文件/AgentProject/src/chat_service.py) 定义了但从未调用，应删除。

8. **`retrieve_node` 缓存策略粒度**：`CachePolicy(ttl=10)` 缓存整个节点输出，若同 thread 内连续追问，改写结果命中但检索重跑——可接受，但注意 10s 过期后相同问题会重复消耗 API。-->thread_id+语义存储作为key存储入redis

9. **重排线程无上限**：`ThreadPoolExecutor(max_workers=len(queries))`，若改写出 5+ 子查询会开 5+ 线程，建议 `min(len(queries), 4)` 封顶。

10. **健康检查只覆盖 DB**：`/health` 不探测 ChromaDB 与 LLM 连通性，可扩展为三合一状态报告。

### 🟢 体验/工程化

11. **流式中断**：前端无"停止生成"按钮，用户无法中断长回答（浪费 token）。可在端点里检测断开连接后取消图执行。
12. **token 用量统计**：未统计每次对话的 token 消耗（LangSmith 有但未接入业务侧），成本核算缺失。
13. **测试缺失**：`requirements` 里有 pytest 但零测试。建议至少补：`embedding.py` 解析/入库单测、`stream()` 过滤逻辑单测、API 冒烟测试。
14. **旧文件清理**：`src/graph.py`（939 行）已被 `chat_service.py + graphs/` 完全取代且无引用，建议删除或归档，避免误导。
15. **`.env` 缺失键校验**：启动时对 `DEEPSEEK_API_KEY`、`POSTGRESQL_DB_URL` 等必填项做显式校验，缺配置直接给出清晰错误。

## 三、建议实施顺序

```
第一阶段（正确性，1 天内）    第二阶段（稳定性/交付）       第三阶段（架构演进）
├─ ① 修复 logger.error(...)   ├─ ④ 统一异常处理 + 限流      ├─ ⑦ Supervisor 多 Agent
├─ ② embedding id 冲突        ├─ ⑤ recursion_limit 超步     ├─ ⑧ tool_graph 工具接入
├─ ③ 相似度阈值过滤           ├─ ⑥ Docker 化 + Redis 缓存   ├─ ⑨ 多模态上传
└─ ⑬ 删除旧 graph.py         └─ ⑫ 评测集 + 单测           └─ ⑩ 鉴权落地
```
优化长期记忆与短期记忆存储策略，动态刷新/自动压缩，防止资源消耗，
抽离向量数据库注入方式，实现可兼容chroma/milvus
优化 model_with_tools 以避免大量无关工具传入模型，采用分层筛选 + 动态绑定的策略