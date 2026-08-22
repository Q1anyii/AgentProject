# Mitta AI 流程图

## 1. 主对话图（main_graph）

```mermaid
flowchart TD
    START([START]) --> CLASSIFY[classify_node<br/>意图分类]

    CLASSIFY -->|LLM 判断是否需要知识库检索| ROUTE{route<br/>needs_retrieval?}

    ROUTE -->|Yes| RETRIEVE[retrieve_node<br/>RAG 知识库检索]
    ROUTE -->|No| LLM

    RETRIEVE -->|检索结果转 dict 存入 state| LLM[llm_node<br/>主对话生成]

    LLM -->|1. 组装 System Prompt<br/>2. 读取长期记忆 Store<br/>3. ToolFilter 筛选工具<br/>4. model.stream 生成| ROUTE_LLM{route_after_llm<br/>tool_calls 非空?}

    ROUTE_LLM -->|有工具调用| TOOL[tool_node<br/>ToolNode 执行 MCP 工具]
    ROUTE_LLM -->|无工具调用| MEMORY[memory_node<br/>长期记忆提取]

    TOOL -->|工具执行结果 ToolMessage| LLM
    TOOL -.->|CachePolicy TTL=10s<br/>同工具同参数复用结果| TOOL

    MEMORY -->|idle 闲聊轮快速跳过<br/>executed/unavailable 轮<br/>LLM 提取记忆写入 Store| END_NODE([END])

    RETRIEVE -.->|CachePolicy TTL=10s<br/>key=input_str| RETRIEVE

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
    START([START]) --> CHECK_CACHE[check_cache<br/>Redis 缓存检查]

    CHECK_CACHE --> CACHE_HIT{缓存命中?}

    CACHE_HIT -->|命中| RETURN_CACHE[直接返回缓存文档<br/>reranked_docs]
    CACHE_HIT -->|未命中| REWRITE[rewrite_node<br/>LLM 查询改写]

    REWRITE -->|主查询 + 2~3 子查询 + 3~5 关键词<br/>解决指代、补全限定词| RETRIEVE[retrieve_node<br/>多路向量召回]

    RETRIEVE -->|每个查询独立向量检索<br/>Top-K=5 / 距离阈值=0.3| RRF[RRF 融合<br/>Reciprocal Rank Fusion]

    RRF -->|多查询结果去重融合<br/>k=60 排名权重衰减| RERANK[rerank_node<br/>在线重排]

    RERANK -->|SiliconFlow bge-reranker-v2-m3<br/>按相关性降序| CACHE_WRITE[写入 Redis 缓存<br/>TTL=900s]

    CACHE_WRITE --> RETURN[返回 Top-K 文档<br/>output: List Document]
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
