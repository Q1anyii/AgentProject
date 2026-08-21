# LangGraph 架构设计实践

> 基于 AgentProject 项目总结，涵盖状态图、节点设计、条件边、Checkpointer、Store、RunnableConfig 等核心概念。

## 一、核心概念

### 1.1 StateGraph（状态图）

LangGraph 的核心是状态图：节点是函数，边是状态流转，状态在节点间传递和累积。

```python
from langgraph.graph.state import StateGraph
from langgraph.constants import START, END
from typing import TypedDict, List, Optional

class RAGState(TypedDict):
    question: str
    history: List[dict]
    rewritten_queries: List[str]
    merged_docs: List[dict]
    reranked_docs: List[dict]
    cache_hit: Optional[bool]

builder = StateGraph(state_schema=RAGState)
```

**状态设计原则**：
- 状态是累积的（additive），节点返回的 dict 会合并到状态中
- 只在状态中存放需要跨节点传递的数据
- 大对象（如完整文档列表）可以放状态，但要注意序列化开销

### 1.2 节点（Node）

节点是普通函数，接收状态，返回需要更新的状态字段。

```python
def rewrite_query(state: RAGState) -> dict:
    """Query 改写节点：用 LLM 生成主查询+子查询+关键词"""
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in state.get("history", []))
    prompt = REWRITE_PROMPT.format(question=state["question"], history=history_text or "无")
    resp = model.invoke(prompt, response_format={"type": "json_object"})
    result = QueryRewriteResult(**extract_json(resp.content))
    queries = [result.main_query] + result.sub_queries
    return {"rewritten_queries": queries}  # 返回需要更新的字段
```

**节点设计原则**：
- 单一职责：每个节点只做一件事
- 纯函数化：尽量不依赖外部状态，输入输出明确
- 返回部分状态：只返回需要更新的字段，不返回整个状态

### 1.3 边（Edge）

#### 普通边

```python
builder.add_edge(START, "check_cache")
builder.add_edge("rewrite", "retrieve")
builder.add_edge("retrieve", "rerank")
builder.add_edge("output_node", END)
```

#### 条件边（Conditional Edges）

根据状态动态决定下一个节点。

```python
builder.add_conditional_edges(
    "check_cache",
    lambda state: "hit" if state.get("cache_hit") else "miss",
    {
        "hit": "output_node",   # 缓存命中：直接输出
        "miss": "rewrite",       # 缓存未命中：走检索流程
    },
)
```

#### 扇出（Send）

并行执行多个节点实例。

```python
from langgraph.types import Send

def map_queries(state: RAGState):
    """为每个改写后的查询创建一个检索任务"""
    return [Send("retrieve", {"query": q}) for q in state["rewritten_queries"]]

builder.add_conditional_edges("rewrite", map_queries)
```

## 二、检索图完整示例

```python
def build_retrieve_graph(collection):
    class RAGState(TypedDict):
        question: str
        history: List[Dict[str, str]]
        rewritten_queries: List[str]
        merged_docs: List[Document]
        reranked_docs: List[Document]
        cache_hit: Optional[bool]

    # 节点定义
    def check_cache(state, config): ...
    def store_cache(state, config): ...
    def rewrite_query(state): ...
    def retrieve(state): ...
    def rerank(state): ...
    def output_node(state): ...

    # 构建图
    builder = StateGraph(state_schema=RAGState)
    builder.add_node("check_cache", check_cache)
    builder.add_node("store_cache", store_cache)
    builder.add_node("rewrite", rewrite_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("rerank", rerank)
    builder.add_node("output_node", output_node)

    # 边
    builder.add_edge(START, "check_cache")
    builder.add_conditional_edges("check_cache", ...)
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "store_cache")
    builder.add_edge("rerank", "output_node")
    builder.add_edge("output_node", END)

    return builder.compile()
```

**流程**：
1. `check_cache` → 命中则直接输出，未命中则继续
2. `rewrite` → Query 改写
3. `retrieve` → 多查询并行检索 + RRF 融合
4. `rerank` → 重排序
5. `store_cache` → 存入缓存（并行）
6. `output_node` → 输出结果

## 三、Checkpointer（检查点）

### 3.1 PostgresSaver

Checkpointer 用于持久化图的执行状态，支持对话历史、中断恢复、人机交互。

```python
from langgraph.checkpoint.postgres import PostgresSaver

# 初始化
checkpointer = PostgresSaver.from_conn_string(os.getenv("POSTGRESQL_DB_URL"))
checkpointer.setup()  # 创建表结构

# 编译图时传入
graph = builder.compile(checkpointer=checkpointer)

# 调用时传入 config
config = {"configurable": {"thread_id": "user-123-session-456"}}
result = graph.invoke({"question": "..."}, config=config)
```

### 3.2 自定义扩展

继承 PostgresSaver 扩展便捷方法，同时保持与父类签名兼容。

```python
class CustomPostgresSaver(PostgresSaver):
    def list(self, config=None, *, thread_id=None, user_id=None, filter=None, before=None, limit=None):
        """扩展支持 thread_id / user_id 便捷过滤，保持与父类签名兼容"""
        if filter is None:
            filter = {}
        if user_id is not None:
            filter["user_id"] = user_id  # 按 metadata 内 user_id 过滤
        if thread_id is not None:
            if config is None:
                config = {}
            config.setdefault("configurable", {})["thread_id"] = thread_id
        return super().list(config, filter=filter, before=before, limit=limit)
```

**关键约束**：
- 扩展方法必须保持与父类签名兼容，否则 LangGraph 内部调用会 TypeError
- 新增参数必须放在 `*` 之后（keyword-only），避免位置参数冲突

## 四、Store（长期记忆）

Store 用于存储跨会话的长期数据，与 Checkpointer（会话级状态）互补。

```python
from langgraph.store.postgres import PostgresStore

store = PostgresStore.from_conn_string(os.getenv("POSTGRESQL_DB_URL"))
store.setup()

# 存储用户全局自定义 prompt
store.put(("user_global", user_id), "custom_prompt", {"content": "..."})

# 读取
item = store.get(("user_global", user_id), "custom_prompt")
if item:
    content = item.value.get("content")
```

**Checkpointer vs Store**：
| 特性 | Checkpointer | Store |
|------|-------------|-------|
| 用途 | 会话执行状态、对话历史 | 跨会话长期记忆、用户配置 |
| 粒度 | thread_id（会话级） | namespace + key（任意层级） |
| 自动管理 | LangGraph 自动写入 | 手动 put/get |
| 典型场景 | 多轮对话、中断恢复 | 用户偏好、知识库、长期记忆 |

## 五、RunnableConfig

`RunnableConfig` 是 LangGraph 传递运行时配置的标准方式。

```python
from langchain_core.runnables.config import RunnableConfig

def check_cache(state: RAGState, config: RunnableConfig) -> dict:
    thread_id = config["configurable"].get("thread_id", None)
    user_id = config["configurable"].get("user_id", None)
    ...
```

**常用配置项**：
- `configurable.thread_id`：会话 ID（Checkpointer 用）
- `configurable.user_id`：用户 ID（业务逻辑用）
- `recursion_limit`：递归深度限制（默认 25）
- `tags`：标签（用于 LangSmith 追踪过滤）

## 六、流式输出

### 6.1 stream 模式

```python
# 流式输出节点状态
for chunk in graph.stream({"question": "..."}, config=config, stream_mode="values"):
    # chunk 是当前状态的快照
    if "answer" in chunk:
        print(chunk["answer"], end="", flush=True)
```

### 6.2 SSE 封装

```python
def stream(self, user_id, thread_id, query):
    """生成器：yield SSE 格式数据"""
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    for chunk in self.graph.stream({"question": query}, config=config, stream_mode="values"):
        if "answer" in chunk:
            yield f"data: {json.dumps({'content': chunk['answer']})}\n\n"
    yield "data: [DONE]\n\n"
```

## 七、常见陷阱

### 7.1 状态累积导致数据膨胀

TypedDict 状态是累积的，如果节点返回大对象且不清理，会导致状态越来越大。

**解决**：只在状态中存放必要数据，大对象用引用或临时存储。

### 7.2 递归深度超限

默认 `recursion_limit=25`，如果图中有循环或过多节点，会触发 `GraphRecursionError`。

**解决**：调用时传入 `config={"recursion_limit": 100}`，或检查是否有无限循环。

### 7.3 Checkpointer 表未创建

使用 PostgresSaver 时必须先调用 `setup()` 创建表结构，否则首次调用会报错。

### 7.4 线程安全

LangGraph 的 stream 是同步生成器，在异步 FastAPI 中调用时会阻塞事件循环。

**解决**：用 `asyncio.to_thread` 托管，或用 `agraph`（异步图）。

## 八、个人见解

1. **状态图是 LangGraph 的核心思维模型**：不要把它当成普通的函数调用链，要思考状态如何在节点间流动和累积。好的状态设计能让图的逻辑清晰易懂。

2. **条件边是灵活性的来源，但也是复杂度的来源**：过多的条件边会让图的执行路径难以追踪。建议条件边不超过 3 层，复杂逻辑用子图封装。

3. **Checkpointer 和 Store 的边界要清晰**：Checkpointer 管"这次对话"，Store 管"这个用户"。不要把用户偏好存在 Checkpointer 里，也不要把对话历史存在 Store 里。

4. **LangGraph 不适合简单的顺序流程**：如果你的流程就是 A→B→C 没有分支和状态，用普通函数链更简单。LangGraph 的价值在于状态管理、中断恢复、人机交互和复杂分支。
