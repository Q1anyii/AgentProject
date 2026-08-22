在实际开发中，优化 `model_with_tools` 以避免大量无关工具传入模型，通常采用**分层筛选 + 动态绑定**的策略。下面我将结合真实项目的常见做法，详细介绍如何设计和实现这套机制。

---

## 1. 实际开发中的典型架构

```
用户输入
   ↓
[第一层：规则/关键词粗筛]  → 快速排除明显无关工具（可选）
   ↓
[第二层：向量检索细筛]    → 基于语义相似度选出 top-N 工具
   ↓
[第三层：LLM 精筛]        → 让轻量模型从 top-N 中选出最终工具（可选）
   ↓
绑定筛选后的工具到主模型 → 生成回复/执行工具
```

**核心思想**：不让主模型看到所有工具，而是先通过多级漏斗将候选工具缩减到可管理的数量（通常 3~10 个），再交给模型进行精确的工具调用。

---

## 2. 工具元数据的标准化

在开发初期，就要为每个工具定义完善的元数据，这是所有筛选机制的基础。一个工具通常包含：

```python
{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": { ... },          # JSON Schema
    "tags": ["weather", "api"],     # 分类标签
    "keywords": ["weather", "temperature", "forecast"],  # 关键词（可选）
    "enabled": True,
    "priority": 1,                  # 优先级
}
```

这些元数据可以存储在数据库、配置文件或代码中，并通过工具注册中心统一管理。

---

## 3. 第一层：基于规则的快速过滤（Cost ≈ 0）

如果工具具有明确的分类或标签，可以先根据用户查询中的关键词进行粗筛。

**示例**：  

- 查询包含“天气”、“温度” → 保留 `tags` 包含 `weather` 的工具。  
- 查询包含“计算”、“加” → 保留 `tags` 包含 `math` 的工具。  
- 如果无法判断分类，则跳过此层，直接进入向量检索。

**实现方式**：

```python
def rule_based_filter(query: str, all_tools: List[Tool]) -> List[Tool]:
    query_lower = query.lower()
    selected = []
    for tool in all_tools:
        # 检查 tags 或 keywords
        if any(tag in query_lower for tag in tool.tags):
            selected.append(tool)
    return selected or all_tools  # 若没有匹配则返回全部，避免漏选
```

实际开发中，这一层通常非常轻量，目的是减少进入向量检索的工具数量，降低向量检索的延迟和成本。

---

## 4. 第二层：向量检索语义筛选（核心层）

当工具数量较大（例如 >50）或规则无法覆盖时，使用向量检索是最可靠的方法。这一层通常借助嵌入模型和向量数据库实现。

### 4.1 离线构建工具向量索引

在系统启动或工具变更时，将所有工具的元数据文本（通常是名称 + 描述 + 参数说明）转换为向量，并存入向量库。

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 工具文本：拼接 name、description、tags 等
tool_texts = [
    f"{tool.name}: {tool.description} (tags: {', '.join(tool.tags)})"
    for tool in all_tools
]

# 构建向量索引
vectorstore = FAISS.from_texts(
    tool_texts,
    embeddings,
    metadatas=[{"tool_index": i} for i in range(len(all_tools))]
)
```

**优化点**：

- 使用异步批量嵌入加快构建速度。
- 向量索引可以持久化到磁盘，避免每次启动重新计算。
- 若工具频繁更新，可使用支持增量添加的向量库（如 Chroma、Qdrant）。

### 4.2 在线检索

每次用户查询时，将查询文本向量化，检索最相似的 top-K 个工具。

```python
def retrieve_tools(query: str, k: int = 5) -> List[Tool]:
    docs = vectorstore.similarity_search_with_score(query, k=k)
    # 可根据相似度得分进一步过滤（例如只保留 score > 阈值）
    tools = []
    for doc, score in docs:
        if score > threshold:   # threshold 需要根据实际数据调整
            index = doc.metadata["tool_index"]
            tools.append(all_tools[index])
    return tools
```

**关键参数**：

- `k`：检索数量，通常取 5~20。太少可能漏掉相关工具，太多则失去筛选意义。
- `threshold`：相似度阈值，用于过滤低分工具。需要根据嵌入模型和工具集特点调整。

### 4.3 使用 LangChain 的 `ToolRetriever`

LangChain 提供了现成的 `ToolRetriever`，可以直接将工具列表转换为检索器，简化代码。

```python
from langchain.retrievers import ToolRetriever

retriever = ToolRetriever(tools=all_tools, embeddings=embeddings, k=5)
relevant_tools = retriever.invoke(user_query)  # 直接返回 List[BaseTool]
```

它内部会处理向量化和检索，并返回工具对象，非常方便。

---

## 5. 第三层：LLM 精筛（可选，用于高准确率场景）

如果第二层检索出的工具仍然较多（例如 20 个），或者需要更精确地判断用户意图，可以引入一个轻量级 LLM 进行二次筛选。

**做法**：

- 使用一个便宜快速的模型（如 GPT-3.5-turbo、Claude Haiku）或本地小模型。
- 输入：用户查询 + 候选工具列表（只包含名称和简短描述）。
- 输出：需要使用的工具名称列表（JSON 格式）。

```python
def llm_refine_tools(query: str, candidate_tools: List[Tool]) -> List[Tool]:
    prompt = f"""Given the user query: "{query}"
Select the most relevant tools from the following list. Return a JSON list of tool names.

Available tools:
{format_tools_for_prompt(candidate_tools)}
"""
    response = selector_llm.invoke(prompt)
    selected_names = json.loads(response.content)
    return [t for t in candidate_tools if t.name in selected_names]
```

**使用场景**：

- 工具之间语义高度重叠，向量检索容易混淆。
- 需要处理多步推理（用户请求可能涉及多个工具）。
- 对成本不敏感，但对工具选择准确性要求极高。

---

## 6. 动态绑定与执行

经过上述筛选后，得到最终的工具集（通常 3~10 个），将其绑定到主模型：

```python
model = ChatOpenAI(model="gpt-4o")
model_with_tools = model.bind_tools(final_tools)

response = model_with_tools.invoke(messages)
```

如果模型返回了工具调用，再执行对应的工具并继续对话循环。

---

## 7. 实际项目中的额外优化

### 7.1 缓存与性能

- **工具向量缓存**：工具描述相对静态，可以定期更新向量索引，不必每次查询都重新嵌入。
- **嵌入批量处理**：离线构建向量时使用批量嵌入 API，提高效率。
- **异步检索**：在 Agent 循环中使用异步检索，避免阻塞。

### 7.2 监控与评估

- 记录每次工具筛选的结果（选择了哪些工具、用户是否满意）。
- 建立离线评估集，计算检索的 recall@k、precision@k。
- 根据评估结果调整 `k` 值、阈值或嵌入模型。

### 7.3 降级策略

- 如果向量检索返回空（例如查询过于模糊），回退到默认工具集或让模型主动询问用户。
- 如果 LLM 精筛失败（返回格式错误），则使用向量检索的结果。
- 如果所有筛选都失败，最后兜底：让模型在没有工具的情况下回答，并提示用户提供更多信息。

### 7.4 支持多轮对话

在多轮对话中，工具选择不能只看当前轮次的用户输入，还需要考虑对话历史。常见的做法是将最近几轮的上下文拼接后用于检索，或者使用对话状态来辅助筛选。

---

## 8. 实际代码示例（简化版）

以下是一个实际项目中可能使用的工具选择器类：

```python
class ToolSelector:
    def __init__(self, all_tools, embeddings, use_llm_refine=False):
        self.all_tools = all_tools
        self.embeddings = embeddings
        self.vectorstore = self._build_vectorstore()
        self.selector_llm = None
        if use_llm_refine:
            self.selector_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def _build_vectorstore(self):
        # 构建向量索引
        texts = [f"{t.name}: {t.description}" for t in self.all_tools]
        return FAISS.from_texts(texts, self.embeddings)

    def select_tools(self, query: str, k: int = 5) -> List[BaseTool]:
        # 第一层：规则粗筛（可选）
        candidates = self._rule_filter(query)
        if len(candidates) <= k:
            return candidates

        # 第二层：向量检索
        retrieved = self._vector_retrieve(query, k=k*2)  # 多取一些供精筛

        # 第三层：LLM 精筛
        if self.selector_llm and len(retrieved) > k:
            retrieved = self._llm_refine(query, retrieved)

        return retrieved[:k]

    def _rule_filter(self, query):
        # 简单的 tag 匹配
        matched = [t for t in self.all_tools if any(tag in query.lower() for tag in t.tags)]
        return matched or self.all_tools

    def _vector_retrieve(self, query, k):
        docs = self.vectorstore.similarity_search(query, k=k)
        indices = [doc.metadata["index"] for doc in docs]
        return [self.all_tools[i] for i in indices]

    def _llm_refine(self, query, tools):
        # 实现略
        pass
```

使用时：

```python
selector = ToolSelector(all_tools, embeddings, use_llm_refine=True)
relevant_tools = selector.select_tools(user_query)

model = ChatOpenAI(model="gpt-4o").bind_tools(relevant_tools)
```

---

## 9. 总结

在实际开发中，优化工具传入的核心是**设计一个工具选择层**，它通常包含规则粗筛、向量检索细筛和可选的 LLM 精筛。具体实施时：

- 从简单的规则过滤开始，快速见效。
- 当工具规模扩大后，引入向量检索，这是最通用的解决方案。
- 对于高准确率场景，增加 LLM 精筛，但要注意成本和延迟。
- 持续监控和评估，调整参数和策略。

没有一种方案适合所有场景，需要根据工具数量、查询复杂度、成本预算等因素权衡选择。