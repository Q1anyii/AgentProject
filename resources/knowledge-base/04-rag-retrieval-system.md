# RAG 检索系统设计实践

> 基于 AgentProject 项目总结，涵盖向量库、嵌入模型、Query 改写、多查询检索、RRF 融合、重排序、距离阈值、缓存策略等完整 RAG  pipeline。

## 一、RAG 整体架构

```
用户问题
    ↓
[缓存检查] → 命中 → 直接返回
    ↓ 未命中
[Query 改写] → 主查询 + 子查询 + 关键词
    ↓
[多查询并行检索] → ChromaDB 向量检索（TOP_K）
    ↓
[距离阈值过滤] → 过滤低相关性结果
    ↓
[RRF 融合] → 多查询结果合并去重
    ↓
[重排序] → bge-reranker-v2-m3 精排
    ↓
[缓存存储] → 存入 Redis（15s TTL）
    ↓
返回最终文档
```

## 二、向量库（ChromaDB）

### 2.1 初始化

```python
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

embedding_function = OpenAIEmbeddingFunction(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    api_base=os.getenv("SILICONFLOW_BASE_URL"),
    model_name="BAAI/bge-m3"
)

client = chromadb.PersistentClient(path="../resources/chroma_db")
collection = client.get_or_create_collection(
    name="mitta_ai_knowledge",
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"},  # 余弦相似度
)
```

### 2.2 文档入库

```python
class EmbeddingProcessor:
    def embed(self, file_path, meta: Meta) -> int:
        # 1. 解析文档（.md 按标题切分，.txt/.pdf 原样加载）
        child_docs = self.split_docs(file_path)
        # 2. 计算稳定 ID（内容 + 元数据哈希，避免重复入库）
        ids = compute_doc_hash_with_meta(child_docs)
        # 3. 组装元数据
        metadatas = [{"source": meta.source, "category": meta.category, **doc.metadata} for doc in child_docs]
        # 4. upsert（存在则更新，不存在则插入）
        self.collection.upsert(
            ids=ids,
            documents=[doc.page_content for doc in child_docs],
            metadatas=metadatas,
        )
        return self.collection.count()
```

### 2.3 文档切分策略

```python
# .md 文件：先按标题切分，再按字符大小切分
headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
parent_docs = splitter.split_text(document)

# 再按字符大小切分（保留上下文重叠）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
child_docs = text_splitter.split_documents(parent_docs)
```

**切分原则**：
- Markdown 按标题切分，保留语义边界
- 字符切分作为兜底，确保单块不超过模型上下文
- overlap 保留上下文，避免关键信息被切断

## 三、嵌入模型

### 3.1 BAAI/bge-m3

```python
from langchain_openai import OpenAIEmbeddings

embed_model = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    api_key=os.getenv("SILICONFLOW_API_KEY")
)
```

**bge-m3 特点**：
- 多语言支持（中英等 100+ 语言）
- 长文本支持（最大 8192 tokens）
- 稠密+稀疏+多向量统一表示
- 开源免费，可通过 SiliconFlow 等平台 API 调用

## 四、Query 改写

### 4.1 LLM 改写

用 LLM 将用户口语化问题改写为结构化查询，提升检索召回率。

```python
REWRITE_PROMPT = """你是一个查询改写专家。请根据用户问题和对话历史，生成：
1. 主查询：最能代表用户意图的检索词
2. 子查询：2-3个相关的补充检索词
3. 关键词：3-5个核心关键词

用户问题：{question}
对话历史：{history}

请以 JSON 格式返回，键名为：主查询、子查询、关键词"""

def rewrite_query(state):
    prompt = REWRITE_PROMPT.format(question=state["question"], history=...)
    resp = model.invoke(prompt, response_format={"type": "json_object"})
    result = QueryRewriteResult(**extract_json(resp.content))
    return {"rewritten_queries": [result.main_query] + result.sub_queries}
```

**改写的价值**：
- 用户问题通常口语化、包含冗余信息
- 改写后生成多个查询角度，提升召回率
- 关键词可用于后续过滤或高亮

### 4.2 JSON 提取容错

LLM 返回的 JSON 可能包含 markdown 代码块或额外文本，需要容错提取。

```python
def extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取 ```json ... ``` 代码块
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # 尝试提取最外层 {}
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        raise
```

## 五、多查询并行检索

### 5.1 ThreadPoolExecutor 并行

```python
from concurrent.futures import ThreadPoolExecutor

TOP_K = 10

with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as ex:
    raw_results = list(
        ex.map(lambda q: collection.query(query_texts=[q], n_results=TOP_K), queries)
    )
```

**ChromaDB query 返回格式**：
```python
{
    "documents": [["doc1", "doc2", ...]],   # 嵌套列表
    "distances": [[0.1, 0.2, ...]],
    "metadatas": [[{"source": "...", ...}, ...]],
    "ids": [["id1", "id2", ...]]
}
```

## 六、距离阈值过滤

### 6.1 过滤低相关性结果

```python
DISTANCE_THRESHOLD = 0.5  # 余弦距离阈值，越小越相关

for res in raw_results:
    docs_list = res["documents"][0]
    dists_list = res["distances"][0]
    meta_list = res["metadatas"][0]
    id_list = res["ids"][0]

    keep_docs, keep_dists, keep_metas, keep_ids = [], [], [], []
    for doc_text, dist, meta, doc_id in zip(docs_list, dists_list, meta_list, id_list):
        if dist < DISTANCE_THRESHOLD:
            meta["_distance"] = dist  # 埋入元数据，便于调试
            keep_docs.append(doc_text)
            keep_dists.append(dist)
            keep_metas.append(meta)
            keep_ids.append(doc_id)
```

**阈值选择**：
- 余弦距离范围 [0, 2]，0 表示完全相同
- 阈值过小 → 召回率低，可能漏检
- 阈值过大 → 精确率低，引入噪声
- 建议根据实际数据分布调整，通常 0.3-0.7 之间

## 七、RRF 融合（Reciprocal Rank Fusion）

### 7.1 算法原理

多查询检索结果按排名融合，排名越靠前权重越高。

```python
RRF_K = 60  # 平滑常数

def rrf_fusion(results: List[List[Document]], k: int = RRF_K) -> List[Document]:
    scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            key = doc.metadata.get("id", doc.page_content)
            if key not in scores:
                scores[key] = {"doc": doc, "score": 0.0}
            scores[key]["score"] += 1.0 / (k + rank + 1)  # RRF 公式

    return [item["doc"] for item in sorted(scores.values(), key=lambda x: x["score"], reverse=True)]
```

**RRF 公式**：`score(d) = Σ 1 / (k + rank_i(d) + 1)`
- k 是平滑常数，通常 60
- rank_i 是文档在第 i 个查询结果中的排名（从 0 开始）
- 多个查询都排前的文档得分最高

**RRF 的优势**：
- 不需要归一化不同查询的相似度分数
- 对排名敏感，能有效合并多视角检索结果
- 实现简单，计算高效

## 八、重排序（Rerank）

### 8.1 bge-reranker-v2-m3

用交叉编码器对融合后的文档做精排，提升 top-k 精确率。

```python
def online_rerank(query: str, documents: list[str], top_n: int = 10) -> list[dict]:
    resp = requests.post(
        f"{os.getenv('SILICONFLOW_BASE_URL')}/rerank",
        headers={"Authorization": f"Bearer {os.getenv('SILICONFLOW_API_KEY')}"},
        json={
            "model": "BAAI/bge-reranker-v2-m3",
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return sorted(resp.json()["results"], key=lambda r: r["relevance_score"], reverse=True)
```

**使用**：
```python
def rerank(state):
    docs = state["merged_docs"]
    if not docs:
        return {"reranked_docs": []}
    query = state["rewritten_queries"][0]  # 用主查询精排
    results = online_rerank(query, [doc.page_content for doc in docs], top_n=10)
    top_docs = [docs[r["index"]] for r in results]
    return {"reranked_docs": top_docs}
```

**检索 vs 重排序**：
| 阶段 | 模型 | 目标 | 速度 | 数据量 |
|------|------|------|------|--------|
| 检索（召回） | 双编码器（bge-m3） | 高召回率，从全库筛 TOP_K | 快（向量索引） | 全库 |
| 重排序（精排） | 交叉编码器（bge-reranker） | 高精确率，从候选中筛 top_n | 慢（逐对计算） | 候选集 |

## 九、缓存策略（Redis）

### 9.1 检索缓存

```python
def check_cache(state, config):
    question = state["question"]
    thread_id = config["configurable"].get("thread_id")
    # 模糊匹配：最近 3 条相似问题直接复用结果
    query_in_cache = cache_service.query_cache(thread_id, question, 3)
    if query_in_cache:
        return {"reranked_docs": query_in_cache, "cache_hit": True}
    return {"cache_hit": False}

def store_cache(state, config):
    if not state.get("cache_hit"):
        cache_service.store_cache(thread_id, state["question"], state["reranked_docs"])
    return {}
```

**缓存设计**：
- TTL 15 秒（短时间内重复问题直接复用）
- 按 thread_id 隔离（不同会话不共享）
- 模糊匹配（最近 N 条相似问题）

## 十、常见陷阱

### 10.1 嵌入函数不一致

入库和检索必须使用同一个嵌入模型，否则向量空间不一致，检索结果无效。

**解决**：全局单例 `embedding_function`，入库和检索都引用它。

### 10.2 ChromaDB 返回嵌套列表

`collection.query()` 返回的 documents/distances/metadatas/ids 都是嵌套列表 `[[...]]`，因为支持多查询。即使只传一个查询，也要取 `[0]`。

### 10.3 距离 vs 相似度

ChromaDB 默认返回 `distances`（距离），不是相似度。余弦距离越小越相似，余弦相似度 = 1 - 距离。

设置阈值时要注意是距离还是相似度。

### 10.4 重排序索引错位

`online_rerank` 返回的 `index` 是传入 documents 列表的索引，不是原始文档 ID。映射回原始文档时要注意。

```python
results = online_rerank(query, [doc.page_content for doc in docs])
top_docs = [docs[r["index"]] for r in results]  # 正确：按传入顺序的索引映射
```

## 十一、个人见解

1. **RAG 的质量瓶颈在数据，不在模型**：再好的检索算法，如果知识库内容质量差、切分不合理，结果也不会好。投入 80% 精力在数据清洗和切分上，回报远高于调参。

2. **Query 改写是性价比最高的优化**：相比换嵌入模型、调阈值，Query 改写能显著提升召回率，且成本低（一次 LLM 调用）。建议所有 RAG 系统都加上。

3. **重排序是精确率的最后一道防线**：向量检索召回的 TOP_K 通常有噪声，重排序能把真正相关的文档排到前面。如果 top_n 很小（如 3-5），重排序的影响非常大。

4. **缓存不是可选，是必须**：LLM 应用的延迟和成本主要在 LLM 调用，但 RAG 检索也有不小开销。15 秒的短缓存能命中大量用户重复提问，显著降低延迟。

5. **距离阈值要根据数据调**：没有通用的最佳阈值。不同领域、不同嵌入模型、不同切分方式，距离分布都不同。建议用标注数据做 ROC 曲线，选择合适的阈值。
