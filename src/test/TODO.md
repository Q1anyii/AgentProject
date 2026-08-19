## 使用 Redis + RediSearch 实现向量相似度语义缓存

以下是完整的实现方案，包括环境准备、索引创建、缓存写入、查询流程和阈值判断。代码以 Python 为例，使用 `redis-py` 官方库。

---

### 1. 环境准备

确保 Redis 已安装 **RediSearch** 模块（或直接使用 Redis Stack，它已包含 RediSearch、RedisJSON 等模块）。

- **Docker 安装 Redis Stack**（推荐）：
  ```bash
  docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
  ```

- **本地编译或加载模块**：参考 [RediSearch 官方文档](https://redis.io/docs/stack/search/)。

- **Python 依赖**：
  ```bash
  pip install redis
  ```
  `redis-py` 从 v4.1.0 开始内置了对 RediSearch 的命令支持（`redis.commands.search`）。

---

### 2. 定义缓存数据结构

每个缓存条目存储为一个 Redis **Hash**，键名可自定义（例如使用 UUID 或基于查询的哈希），包含以下字段：

| 字段名          | 类型   | 说明                                   |
|-----------------|--------|----------------------------------------|
| `thread_id`     | TAG    | 会话线程 ID，用于过滤                   |
| `query_embedding` | VECTOR | 查询文本的向量表示                     |
| `result`        | STRING | 检索结果（JSON 序列化）                |
| `created_at`    | NUMERIC | 创建时间戳（可选）                    |
| `query_text`    | STRING | 原始查询文本（用于调试，可选）         |

键名规则：`retrieve_cache:{uuid}` 或 `retrieve_cache:{thread_id}:{md5}`（后者便于调试但可能产生大量键）。

---

### 3. 创建 RediSearch 索引

需要在 Redis 中创建一个索引，指定 `VECTOR` 字段和 `TAG` 字段。

```python
from redis import Redis
from redis.commands.search.field import VectorField, TagField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# 连接 Redis
r = Redis(host='localhost', port=6379, decode_responses=True)

# 定义索引字段
schema = (
    TagField("thread_id"),                # 用于过滤
    VectorField(
        "query_embedding",
        "FLAT",                           # 向量索引算法（FLAT / HNSW）
        {
            "TYPE": "FLOAT32",            # 向量数据类型
            "DIM": 768,                   # 嵌入向量维度（根据模型而定）
            "DISTANCE_METRIC": "COSINE"   # 距离度量：COSINE / L2 / IP
        }
    ),
    TextField("query_text"),              # 可选，用于调试
    NumericField("created_at")            # 可选
)

# 创建索引（如果不存在）
index_name = "idx:retrieve_cache"
try:
    r.ft(index_name).create_index(schema, definition=IndexDefinition(prefix=["retrieve_cache:"]))
except Exception as e:
    print("Index may already exist:", e)
```

**注意**：
- `DIM` 必须与你的 embedding 模型输出维度一致。
- `DISTANCE_METRIC` 可选 `COSINE`、`L2`、`IP`。不同度量下相似度判断逻辑不同（见后文）。

---

### 4. 生成嵌入向量

使用任意 embedding 模型（如 OpenAI、Sentence-Transformers）将查询文本转换为向量。这里以 `sentence-transformers` 为例：

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # 输出 384 维，需与索引 DIM 一致

def text_to_vector(text: str) -> list[float]:
    return model.encode(text).tolist()
```

---

### 5. 写入缓存

当执行完实际检索后，将结果和查询向量存入 Redis，并设置过期时间（TTL）。

```python
import json
import time
import uuid

def store_cache(thread_id: str, query_text: str, query_vector: list[float], result: dict, ttl: int = 600):
    key = f"retrieve_cache:{uuid.uuid4()}"
    r.hset(key, mapping={
        "thread_id": thread_id,
        "query_embedding": np.array(query_vector, dtype=np.float32).tobytes(),  # 必须转换为二进制
        "query_text": query_text,
        "result": json.dumps(result, ensure_ascii=False),
        "created_at": time.time()
    })
    r.expire(key, ttl)
```

**注意**：`VECTOR` 字段在 Redis Hash 中存储时，必须使用**二进制格式**（`float32` 数组的字节串）。写入时用 `np.array(...).tobytes()`，读取时用 `np.frombuffer(...)`。

---

### 6. 查询缓存（KNN 搜索）

查询时，根据当前查询向量，在相同 `thread_id` 下执行 KNN 搜索，获取最相似的缓存条目。

```python
from redis.commands.search.query import Query
import numpy as np

def query_cache(thread_id: str, query_vector: list[float], top_k: int = 1) -> dict | None:
    # 将向量转为二进制
    query_bytes = np.array(query_vector, dtype=np.float32).tobytes()

    # 构建查询：过滤 thread_id，并按向量相似度排序
    q = (
        Query(f"@thread_id:{{{thread_id}}} => [KNN {top_k} @query_embedding $vec AS vector_score]")
        .sort_by("vector_score")           # 按距离升序排序（COSINE/L2 越小越相似）
        .return_fields("vector_score", "result", "query_text", "created_at")
        .dialect(2)                        # 必须使用 dialect 2 以支持 VECTOR
    )

    # 执行查询，传入向量参数
    params = {"vec": query_bytes}
    res = r.ft(index_name).search(q, query_params=params)

    if res.docs:
        doc = res.docs[0]
        distance = float(doc.vector_score)  # 距离值

        # 判断是否命中（阈值需根据度量调整）
        threshold = 0.1  # 对于 COSINE，距离越小越相似（0 表示完全相同）
        if distance <= threshold:
            cached_result = json.loads(doc.result)
            return cached_result
        else:
            print(f"未命中，距离 {distance} 超过阈值 {threshold}")
            return None
    return None
```

**阈值说明**：
- **COSINE 距离**：范围 0~2，0 表示完全相同，越大越不相似。常用阈值 0.1~0.3。
- **L2 距离**：范围 0~∞，0 表示完全相同，越大越不相似。阈值需根据向量分布设定，例如 0.5。
- **IP（内积）**：越大越相似，需设置最小内积阈值。

---

### 7. 完整检索流程（带语义缓存）

```python
def retrieve_node(thread_id: str, query_text: str) -> dict:
    # 1. 生成查询向量
    query_vec = text_to_vector(query_text)

    # 2. 尝试从缓存获取
    cached = query_cache(thread_id, query_vec)
    if cached is not None:
        print("缓存命中，直接返回")
        return cached

    # 3. 缓存未命中，执行实际检索
    print("缓存未命中，执行实际检索")
    actual_result = actual_retrieve(query_text)  # 你的业务检索逻辑

    # 4. 将结果写入缓存
    store_cache(thread_id, query_text, query_vec, actual_result, ttl=600)

    return actual_result
```

---

### 8. 优化与注意事项

#### 8.1 索引算法选择
- `FLAT`：暴力搜索，精确但较慢，适合数据量小（<1M 条）。
- `HNSW`：近似最近邻，速度快，内存占用略高，推荐大规模使用。创建时使用 `VectorField(..., "HNSW", {...})`。

#### 8.2 键名与过期策略
- 使用 UUID 作为键名可避免覆盖，但需定期清理孤儿键（Redis 的 `EXPIRE` 已处理）。
- 若同一 `thread_id` 下相同查询需要更新，可先删除旧缓存或使用固定键名（如 `retrieve_cache:{thread_id}:{query_hash}`），但需要处理并发写入。

#### 8.3 阈值自适应
- 可通过监控日志调整阈值。若误命中（返回错误结果）较多，降低阈值；若漏命中（总是重算）较多，适当提高阈值。
- 可在缓存值中存储原始查询文本，命中后可选做一次快速向量距离验证（已在 `query_cache` 中实现）。

#### 8.4 跨线程缓存
- 若希望不同线程共享缓存，只需在查询时移除 `@thread_id` 过滤条件，但需谨慎避免隐私泄露。

#### 8.5 Redis 集群
- RediSearch 在集群模式下需要特殊配置，确保索引分布在正确的槽位。单机模式下无需担心。

---

### 总结

通过 Redis + RediSearch 的向量搜索能力，可以实现高精度的语义缓存，显著降低重复 API 调用。实现步骤包括：创建向量索引、存储查询向量和结果、执行 KNN 搜索并判断距离。该方案适用于对语义匹配要求较高的场景，且支持自定义阈值和 TTL。