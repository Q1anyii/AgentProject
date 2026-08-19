import numpy as np
import redis

from redis.commands.search import Search
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.field import TextField, NumericField, TagField, VectorField
from redis.commands.search.query import Query

from init import embed_model

# 连接 Redis
r = redis.Redis(host='localhost', port=6380,  decode_responses=True)

# 定义索引字段
schema = (
    TagField("thread_id"),                # 用于过滤
    VectorField(
        "query_embedding",
        "HNSW",                           # 向量索引算法（FLAT / HNSW）
        {
            "TYPE": "FLOAT32",            # 向量数据类型
            "DIM": 1024,                   # 嵌入向量维度（根据模型而定）
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
    print("success")
except Exception as e:
    print("Index may already exist:", e)

    from sentence_transformers import SentenceTransformer


def text_to_vector(text: str) -> list[float]:
    return embed_model.encode(text).tolist()

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