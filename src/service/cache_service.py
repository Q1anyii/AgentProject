import os

from dotenv import load_dotenv

load_dotenv(override=True)
db_url = os.getenv("REDIS_DB_URL")

prefix, suffix = db_url.split("@")
password = prefix.split(":")[-1]
ip_address,db_id = suffix.split("/")
host, port = ip_address.split(":")

"""4. 实现流程
缓存读取
获取当前 thread_id 和查询文本。

对查询文本计算语义桶 ID。

拼接 Redis key。

尝试 GET 该 key：

命中：反序列化结果，直接返回。

未命中：执行检索逻辑。

缓存写入
执行检索逻辑，得到节点输出。

计算语义桶 ID（与读取时一致）。

拼接 key，将结果序列化后 SETEX 写入，设置合理 TTL。

5. 进阶：使用向量相似度的语义缓存
当简单哈希无法满足语义匹配需求时，可以引入向量搜索。以 Redis + RediSearch 为例：

存储结构
使用 Redis Hash 存储每个缓存条目：

thread_id: 会话 ID

query_embedding: 查询向量（使用 VECTOR 类型）

result: 检索结果

created_at: 时间戳

查询流程
对当前查询生成 embedding。

使用 RediSearch 的 FT.SEARCH 命令，以 query_embedding 为查询向量，限定 thread_id，执行 KNN 搜索，返回最相似的 1 条记录。

检查返回记录的距离是否小于阈值（如 0.1），若满足则视为命中，返回其 result。

否则执行实际检索，并将新条目存入 Redis（自动过期可由 EXPIRE 实现）。

优点
语义匹配精准，即使文本完全不同但语义相近也能命中。

支持跨线程的全局缓存（若去掉 thread_id 限制）
"""

import redis
import hashlib
from typing import Any


r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def get_semantic_bucket_id(query: str) -> str:
    # 示例：使用规范化文本的 MD5 哈希
    normalized = query.lower().strip()
    # 可以加入停用词去除、词干化等
    return hashlib.md5(normalized.encode()).hexdigest()[:12]  # 截断以减小 key

def retrieve_node(thread_id: str, query: str) -> Any:
    bucket_id = get_semantic_bucket_id(query)
    cache_key = f"retrieve_cache:{thread_id}:{bucket_id}"

    # 尝试读缓存
    cached = r.get(cache_key)
    if cached:
        data = json.loads(cached)
        return data["result"]

    # 未命中，执行实际检索
    result = actual_retrieve(query)  # 调用你的检索逻辑

    # 写入缓存，TTL 设为 600 秒（10 分钟）
    cache_value = json.dumps({
        "query": query,
        "result": result,
        "created_at": time.time()
    })
    r.setex(cache_key, 600, cache_value)
    return result