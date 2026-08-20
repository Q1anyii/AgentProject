HEALTH_CHECK_INTERVAL = 30
REDIS_INIT_SUCCESS = "Redis服务器初始化成功"
REDIS_CONNECT_FAILED = "Redis服务器连接失败"
REDIS_CONNECT_CLOSED = "Redis服务器连接池已关闭"

TAG_FIELD = "thread_id"
VECTOR_FIELD_NAME = "query_embedding"
VECTOR_FIELD_ALGORITHM = "HNSW"
VECTOR_ATTRIBUTE = {
    "TYPE": "FLOAT32",            # 向量数据类型
    "DIM": 1024,                   # 嵌入向量维度（根据模型而定）
    "DISTANCE_METRIC": "COSINE"   # 距离度量：COSINE / L2 / IP
}

INDEX_NAME = "idx:retrieve_cache"
KEY_PREFIX = "retrieve_cache:"

CACHE_DEFAULT_TTL = 900