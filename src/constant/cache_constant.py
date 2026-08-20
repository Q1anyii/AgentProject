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

# 登录态 token 的 Redis key 模板（main.py 签发 / jwt_utils 校验共用，避免两处格式漂移）
USER_TOKEN_KEY = "user:{user_id}:token"
USER_REFRESH_TOKEN_KEY = "user:{user_id}:refresh_token"

CACHE_DEFAULT_TTL = 900

# 缓存命中验证阈值：向量初筛召回候选后，用 bge-reranker-v2-m3 验证候选问题与当前问题
# 是否语义等价（实测同义改写 0.89+，无关问题 0.0，取 0.5 有充足间隔）
CACHE_RERANK_HIT_SCORE = 0.5