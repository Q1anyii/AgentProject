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

# retrieve_node 节点级缓存 TTL（秒）：LangGraph CachePolicy 声明，缓存后端为 compile(cache=RedisCache)
# 落地的 Redis（见 chat_service.open）；短窗口去重重复检索，可按知识库更新频率调整
CACHE_RETRIEVE_NODE_TTL = 10

# tool_node 节点级缓存 TTL（秒）：仅工具真实执行后写入（含执行失败返回的错误 ToolMessage），
# TTL 窗口内同工具同参数调用直接复用结果、跳过真实执行；注意写类工具（如 update-record）
# 在窗口内同参数调用会被复用，需严格实时时应缩短 TTL 或从缓存 key 中排除
CACHE_TOOL_NODE_TTL = 10

# memory_node 节点级缓存 TTL（秒）：仅当本轮工具被执行或执行失败（无工具可用）时写入，
# idle 轮（筛选出工具但模型未调用）不缓存；命中时跳过 LLM 记忆提取与 store 写入
CACHE_MEMORY_NODE_TTL = 10

# 缓存命中验证阈值：向量初筛召回候选后，用 bge-reranker-v2-m3 验证候选问题与当前问题
# 是否语义等价（实测同义改写 0.89+，无关问题 0.0，取 0.5 有充足间隔）
CACHE_RERANK_HIT_SCORE = 0.5