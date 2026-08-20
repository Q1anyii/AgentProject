import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from redis.commands.search import Search
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.field import TextField, NumericField, TagField, VectorField
import numpy as np
import redis
from init import embed_model, online_rerank
import json
import time
import uuid
from loguru import logger
from constant.cache_constant import REDIS_INIT_SUCCESS, REDIS_CONNECT_FAILED, REDIS_CONNECT_CLOSED, \
    HEALTH_CHECK_INTERVAL, \
    TAG_FIELD, VECTOR_FIELD_NAME, VECTOR_FIELD_ALGORITHM, VECTOR_ATTRIBUTE, INDEX_NAME, KEY_PREFIX, CACHE_DEFAULT_TTL, \
    CACHE_RERANK_HIT_SCORE
from redis.commands.search.query import Query

from utils.doc_util import documents_to_dicts, dict_to_documents
from utils.lsh_util import RandomProjectionLSH

load_dotenv(override=True)
REDIS_DB_URL = os.getenv("REDIS_DB_URL")


class CacheService:
    db_url: str

    @staticmethod
    #jdbc:redis://localhost:6380
    def parse_url(redis_db_url=REDIS_DB_URL):
        prefix, suffix = redis_db_url.split("//")
        password = prefix.split(":")[-1]
        host, port = suffix.split(":")
        return host, int(port), password

    def __init__(self, redis_db_url: str = REDIS_DB_URL, index_name:str = INDEX_NAME, cache_ttl = CACHE_DEFAULT_TTL):
        self.db_url = redis_db_url or os.getenv("REDIS_DB_URL")
        self.host, self.port, self.password = self.parse_url(self.db_url)
        self.redis = redis.Redis(host=self.host, port= self.port, password=self.password)
        self.index_name = index_name
        self._lsh = None      # LSH 模型复用：planes 必须固定，否则同一 query 每次映射不同 bucket，缓存 key 无限膨胀
        self._lsh_dim = 0
        self.cache_ttl =cache_ttl

    def open(self):
        self.create_index()
        try:
            if self.redis.ping():
                logger.success(REDIS_INIT_SUCCESS)
        except redis.ConnectionError as err:
            logger.error(f"{REDIS_CONNECT_FAILED}:{err}")
            raise

    def close(self):
        if self.redis:
            self.redis.close()
            logger.info(REDIS_CONNECT_CLOSED)

    def __enter__(self):
        self.open()
        return self          # 返回 self，以便在 with 块中使用

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False         # 不抑制异常，异常会继续抛出

    def create_index(self, tag_field=TAG_FIELD,vector_field_name=VECTOR_FIELD_NAME,
        vector_field_algorithm = VECTOR_FIELD_ALGORITHM,
        vector_attribute=VECTOR_ATTRIBUTE):
        # 定义索引字段
        schema = (
            TagField(tag_field),                # 用于过滤
            VectorField(
                vector_field_name,
                vector_field_algorithm,          # 向量索引算法（flat / hnsw）
                vector_attribute
            ),
            TextField("query_text"),              # 可选，用于调试
            NumericField("created_at")            # 可选
        )

        # 创建索引（如果不存在)
        try:
            self.redis.ft(self.index_name).create_index(schema, definition=IndexDefinition(prefix=[KEY_PREFIX]))
            logger.success("success")
        except Exception as e:
            logger.info("Index may already exist:", e)

    @staticmethod
    def query_to_vector(query: str) -> list[float]:
        return embed_model.embed_query(query)

    def _get_lsh(self, dim: int) -> RandomProjectionLSH:
        """惰性创建并复用 LSH 模型（planes 固定，保证同一 query 稳定映射同一 bucket）"""
        if self._lsh is None or self._lsh_dim != dim:
            self._lsh = RandomProjectionLSH(dim=dim, num_bits=64)
            self._lsh_dim = dim
        return self._lsh

    def set_key(self, thread_id: str, query_vector: list[float]):
        dim = len(query_vector)
        lsh_model = self._get_lsh(dim)
        # get_bucket_id 内部会再做一次 hash，这里必须传原始向量（1024 维）；
        # 传 hash 后的 64 位结果会 np.dot((64,1024),(64,)) 维度不匹配报错
        bucket_id = lsh_model.get_bucket_id(query_vector)
        key = f"retrieve_cache:{thread_id}:{bucket_id}"
        return key

    def store_cache(self, thread_id: str, query_text: str,  result: List[Document]):
        query_vector = self.query_to_vector(query_text)
        key = self.set_key(thread_id, query_vector)
        serializable_result = documents_to_dicts(result)
        self.redis.hset(key, mapping={
            "thread_id": thread_id,  # 索引的 Tag 字段，query_cache 按它过滤（缺失会导致 KNN 永远查不到）
            "query_embedding": np.array(query_vector, dtype=np.float32).tobytes(),  # 必须转换为二进制
            "query_text": query_text,
            "result": json.dumps(serializable_result, ensure_ascii=False),
            "created_at": time.time()
        })
        logger.info(f"{key}已存储")
        self.redis.expire(key, self.cache_ttl)

    def query_cache(self, thread_id: str, query: str, top_k: int = 3) -> List[Document] | None:
        query_vector = self.query_to_vector(query)
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
        res = self.redis.ft(self.index_name).search(q, query_params=params)

        if not res.docs:
            return None

        # 向量初筛只做候选召回，不做命中判定：实测 bge-m3 原始 query 向量对短问题的
        # 语义区分度很差（同义改写距离 0.6+，比无关问题还远），绝对距离阈值不可靠，
        # 必须再用重排模型验证候选问题与当前问题是否语义等价，才允许命中缓存
        candidate_texts = [doc.query_text for doc in res.docs]
        try:
            results = online_rerank(query, candidate_texts, top_n=len(candidate_texts))
        except Exception as e:
            # 缓存是优化而非正确性依赖：验证服务不可用时按未命中降级，不阻塞主流程
            logger.warning(f"缓存验证重排调用失败，按未命中处理：{e}")
            return None
        if not results:
            logger.warning("缓存验证重排返回空结果，按未命中处理")
            return None

        best = results[0]
        if best["relevance_score"] >= CACHE_RERANK_HIT_SCORE:
            hit = res.docs[best["index"]]
            # 滑动过期：给真实命中的条目续期（hit.id 才是该条目的 Redis key，
            # 按查询向量 set_key 算出的桶 key 未必存在）
            self.redis.expire(hit.id, self.cache_ttl)
            return dict_to_documents(json.loads(hit.result))

        logger.error(f"未命中，重排最高分 {best['relevance_score']} 低于阈值 {CACHE_RERANK_HIT_SCORE}")
        return None


# 全局单例：连接生命周期由 main.py 的 lifespan 统一 open()/close()，
# 业务模块（retrieve_graph 等）直接 import 本实例，不自行创建/关闭
cache_service = CacheService()