"""向量库抽象与实现（Chroma / Milvus 可插拔）。

契约（所有实现必须一致）：
- distance 统一为"距离，越小越近"：Chroma 原生即 cosine distance；Milvus 的
  COSINE 度量返回 similarity（越大越近），实现内部换算为 1 - similarity
- query 返回 list[list[RetrievedDoc]]：每个 query_texts 对应一个结果列表，
  已按 DISTANCE_THRESHOLD 过滤且按距离升序（topK 截断由 n_results 控制）
- upsert 使用与 Chroma 一致的 sha256 哈希 id（rrf_fusion 的融合 key 依赖它）


"""
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from constant.embedding_constants import COLLECTION_NAME
from constant.retrieval_constants import DISTANCE_THRESHOLD
from vector.retrieve_doc import RetrievedDoc


class VectorStore(Protocol):
    """向量库统一接口：只声明契约，不提供默认实现。

    实现类：ChromaVectorStore（当前默认）/ MilvusVectorStore（需 pymilvus）。
    业务层（retrieve_graph / embedding / chat_service）只依赖本接口，
    不直接 import chromadb / pymilvus。
    """

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None: ...

    def query(self, query_texts: list[str], n_results: int, distance_threshold: float) -> list[list[RetrievedDoc]]: ...

    def count(self) -> int: ...


class ChromaVectorStore:
    """ChromaDB 实现：包装 collection，把四键返回结构归一为 RetrievedDoc。

    collection 自带 embedding_function（工厂创建时注入），
    query / upsert 均无需手动向量化。
    """

    def __init__(self, collection: Any):
        self.collection = collection

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, query_texts: list[str], n_results: int, distance_threshold: float = DISTANCE_THRESHOLD) -> list[list[RetrievedDoc]]:
        # chroma 同步阻塞调用丢线程池并发执行，避免多 query 串行拖慢检索
        with ThreadPoolExecutor(max_workers=min(max(len(query_texts), 1), 4)) as ex:
            raw_results = list(
                ex.map(
                    lambda q: self.collection.query(query_texts=[q], n_results=n_results),
                    query_texts,
                )
            )

        # chroma collection.query 返回格式：{documents:[[...]], distances:[[...]], metadatas:[[...]], ids:[[...]]}
        results: list[list[RetrievedDoc]] = []
        for res in raw_results:
            retrieved: list[RetrievedDoc] = []
            for doc_text, dist, meta, doc_id in zip(
                res["documents"][0], res["distances"][0],
                res["metadatas"][0], res["ids"][0],
            ):
                if dist >= distance_threshold:
                    continue
                meta = dict(meta or {})  # 无元数据文档返回 None，兜底为空 dict（并拷贝避免污染原对象）
                meta["_distance"] = dist  # 埋入元数据，用于 langsmith 调试看距离
                retrieved.append(RetrievedDoc(text=doc_text, distance=dist, metadata=meta, id=doc_id))
            results.append(retrieved)  # 每个 query 的结果独立成列表，避免跨 query 交叉污染

        return results

    def count(self) -> int:
        return self.collection.count()


class MilvusVectorStore:
    """Milvus 实现（pymilvus，需 pip install pymilvus 后启用）。

    与 Chroma 的三个关键差异（正确性前提）：
    1. embedding 前置：Milvus 不做自动向量化，query / upsert 前必须手动 embed
    2. 距离换算：COSINE 度量返回 similarity（越大越近），统一转为 1 - similarity，
       否则 retrieve 节点的 DISTANCE_THRESHOLD 过滤语义会完全颠倒
    3. metadata 为 JSON 字段，返回时可能是 dict 或 JSON 字符串，统一归一为 dict

    embedding_function 兼容两种形态：langchain Embeddings（embed_documents/
    embed_query）与 chroma EmbeddingFunction（仅 __call__），见 _embed_texts/_embed_query。
    """

    def __init__(self, uri: str, collection_name: str, embedding_function: Any, dimension: int = 1024):
        from pymilvus import MilvusClient  # 延迟导入：未安装 pymilvus 时不影响 chroma 路径

        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        if not self.client.has_collection(collection_name):
            # 主键沿用 Chroma 的 sha256 哈希 id（VarChar），保证 rrf_fusion key 与数据迁移一致性
            self.client.create_collection(
                collection_name=collection_name,
                dimension=dimension,  # bge-m3 向量维度 1024
                primary_field_name="id",
                vector_field_name="vector",
                id_type="string",
                max_length=64,  # VarChar 主键必须显式指定 max_length：sha256 哈希 id 为 64 字符
                metric_type="COSINE",
            )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量向量化：优先 langchain 接口，退化用 chroma 风格 __call__（单次调用即批量）。

        输出统一归一为 list[list[float]]：chroma 风格返回 list[numpy.ndarray]，
        langchain 风格返回 list[list[float]]，Milvus 需要纯 float 序列（struct.pack 要求）。
        """
        fn = self.embedding_function
        if hasattr(fn, "embed_documents"):
            vectors = fn.embed_documents(texts)
        else:
            vectors = fn(texts)
        return [self._normalize_vector(v) for v in vectors]

    @staticmethod
    def _normalize_vector(v) -> list[float]:
        """向量形态归一：兼容 [向量] 嵌套 / numpy 数组 / 原生列表 → list[float]。

        判据：若 v 恰好含一个元素且该元素是序列（非标量），剥掉外层
        （chroma 风格 embed_query 返回 [向量]，langchain 风格返回向量本身）；
        再逐元素转 float，非数值元素抛出带上下文的结构化错误。
        """
        if len(v) == 1 and hasattr(v[0], "__len__"):
            v = v[0]  # 剥掉外层： [向量] → 向量
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"embedding 输出含非数值元素：{type(v)} 前几个元素={list(v)[:3]!r}"
            ) from e

    def _embed_query(self, text: str) -> list[float]:
        """单条向量化：优先 embed_query，退化用 __call__ 包一层取首个结果。

        形态差异陷阱：chroma 风格的 embed_query 返回 [向量]（批量语义），
        langchain 风格返回向量本身；统一经 _normalize_vector 归一为 list[float]。
        """
        fn = self.embedding_function
        if hasattr(fn, "embed_query"):
            v = fn.embed_query(text)
        else:
            v = fn([text])[0]
        return self._normalize_vector(v)

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        from pymilvus import MilvusClient

        vectors = self._embed_texts(documents)
        data = [
            {"id": doc_id, "vector": vector, "document": doc, "metadata": meta}
            for doc_id, vector, doc, meta in zip(ids, vectors, documents, metadatas)
        ]
        self.client.upsert(collection_name=self.collection_name, data=data)

    def query(self, query_texts: list[str], n_results: int, distance_threshold: float = DISTANCE_THRESHOLD) -> list[list[RetrievedDoc]]:
        # 同步实现：与 ChromaVectorStore / VectorStore 协议契约一致，
        # 调用链（retrieve_graph / tool_filter / embedding）均为同步；
        # 如需异步检索需全链路改造（AsyncMilvusClient + async 节点），见 toolsTODO 7.1
        from pymilvus import MilvusClient

        # ① embedding 前置：Milvus 不自动向量化查询文本
        vectors = [self._embed_query(q) for q in query_texts]
        hits = self.client.search(
            collection_name=self.collection_name,
            data=vectors,
            limit=n_results,
            output_fields=["document", "metadata"],
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        )

        results: list[list[RetrievedDoc]] = []
        for hits_per_query in hits:
            retrieved: list[RetrievedDoc] = []
            for hit in hits_per_query:
                distance = 1 - hit["distance"]  # ② COSINE similarity → 统一距离语义
                if distance >= distance_threshold:
                    continue
                meta = hit["entity"].get("metadata")
                if isinstance(meta, str):
                    meta = json.loads(meta or "{}")  # ③ JSON 字符串归一为 dict
                meta = dict(meta or {})
                meta["_distance"] = distance
                retrieved.append(
                    RetrievedDoc(
                        text=hit["entity"].get("document", ""),
                        distance=distance,
                        metadata=meta,
                        id=hit["id"],
                    )
                )
            results.append(retrieved)

        return results

    def count(self) -> int:
        stats = self.client.get_collection_stats(self.collection_name)
        return int(stats.get("row_count", 0))


def create_vector_store(cfg: dict[str, Any]) -> VectorStore:
    """工厂：按配置创建向量库实例（注入方式落点）。

    cfg 来自 resources/config/vector_db.json：
      {"type": "chroma", "persist_path": "../resources/chroma_db", "collection": "FAQ_KNOWLEDGE_BASE"}
      {"type": "milvus", "uri": "http://localhost:19530", "collection": "FAQ_KNOWLEDGE_BASE", "dimension": 1024}
    换库只改配置 + 保证实现类存在，业务层零改动。
    """
    collection_name = cfg.get("collection") or COLLECTION_NAME

    if cfg.get("type") == "milvus":
        from init import embedding_function  # 延迟导入：init 会初始化模型等重资源

        return MilvusVectorStore(
            uri=cfg["uri"],
            collection_name=collection_name,
            embedding_function=embedding_function,
            dimension=cfg.get("dimension", 1024),
        )

    # 默认 chroma
    import chromadb
    from init import embedding_function

    client = chromadb.PersistentClient(path=str(cfg.get("persist_path", "../resources/chroma_db")))
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,  # 与 RAG 侧保持一致
        metadata={"hnsw:space": "cosine"},
    )
    return ChromaVectorStore(collection)
