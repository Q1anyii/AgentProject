import json
from typing import TYPE_CHECKING, TypedDict, List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig
from langgraph.constants import START, END
from langgraph.graph.state import StateGraph
from langgraph.types import Send
from loguru import logger
from pydantic import Field, BaseModel, ConfigDict
from service.cache_service import cache_service as _cache_service
from utils.doc_util import unpack_query_results, documents_to_dicts
import json, re
if TYPE_CHECKING:
    # 仅用于类型注解，运行时导入会与 chat_service 形成循环依赖
    from service.chat_service import ChatService
from init import model, online_rerank
from constant.retrieval_constants import TOP_K, DISTANCE_THRESHOLD, REWRITE_PROMPT, RRF_K


def build_retrieve_graph(self: "ChatService"):  # ← 原 query_rerank_graph 逻辑整体搬入
    collection = self.collection  # ← 关键：闭包捕获 self.collection，retrieve 节点内继续用 collection 变量
    # Redis 检索缓存：全局单例（main.py lifespan 统一 open/close），节点内直接使用，不自行管理生命周期
    cache_service = _cache_service

    class OutputState(TypedDict):
        output: List[Document]


    class RAGState(TypedDict):
        question: str
        history: List[Dict[str, str]]  # 多轮对话历史，只用于 query 改写
        rewritten_queries: List[str]  # 改写后的查询列表
        merged_docs: List[Document]  # 多查询召回 + RRF 融合后的候选
        reranked_docs: List[Document]  # 重排后的最终文档
        cache_hit: Optional[bool]

    class QueryRewriteResult(BaseModel):
        model_config = ConfigDict(populate_by_name=True)  # 关键配置

        main_query: str = Field(..., alias="主查询")
        sub_queries: List[str] = Field(default_factory=list, alias="子查询")
        keywords: List[str] = Field(default_factory=list, alias="关键词")

    def extract_json(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end + 1])
            raise

    def check_cache(state: RAGState, config: RunnableConfig) -> dict:
        question = state["question"]
        thread_id = config["configurable"].get("thread_id", None)
        query_in_cache = cache_service.query_cache(thread_id, question, 3)
        if query_in_cache:
            logger.info("缓存命中，直接返回")
            return {"reranked_docs": query_in_cache, "cache_hit": True}
        return {"cache_hit": False}

    def store_cache(state: RAGState, config: RunnableConfig) -> dict:
        thread_id = config["configurable"].get("thread_id", None)
        if not state.get("cache_hit"):
            # ttl 走 CacheService.store_cache 默认值（15s）；不要对 user_id 调 redis TTL——
            # TTL 只能查已存在 key 的剩余时间，user_id 不是 key，返回 -2 会导致 expire 异常
            cache_service.store_cache(thread_id, state["question"], state["reranked_docs"])
        return {}

    def rewrite_query(state: RAGState) -> dict:
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in state.get("history", [])
        )

        prompt = REWRITE_PROMPT.format(
            question=state["question"],
            history=history_text or "无",
        )
        logger.info("正在进行Query 改写 + 重排序")

        resp = model.invoke(prompt, response_format={"type": "json_object"})
        raw_json = extract_json(resp.content)
        result = QueryRewriteResult(**raw_json)

        queries = [result.main_query] + result.sub_queries
        logger.info(f"重写后问题:{queries}")
        return {"rewritten_queries": queries}

    from concurrent.futures import ThreadPoolExecutor

    def rrf_fusion(results: List[List[Document]], k: int = RRF_K) -> List[Document]:
        scores = {}

        for docs in results:
            for rank, doc in enumerate(docs):
                key = doc.metadata.get("id", doc.page_content)
                if key not in scores:
                    scores[key] = {"doc": doc, "score": 0.0}
                scores[key]["score"] += 1.0 / (k + rank + 1)

        return [
            item["doc"]
            for item in sorted(
                scores.values(),
                key=lambda x: x["score"],
                reverse=True,
            )
        ]

    def retrieve(state: RAGState) -> dict:
        # TOP_K 和 DISTANCE_THRESHOLD 已移至 constant/retrieval_constants.py 统一管理
        queries = state["rewritten_queries"]

        with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as ex:
            raw_chroma_results = list(
                ex.map(
                    lambda q: collection.query(query_texts=[q], n_results=TOP_K),
                    queries,
                )
            )

            # 对每一个query的检索结果，先执行distance阈值过滤
            filtered_results: List[Dict[str, Any]] = []
            for res in raw_chroma_results:
                # chroma collection.query 返回格式：{documents:[[...]], distances:[[...]], metadatas:[[...]], ids:[[...]]}
                docs_list = res["documents"][0]
                dists_list = res["distances"][0]
                meta_list = res["metadatas"][0]
                id_list = res["ids"][0]

                keep_docs = []
                keep_dists = []
                keep_metas = []
                keep_ids = []

                for doc_text, dist, meta, doc_id in zip(docs_list, dists_list, meta_list, id_list):
                    if dist < DISTANCE_THRESHOLD:
                        meta["_distance"] = dist  # 埋入元数据，用于langsmith调试看距离
                        keep_docs.append(doc_text)
                        keep_dists.append(dist)
                        keep_metas.append(meta)
                        keep_ids.append(doc_id)

                # 把过滤之后的结果重新组装成chroma相同结构，给unpack_query_results复用
                filtered_results.append({
                    "documents": [keep_docs],
                    "distances": [keep_dists],
                    "metadatas": [keep_metas],
                    "ids": [keep_ids]
                })

            # 现在传给unpack的已经是过滤噪声之后的数据
            docs_per_query = unpack_query_results(filtered_results)
            merged_docs = rrf_fusion(docs_per_query)

            return {"merged_docs": merged_docs}

    def rerank(state: RAGState) -> dict:
        docs = state["merged_docs"]
        if not docs:
            return {"reranked_docs": []}

        # 用改写后的主查询做精排，通常比原始口语问题更稳定
        query = state["rewritten_queries"][0]

        results = online_rerank(query, [doc.page_content for doc in docs], top_n=10)
        top_docs = [docs[r["index"]] for r in results]

        return {"reranked_docs": top_docs}

    def output_node(state: RAGState) -> dict:
        return {"output": state["reranked_docs"]}

    builder = StateGraph(state_schema=RAGState, output_schema=OutputState)

    builder.add_node("check_cache", check_cache)
    builder.add_node("store_cache", store_cache)
    builder.add_node("rewrite", rewrite_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("rerank", rerank)
    builder.add_node("output_node", output_node)

    builder.add_edge(START, "check_cache")
    builder.add_conditional_edges(
        "check_cache",
        lambda state: "hit" if state.get("cache_hit") else "miss",
        {
            "hit": "output_node",
            "miss": "rewrite",
        },
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "store_cache")
    builder.add_edge("rerank", "output_node")
    builder.add_edge("output_node", END)

    rerank_graph = builder.compile()
    return rerank_graph