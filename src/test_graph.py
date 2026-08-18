import asyncio
import json
import os

from pathlib import Path
from typing import Annotated, Optional, TypedDict, List, Dict, Any

import chromadb
from langchain_core.documents import Document
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.types import CachePolicy, Send
from psycopg_pool import ConnectionPool
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from chromadb import QueryResult
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, AIMessageChunk
from langgraph.constants import START, END
from langgraph.graph.message import MessagesState
from langgraph.graph.state import StateGraph, CompiledStateGraph
from loguru import logger
from pydantic import Field, BaseModel
from init import model, system_prompt, online_rerank, COLLECTION_NAME


"""
ChatService（类，收拢全部资源与业务方法）
│
├── __init__()                    # 只存配置（.env 的 DB URL），不建任何重资源
│
├── open()  ←→  close()           # 幂等；对应 FastAPI lifespan 的启动/关闭
│   ├── chroma client + collection          ← 原模块级单例收进来
│   ├── ConnectionPool + setup()            ← 原 build_chat_graph 内逻辑
│   ├── PostgresSaver / PostgresStore       ← 共用 pool
│   ├── InMemoryCache()                     ← 实例级，随 open/close 同生命周期
│   └── _build_rerank_graph() / 主图 compile（只 build 一次）
│
├── 业务方法（main.py 端点的全部逻辑收编）
│   ├── invoke(user_id, thread_id, input_str) -> str   # config 组装 + invoke + 取末条
│   ├── a_invoke(...)                                  # asyncio.to_thread(invoke)
│   ├── get_history(thread_id) / list_sessions(user_id)
│   ├── get_memory(user_id) / delete_session(thread_id)
│   └── 属性暴露: .graph / .pool / .collection（端点特殊场景兜底）
│
└── 模块级工厂 get_chat_service()   # 或直接由 lifespan new + open/close
"""

class ChatService:


    POSTGRESQL_DB_URL = os.getenv("POSTGRESQL_DB_URL")
    persist_path: str | Path
    db_url: str


    def __init__(self, persist_path="../recourses/chroma_db", db_url=None):
        self.persist_path = persist_path
        self.db_url = db_url or os.getenv("POSTGRESQL_DB_URL")
        # 资源占位，open() 里真正创建，close() 里释放
        self.client = self.collection = None
        self.pool = self.checkpointer = self.store = None
        self.cache = None
        self.graph = None            # 主对话图
        self.rerank_graph = None     # 改写+重排图

    def open(self):
        from init import embedding_function
        self.client = chromadb.PersistentClient(path=str(self.persist_path))
        self.collection = self.client.get_collection(
            name= COLLECTION_NAME,
            embedding_function=embedding_function,  # 与 RAG 侧保持一致
            )
        self.pool = ConnectionPool(
            conninfo=self.db_url,
            kwargs={"autocommit": True},
            min_size=1,
            max_size=10,
            timeout=5,  # 借连接 5 秒快速失败，不干等 30 秒
            open=True,
        )  # ← self.
        try:
            self.pool.check()
        except Exception:
            logger.error(...)
            raise
        self.checkpointer = PostgresSaver(self.pool)  # ← self.
        self.store = PostgresStore(self.pool)  # ← self.
        self.checkpointer.setup()
        self.store.setup()
        self.cache = InMemoryCache()  # ← self.，且 compile 用它
        self.rerank_graph = self._build_rerank_graph()  # 只 build 一次，替代 @lru_cache
        self.graph = self._build_chat_graph()

    def close(self, timeout:int =10):
        if self.pool:
            self.pool.close(timeout=timeout)

    def _build_rerank_graph(self):  # ← 原 query_rerank_graph 逻辑整体搬入
        collection = self.collection  # ← 关键：闭包捕获 self.collection，retrieve 节点内继续用 collection 变量

        logger.info("正在进行Query 改写 + 重排序")

        class RAGState(TypedDict):
            question: str
            history: List[Dict[str, str]]  # 多轮对话历史，只用于 query 改写
            rewritten_queries: List[str]  # 改写后的查询列表
            merged_docs: List[Document]  # 多查询召回 + RRF 融合后的候选
            reranked_docs: List[Document]  # 重排后的最终文档

        class QueryRewriteResult(BaseModel):
            main_query: str
            sub_queries: List[str] = Field(default_factory=list)
            keywords: List[str] = Field(default_factory=list)

        REWRITE_PROMPT = """你是查询改写专家。根据对话历史，将用户问题改写成适合向量检索的独立查询。

        要求：
        1. 解决指代，如“他/它/这个/那个”必须替换成明确实体；
        2. 多义词根据上下文补全限定词；
        3. 生成 1 个主查询 + 2~3 个子查询，覆盖不同语义角度；
        4. 再提取 3~5 个关键词。

        对话历史：
        {history}

        用户当前问题：
        {question}
        """

        def rewrite_query(state: RAGState) -> dict:
            history_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in state.get("history", [])
            )

            prompt = REWRITE_PROMPT.format(
                question=state["question"],
                history=history_text or "无",
            )

            result = model.with_structured_output(QueryRewriteResult).invoke(prompt)

            queries = [result.main_query] + result.sub_queries
            logger.info(f"重新后问题:{queries}")
            return {"rewritten_queries": queries}

        from concurrent.futures import ThreadPoolExecutor

        def rrf_fusion(results: List[List[Document]], k: int = 60) -> List[Document]:
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

        def unpack_query_results(results: List[Any]) -> List[List[Document]]:
            """
            将 List[QueryResult] 拆解为 List[List[Document]]，
            每个内层列表对应一个查询的文档列表。
            """
            all_query_docs = []

            for result in results:
                # 防御：某些版本/配置下 ChromaDB 会再包一层 list，先解包
                while isinstance(result, list) and len(result) == 1:
                    result = result[0]

                if not isinstance(result, dict):
                    logger.warning(
                        f"Unexpected query result type: {type(result)}, value: {result[:200] if isinstance(result, (list, str)) else result}")
                    all_query_docs.append([])
                    continue

                ids = result.get("ids", [])
                documents = result.get("documents", [])
                metadatas = result.get("metadatas", [])

                # ChromaDB 返回的是 batch 嵌套结构：List[List[...]]
                if ids and isinstance(ids[0], list):
                    for sub_ids, sub_docs, sub_metas in zip(ids, documents, metadatas):
                        query_docs = []
                        for i, doc_id in enumerate(sub_ids):
                            meta = dict(sub_metas[i]) if i < len(sub_metas) else {}
                            meta["id"] = doc_id
                            text = sub_docs[i] if i < len(sub_docs) else ""
                            query_docs.append(Document(page_content=text, metadata=meta))
                        all_query_docs.append(query_docs)
                else:
                    # 兜底：一维结构
                    query_docs = []
                    for i, doc_id in enumerate(ids):
                        meta = dict(metadatas[i]) if i < len(metadatas) else {}
                        meta["id"] = doc_id
                        text = documents[i] if i < len(documents) else ""
                        query_docs.append(Document(page_content=text, metadata=meta))
                    all_query_docs.append(query_docs)

            return all_query_docs

        def retrieve(state: RAGState) -> dict:
            queries = state["rewritten_queries"]
            top_k = 8

            with ThreadPoolExecutor(max_workers=len(queries)) as ex:
                results = list(
                    ex.map(
                        lambda q: collection.query(query_texts=[q], n_results=top_k),
                        queries,
                    )
                )

            docs_per_query = unpack_query_results(results)
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

        builder = StateGraph(RAGState)

        builder.add_node("rewrite", rewrite_query)
        builder.add_node("retrieve", retrieve)
        builder.add_node("rerank", rerank)

        builder.add_edge(START, "rewrite")
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("retrieve", "rerank")
        builder.add_edge("rerank", END)

        rerank_graph = builder.compile()
        return rerank_graph

    def _build_chat_graph(self):  # ← 原 build_chat_graph 逻辑整体搬入
        rerank_graph = self.rerank_graph

        class OverAllState(MessagesState):
            input_str: Annotated[str, Field(description="用户输入")]
            retrieve_res: Annotated[Optional[QueryResult], "检索结果"] = None
            needs_retrieval: Annotated[bool, Field(description="是否需要检索知识库")] = False

        def retrieve_node(state: OverAllState) -> OverAllState:
            input_str = state["input_str"]
            logger.info(f"执行知识库检索：{input_str}")

            history = [
                {"role": "user" if m.type == "human" else "assistant", "content": m.content}
                for m in state.get("messages", [])
                if m.type in ("human", "ai")
            ]

            retrieve_res = rerank_graph.invoke({
                "question": input_str,
                "history": history
            }
            )

            return {
                "retrieve_res": retrieve_res
            }

        def llm_node(state: OverAllState, config: RunnableConfig, store: BaseStore) -> OverAllState:
            input_str = state["input_str"]
            retrieval_res = state.get("retrieve_res")

            if retrieval_res is not None:
                # 检索分支：在线重排结果已按相关性降序，直接取文档文本
                docs = [d.page_content for d in retrieval_res.get("reranked_docs", [])]
                if docs:
                    context = "\n\n".join(f"[文档 {i + 1}] {doc}" for i, doc in enumerate(docs[:5]))
                else:
                    context = "（知识库中未检索到相关内容）"
                user_content = (
                    f"请严格依据下面检索到的资料回答用户问题，资料中没有的内容不要编造。\n\n"
                    f"【检索资料】\n{context}\n\n"
                    f"【用户问题】\n{input_str}"
                )
            else:
                # 无需检索分支：直接回答
                user_content = input_str

            # 长期记忆：从 store 读取该用户的档案（跨会话保存）
            user_id = config["configurable"].get("user_id", "default")
            long_term = ""
            item = store.get(("rag_chat", user_id), "user_profile")
            if item and item.value.get("profile"):
                long_term = item.value["profile"]

            # 短期记忆：checkpointer 按 thread_id 恢复的历史对话
            history = state.get("messages", [])

            # 组装消息：系统提示（含长期记忆）+ 历史对话 + 检索资料与当前问题
            system_content = system_prompt
            if long_term:
                system_content += f"\n\n【用户长期记忆】\n{long_term}"

            messages = [SystemMessage(content=system_content)] + list(history)
            messages.append(HumanMessage(content=user_content))

            # 流式生成：LangGraph 会通过 callback 机制自动捕获 model.stream 的每个 token，
            # 以 stream_mode="messages" 输出（前端逐片累加即打字机效果）。
            # 注意：节点不能返回生成器——langgraph 1.x 会把生成器当单条消息交给
            # add_messages/_convert_to_message 转换，报 "Unsupported message type: generator"。
            chunks = []
            for chunk in model.stream(messages):
                chunks.append(chunk)
            ai_reply = AIMessage(content="".join(c.content for c in chunks if isinstance(c.content, str)))

            return {"messages": [HumanMessage(content=input_str), ai_reply]}

        MEMORY_EXTRACT_PROMPT = """你是长期记忆管理器，负责维护用户档案。

        根据【本轮对话】，更新【已有档案】：
        1. 只记录长期有效的事实（如姓名、职业、身份、偏好、习惯、目标等），忽略一次性请求与寒暄。
        2. 若本轮没有值得记录的新信息，只输出一个词：无。
        3. 有新信息则输出合并后的完整档案，直接输出文本，不要任何解释或 JSON。

        【已有档案】
        {old_profile}

        【本轮对话】
        用户：{input_str}
        助手：{llm_output}"""

        def memory_node(state: OverAllState, config: RunnableConfig, store: BaseStore) -> None:
            """将本轮对话中的长期信息提取并写入 store（按用户隔离）。"""
            user_id = config["configurable"].get("user_id", "default")
            namespace = ("rag_chat", user_id)

            # memory_node 中，把写入条件从"非空"改为"非占位符且内容有变化"
            NO_INFO_MARKS = {"（无）", "无", "无新信息", "暂无", "无新增信息"}

            # 读取已有档案
            item = store.get(namespace, "user_profile")
            old_profile = item.value["profile"] if item else "（暂无档案）"

            # 用 LLM 提取/合并长期记忆（AI 回答已由 add_messages 合并为完整消息）
            ai_reply = state["messages"][-1].content
            response = model.invoke(
                [HumanMessage(content=MEMORY_EXTRACT_PROMPT.format(
                    old_profile=old_profile,
                    input_str=state["input_str"],
                    llm_output=ai_reply,
                ))]
            )
            new_profile = response.content.strip()
            if new_profile and new_profile not in NO_INFO_MARKS and new_profile != old_profile:
                store.put(namespace, "user_profile", {"profile": new_profile})
                logger.info(f"长期记忆已更新（user_id={user_id}）：{new_profile[:100]}")

        CLASSIFIER_PROMPT = """你是问答路由，判断用户问题是否需要检索"在线学习平台"知识库。

        需要检索：涉及平台业务的具体问题，如账号登录、密码重置、课程购买、作业提交、学习记录、费用等。
        不需要检索：寒暄问候、自我介绍、闲聊、与平台无关的常识问题，或仅凭已有对话即可回答的问题。

        只输出一个词：yes 或 no。"""

        def classify_node(state: OverAllState) -> OverAllState:
            """判断本轮问题是否需要知识库检索（仅在需要时走 retrieval_node）。"""
            response = model.invoke(
                [
                    SystemMessage(content=CLASSIFIER_PROMPT),
                    HumanMessage(content=state["input_str"]),
                ]
            )
            needs_retrieval = response.content.strip().lower().startswith("yes")
            logger.info(f"分类结果（needs_retrieval={needs_retrieval}）：{state['input_str'][:50]}")
            return {"needs_retrieval": needs_retrieval}

        def route(state: OverAllState) -> list[Send]:
            """条件路由：需要检索才 Send 到 retrieval_node，否则直接 Send 到 llm_node。

            Send 任务不会继承父 state，必须把节点所需的数据显式放进 payload。
            """
            payload = {
                "input_str": state["input_str"],
                "messages": state.get("messages", []),  # 历史对话（短期记忆）
            }
            if state.get("needs_retrieval"):
                return [Send("retrieve_node", payload)]
            return [Send("llm_node", payload)]

        builder = StateGraph(state_schema=OverAllState)
        builder.add_node("classify_node", classify_node)
        builder.add_node("retrieve_node", retrieve_node, cache_policy=CachePolicy(ttl=10))
        builder.add_node("llm_node", llm_node)
        builder.add_node("memory_node", memory_node)

        builder.add_edge(START, "classify_node")
        builder.add_conditional_edges(
            "classify_node",
            route,
            ["retrieve_node", "llm_node"],
        )
        builder.add_edge("retrieve_node", "llm_node")
        builder.add_edge("llm_node", "memory_node")
        builder.add_edge("memory_node", END)

        # 创建连接池（open=True 表示立即打开连接）
        # 必须开启 autocommit：迁移脚本含 CREATE INDEX CONCURRENTLY，不能在事务块中执行
        pool = self.pool
        try:
            pool.check()
        except Exception as e:
            logger.error(f"数据库连接失败，请检查 .env 的 POSTGRESQL_DB_URL 与 PostgreSQL 服务")
            logger.error(f"真实错误：{e}")
            raise

        # 直接实例化 PostgresSaver（短期记忆：按 thread_id 恢复历史对话）
        checkpointer = self.checkpointer

        # PostgresStore（长期记忆：跨会话保存用户档案），与 checkpointer 共用连接池
        store = self.store

        checkpointer.setup()
        store.setup()

        graph = builder.compile(checkpointer=checkpointer, store=store, cache=self.cache)

        return graph

    def invoke(self, user_id, thread_id, query) -> str:
        config = {
            "configurable": {"thread_id": thread_id, "user_id": user_id},
            "metadata": {"user_id": user_id},  # 随 checkpoint 写入 metadata
        }
        result = self.graph.invoke({"input_str": query}, config=config)
        ai_msg = result["messages"][-1]
        return ai_msg.content

    async def a_invoke(self, user_id, thread_id, input_str) -> str:
        """异步版 invoke：同步调用丢进线程池，不阻塞事件循环。"""
        return await asyncio.to_thread(self.invoke, user_id, thread_id, input_str)

    def stream(self, user_id, thread_id, input_str):

        def make_serializable(obj):
            # 处理 LangChain 消息对象
            if isinstance(obj, BaseMessage):
                return {
                    "type": obj.type,
                    "content": obj.content,
                }
            # 递归处理字典
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            # 递归处理列表/元组
            if isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            # 其他类型直接返回
            return obj

        config = {
            "configurable": {"thread_id": thread_id, "user_id": user_id},
            "metadata": {"user_id": user_id},  # 随 checkpoint 写入 metadata
        }  # 会话隔离

        # stream_mode="messages" 会捕获图中所有 LLM 调用的 token 事件，
        # 包括 classify_node 的 yes/no 与 memory_node 的记忆提取输出，
        # 必须按 meta["langgraph_node"] 过滤，只输出 llm_node 的增量，
        # 否则分类器的 "no" 会混入流式回答出现在前端。
        for chunk, meta in self.graph.stream(
                {"input_str": input_str},
                config=config,
                stream_mode="messages",
        ):
            if meta.get("langgraph_node") != "llm_node":
                continue
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield f"data: {json.dumps({'content': chunk.content})}\n\n"

    def get_history_session(self, thread_id: str):
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        snapshot = self.graph.get_state(config)
        if not snapshot or len(snapshot) == 0:
            logger.error(f"会话:{thread_id}记录不存在")
            return {"code": 404, "message": f"会话:{thread_id}记录不存在"}
        state_data = snapshot.values
        history_messages = state_data.get("messages", [])
        if not history_messages:
            logger.error(f"会话:{thread_id}记录不存在")
            return []
        history_session = []
        for message in history_messages:
            history_session.append(
                {
                    "role": f"{message.type}",
                    "content": message.content
                }
            )
        return history_session

    def get_memory(self, user_id: str):
        store = self.store
        memory = ""
        item = store.get(("rag_chat", user_id), "user_profile")
        if item and item.value.get("profile"):
            memory = item.value["profile"]
        return memory

    def delete_session_by_id(self, thread_id: str):
        checkpointer = self.checkpointer
        history_msg = self.get_history_session(thread_id)
        if not history_msg:
            logger.error(f"会话:{thread_id}记录不存在")
            return f"会话:{thread_id}记录不存在"
        checkpointer.delete_thread(thread_id)
        logger.info(f"删除会话:{thread_id}成功")
        return {
            "title": f"删除会话:{thread_id}成功",
        }

    def get_user_sessions(self, user_id: str):
        checkpointer = self.checkpointer

        from langgraph.checkpoint.base import CheckpointTuple

        latest_by_thread: dict[str, CheckpointTuple] = {}

        for item in checkpointer.list(None):
            tid = item.config["configurable"]["thread_id"]
            checkpoint_data = item.checkpoint
            checkpoint_user_id = (
                    checkpoint_data.get("metadata", {}).get("user_id") or
                    item.config["configurable"].get("user_id") or
                    item.metadata.get("user_id")  # CheckpointTuple 可能有 metadata
            )
            if checkpoint_user_id != user_id:
                continue
            if tid not in latest_by_thread:
                latest_by_thread[tid] = item

        sessions = []
        for tid, item in latest_by_thread.items():
            messages = item.checkpoint["channel_values"].get("messages", [])
            first_user = next((m for m in messages if m.type == "human"), None)
            sessions.append({
                "thread_id": tid,
                "title": first_user.content[:20] if first_user else "新会话",
                "last_updated": item.checkpoint["ts"],
            })
        sessions.sort(key=lambda s: s["last_updated"], reverse=True)
        return sessions

    def check_db_health(self):
        import psycopg
        from psycopg_pool import PoolTimeout
        pool = self.pool
        try:
            with pool.connection() as conn:  # 从池中借连接（空闲不足会抛 PoolTimeout）
                conn.execute("SELECT 1")  # 真正发一条查询验证链路
            return {"status": "ok", "db": True}
        except PoolTimeout:
            logger.warning("数据库连接池已满或无法建立连接")
            return {"status": "degraded", "db": False}
        except psycopg.OperationalError as e:  # psycopg3 的异常就在 psycopg 顶层
            logger.error(f"数据库不可用: {e}")
            return {"status": "degraded", "db": False}


# @lru_cache(maxsize=1)
# def query_rerank_graph():
#     logger.info("正在进行Query 改写 + 重排序")
#     class RAGState(TypedDict):
#         question: str
#         history: List[Dict[str, str]]  # 多轮对话历史，只用于 query 改写
#         rewritten_queries: List[str]  # 改写后的查询列表
#         merged_docs: List[Document]  # 多查询召回 + RRF 融合后的候选
#         reranked_docs: List[Document]  # 重排后的最终文档
#
#     class QueryRewriteResult(BaseModel):
#         main_query: str
#         sub_queries: List[str] = Field(default_factory=list)
#         keywords: List[str] = Field(default_factory=list)
#
#     REWRITE_PROMPT = """你是查询改写专家。根据对话历史，将用户问题改写成适合向量检索的独立查询。
#
#     要求：
#     1. 解决指代，如“他/它/这个/那个”必须替换成明确实体；
#     2. 多义词根据上下文补全限定词；
#     3. 生成 1 个主查询 + 2~3 个子查询，覆盖不同语义角度；
#     4. 再提取 3~5 个关键词。
#
#     对话历史：
#     {history}
#
#     用户当前问题：
#     {question}
#     """
#
#     def rewrite_query(state: RAGState) -> dict:
#         history_text = "\n".join(
#             f"{m['role']}: {m['content']}" for m in state.get("history", [])
#         )
#
#         prompt = REWRITE_PROMPT.format(
#             question=state["question"],
#             history=history_text or "无",
#         )
#
#         result = model.with_structured_output(QueryRewriteResult).invoke(prompt)
#
#         queries = [result.main_query] + result.sub_queries
#         logger.info(f"重新后问题:{queries}")
#         return {"rewritten_queries": queries}
#
#     from concurrent.futures import ThreadPoolExecutor
#
#     def rrf_fusion(results: List[List[Document]], k: int = 60) -> List[Document]:
#         scores = {}
#
#         for docs in results:
#             for rank, doc in enumerate(docs):
#                 key = doc.metadata.get("id", doc.page_content)
#                 if key not in scores:
#                     scores[key] = {"doc": doc, "score": 0.0}
#                 scores[key]["score"] += 1.0 / (k + rank + 1)
#
#         return [
#             item["doc"]
#             for item in sorted(
#                 scores.values(),
#                 key=lambda x: x["score"],
#                 reverse=True,
#             )
#         ]
#
#
#     def unpack_query_results(results: List[Any]) -> List[List[Document]]:
#         """
#         将 List[QueryResult] 拆解为 List[List[Document]]，
#         每个内层列表对应一个查询的文档列表。
#         """
#         all_query_docs = []
#
#         for result in results:
#             # 防御：某些版本/配置下 ChromaDB 会再包一层 list，先解包
#             while isinstance(result, list) and len(result) == 1:
#                 result = result[0]
#
#             if not isinstance(result, dict):
#                 logger.warning(f"Unexpected query result type: {type(result)}, value: {result[:200] if isinstance(result, (list, str)) else result}")
#                 all_query_docs.append([])
#                 continue
#
#             ids = result.get("ids", [])
#             documents = result.get("documents", [])
#             metadatas = result.get("metadatas", [])
#
#             # ChromaDB 返回的是 batch 嵌套结构：List[List[...]]
#             if ids and isinstance(ids[0], list):
#                 for sub_ids, sub_docs, sub_metas in zip(ids, documents, metadatas):
#                     query_docs = []
#                     for i, doc_id in enumerate(sub_ids):
#                         meta = dict(sub_metas[i]) if i < len(sub_metas) else {}
#                         meta["id"] = doc_id
#                         text = sub_docs[i] if i < len(sub_docs) else ""
#                         query_docs.append(Document(page_content=text, metadata=meta))
#                     all_query_docs.append(query_docs)
#             else:
#                 # 兜底：一维结构
#                 query_docs = []
#                 for i, doc_id in enumerate(ids):
#                     meta = dict(metadatas[i]) if i < len(metadatas) else {}
#                     meta["id"] = doc_id
#                     text = documents[i] if i < len(documents) else ""
#                     query_docs.append(Document(page_content=text, metadata=meta))
#                 all_query_docs.append(query_docs)
#
#         return all_query_docs
#
#     def retrieve(state: RAGState) -> dict:
#         queries = state["rewritten_queries"]
#         top_k = 8
#
#         with ThreadPoolExecutor(max_workers=len(queries)) as ex:
#             results = list(
#                 ex.map(
#                     lambda q: collection.query(query_texts=[q], n_results=top_k),
#                     queries,
#                 )
#             )
#
#         docs_per_query = unpack_query_results(results)
#         merged_docs = rrf_fusion(docs_per_query)
#         return {"merged_docs": merged_docs}
#
#     def rerank(state: RAGState) -> dict:
#         docs = state["merged_docs"]
#         if not docs:
#             return {"reranked_docs": []}
#
#         # 用改写后的主查询做精排，通常比原始口语问题更稳定
#         query = state["rewritten_queries"][0]
#
#         results = online_rerank(query, [doc.page_content for doc in docs], top_n=10)
#         top_docs = [docs[r["index"]] for r in results]
#
#         return {"reranked_docs": top_docs}
#
#     builder = StateGraph(RAGState)
#
#     builder.add_node("rewrite", rewrite_query)
#     builder.add_node("retrieve", retrieve)
#     builder.add_node("rerank", rerank)
#
#     builder.add_edge(START, "rewrite")
#     builder.add_edge("rewrite", "retrieve")
#     builder.add_edge("retrieve", "rerank")
#     builder.add_edge("rerank",  END)
#
#     rerank_graph = builder.compile()
#     return rerank_graph
#
# def build_chat_graph() -> tuple[CompiledStateGraph, ConnectionPool]:
#
#     class OverAllState(MessagesState):
#         input_str: Annotated[str, Field(description="用户输入")]
#         retrieve_res: Annotated[Optional[QueryResult], "检索结果"] = None
#         llm_output: Annotated[str, Field(description="模型输出")]
#         needs_retrieval: Annotated[bool, Field(description="是否需要检索知识库")] = False
#
#     def retrieve_node(state: OverAllState) -> OverAllState:
#         input_str = state["input_str"]
#         logger.info(f"执行知识库检索：{input_str}")
#
#         rerank_graph = query_rerank_graph()
#
#         history = [
#             {"role": "user" if m.type == "human" else "assistant", "content": m.content}
#             for m in state.get("messages", [])
#             if m.type in ("human", "ai")
#         ]
#
#         retrieve_res = rerank_graph.invoke({
#             "question": input_str,
#             "histroy": history
#             }
#         )
#
#         return {
#             "retrieve_res": retrieve_res
#         }
#
#     def llm_node(state: OverAllState, config: RunnableConfig, store: BaseStore) -> OverAllState:
#         input_str = state["input_str"]
#         retrieval_res = state.get("retrieve_res")
#
#         if retrieval_res is not None:
#             # 检索分支：在线重排结果已按相关性降序，直接取文档文本
#             docs = [d.page_content for d in retrieval_res.get("reranked_docs", [])]
#             if docs:
#                 context = "\n\n".join(f"[文档 {i + 1}] {doc}" for i, doc in enumerate(docs[:5]))
#             else:
#                 context = "（知识库中未检索到相关内容）"
#             user_content = (
#                 f"请严格依据下面检索到的资料回答用户问题，资料中没有的内容不要编造。\n\n"
#                 f"【检索资料】\n{context}\n\n"
#                 f"【用户问题】\n{input_str}"
#             )
#         else:
#             # 无需检索分支：直接回答
#             user_content = input_str
#
#         # 长期记忆：从 store 读取该用户的档案（跨会话保存）
#         user_id = config["configurable"].get("user_id", "default")
#         long_term = ""
#         item = store.get(("rag_chat", user_id), "user_profile")
#         if item and item.value.get("profile"):
#             long_term = item.value["profile"]
#
#         # 短期记忆：checkpointer 按 thread_id 恢复的历史对话
#         history = state.get("messages", [])
#
#         # 组装消息：系统提示（含长期记忆）+ 历史对话 + 检索资料与当前问题
#         system_content = system_prompt
#         if long_term:
#             system_content += f"\n\n【用户长期记忆】\n{long_term}"
#
#         messages = [SystemMessage(content=system_content)] + list(history)
#         messages.append(HumanMessage(content=user_content))
#
#         response = model.invoke(messages)
#         llm_output = response.content
#
#         return {
#             "llm_output": llm_output,
#             "messages": [HumanMessage(content=input_str), AIMessage(content=llm_output)],
#         }
#
#     MEMORY_EXTRACT_PROMPT = """你是长期记忆管理器，负责维护用户档案。
#
#     根据【本轮对话】，更新【已有档案】：
#     1. 只记录长期有效的事实（如姓名、职业、身份、偏好、习惯、目标等），忽略一次性请求与寒暄。
#     2. 若本轮没有值得记录的新信息，只输出一个词：无。
#     3. 有新信息则输出合并后的完整档案，直接输出文本，不要任何解释或 JSON。
#
#     【已有档案】
#     {old_profile}
#
#     【本轮对话】
#     用户：{input_str}
#     助手：{llm_output}"""
#
#
#     def memory_node(state: OverAllState, config: RunnableConfig, store: BaseStore) -> None:
#         """将本轮对话中的长期信息提取并写入 store（按用户隔离）。"""
#         user_id = config["configurable"].get("user_id", "default")
#         namespace = ("rag_chat", user_id)
#
#         # memory_node 中，把写入条件从"非空"改为"非占位符且内容有变化"
#         NO_INFO_MARKS = {"（无）", "无", "无新信息", "暂无", "无新增信息"}
#
#         # 读取已有档案
#         item = store.get(namespace, "user_profile")
#         old_profile = item.value["profile"] if item else "（暂无档案）"
#
#         # 用 LLM 提取/合并长期记忆
#         response = model.invoke(
#             [HumanMessage(content=MEMORY_EXTRACT_PROMPT.format(
#                 old_profile=old_profile,
#                 input_str=state["input_str"],
#                 llm_output=state["llm_output"],
#             ))]
#         )
#         new_profile = response.content.strip()
#         if new_profile and new_profile not in NO_INFO_MARKS and new_profile != old_profile:
#             store.put(namespace, "user_profile", {"profile": new_profile})
#             logger.info(f"长期记忆已更新（user_id={user_id}）：{new_profile[:100]}")
#
#
#     CLASSIFIER_PROMPT = """你是问答路由，判断用户问题是否需要检索"在线学习平台"知识库。
#
#     需要检索：涉及平台业务的具体问题，如账号登录、密码重置、课程购买、作业提交、学习记录、费用等。
#     不需要检索：寒暄问候、自我介绍、闲聊、与平台无关的常识问题，或仅凭已有对话即可回答的问题。
#
#     只输出一个词：yes 或 no。"""
#
#
#     def classify_node(state: OverAllState) -> OverAllState:
#         """判断本轮问题是否需要知识库检索（仅在需要时走 retrieval_node）。"""
#         response = model.invoke(
#             [
#                 SystemMessage(content=CLASSIFIER_PROMPT),
#                 HumanMessage(content=state["input_str"]),
#             ]
#         )
#         needs_retrieval = response.content.strip().lower().startswith("yes")
#         logger.info(f"分类结果（needs_retrieval={needs_retrieval}）：{state['input_str'][:50]}")
#         return {"needs_retrieval": needs_retrieval}
#
#
#     def route(state: OverAllState) -> list[Send]:
#         """条件路由：需要检索才 Send 到 retrieval_node，否则直接 Send 到 llm_node。
#
#         Send 任务不会继承父 state，必须把节点所需的数据显式放进 payload。
#         """
#         payload = {
#             "input_str": state["input_str"],
#             "messages": state.get("messages", []),  # 历史对话（短期记忆）
#         }
#         if state.get("needs_retrieval"):
#             return [Send("retrieve_node", payload)]
#         return [Send("llm_node", payload)]
#
#     builder = StateGraph(state_schema=OverAllState)
#     builder.add_node("classify_node", classify_node)
#     builder.add_node("retrieve_node", retrieve_node, cache_policy=CachePolicy(ttl=10))
#     builder.add_node("llm_node", llm_node)
#     builder.add_node("memory_node", memory_node)
#
#     builder.add_edge(START, "classify_node")
#     builder.add_conditional_edges(
#         "classify_node",
#         route,
#         ["retrieve_node", "llm_node"],
#     )
#     builder.add_edge("retrieve_node", "llm_node")
#     builder.add_edge("llm_node", "memory_node")
#     builder.add_edge("memory_node", END)
#
#     # 创建连接池（open=True 表示立即打开连接）
#     # 必须开启 autocommit：迁移脚本含 CREATE INDEX CONCURRENTLY，不能在事务块中执行
#     pool = ConnectionPool(
#         conninfo=POSTGRESQL_DB_URL,
#         kwargs={"autocommit": True},
#         min_size=1,
#         max_size=10,
#         timeout=5,  # 借连接 5 秒快速失败，不干等 30 秒
#         open=True,
#     )
#     try:
#         pool.check()
#     except Exception as e:
#         logger.error(f"数据库连接失败，请检查 .env 的 POSTGRESQL_DB_URL 与 PostgreSQL 服务")
#         logger.error(f"真实错误：{e}")
#         raise
#
#     # 直接实例化 PostgresSaver（短期记忆：按 thread_id 恢复历史对话）
#     checkpointer = PostgresSaver(pool)
#
#     # PostgresStore（长期记忆：跨会话保存用户档案），与 checkpointer 共用连接池
#     store = PostgresStore(pool)
#
#     checkpointer.setup()
#     store.setup()
#
#     graph = builder.compile(checkpointer=checkpointer, store=store, cache=InMemoryCache())
#
#     return graph, pool



