import asyncio
import json
import os
import chromadb

from pathlib import Path
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg_pool import ConnectionPool
from langchain_core.messages import BaseMessage, AIMessageChunk
from loguru import logger
from graphs.main_graph import build_main_graph
from graphs.rerank_graph import build_rerank_graph
from init import COLLECTION_NAME


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
        self.main_graph = None            # 主对话图
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
            logger.success("PostgreSQL连接池初始化成功")
        except Exception as e:
            logger.error(f"PostgreSQL数据库连接失败：{e}")
            raise
        self.checkpointer = PostgresSaver(self.pool)  # ← self.
        self.store = PostgresStore(self.pool)  # ← self.
        self.checkpointer.setup()
        self.store.setup()
        self.cache = InMemoryCache()  # ← self.，且 compile 用它
        self.rerank_graph = build_rerank_graph(self)  # 只 build 一次，替代 @lru_cache
        self.main_graph = build_main_graph(self)

    def close(self, timeout:int =10):
        if self.pool:
            self.pool.close(timeout=timeout)
            logger.info("PostgreSQL连接池已关闭")


    def invoke(self, user_id, thread_id, query) -> str:
        config = {
            "configurable": {"thread_id": thread_id, "user_id": user_id},
            "metadata": {"user_id": user_id},  # 随 checkpoint 写入 metadata
        }
        result = self.main_graph.invoke({"input_str": query}, config=config)
        ai_msg = result["messages"][-1]
        return ai_msg.content

    async def a_invoke(self, user_id, thread_id, input_str) -> str:
        """异步版 invoke：同步调用丢进线程池，不阻塞事件循环。"""
        return await asyncio.to_thread(self.invoke, user_id, thread_id, input_str)

    def stream(self, user_id, thread_id, input_str, user_info=None):

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
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                # 请求级用户上下文随 config 传入图（工具通过 RunnableConfig 参数读取），
                # 不依赖 contextvars：StreamingResponse 每次 next() 都在新线程/新 context 执行，
                # contextvars 的 set/reset 会跨 context 报错且 get() 拿不到值
                "user_info": user_info,
            },
            "metadata": {"user_id": user_id},  # 随 checkpoint 写入 metadata
        }  # 会话隔离

        try:
            # stream_mode="messages" 会捕获图中所有 LLM 调用的 token 事件，
            # 包括 classify_node 的 yes/no 与 memory_node 的记忆提取输出，
            # 必须按 meta["langgraph_node"] 过滤，只输出 llm_node 的增量，
            # 否则分类器的 "no" 会混入流式回答出现在前端。
            for chunk, meta in self.main_graph.stream(
                    {"input_str": input_str},
                    config=config,
                    stream_mode="messages",
            ):
                if meta.get("langgraph_node") != "llm_node":
                    continue
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    # content 可能是字符串或 list（多模态/工具消息），统一归一为纯文本，
                    # 否则前端 typeof === 'string' 检查失败会静默跳过，导致整轮流式不输出
                    content = chunk.content
                    if isinstance(content, list):
                        content = "".join(
                            part.get("text", "") if isinstance(part, dict) else str(part)
                            for part in content
                        )
                    if content:
                        yield f"data: {json.dumps({'content': content})}\n\n"
            yield f"data: [DONE]\n\n"
        except GeneratorExit:
            # 客户端断开连接时 StreamingResponse 会关闭生成器，这里静默退出即可
            raise

    def get_history_session(self, thread_id: str):
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        snapshot = self.main_graph.get_state(config)
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

    def get_thread_user_id(self, thread_id: str):
        """查询会话归属用户（用于 history/delete 接口的归属校验）"""
        from langgraph.checkpoint.base import CheckpointTuple

        for item in self.checkpointer.list(None):
            if item.config["configurable"]["thread_id"] != thread_id:
                continue
            checkpoint_data = item.checkpoint
            owner = (
                    checkpoint_data.get("metadata", {}).get("user_id") or
                    item.config["configurable"].get("user_id") or
                    item.metadata.get("user_id")  # CheckpointTuple 可能有 metadata
            )
            if owner:
                return str(owner)
        return None

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
        flag = False
        if not history_msg:
            logger.error(f"会话:{thread_id}记录不存在")
            return flag, f"会话:{thread_id}记录不存在:none"
        try:
            checkpointer.delete_thread(thread_id)
            logger.info(f"删除会话:{thread_id}成功:success")
            flag = True
            return flag, f"删除会话:{thread_id}成功"
        except TypeError:
            return flag, f"删除会话失败:failed"


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




