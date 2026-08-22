import asyncio
import json
import os
import chromadb

from pathlib import Path
from langgraph.store.postgres import PostgresStore
from psycopg_pool import ConnectionPool
from langchain_core.messages import BaseMessage, AIMessageChunk, ToolMessage
from loguru import logger
from langgraph.cache.redis import RedisCache  # 可能需要 langgraph-checkpoint-redis 扩展

from config import load_vector_db_config
from graphs.main_graph import build_main_graph
from graphs.retrieve_graph import build_retrieve_graph
from init import COLLECTION_NAME, CustomPostgresSaver
from service.cache_service import cache_service
from service.file_upload_service import file_upload_service
from vector.vector_store import create_vector_store


class ChatService:

    POSTGRESQL_DB_URL = os.getenv("POSTGRESQL_DB_URL")
    persist_path: str | Path
    db_url: str


    def __init__(self, db_url=None):
        self.db_url = db_url or os.getenv("POSTGRESQL_DB_URL")
        # 资源占位，open() 里真正创建，close() 里释放
        self.vector_store = None
        self.pool = self.checkpointer = self.store = None
        self.main_graph = None            # 主对话图
        self.retrieve_graph = None     # 改写+重排图
        self.cache = None
        self.redis_client = None
        # 上传文件解析内容缓存：key="{user_id}:{file_id}"，value={"name":..., "content":...}
        # 上传后立即解析并存入，发送消息时从缓存读取拼接到 input_str，避免重复解析
        self._file_content_cache: dict[str, dict] = {}

    def open(self, mcp_tools: list | None = None):
        self.vector_store = create_vector_store(load_vector_db_config())
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
        self.checkpointer = CustomPostgresSaver(self.pool)  # ← self.
        self.store = PostgresStore(self.pool)  # ← self.
        self.checkpointer.setup()
        self.store.setup()
        # 图级缓存后端：compile(cache=...) 传入 RedisCache 后，图中所有 CachePolicy 标记的
        # 节点（如 retrieve_node）命中/写入都走 Redis（键前缀 langgraph:cache:，带 TTL），
        # 多 worker 间共享；Redis 不可用时 RedisCache 内部静默降级为不缓存
        self.cache = RedisCache(cache_service.redis)  # ← self.，且 compile 用它
        self.retrieve_graph = build_retrieve_graph(self.vector_store)   # 只 build 一次，替代 @lru_cache
        self.main_graph = build_main_graph(  # 改：显式传参
            retrieve_graph=self.retrieve_graph,
            pool=self.pool,
            checkpointer=self.checkpointer,
            store=self.store,
            cache=self.cache,
            mcp_tools=mcp_tools,
        )


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

    def parse_and_cache_file(self, file_id: int, user_id: str) -> dict:
        """上传文件后立即解析文本内容并缓存（阻塞执行，解析完成才返回）。

        支持 txt/md/csv/json/xml/html/py/js 等纯文本格式；PDF/docx 等二进制格式
        暂不支持解析，返回 content=None。解析结果存入 _file_content_cache，
        发送消息时从缓存读取拼接到 input_str。

        Args:
            file_id: 文件 ID（上传接口返回）
            user_id: 用户 ID

        Returns:
            {"file_id": int, "file_name": str, "content": str|None, "parsed": bool}
        """
        cache_key = f"{user_id}:{file_id}"
        # 已缓存则直接返回，避免重复解析
        if cache_key in self._file_content_cache:
            return self._file_content_cache[cache_key]

        # 获取文件元信息
        file_info = file_upload_service.get_file(file_id, user_id)
        file_name = file_info.get("file_name", f"file_{file_id}") if file_info else f"file_{file_id}"

        # 解析文本内容
        content = file_upload_service.extract_text_from_file(file_id, user_id)
        parsed = content is not None and len(content) > 0

        result = {
            "file_id": file_id,
            "file_name": file_name,
            "content": content if parsed else None,
            "parsed": parsed,
        }
        self._file_content_cache[cache_key] = result
        logger.info(f"文件解析完成 file_id={file_id}, user_id={user_id}, parsed={parsed}, content_len={len(content) if content else 0}")
        return result

    def _build_input_with_files(self, input_str: str, file_ids: list[int], user_id: str) -> str:
        """将用户输入与上传文件解析内容拼接，作为最终 input_str 传入 llm_node。

        拼接格式：
            用户输入：{input_str}

            --- 以下为用户上传的文件内容 ---
            【文件名1】
            {文件内容1}

            【文件名2】
            {文件内容2}

        Args:
            input_str: 用户原始输入
            file_ids: 上传文件 ID 列表
            user_id: 用户 ID

        Returns:
            拼接后的完整输入文本
        """
        if not file_ids:
            return input_str

        file_sections = []
        for fid in file_ids:
            cache_key = f"{user_id}:{fid}"
            cached = self._file_content_cache.get(cache_key)
            # 缓存未命中则实时解析（兜底）
            if not cached:
                cached = self.parse_and_cache_file(fid, user_id)
            if cached.get("parsed") and cached.get("content"):
                file_sections.append(f"【{cached['file_name']}】\n{cached['content']}")

        if not file_sections:
            return input_str

        file_block = "--- 以下为用户上传的文件内容 ---\n" + "\n\n".join(file_sections)
        if input_str.strip():
            return f"{input_str}\n\n{file_block}"
        return file_block

    def clear_file_cache(self, user_id: str, file_id: int = None):
        """清除文件解析缓存（删除文件时调用）。

        Args:
            user_id: 用户 ID
            file_id: 文件 ID，None 表示清除该用户所有缓存
        """
        if file_id is not None:
            self._file_content_cache.pop(f"{user_id}:{file_id}", None)
        else:
            keys_to_remove = [k for k in self._file_content_cache if k.startswith(f"{user_id}:")]
            for k in keys_to_remove:
                del self._file_content_cache[k]

    def stream(self, user_id, thread_id, input_str, user_info=None, file_ids: list[int] = None):
        """流式对话生成。

        Args:
            user_id: 用户 ID
            thread_id: 会话 ID
            input_str: 用户输入文本
            user_info: 用户上下文信息
            file_ids: 上传文件 ID 列表，解析内容会拼接到 input_str 传入 llm_node
        """
        # 拼接用户输入与上传文件解析内容（文件内容作为上下文传入 LLM）
        if file_ids:
            input_str = self._build_input_with_files(input_str, file_ids, user_id)
            logger.info(f"拼接文件内容后 input_str 长度: {len(input_str)}, 文件数: {len(file_ids)}")
            # 文件内容已拼入本次消息，清除内存解析缓存避免累积（数据库文件记录保留）
            for fid in file_ids:
                self._file_content_cache.pop(f"{user_id}:{fid}", None)

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
            # 同时捕获 tool_call 开始/结束事件，供前端显示工具调用加载界面。
            for chunk, meta in self.main_graph.stream(
                    {"input_str": input_str},
                    config=config,
                    stream_mode="messages",
            ):
                node = meta.get("langgraph_node")
                # llm_node：输出内容 + 检测工具调用开始
                if node == "llm_node":
                    if isinstance(chunk, AIMessageChunk):
                        # 检测工具调用开始：AIMessageChunk 含 tool_calls 字段
                        if chunk.tool_calls:
                            for tc in chunk.tool_calls:
                                yield f"data: {json.dumps({'tool_call_start': {'name': tc.get('name', ''), 'args': tc.get('args', {})}})}\n\n"
                        # 输出文本内容
                        if chunk.content:
                            content = chunk.content
                            if isinstance(content, list):
                                content = "".join(
                                    part.get("text", "") if isinstance(part, dict) else str(part)
                                    for part in content
                                )
                            if content:
                                yield f"data: {json.dumps({'content': content})}\n\n"
                # tool_node：工具执行结果，发送工具调用结束事件
                elif node == "tool_node":
                    if isinstance(chunk, ToolMessage):
                        yield f"data: {json.dumps({'tool_call_end': {'name': chunk.name, 'content': str(chunk.content)[:200]}})}\n\n"
            yield f"data: [DONE]\n\n"
        except GeneratorExit:
            # 客户端断开连接时 StreamingResponse 会关闭生成器，这里静默退出即可
            raise
        except Exception as e:
            # 图执行异常（工具执行跨事件循环/checkpoint 序列化等）：记录完整堆栈并
            # 向前端推送错误事件，避免 SSE 静默断流导致前端报"发送消息失败"而日志无迹
            logger.exception(f"对话流生成异常（thread_id={thread_id}）：{e}")
            yield f"data: {json.dumps({'error': str(e), 'error_type': type(e).__name__})}\n\n"
            yield f"data: [DONE]\n\n"

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
        # CustomPostgresSaver 扩展参数：SQL 层 WHERE thread_id = %s 精确定位该会话（最新在前），
        # 不再全量遍历所有线程
        for item in self.checkpointer.list(thread_id=thread_id):
            # LangGraph 1.x：config 的 metadata 落在 CheckpointTuple.metadata（checkpoints 表 metadata 列），
            # checkpoint JSON 内部没有 metadata 字段；configurable 只持久化 thread_id，也不含 user_id
            owner = None
            if isinstance(item.metadata, dict):
                owner = item.metadata.get("user_id")
            if not owner and isinstance(item.checkpoint, dict):
                owner = item.checkpoint.get("metadata", {}).get("user_id")  # 兼容旧版本存储
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
        flag = False
        try:
            checkpointer.delete_thread(thread_id)
            logger.info(f"删除会话:{thread_id}成功")
            flag = True
            return flag, f"删除会话:{thread_id}成功"
        except KeyError:
            # LangGraph checkpointer: thread不存在抛出 KeyError
            logger.warning(f"删除会话:{thread_id}，会话记录不存在")
            return flag, f"会话:{thread_id}记录不存在"
        except Exception as e:
            logger.error(f"删除会话{thread_id}异常，err={repr(e)}", exc_info=True)
            return flag, f"删除会话失败：{str(e)}"


    def get_user_sessions(self, user_id: str):
        checkpointer = self.checkpointer

        from langgraph.checkpoint.base import CheckpointTuple

        latest_by_thread: dict[str, CheckpointTuple] = {}

        # CustomPostgresSaver 扩展参数 user_id：数据库层执行 metadata @> '{"user_id": ...}' 过滤，
        # 只返回该用户的 checkpoint，避免全表扫描后在 Python 层逐个跳过
        for item in checkpointer.list(None, user_id=user_id):
            tid = item.config["configurable"]["thread_id"]
            # 兼容旧版本存储：checkpoint JSON 内可能没有 metadata 字段
            owner = None
            if isinstance(item.metadata, dict):
                owner = item.metadata.get("user_id")
            if not owner and isinstance(item.checkpoint, dict):
                owner = item.checkpoint.get("metadata", {}).get("user_id")  # 兼容旧版本存储
            if owner != user_id:
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

chat_service = ChatService()


