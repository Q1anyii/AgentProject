import os
from typing import Annotated, Optional
from operator import add

from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy, Send
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from chromadb import QueryResult
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.constants import START, END
from langgraph.graph.message import MessagesState
import chromadb
from langgraph.graph.state import StateGraph, CompiledStateGraph
from loguru import logger
from pydantic import Field
from init import model, system_prompt, embedding_function


def build_chat_graph() -> tuple[CompiledStateGraph, ConnectionPool]:
    client = chromadb.PersistentClient("../recourses/chroma_db")
    collection = client.get_collection(
        name="FAQ_KNOWLEDGE_BASE",
        embedding_function=embedding_function,
    )

    class OverAllState(MessagesState):
        input_str: Annotated[str, Field(description="用户输入")]
        retrieval_res: Annotated[Optional[QueryResult], "检索结果"] = None
        llm_output: Annotated[str, Field(description="模型输出")]
        needs_retrieval: Annotated[bool, Field(description="是否需要检索知识库")] = False

    def retrieval_node(state: OverAllState) -> OverAllState:
        input_str = state["input_str"]
        logger.info(f"执行知识库检索：{input_str}")

        retrieval_res = collection.query(
        query_texts=[input_str], # 查询文本
        n_results=3                     # 返回最相似的两个结果
        )
        return {
            "retrieval_res": retrieval_res
        }

    def llm_node(state: OverAllState, config: RunnableConfig, store: BaseStore) -> OverAllState:
        input_str = state["input_str"]
        retrieval_res = state.get("retrieval_res")

        if retrieval_res is not None:
            # 检索分支：只保留语义相关的检索结果（l2² < 1.0 约等于余弦相似度 > 0.5）
            docs = retrieval_res["documents"][0] if retrieval_res["documents"] else []
            distances = retrieval_res["distances"][0] if retrieval_res["distances"] else []
            filtered = [(d, s) for d, s in zip(docs, distances) if s < 1.0]
            if filtered:
                context = "\n\n".join(f"[文档 {i + 1}] {doc}" for i, (doc, _) in enumerate(filtered))
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

        response = model.invoke(messages)
        llm_output = response.content

        return {
            "llm_output": llm_output,
            "messages": [HumanMessage(content=input_str), AIMessage(content=llm_output)],
        }

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

        # 用 LLM 提取/合并长期记忆
        response = model.invoke(
            [HumanMessage(content=MEMORY_EXTRACT_PROMPT.format(
                old_profile=old_profile,
                input_str=state["input_str"],
                llm_output=state["llm_output"],
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
            return [Send("retrieval_node", payload)]
        return [Send("llm_node", payload)]

    builder = StateGraph(state_schema=OverAllState)
    builder.add_node("classify_node", classify_node)
    builder.add_node("retrieval_node", retrieval_node, cache_policy=CachePolicy(ttl=10))
    builder.add_node("llm_node", llm_node)
    builder.add_node("memory_node", memory_node)

    builder.add_edge(START, "classify_node")
    builder.add_conditional_edges(
        "classify_node",
        route,
        ["retrieval_node", "llm_node"],
    )
    builder.add_edge("retrieval_node", "llm_node")
    builder.add_edge("llm_node", "memory_node")
    builder.add_edge("memory_node", END)

    POSTGRESQL_DB_URL = os.getenv("POSTGRESQL_DB_URL")

    # 创建连接池（open=True 表示立即打开连接）
    # 必须开启 autocommit：迁移脚本含 CREATE INDEX CONCURRENTLY，不能在事务块中执行
    pool = ConnectionPool(
        conninfo=POSTGRESQL_DB_URL,
        kwargs={"autocommit": True},
        open=True,
    )

    # 直接实例化 PostgresSaver（短期记忆：按 thread_id 恢复历史对话）
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()

    # PostgresStore（长期记忆：跨会话保存用户档案），与 checkpointer 共用连接池
    store = PostgresStore(pool)
    store.setup()

    graph = builder.compile(checkpointer=checkpointer, store=store, cache=InMemoryCache())

    return graph, pool



