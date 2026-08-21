from typing import Annotated, Optional, TYPE_CHECKING, Any

from langgraph.types import CachePolicy, Send
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from chromadb import QueryResult
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.constants import START, END
from langgraph.graph.message import MessagesState
from langgraph.graph.state import StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger
from pydantic import Field
from rich import print as rprint

if TYPE_CHECKING:
    # 仅用于类型注解，运行时导入会与 chat_service 形成循环依赖
    from service.chat_service import ChatService
from graphs.retrieve_graph import build_retrieve_graph
from graphs.tool_graph import build_tool_graph
from init import model, system_prompt
from utils.doc_util import documents_to_dicts
from constant.prompt_constants import MEMORY_EXTRACT_PROMPT, CLASSIFIER_PROMPT, NO_INFO_MARKS


def build_main_graph(self, mcp_tools: list | None = None):

    retrieve_graph = self.retrieve_graph or build_retrieve_graph(self=self)

    # 工具：LLM 需要时可调用（用户信息/会话总结），工具通过请求级上下文读取数据
    tools = build_tool_graph(mcp_tools)  # 原 build_tool_graph() 调用点替换
    tool_node = ToolNode(tools)
    model_with_tools = model.bind_tools(tools)  # ← 原 build_chat_graph 逻辑整体搬入

    class OverAllState(MessagesState):
        input_str: Annotated[str, Field(description="用户输入")]
        retrieve_res: Annotated[Optional[list[Any] | dict[str, Any] | Any], "检索结果"] = None
        needs_retrieval: Annotated[bool, Field(description="是否需要检索知识库")] = False

    def retrieve_node(state: OverAllState) -> OverAllState:
        input_str = state["input_str"]
        logger.info(f"执行知识库检索：{input_str}")

        history = [
            {"role": "user" if m.type == "human" else "assistant", "content": m.content}
            for m in state.get("messages", [])
            if m.type in ("human", "ai")
        ]

        retrieve_res = retrieve_graph.invoke({
            "question": input_str,
            "history": history,
        })

        # Document 无法被 checkpointer 正确反序列化：恢复会话时会被还原成 dict，
        # 导致 llm_node 里 doc.page_content 报 AttributeError。
        # 统一在入 state 前转成 dict，llm_node 侧兼容两种形态读取。
        output = retrieve_res.get("output", [])
        if output and hasattr(output[0], "page_content"):
            retrieve_res["output"] = documents_to_dicts(output)

        return {
            "retrieve_res": retrieve_res
        }

    def _get_username(config: RunnableConfig) -> str | None:
        """取当前用户 username：优先用鉴权解析结果（chat 路由已从 JWT 解析），
        invoke 等无 user_info 的路径回退到 Redis 登录态 token 解析。"""
        user_info = config["configurable"].get("user_info")
        if user_info is not None:
            username = getattr(user_info, "username", None)
            if username:
                return username
        user_id = config["configurable"].get("user_id", "default")
        from constant.cache_constant import USER_TOKEN_KEY
        from service.cache_service import cache_service
        from utils.jwt_utils import get_username_from_token
        try:
            token = cache_service.redis.get(USER_TOKEN_KEY.format(user_id=user_id))
        except Exception as e:
            logger.warning(f"读取登录态 token 失败，跳过用户名解析：{e}")
            return None
        return get_username_from_token(token) if token else None

    def _ensure_username_profile(store: BaseStore, user_id: str, username: str | None) -> str:
        """把 username 并入长期记忆档案并立即落库（回答前完成），返回合并后档案。

        username 属长期事实，先写入再组装提示词，AI 首轮就能识别用户；
        档案按 (rag_chat, user_id) 命名空间隔离，各用户独立存储。
        """
        namespace = ("rag_chat", user_id)
        item = store.get(namespace, "user_profile")
        profile = item.value["profile"] if item else "（暂无档案）"
        base_profile = f"用户名：{username}" if username else ""
        if base_profile and base_profile not in profile:
            profile = f"{profile}\n{base_profile}" if profile != "（暂无档案）" else base_profile
            store.put(namespace, "user_profile", {"profile": profile})
            logger.info(f"用户名基础档案已落库（user_id={user_id}）")
        return profile

    def llm_node(state: OverAllState, config: RunnableConfig, store: BaseStore) -> OverAllState:
        input_str = state["input_str"]
        retrieval_res = state.get("retrieve_res")


        if retrieval_res and "output" in retrieval_res:
            # 检索分支：在线重排结果已按相关性降序，直接取文档文本。
            # 兼容 Document 对象与 dict 两种形态（checkpoint 恢复/旧缓存里是 dict）
            raw_docs = retrieval_res.get("output", [])
            docs = [
                doc.page_content if hasattr(doc, "page_content") else doc.get("page_content", "")
                for doc in raw_docs
            ]
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

        # 长期记忆：先把 username 并入档案并立即落库，再读取组装提示词（回答前完成）。
        # 若等回答完 memory_node 才写入，首轮 AI 会先回答不认识、记忆随后才落库
        user_id = config["configurable"].get("user_id", "default")
        username = _get_username(config)
        long_term = _ensure_username_profile(store, user_id, username)

        # 短期记忆：checkpointer 按 thread_id 恢复的历史对话
        history = state.get("messages", [])

        # 组装消息：系统提示（含长期记忆）+ 历史对话 + 检索资料与当前问题
        system_content = system_prompt
        if long_term and long_term != "（暂无档案）":
            system_content += f"\n\n【用户长期记忆】\n{long_term}"

        messages = [SystemMessage(content=system_content)] + list(history)
        messages.append(HumanMessage(content=user_content))

        # 流式生成：LangGraph 会通过 callback 机制自动捕获 model.stream 的每个 token，
        # 以 stream_mode="messages" 输出（前端逐片累加即打字机效果）。
        # 注意：节点不能返回生成器——langgraph 1.x 会把生成器当单条消息交给
        # add_messages/_convert_to_message 转换，报 "Unsupported message type: generator"。
        chunks = []
        for chunk in model_with_tools.stream(messages):
            chunks.append(chunk)
        content = "".join(c.content for c in chunks if isinstance(c.content, str))
        # 工具调用：模型可能返回 tool_calls（跨 chunk 累积，取最后一个完整结果），
        # 有 tool_calls 时由 route_after_llm 交给 ToolNode 执行，再回到本节点生成最终回答
        tool_calls = chunks[-1].tool_calls if chunks else []
        ai_reply = AIMessage(content=content, tool_calls=tool_calls)

        return {"messages": [HumanMessage(content=input_str), ai_reply]}

    def memory_node(state: OverAllState, config: RunnableConfig, store: BaseStore) -> None:
        """将本轮对话中的长期信息提取并写入 store（按用户隔离）。"""
        user_id = config["configurable"].get("user_id", "default")
        namespace = ("rag_chat", user_id)

        # NO_INFO_MARKS 已移至 constant/prompt_constants.py 统一管理

        # 读取已有档案
        item = store.get(namespace, "user_profile")
        original_profile = item.value["profile"] if item else "（暂无档案）"
        old_profile = original_profile

        # 用户名基础档案已由 llm_node 在组装提示词前写入（首轮对话即落库），
        # 这里只负责增量提取与防丢失兜底，不再重复解析 Redis token

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
        # 本轮无新信息时（LLM 返回占位符），至少把已有档案持久化
        if new_profile in NO_INFO_MARKS:
            new_profile = old_profile
        # 兜底：LLM 合并结果若丢失了"用户名"行，从原档案补回（llm_node 已保证原档案含该行）
        import re
        m = re.search(r"^用户名：.+$", original_profile, re.MULTILINE)
        if m and m.group(0) not in new_profile:
            new_profile = f"{new_profile}\n{m.group(0)}"
        # 与「合并前」档案比较：首次对话（无档案→含用户名）也会触发写入
        if new_profile and new_profile != original_profile:
            store.put(namespace, "user_profile", {"profile": new_profile})
            logger.info(f"长期记忆已更新（user_id={user_id}）：{new_profile[:100]}")

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

    def route_after_llm(state: OverAllState) -> str:
        """llm_node 之后：有工具调用则执行 ToolNode，否则进入记忆节点收尾"""
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "memory_node"

    builder = StateGraph(state_schema=OverAllState)
    builder.add_node("classify_node", classify_node)
    builder.add_node("retrieve_node", retrieve_node, cache_policy=CachePolicy(ttl=10))
    builder.add_node("llm_node", llm_node)
    builder.add_node("tools", tool_node)
    builder.add_node("memory_node", memory_node)

    builder.add_edge(START, "classify_node")
    builder.add_conditional_edges(
        "classify_node",
        route,
        ["retrieve_node", "llm_node"],
    )
    builder.add_edge("retrieve_node", "llm_node")
    builder.add_conditional_edges(
        "llm_node",
        route_after_llm,
        ["tools", "memory_node"],
    )
    builder.add_edge("tools", "llm_node")  # 工具执行结果回到 LLM，生成最终回答
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

    main_graph = builder.compile(checkpointer=checkpointer, store=store, cache=self.cache)

    return main_graph