from typing import Annotated, Optional, Any
import json
import uuid

from langchain_core.tools import BaseTool
from langgraph.types import CachePolicy, Send
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.constants import START, END
from langgraph.graph.message import MessagesState
from langgraph.graph.state import StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger
from pydantic import Field
from graphs.tool_filter import ToolFilter
from init import model, system_prompt
from utils.doc_util import documents_to_dicts
from constant.prompt_constants import MEMORY_EXTRACT_PROMPT, CLASSIFIER_PROMPT, NO_INFO_MARKS
from constant.cache_constant import (
    CACHE_RETRIEVE_NODE_TTL,
    CACHE_TOOL_NODE_TTL,
    CACHE_MEMORY_NODE_TTL,
)


def build_main_graph(retrieve_graph,
    pool,
    checkpointer,
    store,
    cache=None,
    mcp_tools: list[BaseTool] | None = None,):
    # 工具：ToolNode 绑定全量安全工具（按 name 路由执行），
    # LLM 侧在 llm_node 里按本轮 query 运行时筛选后 bind_tools（见 llm_node）
    tool_filter = ToolFilter()
    tools = list(mcp_tools or [])  # build 期无用户 query，不做筛选，直接全量绑定路由
    tool_node = ToolNode(tools)

    class OverAllState(MessagesState):
        input_str: Annotated[str, Field(description="用户输入")]
        retrieve_res: Annotated[Optional[list[Any] | dict[str, Any] | Any], "检索结果"] = None
        needs_retrieval: Annotated[bool, Field(description="是否需要检索知识库")] = False
        tool_status: Annotated[str, Field(
            description="本轮工具状态：executed=工具被执行；unavailable=无工具可用；idle=筛选出工具但模型未调用"
        )] = "idle"

    def _retrieve_cache_key(state: dict) -> str:
        """retrieve_node 缓存键只取用户输入，不取整个 Send payload。

        默认 key_func 会对节点输入整体 pickle 哈希，而 route() 的 Send payload
        含每轮累积变化的 messages，会导致缓存键每轮都变、缓存永不命中；
        检索结果本身只取决于问题（知识库全局共享），历史仅用于指代改写，
        同一问题短窗口内直接复用即可（Redis 键见 cache.redis.RedisCache 的 prefix）。
        """
        if not isinstance(state, dict):
            return str(state)
        return state.get("input_str", "")

    def _tool_cache_key(state: dict) -> str:
        """tool_node 缓存键：取本轮 AI 消息的 tool_calls（工具名+参数，排除每次变化的调用 id）。

        只有 route_after_llm 在 tool_calls 非空时才路由到 tool_node，因此缓存条目
        必然产生于工具真实执行之后（含执行失败返回的错误 ToolMessage），满足
        "工具被执行才存入缓存"；TTL 窗口内同工具同参数调用直接复用结果、跳过真实执行。
        防御分支（理论不可达）返回随机键，避免无工具调用时互相误命中。
        """
        if not isinstance(state, dict):
            return f"nocache-{uuid.uuid4().hex}"
        msgs = state.get("messages", []) if isinstance(state.get("messages", []), list) else []
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and m.tool_calls:
                calls = sorted(
                    (tc.get("name", ""), json.dumps(tc.get("args", {}), ensure_ascii=False, sort_keys=True))
                    for tc in m.tool_calls
                )
                return f"{len(msgs)}|{json.dumps(calls, ensure_ascii=False)}"
        return f"nocache-{uuid.uuid4().hex}"

    def _memory_cache_key(state: dict) -> str:
        """memory_node 缓存键：仅当本轮工具被执行（executed）或执行失败/无工具可用（unavailable）
        时返回确定性 key（可命中）；idle 轮（筛选出工具但模型未调用）返回随机键，永不命中、
        不与其他轮次共享缓存。key 含消息轮次（len(msgs)），避免跨轮同文误命中导致
        store 记忆写入被跳过；命中时跳过 LLM 记忆提取与 store.put（仅限 TTL 内同轮重复到达）。
        """
        if not isinstance(state, dict):
            return f"nocache-{uuid.uuid4().hex}"
        if state.get("tool_status") not in ("executed", "unavailable"):
            return f"nocache-{uuid.uuid4().hex}"
        msgs = state.get("messages", []) if isinstance(state.get("messages", []), list) else []
        last_ai = ""
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and m.content:
                last_ai = m.content if isinstance(m.content, str) else str(m.content)
                break
        return f"{len(msgs)}|{state.get('input_str', '')}|{last_ai}"

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
        if output and (hasattr(output[0], "page_content") or hasattr(output[0], "text")):
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

        # 组装消息：用户级 system prompt（基础默认 + 用户自定义）+ 长期记忆 + 历史对话 + 检索资料
        # get_user_system_prompt 内部从 MySQL user_profile 表按 user_id 读取用户自定义内容
        from init import get_user_system_prompt
        system_content = get_user_system_prompt(user_id, system_prompt)
        if long_term and long_term != "（暂无档案）":
            system_content += f"\n\n【用户长期记忆】\n{long_term}"

        messages = [SystemMessage(content=system_content)] + list(history)
        messages.append(HumanMessage(content=user_content))

        # 运行时工具筛选：规则命中 + 语义检索并集，只把候选工具暴露给 LLM 自主决策。
        # build 期绑定的全量 tools 仍由 ToolNode 按 name 路由兜底，模型幻觉调用不会 KeyError
        # 多轮增强（toolsTODO 7.4）：仅当输入含指代/承接信号时才拼接最近一轮 AI 回复，
        # 弥补指代消解（如"继续""用刚才那个工具"）；话题切换时拼接反而污染检索信号
        # （日志实证：query='软件工程\n好哒，我来帮你搜搜今天的新闻热点！' 命中了 git/fetch 等弱相关工具）
        import re
        last_ai = next(
            (m.content for m in reversed(history) if isinstance(m, AIMessage) and m.content),
            "",
        )
        if last_ai and re.search(r"(继续|刚才|那个|这个|它|同样|跟刚才)", input_str):
            # 截断防 token 膨胀：AI 回复只取前 200 字符，主信号仍是用户输入（放最前）
            filter_query = f"{input_str}\n{last_ai[:200]}"
        else:
            filter_query = input_str  # 话题切换/首轮：纯用户输入，检索信号纯净
        selected_tools = tool_filter.select_tools(filter_query, tools)
        if selected_tools:
            model_with_tools = model.bind_tools(selected_tools)
        else:
            # 两路均未命中（tags 空 + 语义检索无果）：不 bind 空列表（OpenAI 兼容 API 会 400），
            # 改用裸模型并注入提示，让 AI 如实告知无法处理，避免编造结果或假装已执行
            messages.append(SystemMessage(
                content="注意：当前没有可用的工具。若用户的请求依赖工具能力（如查文件、查数据库、"
                        "操作外部服务），请如实告知暂时无法处理，不要编造结果或假装已执行。"
            ))
            model_with_tools = model

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
        logger.info(
            f"llm_node 生成完成：tool_calls={[tc['name'] for tc in tool_calls]} "
            f"content_len={len(content)}"
        )
        ai_reply = AIMessage(content=content, tool_calls=tool_calls)

        # 本轮工具状态（memory_node 缓存策略的依据）：
        # - 有 tool_calls → executed（工具将被 ToolNode 执行）
        # - 无 tool_calls 但上一步已执行过工具（messages 末尾是 ToolMessage）→ executed
        # - 无工具可用（筛选为空，裸模型兜底回答）→ unavailable
        # - 其余（筛选出工具但模型未调用）→ idle，不参与 memory_node 缓存
        tool_status = "idle"
        if tool_calls:
            tool_status = "executed"
        elif history and isinstance(history[-1], ToolMessage):
            tool_status = "executed"
        elif not selected_tools:
            tool_status = "unavailable"

        return {"messages": [HumanMessage(content=input_str), ai_reply], "tool_status": tool_status}

    def memory_node(state: OverAllState, config: RunnableConfig, store: BaseStore) -> None:
        """将本轮对话中的长期信息提取并写入 store（按用户隔离）。"""
        user_id = config["configurable"].get("user_id", "default")
        namespace = ("rag_chat", user_id)

        # 快速路径：idle 轮（筛选出工具但模型未调用，通常是简单闲聊）且无需检索时，
        # 直接跳过记忆提取，避免 model.invoke() 阻塞 SSE 流导致前端消息超时失效。
        # 用户名基础档案已由 llm_node 在回答前写入，闲聊场景通常无新增长期信息。
        if state.get("tool_status") == "idle" and not state.get("needs_retrieval"):
            logger.debug(f"memory_node 跳过（idle 闲聊轮）user_id={user_id}")
            return

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
        target = "tool_node" if getattr(last, "tool_calls", None) else "memory_node"
        logger.info(f"llm_node 路由：{target}（last={type(last).__name__}）")
        return target

    builder = StateGraph(state_schema=OverAllState)
    builder.add_node("classify_node", classify_node)
    # 节点级缓存：compile(cache=RedisCache) 已把缓存后端落到 Redis（见 chat_service.open），
    # CachePolicy 只声明策略：TTL 常量化 + 自定义 key（仅 input_str，见 _retrieve_cache_key），
    # 命中时直接复用检索结果，跳过向量检索与重排，降低 API/算力消耗。
    builder.add_node(
        "retrieve_node",
        retrieve_node,
        cache_policy=CachePolicy(ttl=CACHE_RETRIEVE_NODE_TTL, key_func=_retrieve_cache_key),
    )
    builder.add_node("llm_node", llm_node)
    # tool_node：仅工具真实执行后才有缓存条目（路由保证 tool_calls 非空才到达）；
    # key 取工具名+参数（排除每次变化的调用 id），TTL 窗口内同调用复用结果
    builder.add_node(
        "tool_node",
        tool_node,
        cache_policy=CachePolicy(ttl=CACHE_TOOL_NODE_TTL, key_func=_tool_cache_key),
    )
    # memory_node：仅 executed/unavailable 轮写入缓存（见 _memory_cache_key），
    # 命中时跳过 LLM 记忆提取与 store 写入，省一次模型调用
    builder.add_node(
        "memory_node",
        memory_node,
        cache_policy=CachePolicy(ttl=CACHE_MEMORY_NODE_TTL, key_func=_memory_cache_key),
    )

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
        ["tool_node", "memory_node"],
    )
    builder.add_edge("tool_node", "llm_node")  # 工具执行结果回到 LLM，生成最终回答
    builder.add_edge("memory_node", END)

    # 创建连接池（open=True 表示立即打开连接）
    # 必须开启 autocommit：迁移脚本含 CREATE INDEX CONCURRENTLY，不能在事务块中执行
    try:
        pool.check()
    except Exception as e:
        logger.error(f"数据库连接失败，请检查 .env 的 POSTGRESQL_DB_URL 与 PostgreSQL 服务")
        logger.error(f"真实错误：{e}")
        raise

    checkpointer.setup()
    store.setup()

    main_graph = builder.compile(checkpointer=checkpointer, store=store, cache=cache)

    return main_graph