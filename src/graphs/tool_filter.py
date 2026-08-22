
from langchain_core.tools import tool, BaseTool


from loguru import logger

from config import load_vector_db_config
from constant.tool_constant import TOOLS_COLLECTION, TOOL_DISTANCE_THRESHOLD, TOOLS_EMBEDDING_LIMIT, TOP_FILTER_TOOLS
from init import selector_llm
from utils.tools_util import format_tools_for_prompt, parse_tool_names
from vector.vector_store import create_vector_store


class ToolFilter:
    """工具筛选装配：规则层（tags 命中）+ 语义层（工具向量索引）并集。

    语义层是加分项：任何失败（向量库不可用/索引为空/空 query）都降级为规则层，
    绝不让工具筛选阻塞主对话链路；失败一次后熔断，避免每轮重试刷屏。
    """

    def __init__(self):
        self._semantic_available = True  # 语义层熔断开关：失败后降级规则层，重启恢复
        cfg = dict(load_vector_db_config())
        cfg["collection"] = TOOLS_COLLECTION
        vector_store = create_vector_store(cfg)
        self.vector_store = vector_store
    @staticmethod
    def rule_based_filter(query, tools: list[BaseTool]):
        query_lower = query.lower()
        selected = []
        for t in tools:
            # 检查 tags 或 keywords（tags 可能为 None，防御性判空）
            if any(tag in query_lower for tag in (t.tags or [])):
                selected.append(t)
        return selected  # 可空：无 tags 命中时不再兜底全量，由 select_tools 统一决策

    def query_available_tools(self, query: str, tools: list[BaseTool], top_k: int = 5) -> list[BaseTool]:
        """语义检索候选工具：命中返回对应 BaseTool，失败降级返回空列表。"""
        if not self._semantic_available:
            return []  # 已熔断：跳过语义检索，只走规则层
        if not query or not query.strip():
            return []  # 空 query 语义检索无意义，交由 select_tools 兜底
        tool_map = {t.name: t for t in tools}        # 内存映射表：外键 -> 对象
        try:
            hits = self.vector_store.query([query], top_k, TOOL_DISTANCE_THRESHOLD)[0]  # 已按距离升序 + 阈值过滤
            selected = [
                tool_map[h.metadata["tool_name"]] for h in hits
                if h.metadata.get("tool_name") in tool_map  # 两道防线：缺外键的旧数据丢弃、工具已移除丢弃
            ]
            if len(selected) > TOOLS_EMBEDDING_LIMIT:
                selected = self.llm_refine_tools(query, selected)
                
        except Exception as e:
            # 语义层是加分项，失败不能阻塞主链路：降级为规则层筛选，并熔断避免每轮重试
            self._semantic_available = False
            logger.warning(f"工具语义检索失败，已降级为规则层筛选（重启后恢复）：{e}")
            return []
        return selected  # 可空：无命中时不再兜底全量，由 select_tools 统一决策

    def select_tools(self, query: str, tools: list[BaseTool]) -> list[BaseTool]:
        try:
            rule_hit = self.rule_based_filter(query, tools)          # 强相关：tags 命中
            semantic_hit = self.query_available_tools(query, tools, top_k=TOP_FILTER_TOOLS)  # 弱相关：语义补充
            # 按 name 去重保序：StructuredTool 不可哈希（dict.fromkeys 会炸 TypeError），
            # 工具名是稳定唯一标识；同名（多 server 冲突）后者覆盖
            merged = list({t.name: t for t in rule_hit + semantic_hit}.values())
            if not merged:
                # 两路均未命中：返回空列表，由 llm_node 注入"无工具可用"提示；
                # 不再全量兜底——全量暴露稀释模型注意力，诱发无关/幻觉调用
                logger.warning(f"工具筛选：两路均未命中，本轮无可用工具 query={query[:60]!r}")
            # 7.2 监控埋点：记录每轮筛选结果，为调整 k/阈值提供数据（toolsTODO 7.2）
            logger.info(
                f"工具筛选：query={query[:60]!r} 规则命中={len(rule_hit)} "
                f"语义命中={len(semantic_hit)} 最终={len(merged)} 工具={[t.name for t in merged]}"
            )
            return merged
        except Exception as e:
            # 最终防线：筛选过程任何意外异常都不阻塞主链路，本轮直接全量兜底
            logger.exception(f"工具筛选异常，本轮使用全量工具：{e}")
            return tools

    @staticmethod
    def llm_refine_tools(query: str, candidate_tools: list[BaseTool]) -> list[BaseTool]:
        prompt = f"""Given the user query: "{query}"
    Select the most relevant tools from the following list. Return a JSON list of tool names.

    Available tools:
    {format_tools_for_prompt(candidate_tools)}
    """
        response = selector_llm.invoke(prompt)
        selected_names = parse_tool_names(response.content)
        if not selected_names:
            # 精筛失败（格式错/无匹配）：回退向量检索结果，宁多勿漏，不缩水
            logger.warning("LLM 精筛返回无法解析，回退为向量检索结果")
            return candidate_tools
        logger.info(f"LLM 精筛：{len(candidate_tools)} -> {len(selected_names)} 个工具")
        # 只保留候选内的名字：防模型幻觉输出不存在的工具
        return [t for t in candidate_tools if t.name in selected_names]

