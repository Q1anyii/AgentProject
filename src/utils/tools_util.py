import asyncio
from typing import List

from langchain_core.tools import BaseTool
import json
import re
from config import load_mcp_server_configs, load_vector_db_config
from constant.tool_constant import TOOLS_COLLECTION
from mcp_client.client import init_mcp_holders
from loguru import logger
from rich import print as rprint

from vector.vector_store import create_vector_store


async def list_mcp_tools():
    mcp_holders = await init_mcp_holders(load_mcp_server_configs())
    tools_list = List
    for h in mcp_holders:
        for t in h.tools:
            logger.success(f"已加载 MCP 工具：{t.name}")
            rprint(t)

    mcp_tools = [t for h in mcp_holders for t in h.tools]


def safety_filter(tools: list[BaseTool], user_role: str = "user") -> list[BaseTool]:
    """安全硬过滤：破坏性工具需管理员权限，写操作需场景允许"""
    result = []
    for tool in tools:
        meta = tool.metadata or {}
        # 破坏性操作（如 git_reset、delete_file）：仅管理员可用
        if meta.get("destructiveHint") and user_role != "admin":
            continue
        # 写操作（如 git_add、write_file）：只读模式下禁用
        # if not meta.get("readOnlyHint") and read_only_mode:
        #     continue
        result.append(tool)
    return result


def tools_embedding(mcp_tools: list[BaseTool]) -> None:
    """启动期构建工具向量索引：独立 collection + 工具名 id + metadata 存 tool_name。

    document 拼接 name + description + tags（toolsTODO 4.1），检索命中后靠
    metadata["tool_name"] 回查内存映射表还原 BaseTool（见 ToolFilter.query_available_tools）。
    构建失败不阻断启动：语义层查不到工具时会自然降级（规则层/全量兜底）。
    """
    if not mcp_tools:
        logger.info("无 MCP 工具，跳过工具索引构建")
        return
    cfg = dict(load_vector_db_config())  # 拷贝，避免污染共享配置对象
    cfg["collection"] = TOOLS_COLLECTION  # 与知识库 FAQ_KNOWLEDGE_BASE 隔离
    vector_store = create_vector_store(cfg)
    ids, documents, metadatas = [], [], []
    for t in mcp_tools:
        ids.append(t.name)  # 工具名做 id：天然唯一、upsert 幂等覆盖，重复启动不产生脏数据
        documents.append(f"{t.name}\n{t.description or ''}\n{'、'.join(t.tags or [])}")
        metadatas.append({"tool_name": t.name, "source": "mcp"})  # 外键字段，供检索后映射回 BaseTool
    try:
        vector_store.upsert(ids, documents, metadatas)
        logger.success(f"工具向量索引构建完成：{len(mcp_tools)} 个工具 -> collection={TOOLS_COLLECTION}")
    except Exception as e:
        logger.error(f"工具向量索引构建失败：{e}")


def format_tools_for_prompt(candidate_tools: list[BaseTool]) -> str:
    """把候选工具格式化为 LLM 可读文本：每工具一行，含参数名，控制 token 量。"""
    lines = []
    for t in candidate_tools:
        args_schema = getattr(t, "args", None) or {}          # JSON Schema dict
        param_names = list((args_schema.get("properties") or {}).keys())
        desc = (t.description or "").strip().replace("\n", " ")[:200]  # 截断防爆
        lines.append(f"- {t.name}: {desc}（参数：{', '.join(param_names) or '无'}）")
    return "\n".join(lines)

def parse_tool_names(content: str) -> list[str]:
    """清洗 LLM 返回：去围栏 → 提取 JSON 数组 → 解析失败返回空列表。"""

    text = content.strip()
    # ① 去掉 ```json ... ``` 围栏（模型常把 JSON 包在代码块里）
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # ② 提取第一个 [ ... ] 数组（容忍前后夹带的解释文本）
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        text = text[start:end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    # ③ 兼容两种返回形态：纯数组 / {"tools": [...]} 包装
    if isinstance(data, list):
        return [str(n) for n in data]
    if isinstance(data, dict) and data.get("tools"):
        return [str(n) for n in data["tools"]]
    return []


if __name__ == "__main__":
    asyncio.run(list_mcp_tools())