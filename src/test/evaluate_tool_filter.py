"""工具筛选离线评估（toolsTODO 7.2）：对测试集计算 recall@k / precision@k。

用法：
    conda activate langchain1.2
    cd src && python -m test.evaluate_tool_filter

先按实际 MCP 工具名调整 TEST_CASES 的期望命中工具，运行后输出各 query 的
命中情况与汇总指标，据此调整 TOP_FILTER_TOOLS / TOOL_DISTANCE_THRESHOLD。
"""
import asyncio

from loguru import logger
from rich.console import Console
from rich.table import Table

from config import load_mcp_server_configs
from constant.tool_constant import TOP_FILTER_TOOLS
from graphs.tool_filter import ToolFilter
from mcp_client.client import init_mcp_holders
from utils.tools_util import safety_filter

console = Console()

# 测试集：query -> 期望命中的工具名（按实际 MCP 工具名调整）
TEST_CASES = [
    {"query": "今天北京天气怎么样", "expected": ["get_weather"]},
    # {"query": "帮我写一份周报", "expected": ["write_report"]},
]


def recall_at_k(selected: list[str], expected: list[str], k: int) -> float:
    """命中的期望工具数 / 期望工具总数（k 内截断）。"""
    if not expected:
        return 1.0
    return len(set(selected[:k]) & set(expected)) / len(expected)


def precision_at_k(selected: list[str], expected: list[str], k: int) -> float:
    """命中的期望工具数 / 实际返回工具数（k 内截断）。"""
    if not selected[:k]:
        return 0.0
    return len(set(selected[:k]) & set(expected)) / len(selected[:k])


async def main():
    mcp_holders = await init_mcp_holders(load_mcp_server_configs())
    tools = safety_filter([t for h in mcp_holders for t in h.tools])
    if not tools:
        logger.error("无 MCP 工具，请先配置 MCP 服务器")
        return

    tool_filter = ToolFilter()
    table = Table(title=f"工具筛选离线评估（k={TOP_FILTER_TOOLS}，共 {len(tools)} 个工具）")
    table.add_column("query")
    table.add_column("期望")
    table.add_column("命中")
    table.add_column(f"recall@{TOP_FILTER_TOOLS}")
    table.add_column(f"precision@{TOP_FILTER_TOOLS}")

    recall_sum = precision_sum = 0.0
    for case in TEST_CASES:
        expected = case["expected"]
        selected = [t.name for t in tool_filter.select_tools(case["query"], tools)][:TOP_FILTER_TOOLS]
        recall = recall_at_k(selected, expected, TOP_FILTER_TOOLS)
        precision = precision_at_k(selected, expected, TOP_FILTER_TOOLS)
        recall_sum += recall
        precision_sum += precision
        table.add_row(case["query"], str(expected), str(selected), f"{recall:.2f}", f"{precision:.2f}")

    console.print(table)
    n = len(TEST_CASES)
    console.print(
        f"平均 recall@{TOP_FILTER_TOOLS} = {recall_sum / n:.2f}，"
        f"平均 precision@{TOP_FILTER_TOOLS} = {precision_sum / n:.2f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
