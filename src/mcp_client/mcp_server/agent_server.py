import sys
from pathlib import Path

# stdio 子进程以 cwd=本目录启动，需手动把项目 src/ 加入 sys.path 才能 import service.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastmcp import FastMCP

from service.login_service import login_service
from service.chat_service import chat_service     # 子进程内新建单例，需调用 open() 后方可用

mcp = FastMCP("agent_server")

@mcp.tool()
async def chat(query: str, thread_id: str = "mcp-default", user_id: str = "mcp-user") -> str:
    """客服对话：含知识库检索、短期记忆（thread_id 隔离）、长期记忆（user_id 隔离）"""
    return await chat_service.a_invoke(user_id, thread_id, query)   # 已有异步壳，流式降级为整段返回

@mcp.tool()
async def get_current_user(user_id: str) -> dict:
    """查询用户账户信息（复用 login_service，补全现有空壳实现）"""
    row = login_service.get_user_by_id(user_id)
    return {"user_id": ..., "username": ..., "create_time": ...} | {"error": "用户不存在"}

@mcp.tool()
async def summarize(thread_id: str, user_id: str) -> str:
    """按会话压缩上下文：取 checkpointer 历史（chat_service.get_history_session）交给 model 压缩"""
    ...


if __name__ == "__main__":
    # stdio 子进程入口：缺了它脚本执行完就退出，客户端握手直接失败
    mcp.run(transport="stdio")