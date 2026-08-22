"""
全局 MCP 配置路由：本地文件存储的 MCP 服务器配置读写

对应原 main.py 中的 /api/mcp/config 接口。
MCP 配置存储在用户本地 JSON 文件中，用户可指定路径，读写均走该文件。
"""

from fastapi import APIRouter, Depends

from config import get_mcp_config_path, load_mcp_server_configs, save_mcp_server_configs
from utils.jwt_utils import get_current_user, TokenData
from utils.response_util import Response

router = APIRouter(tags=["MCP 配置"])


@router.get("/api/mcp/config")
def get_global_mcp_config(current_user: TokenData = Depends(get_current_user)):
    """获取全局 MCP 配置（从本地文件读取）。

    Returns:
        { ok, data: { path, mcp_servers } }
    """
    config_path = get_mcp_config_path()
    mcp_servers = load_mcp_server_configs()
    return {"ok": True, "data": {"path": config_path, "mcp_servers": mcp_servers}}


@router.put("/api/mcp/config")
def update_global_mcp_config(
    request_body: dict,
    current_user: TokenData = Depends(get_current_user)
):
    """更新全局 MCP 配置（保存到本地文件）。

    Request body:
        - path: 配置文件路径（可选，不指定则使用当前路径）
        - mcp_servers: MCP 配置列表（必填）

    Returns:
        { ok, detail, data: { path } }
    """
    mcp_servers = request_body.get("mcp_servers")
    path = request_body.get("path")

    if mcp_servers is None:
        return Response.failed("缺少 mcp_servers 字段")
    if not isinstance(mcp_servers, list):
        return Response.failed("mcp_servers 必须是 JSON 数组")

    try:
        saved_path = save_mcp_server_configs(mcp_servers, path)
        return {
            "ok": True,
            "detail": "MCP 配置已保存到本地文件，重启后端服务后生效",
            "data": {"path": saved_path}
        }
    except ValueError as e:
        return Response.failed(str(e))
