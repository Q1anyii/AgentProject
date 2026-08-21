# ============================================================
# 配置管理模块
# 作用：集中加载环境变量、校验必填项、提供类型安全的配置访问
# 使用：在 main.py 启动时调用 validate_config() 校验必填项
# ============================================================

import json
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from loguru import logger

# 加载 .env 文件（override=True 确保 .env 优先于系统环境变量）
load_dotenv(override=True)


class ConfigError(Exception):
    """配置错误异常：缺少必填环境变量或值格式错误时抛出"""
    pass


# 必填环境变量清单：(变量名, 说明)
REQUIRED_ENV_VARS = [
    ("DEEPSEEK_API_KEY", "DeepSeek 平台 API 密钥"),
    ("SILICONFLOW_API_KEY", "SiliconFlow 平台 API 密钥（Embedding + 重排）"),
    ("SILICONFLOW_BASE_URL", "SiliconFlow 接口地址"),
    ("POSTGRESQL_DB_URL", "PostgreSQL 连接串（LangGraph Checkpointer/Store）"),
    ("MYSQL_DB_URL", "MySQL 连接串（用户表 userInfo）"),
    ("REDIS_DB_URL", "Redis 连接串（检索缓存 + JWT 登录态）"),
    ("JWT_SECRET_KEY", "JWT 签名密钥"),
]

# 可选环境变量及默认值：(变量名, 默认值, 说明)
OPTIONAL_ENV_VARS = [
    ("MODEL_NAME", "deepseek:deepseek-v4-flash", "模型名称"),
    ("BASE_URL", "https://api.deepseek.com", "DeepSeek 接口地址"),
    ("JWT_ALGORITHM", "HS256", "JWT 签名算法"),
    ("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "15", "access token 有效期（分钟）"),
    ("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30", "refresh token 有效期（天）"),
    ("LANGSMITH_TRACING", "false", "是否开启 LangSmith 追踪"),
]


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """获取环境变量值。

    Args:
        key: 环境变量名
        default: 默认值（变量不存在时返回）

    Returns:
        环境变量值或默认值
    """
    return os.getenv(key, default)


def get_env_int(key: str, default: int) -> int:
    """获取整数类型环境变量，格式错误时返回默认值并记录警告。

    Args:
        key: 环境变量名
        default: 默认值

    Returns:
        整数值
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(f"环境变量 {key} 值 '{value}' 不是有效整数，使用默认值 {default}")
        return default


def get_env_bool(key: str, default: bool = False) -> bool:
    """获取布尔类型环境变量。

    支持的值：true/false, 1/0, yes/no, on/off（不区分大小写）

    Args:
        key: 环境变量名
        default: 默认值

    Returns:
        布尔值
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes", "on")


def validate_config() -> None:
    """校验所有必填环境变量，缺失时抛出 ConfigError 并列出所有缺失项。

    Raises:
        ConfigError: 存在缺失的必填环境变量时抛出
    """
    missing = []
    for key, desc in REQUIRED_ENV_VARS:
        value = os.getenv(key)
        if not value or value.strip() == "":
            missing.append(f"  - {key}: {desc}")

    if missing:
        error_msg = (
            "缺少必填环境变量，请在 .env 文件中配置以下项：\n"
            + "\n".join(missing)
            + "\n\n可复制 .env.example 为 .env 并填写实际值。"
        )
        logger.error(error_msg)
        raise ConfigError(error_msg)

    logger.success("环境变量校验通过，所有必填项已配置")


def print_config_summary() -> None:
    """打印配置摘要（不打印敏感值），用于启动时确认。"""
    logger.info("=== 配置摘要 ===")
    for key, desc in REQUIRED_ENV_VARS:
        value = os.getenv(key)
        if value:
            # 敏感信息只显示前4位和后4位，中间用*代替
            if "KEY" in key or "SECRET" in key or "PASSWORD" in key:
                masked = value[:4] + "*" * (len(value) - 8) + value[-4:] if len(value) > 8 else "****"
                logger.info(f"  {key}: {masked} (已配置)")
            else:
                logger.info(f"  {key}: {value}")
        else:
            logger.warning(f"  {key}: 未配置")
    logger.info("================")

def load_mcp_server_configs() -> list[dict]:
    """加载 MCP 服务器配置列表（供 mcp_client.init_mcp_holders 连接外部 MCP 服务器）。

    从环境变量 MCP_SERVERS 读取 JSON 数组，每项支持：
      - type: "stdio"（默认）或 "sse"
      - stdio: command（可执行文件）、args、cwd、env
      - sse: url
    cwd 若为相对路径，按项目根目录解析为绝对路径（与 .env 中
    "src/mcp_server" 这类写法一致，不受进程工作目录影响）。

    Returns:
        校验通过的配置列表；环境变量缺失 / JSON 解析失败 / 无有效条目时返回空列表。
        MCP 是可选项，配置错误不阻塞应用启动，单条无效只跳过该条。
    """
    raw = os.getenv("MCP_SERVERS")
    if not raw or not raw.strip():
        logger.info("未配置 MCP_SERVERS，跳过 MCP 工具加载")
        return []
    try:
        servers = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"MCP_SERVERS 不是合法 JSON，跳过 MCP 工具加载：{e}")
        return []
    if not isinstance(servers, list):
        logger.warning("MCP_SERVERS 应为 JSON 数组，跳过 MCP 工具加载")
        return []

    project_root = Path(__file__).resolve().parent.parent
    validated = []
    for cfg in servers:
        if not isinstance(cfg, dict):
            logger.warning(f"忽略无效的 MCP 服务器配置项（非对象）：{cfg}")
            continue
        server_type = cfg.get("type", "stdio")
        if server_type == "stdio":
            if not cfg.get("command"):
                logger.warning(f"忽略 MCP 服务器配置项（stdio 缺少 command）：{cfg}")
                continue
            cwd = cfg.get("cwd")
            if cwd and not Path(cwd).is_absolute():
                cwd = str(project_root / cwd)
            item = {**cfg, "type": "stdio", "cwd": cwd}
        elif server_type == "sse":
            if not cfg.get("url"):
                logger.warning(f"忽略 MCP 服务器配置项（sse 缺少 url）：{cfg}")
                continue
            item = {**cfg, "type": "sse"}
        else:
            logger.warning(f"忽略 MCP 服务器配置项（不支持的 type={server_type}）：{cfg}")
            continue
        validated.append(item)
    logger.info(f"MCP 服务器配置加载完成，共 {len(validated)} 个："
                f"{[c.get('name', c.get('type')) for c in validated]}")
    return validated