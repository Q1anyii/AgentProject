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

_mcp_config_cache: list[dict] | None = None

def load_mcp_server_configs() -> list[dict]:
    """加载 MCP 服务器配置列表（供 mcp_client.init_mcp_holders 连接外部 MCP 服务器）。

    【架构变更】MCP 配置改为本地 JSON 文件存储，用户可在前端指定文件路径。
    路径记录在 resources/config/.mcp_config_path 中，默认路径为 resources/config/mcp_servers.json。
    启动时从该路径读取配置；文件不存在或解析失败时返回空列表（不阻塞启动）。

    Returns:
        校验通过的配置列表；文件不存在 / JSON 解析失败 / 无有效条目时返回空列表。
    """
    global _mcp_config_cache
    if _mcp_config_cache is not None:
        return _mcp_config_cache
    config_path = get_mcp_config_path()
    if not config_path or not Path(config_path).is_file():
        logger.info(f"MCP 配置文件不存在（路径: {config_path}），跳过 MCP 工具加载")
        return []
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            servers = json.load(f)
    except (json.JSONDecodeError, TypeError, OSError) as e:
        logger.warning(f"MCP 配置文件读取失败（路径: {config_path}）：{e}，跳过 MCP 工具加载")
        return []
    if not isinstance(servers, list):
        logger.warning(f"MCP 配置文件内容不是 JSON 数组（路径: {config_path}），跳过 MCP 工具加载")
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
    logger.info(f"MCP 服务器配置加载完成（路径: {config_path}），共 {len(validated)} 个："
                f"{[c.get('name', c.get('type')) for c in validated]}")
    _mcp_config_cache = validated
    return validated


# ============================================================
# MCP 配置文件路径管理
# 作用：记录用户指定的 MCP 配置文件路径，读写均走该路径
# 路径存储文件：resources/config/.mcp_config_path
# 默认配置文件：resources/config/mcp_servers.json
# ============================================================

# 路径记录文件：存储用户指定的 MCP 配置文件路径
_MCP_PATH_FILE = Path(__file__).resolve().parent.parent / "resources" / "config" / ".mcp_config_path"
# 默认配置文件路径
_DEFAULT_MCP_CONFIG_PATH = Path(__file__).resolve().parent.parent / "resources" / "config" / "mcp_servers.json"
# 允许的配置文件目录白名单（安全限制，防止写入系统敏感目录）
_ALLOWED_MCP_CONFIG_DIRS = [
    Path(__file__).resolve().parent.parent / "resources",
    Path(__file__).resolve().parent.parent / "config",
    Path.home() / ".mitta",
]


def get_mcp_config_path() -> str:
    """获取当前 MCP 配置文件路径。

    优先从路径记录文件读取；记录文件不存在或为空时返回默认路径。

    Returns:
        MCP 配置文件的绝对路径字符串
    """
    try:
        if _MCP_PATH_FILE.is_file():
            content = _MCP_PATH_FILE.read_text(encoding="utf-8").strip()
            if content:
                return str(Path(content).resolve())
    except OSError as e:
        logger.warning(f"读取 MCP 配置路径记录失败：{e}，使用默认路径")
    return str(_DEFAULT_MCP_CONFIG_PATH.resolve())


def set_mcp_config_path(path: str) -> str:
    """设置 MCP 配置文件路径，并写入路径记录文件。

    会自动创建父目录（如果不存在）。路径需通过安全校验。

    Args:
        path: 配置文件路径（绝对路径或相对项目根目录的路径）

    Returns:
        解析后的绝对路径字符串

    Raises:
        ValueError: 路径未通过安全校验
    """
    if not path or not path.strip():
        raise ValueError("配置文件路径不能为空")

    resolved = Path(path).resolve()
    # 安全校验：路径必须在允许的目录白名单内
    _validate_mcp_config_path(resolved)

    # 确保父目录存在
    resolved.parent.mkdir(parents=True, exist_ok=True)

    # 写入路径记录文件
    try:
        _MCP_PATH_FILE.parent.mkdir(parents=True, exist_ok=True)
        _MCP_PATH_FILE.write_text(str(resolved), encoding="utf-8")
    except OSError as e:
        raise ValueError(f"写入路径记录文件失败：{e}")

    logger.info(f"MCP 配置文件路径已更新：{resolved}")
    return str(resolved)


def _validate_mcp_config_path(path: Path) -> None:
    """校验 MCP 配置文件路径是否在允许的目录白名单内。

    防止用户指定系统敏感目录（如 C:/Windows、C:/Program Files）造成安全风险。

    Args:
        path: 解析后的绝对路径

    Raises:
        ValueError: 路径不在允许的目录内
    """
    # 允许用户主目录下的任意路径（个人配置）
    if str(path).startswith(str(Path.home())):
        return
    # 检查白名单目录
    for allowed_dir in _ALLOWED_MCP_CONFIG_DIRS:
        try:
            path.relative_to(allowed_dir.resolve())
            return
        except ValueError:
            continue
    raise ValueError(
        f"配置文件路径不在允许的目录内。允许的目录：\n"
        f"  - 项目 resources/ 目录\n"
        f"  - 项目 config/ 目录\n"
        f"  - 用户主目录（{Path.home()}）下任意路径\n"
        f"当前路径：{path}"
    )


def save_mcp_server_configs(configs: list[dict], path: str = None) -> str:
    """保存 MCP 服务器配置到本地 JSON 文件。

    Args:
        configs: MCP 配置列表
        path: 配置文件路径（可选，不指定则使用当前路径）

    Returns:
        保存的文件绝对路径

    Raises:
        ValueError: 路径未通过安全校验
    """
    global _mcp_config_cache
    if path:
        config_path = set_mcp_config_path(path)
    else:
        config_path = get_mcp_config_path()

    resolved = Path(config_path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(resolved, "w", encoding="utf-8") as f:
            json.dump(configs, f, ensure_ascii=False, indent=2)
    except OSError as e:
        raise ValueError(f"写入 MCP 配置文件失败：{e}")

    logger.info(f"MCP 配置已保存到文件：{resolved}（共 {len(configs)} 个服务器）")
    _mcp_config_cache = None  # 配置已变更，下次读取时重读文件
    return str(resolved)