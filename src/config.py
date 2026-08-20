# ============================================================
# 配置管理模块
# 作用：集中加载环境变量、校验必填项、提供类型安全的配置访问
# 使用：在 main.py 启动时调用 validate_config() 校验必填项
# ============================================================

import os
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
