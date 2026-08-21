# ============================================================
# 用户扩展信息服务层
# 作用：管理用户个人信息（头像、助手风格、自定义 system prompt、主题、MCP 配置）
# 存储：MySQL user_profile 表（与 userInfo 表通过 user_id 关联）
# ============================================================

import json
from datetime import datetime
from typing import Optional, Dict, Any

import pymysql
from dbutils.pooled_db import PooledDB
from loguru import logger
from dotenv import load_dotenv
import os

load_dotenv()


class UserProfileService:
    """用户扩展信息服务。

    与 LoginService 共用 MySQL 连接池思路，但独立管理 user_profile 表。
    所有方法均为同步实现，FastAPI 路由层用普通 def 自动放入线程池。
    """

    def __init__(self):
        self._pool: Optional[PooledDB] = None

    def open(self):
        """初始化连接池（与 ChatService.open() 同期调用）。"""
        if self._pool:
            return
        db_url = os.getenv("MYSQL_DB_URL")
        # 解析 mysql+pymysql://user:pass@host:port/dbname
        # 格式：mysql+pymysql://root:1234@127.0.0.1:3306/Mitta
        rest = db_url.split("://", 1)[1]
        user_pass, host_port_db = rest.split("@", 1)
        user, password = user_pass.split(":", 1)
        host_port, dbname = host_port_db.split("/", 1)
        host, port = host_port.split(":", 1)

        self._pool = PooledDB(
            creator=pymysql,
            maxconnections=10,
            mincached=1,
            maxcached=5,
            blocking=True,
            maxusage=None,
            ping=1,
            host=host,
            port=int(port),
            user=user,
            password=password,
            database=dbname,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )
        self._ensure_table()
        logger.info("UserProfileService 连接池已初始化")

    def close(self, timeout: int = 5):
        """关闭连接池。"""
        if self._pool:
            self._pool.close()
            self._pool = None
            logger.info("UserProfileService 连接池已关闭")

    def _get_conn(self):
        if not self._pool:
            raise RuntimeError("UserProfileService 未初始化，请先调用 open()")
        return self._pool.connection()

    def _ensure_table(self):
        """确保 user_profile 表存在，不存在则创建。"""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    user_id VARCHAR(64) PRIMARY KEY COMMENT '用户ID，关联 userInfo.user_id',
                    username VARCHAR(64) DEFAULT NULL COMMENT '显示用户名',
                    avatar TEXT DEFAULT NULL COMMENT '头像（base64 data URL 或 颜色标识）',
                    assistant_style TEXT DEFAULT NULL COMMENT '助手风格设定（用户自定义）',
                    system_prompt TEXT DEFAULT NULL COMMENT '用户自定义 system prompt（全局）',
                    theme VARCHAR(32) DEFAULT 'default' COMMENT '前端主题名称',
                    mcp_config JSON DEFAULT NULL COMMENT 'MCP 服务器配置（JSON 数组）',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_theme (theme)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户扩展信息表';
            """)
            conn.commit()
        except pymysql.MySQLError as e:
            logger.error(f"创建 user_profile 表失败: {e}")
            raise
        finally:
            conn.close()

    # ==================== 基础 CRUD ====================

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户扩展信息，不存在返回 None。"""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM user_profile WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if row and row.get("mcp_config"):
                # mcp_config 是 JSON 字符串，解析为 dict
                if isinstance(row["mcp_config"], str):
                    try:
                        row["mcp_config"] = json.loads(row["mcp_config"])
                    except json.JSONDecodeError:
                        row["mcp_config"] = []
            return row
        except pymysql.MySQLError as e:
            logger.error(f"获取用户扩展信息失败 user_id={user_id}: {e}")
            raise
        finally:
            conn.close()

    def _upsert_profile(self, user_id: str, fields: Dict[str, Any]):
        """内部方法：插入或更新用户扩展信息。

        Args:
            user_id: 用户 ID
            fields: 要更新的字段字典（字段名 -> 值）
        """
        if not fields:
            return
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            # 先检查是否存在
            cur.execute("SELECT user_id FROM user_profile WHERE user_id = %s", (user_id,))
            exists = cur.fetchone()

            if exists:
                # UPDATE
                set_clause = ", ".join([f"{k} = %s" for k in fields.keys()])
                values = list(fields.values()) + [user_id]
                cur.execute(f"UPDATE user_profile SET {set_clause} WHERE user_id = %s", values)
            else:
                # INSERT
                columns = ", ".join(["user_id"] + list(fields.keys()))
                placeholders = ", ".join(["%s"] * (len(fields) + 1))
                values = [user_id] + list(fields.values())
                cur.execute(f"INSERT INTO user_profile ({columns}) VALUES ({placeholders})", values)
            conn.commit()
        except pymysql.MySQLError as e:
            conn.rollback()
            logger.error(f"更新用户扩展信息失败 user_id={user_id}: {e}")
            raise
        finally:
            conn.close()

    # ==================== 个人信息 ====================

    def update_basic_info(self, user_id: str, username: Optional[str] = None,
                           avatar: Optional[str] = None,
                           assistant_style: Optional[str] = None) -> bool:
        """更新用户基本信息（用户名、头像、助手风格）。

        Args:
            user_id: 用户 ID
            username: 显示用户名（None 表示不更新）
            avatar: 头像（base64 data URL，None 表示不更新）
            assistant_style: 助手风格设定（None 表示不更新）

        Returns:
            bool: 是否成功
        """
        fields = {}
        if username is not None:
            fields["username"] = username
        if avatar is not None:
            fields["avatar"] = avatar
        if assistant_style is not None:
            fields["assistant_style"] = assistant_style
        if not fields:
            return False
        self._upsert_profile(user_id, fields)
        return True

    # ==================== 自定义 System Prompt ====================

    def get_system_prompt(self, user_id: str) -> Optional[str]:
        """获取用户自定义 system prompt，不存在或为空返回 None。"""
        profile = self.get_profile(user_id)
        if profile and profile.get("system_prompt"):
            return profile["system_prompt"]
        return None

    def update_system_prompt(self, user_id: str, content: str) -> bool:
        """更新用户自定义 system prompt。

        Args:
            user_id: 用户 ID
            content: system prompt 内容（空字符串表示清除）

        Returns:
            bool: 是否成功
        """
        self._upsert_profile(user_id, {"system_prompt": content})
        return True

    # ==================== 主题配置 ====================

    def get_theme(self, user_id: str) -> str:
        """获取用户主题名称，默认 'default'。"""
        profile = self.get_profile(user_id)
        if profile and profile.get("theme"):
            return profile["theme"]
        return "default"

    def update_theme(self, user_id: str, theme: str) -> bool:
        """更新用户主题。"""
        self._upsert_profile(user_id, {"theme": theme})
        return True

    # ==================== MCP 配置 ====================

    def get_mcp_config(self, user_id: str) -> list:
        """获取用户 MCP 服务器配置，返回 list（每个元素是一个 dict）。"""
        profile = self.get_profile(user_id)
        if profile and profile.get("mcp_config"):
            config = profile["mcp_config"]
            if isinstance(config, list):
                return config
            if isinstance(config, dict):
                return [config]
        return []

    def update_mcp_config(self, user_id: str, mcp_servers: list) -> bool:
        """更新用户 MCP 服务器配置。

        Args:
            user_id: 用户 ID
            mcp_servers: MCP 服务器配置列表，每个元素是 dict
                         格式示例：[{"name": "server1", "command": "npx", "args": ["-y", "mcp-server"], "env": {}}]

        Returns:
            bool: 是否成功
        """
        if not isinstance(mcp_servers, list):
            raise ValueError("mcp_servers 必须是列表")
        # 序列化为 JSON 字符串存储
        mcp_json = json.dumps(mcp_servers, ensure_ascii=False)
        self._upsert_profile(user_id, {"mcp_config": mcp_json})
        return True


# 模块级单例（与 chat_service / login_service 一致）
user_profile_service = UserProfileService()
