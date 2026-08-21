# ============================================================
# 文件上传服务层
# 作用：管理用户上传的文件（多种格式，base64 存储在 MySQL）
# 存储：MySQL user_files 表
# 支持格式：txt, md, pdf, docx, pptx, xlsx, csv, json, 图片(png/jpg/gif/webp)等
# ============================================================

import base64
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import pymysql
from dbutils.pooled_db import PooledDB
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {
    # 文本
    ".txt", ".md", ".markdown", ".csv", ".json", ".xml", ".html", ".htm",
    # 文档
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    # 图片
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg",
    # 代码
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go", ".rs",
    ".sh", ".bat", ".ps1", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    # 压缩
    ".zip", ".rar", ".7z", ".tar", ".gz",
}

# 单文件大小限制：10MB
MAX_FILE_SIZE = 10 * 1024 * 1024


class FileUploadService:
    """文件上传服务。

    文件内容以 base64 编码存储在 MySQL LONGTEXT 字段中。
    10MB 文件 base64 后约 13.3MB，在 LONGTEXT 最大 4GB 限制内。
    """

    def __init__(self):
        self._pool: Optional[PooledDB] = None

    def open(self):
        """初始化连接池。"""
        if self._pool:
            return
        db_url = os.getenv("MYSQL_DB_URL")
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
        logger.info("FileUploadService 连接池已初始化")

    def close(self, timeout: int = 5):
        """关闭连接池。"""
        if self._pool:
            self._pool.close()
            self._pool = None
            logger.info("FileUploadService 连接池已关闭")

    def _get_conn(self):
        if not self._pool:
            raise RuntimeError("FileUploadService 未初始化，请先调用 open()")
        return self._pool.connection()

    def _ensure_table(self):
        """确保 user_files 表存在。"""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_files (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(64) NOT NULL COMMENT '上传用户ID',
                    thread_id VARCHAR(128) DEFAULT NULL COMMENT '关联会话ID（NULL表示全局文件）',
                    file_name VARCHAR(255) NOT NULL COMMENT '原始文件名',
                    file_type VARCHAR(64) DEFAULT NULL COMMENT 'MIME类型',
                    file_ext VARCHAR(16) DEFAULT NULL COMMENT '文件扩展名',
                    file_size INT DEFAULT 0 COMMENT '文件大小（字节）',
                    file_content LONGTEXT COMMENT 'base64编码的文件内容',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_user (user_id),
                    INDEX idx_thread (thread_id),
                    INDEX idx_user_thread (user_id, thread_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户上传文件表';
            """)
            conn.commit()
        except pymysql.MySQLError as e:
            logger.error(f"创建 user_files 表失败: {e}")
            raise
        finally:
            conn.close()

    @staticmethod
    def validate_file(filename: str, file_size: int) -> Tuple[bool, str]:
        """验证文件是否允许上传。

        Args:
            filename: 文件名
            file_size: 文件大小（字节）

        Returns:
            (是否允许, 错误信息)
        """
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"不支持的文件格式: {ext}"
        if file_size > MAX_FILE_SIZE:
            return False, f"文件大小超过限制: {file_size // 1024 // 1024}MB > {MAX_FILE_SIZE // 1024 // 1024}MB"
        if file_size == 0:
            return False, "文件为空"
        return True, ""

    def save_file(self, user_id: str, file_name: str, file_content_bytes: bytes,
                   file_type: Optional[str] = None,
                   thread_id: Optional[str] = None) -> Dict[str, Any]:
        """保存用户上传的文件。

        Args:
            user_id: 用户 ID
            file_name: 原始文件名
            file_content_bytes: 文件内容（字节）
            file_type: MIME 类型
            thread_id: 关联会话 ID（None 表示全局文件）

        Returns:
            保存结果字典，含 file_id, file_name, file_size, file_ext

        Raises:
            ValueError: 文件验证失败
        """
        file_size = len(file_content_bytes)
        valid, error = self.validate_file(file_name, file_size)
        if not valid:
            raise ValueError(error)

        ext = os.path.splitext(file_name)[1].lower()
        # base64 编码
        content_b64 = base64.b64encode(file_content_bytes).decode("utf-8")

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO user_files (user_id, thread_id, file_name, file_type, file_ext, file_size, file_content)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (user_id, thread_id, file_name, file_type, ext, file_size, content_b64))
            file_id = cur.lastrowid
            conn.commit()
            logger.info(f"文件保存成功 file_id={file_id}, user_id={user_id}, size={file_size}")
            return {
                "file_id": file_id,
                "file_name": file_name,
                "file_size": file_size,
                "file_ext": ext,
                "file_type": file_type,
            }
        except pymysql.MySQLError as e:
            conn.rollback()
            logger.error(f"文件保存失败 user_id={user_id}: {e}")
            raise
        finally:
            conn.close()

    def get_file(self, file_id: int, user_id: str) -> Optional[Dict[str, Any]]:
        """获取文件信息（含 base64 内容）。

        Args:
            file_id: 文件 ID
            user_id: 用户 ID（用于权限校验）

        Returns:
            文件信息字典，不存在或无权限返回 None
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM user_files WHERE id = %s AND user_id = %s", (file_id, user_id))
            row = cur.fetchone()
            return row
        except pymysql.MySQLError as e:
            logger.error(f"获取文件失败 file_id={file_id}: {e}")
            raise
        finally:
            conn.close()

    def get_file_content_bytes(self, file_id: int, user_id: str) -> Optional[bytes]:
        """获取文件原始字节内容。

        Args:
            file_id: 文件 ID
            user_id: 用户 ID

        Returns:
            文件字节内容，不存在返回 None
        """
        row = self.get_file(file_id, user_id)
        if not row or not row.get("file_content"):
            return None
        return base64.b64decode(row["file_content"])

    def list_files(self, user_id: str, thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出用户的文件（不返回 file_content 大字段）。

        Args:
            user_id: 用户 ID
            thread_id: 会话 ID（None 表示列出所有，含全局和会话）

        Returns:
            文件信息列表（不含 file_content）
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            if thread_id is not None:
                cur.execute("""
                    SELECT id, user_id, thread_id, file_name, file_type, file_ext, file_size, created_at
                    FROM user_files WHERE user_id = %s AND thread_id = %s
                    ORDER BY created_at DESC
                """, (user_id, thread_id))
            else:
                cur.execute("""
                    SELECT id, user_id, thread_id, file_name, file_type, file_ext, file_size, created_at
                    FROM user_files WHERE user_id = %s
                    ORDER BY created_at DESC
                """, (user_id,))
            return cur.fetchall()
        except pymysql.MySQLError as e:
            logger.error(f"列出文件失败 user_id={user_id}: {e}")
            raise
        finally:
            conn.close()

    def delete_file(self, file_id: int, user_id: str) -> bool:
        """删除用户文件。

        Args:
            file_id: 文件 ID
            user_id: 用户 ID（权限校验）

        Returns:
            是否删除成功
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM user_files WHERE id = %s AND user_id = %s", (file_id, user_id))
            affected = cur.rowcount
            conn.commit()
            if affected > 0:
                logger.info(f"文件删除成功 file_id={file_id}, user_id={user_id}")
            return affected > 0
        except pymysql.MySQLError as e:
            conn.rollback()
            logger.error(f"文件删除失败 file_id={file_id}: {e}")
            raise
        finally:
            conn.close()

    def extract_text_from_file(self, file_id: int, user_id: str) -> Optional[str]:
        """从文件中提取文本内容（用于 RAG 上下文）。

        支持：txt, md, csv, json, xml, html 等纯文本格式直接读取。
        图片、PDF、docx 等二进制格式返回 None（需额外解析库）。

        Args:
            file_id: 文件 ID
            user_id: 用户 ID

        Returns:
            提取的文本内容，不支持的格式返回 None
        """
        row = self.get_file(file_id, user_id)
        if not row:
            return None
        ext = row.get("file_ext", "").lower()
        text_extensions = {".txt", ".md", ".markdown", ".csv", ".json", ".xml", ".html", ".htm",
                            ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go", ".rs",
                            ".sh", ".bat", ".ps1", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
        if ext in text_extensions:
            content_bytes = self.get_file_content_bytes(file_id, user_id)
            if content_bytes:
                try:
                    return content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        return content_bytes.decode("gbk")
                    except UnicodeDecodeError:
                        return None
        return None


# 模块级单例
file_upload_service = FileUploadService()
