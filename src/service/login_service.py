import os
from pathlib import Path
from typing import Optional

from dbutils.pooled_db import PooledDB
import pymysql
from loguru import logger

from schemas.request_schemas.login_schema import LoginRequest
from schemas.response_schemas.login_schema import LoginResponse


class LoginService:
    # 环境变量 key 统一
    ENV_DB_URL = "MYSQL_DB_URL"
    persist_path: str | Path

    def __init__(self, db_url: Optional[str] = None):
        # 优先传入参数，其次读取环境变量
        self.db_url: Optional[str] = db_url or os.getenv(self.ENV_DB_URL)

        # 数据库连接池对象
        self._pool: Optional[PooledDB] = None

        # 【移除无关LangGraph变量,LoginService只负责登录数据库，不要混入graph、checkpointer】
        self.persist_path = ""

    def open(self) -> None:
        """初始化MySQL连接池，打开连接"""
        if not self.db_url:
            raise ValueError(f"数据库配置缺失，请设置环境变量 {self.ENV_DB_URL} 或者传入 db_url 参数")

        # 解析 mysql url: mysql+pymysql://root:1234@127.0.0.1:3306/Mitta?charset=utf8mb4
        # 去掉前缀 mysql+pymysql://
        prefix = "mysql+pymysql://"
        if self.db_url.startswith(prefix):
            dsn = self.db_url[len(prefix):]
        else:
            dsn = self.db_url

        user_pass, host_db = dsn.split("@")
        user_name, password = user_pass.split(":")
        host_port, database = host_db.split("/")
        host, port_str = host_port.split(":")
        port = int(port_str)

        self._pool = PooledDB(
            creator=pymysql,
            mincached=1,  # 最小空闲连接
            maxcached=10,  # 最大空闲连接
            maxconnections=10,  # 总最大连接
            blocking=False,  # 拿不到连接直接抛异常，不阻塞等待
            host=host,
            port=port,
            user=user_name,
            password=password,
            database=database,
            charset="utf8mb4",
            autocommit=True,
            cursorclass=pymysql.cursors.DictCursor,
        )
        # 拿一条连接测试连通性
        try:
            conn = self._pool.connection()
            conn.close()
            logger.success("MySQL连接池初始化成功")
        except Exception as e:
            logger.error(f"MySQL数据库连接失败：{e}")
            raise

    def get_connection(self):
        """从池中获取连接"""
        if self._pool is None:
            raise RuntimeError("请先调用 open() 初始化连接池")
        return self._pool.connection()

    def close(self, timeout: int = 10) -> None:
        """关闭连接池，释放全部资源"""
        if self._pool:
            self._pool.close()
            self._pool = None
            logger.info("MySQL连接池已关闭")

    # 支持 with 上下文管理器
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def login(self, user_id, password)  :
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM userInfo WHERE user_id=%s AND password=%s",
                (user_id, password)
            )
            user_info = cur.fetchone()
            if user_info:
                return user_info
        except pymysql.OperationalError as e:
            logger.error(f"数据库连接异常 {e}")
            raise
        except pymysql.ProgrammingError as e:
            logger.error(f"SQL错误 {e}")
            raise
        except pymysql.MySQLError as e:
            logger.error(f"数据库执行异常 {e}")
            raise





# ---------------- 业务示例 登录查询用户 ----------------
if __name__ == "__main__":
    # .env 文件配置： MYSQL_DB_URL=mysql+pymysql://root:1234@127.0.0.1:3306/Mitta
    from dotenv import load_dotenv
    load_dotenv()

    with LoginService() as service:
        conn = service.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM userInfo WHERE user_id=%s", ("user_01",))
        user = cursor.fetchone()
        print(user)
        cursor.close()
        conn.close()
