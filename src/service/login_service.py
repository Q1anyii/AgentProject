import datetime
import os
import random
from pathlib import Path
from typing import Optional
from utils.rand_id_util import Random
from dbutils.pooled_db import PooledDB
import pymysql
from loguru import logger
from datetime import datetime

from utils.jwt_utils import verify_password, get_password_hash


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

    def login(self, user_id, password):
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            # 修复：参数必须以元组形式传递 (user_id,)，不能直接传字符串
            cur.execute(
                "SELECT * FROM userInfo WHERE user_id=%s",
                (user_id,)
            )
            user_info = cur.fetchone()
            if user_info:
                if verify_password(password, user_info["password"]):
                    return user_info
                else:
                    return "密码错误"
            else:
                return f"用户{user_id}不存在"
        except pymysql.OperationalError as e:
            logger.error(f"数据库连接异常 {e}")
            raise
        except pymysql.ProgrammingError as e:
            logger.error(f"SQL错误 {e}")
            raise
        except pymysql.MySQLError as e:
            logger.error(f"数据库执行异常 {e}")
            raise

    def get_user_by_id(self, user_id):
        """按用户 ID 查询用户信息（聊天接口 JWT 认证后获取当前用户）"""
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM userInfo WHERE user_id=%s", (user_id,))
            return cur.fetchone()
        except pymysql.MySQLError as e:
            logger.error(f"数据库执行异常 {e}")
            raise
        finally:
            conn.close()

    def register(self, username, user_id, password):
        """用户注册。

        Args:
            username: 用户名
            user_id: 用户 ID
            password: 明文密码（内部会 bcrypt 哈希）

        Returns:
            tuple: (flag, response)，flag=True 表示成功
        """
        conn = self.get_connection()
        create_time = datetime.now()
        update_time = datetime.now()
        flag = False
        try:
            cur = conn.cursor()
            sql = "SELECT user_id FROM userInfo WHERE user_id=%s"
            # 修复：参数必须以元组形式传递 (user_id,)
            if cur.execute(sql, (user_id,)):
                return flag, "该用户已存在"
            password = get_password_hash(password)
            # 显式列出 6 个占位符，对应 (id, user_id, password, username, create_time, update_time)
            sql = "INSERT into userinfo(id, user_id, password, username, create_time, update_time) VALUES (%s, %s, %s, %s, %s, %s)"
            success = cur.execute(sql, (Random.gen_simple_inc_random(), user_id, password, username, create_time, update_time))
            flag = True
            return flag, success
        except pymysql.MySQLError as e:
            logger.error(f"数据库执行异常 {e}")
            return flag, f"数据库执行异常 {e}"
        finally:
            conn.close()

    def recover(self, user_id, new_password):
        update_time = datetime.now()
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            sql = "SELECT user_id, password FROM userInfo WHERE user_id = %s"
            # 修复：参数必须以元组形式传递 (user_id,)
            exist = cur.execute(sql, (user_id,))
            if not exist:
                return f"用户{user_id}不存在"
            user_info = cur.fetchone()
            user_id = user_info["user_id"]
            old_password = user_info["password"]
            if verify_password(new_password, old_password):
                return f"密码不可与原密码相同"
            new_password = get_password_hash(new_password)
            sql = "UPDATE userInfo SET password = %s, update_time = %s WHERE user_id = %s"
            success = cur.execute(sql, (new_password, update_time, user_id))
            return success
        except pymysql.MySQLError as e:
            logger.error(f"数据库执行异常 {e}, 联系管理员")
            raise
        finally:
            conn.close()

login_service = LoginService()

# ---------------- 业务示例 登录查询用户 ----------------
if __name__ == "__main__":
    # .env 文件配置： MYSQL_DB_URL=mysql+pymysql://root:1234@127.0.0.1:3306/Mitta
    from dotenv import load_dotenv
    load_dotenv()

    with LoginService() as service:
        conn = service.get_connection()
        cursor = conn.cursor()
        execute = cursor.execute("SELECT * FROM userInfo WHERE user_id=%s", ("user",))
        user = cursor.fetchone()
        print(user)
        print(execute)
        cursor.close()
        conn.close()
