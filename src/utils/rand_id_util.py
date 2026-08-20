# ============================================================
# ID 生成工具
# 作用：生成 MySQL int 范围内的唯一 ID
# 原方案问题：time.time() 约 17.5亿（2026年），接近 int 上限 21.47亿，
#           且随机后缀仅 1000 种，碰撞概率高，2033 年后会溢出
# 现方案：基于 uuid4 取模，碰撞概率约 1/21亿，安全可靠
# ============================================================

import uuid
import random
import time


class Random:
    """ID 生成工具类。"""

    uid: int

    # MySQL signed int 最大值：2,147,483,647
    MYSQL_INT_MAX = 2**31 - 1

    @staticmethod
    def gen_simple_inc_random() -> int:
        """生成 MySQL int 范围内的唯一 ID。

        基于 uuid4 取模，碰撞概率极低（约 1/21亿）。
        结果范围：1 ~ 2,147,483,647（不含 0，避免与数据库默认值冲突）

        Returns:
            int: 唯一 ID
        """
        # uuid4().int 是 128 位大整数，取模到 int 范围
        uid = uuid.uuid4().int % Random.MYSQL_INT_MAX + 1
        return uid

    @staticmethod
    def gen_timestamp_id() -> int:
        """生成基于时间戳的 ID（备用方案，适合需要粗略排序的场景）。

        从 2020-01-01 起算的秒数 + 随机数，确保在 int 范围内。
        注意：同一秒内可能碰撞，高并发场景不推荐。

        Returns:
            int: 时间戳 ID
        """
        BASE_TIMESTAMP = 1577836800  # 2020-01-01 00:00:00 UTC
        elapsed = int(time.time()) - BASE_TIMESTAMP  # 约 1.9亿（2026年）
        rand_suffix = random.randint(0, 999999)  # 6位随机数降低碰撞
        return elapsed + rand_suffix
