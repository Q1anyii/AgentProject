# ============================================================
# ID 生成工具单元测试
# 覆盖：gen_simple_inc_random、gen_timestamp_id
# 运行：pytest tests/test_rand_id_util.py -v
# ============================================================

import os
import sys
import pytest

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.rand_id_util import Random


class TestGenSimpleIncRandom:
    """gen_simple_inc_random 测试。"""

    def test_returns_int(self):
        """应返回整数。"""
        uid = Random.gen_simple_inc_random()
        assert isinstance(uid, int)

    def test_positive(self):
        """应返回正整数（不含 0）。"""
        uid = Random.gen_simple_inc_random()
        assert uid > 0

    def test_within_mysql_int_range(self):
        """应在 MySQL signed int 范围内（1 ~ 2,147,483,647）。"""
        for _ in range(100):
            uid = Random.gen_simple_inc_random()
            assert 1 <= uid <= 2**31 - 1, f"ID {uid} 超出 MySQL int 范围"

    def test_uniqueness_large_sample(self):
        """大样本下应几乎不重复（碰撞概率约 1/21亿）。"""
        ids = set()
        for _ in range(10000):
            uid = Random.gen_simple_inc_random()
            ids.add(uid)
        # 10000 个 ID 中重复数应极少（理论上几乎为 0）
        assert len(ids) > 9990, f"重复过多：生成 10000 个，唯一 {len(ids)} 个"

    def test_not_zero(self):
        """不应返回 0（避免与数据库默认值冲突）。"""
        for _ in range(100):
            assert Random.gen_simple_inc_random() != 0


class TestGenTimestampId:
    """gen_timestamp_id 测试。"""

    def test_returns_int(self):
        """应返回整数。"""
        uid = Random.gen_timestamp_id()
        assert isinstance(uid, int)

    def test_positive(self):
        """应返回正整数。"""
        uid = Random.gen_timestamp_id()
        assert uid > 0

    def test_within_mysql_int_range(self):
        """应在 MySQL signed int 范围内。"""
        for _ in range(100):
            uid = Random.gen_timestamp_id()
            assert 1 <= uid <= 2**31 - 1, f"ID {uid} 超出 MySQL int 范围"

    def test_roughly_increasing(self):
        """连续生成的 ID 应大致递增（基于时间戳）。"""
        id1 = Random.gen_timestamp_id()
        id2 = Random.gen_timestamp_id()
        # 由于有随机后缀，不一定严格递增，但应在相近范围内
        assert abs(id2 - id1) < 1000000, f"连续 ID 差距过大：{id1} -> {id2}"


class TestRandomClass:
    """Random 类常量测试。"""

    def test_mysql_int_max_constant(self):
        """MYSQL_INT_MAX 应为 2^31 - 1。"""
        assert Random.MYSQL_INT_MAX == 2**31 - 1
        assert Random.MYSQL_INT_MAX == 2147483647
