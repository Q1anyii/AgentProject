# ============================================================
# JWT 工具单元测试
# 覆盖：密码哈希、JWT 签发与验证
# 运行：pytest tests/test_jwt_utils.py -v
# ============================================================

import os
import sys
import pytest
from datetime import datetime, timedelta, UTC

# 设置测试环境变量（必须在导入模块前设置）
os.environ["JWT_SECRET_KEY"] = "test_secret_key_for_unit_testing_only"
os.environ["JWT_ALGORITHM"] = "HS256"
os.environ["JWT_ACCESS_TOKEN_EXPIRE_MINUTES"] = "15"
os.environ["JWT_REFRESH_TOKEN_EXPIRE_DAYS"] = "30"

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.jwt_utils import (
    get_password_hash,
    verify_password,
    create_access_token,
    create_refresh_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
)
import jwt


class TestPasswordHash:
    """密码哈希测试。"""

    def test_hash_and_verify_success(self):
        """正常密码哈希与验证应通过。"""
        password = "test_password_123"
        hashed = get_password_hash(password)
        assert hashed != password  # 哈希后不应等于明文
        assert verify_password(password, hashed) is True

    def test_verify_wrong_password(self):
        """错误密码应验证失败。"""
        password = "correct_password"
        wrong_password = "wrong_password"
        hashed = get_password_hash(password)
        assert verify_password(wrong_password, hashed) is False

    def test_hash_is_salted(self):
        """相同密码两次哈希结果应不同（bcrypt 自动加盐）。"""
        password = "same_password"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)
        assert hash1 != hash2

    def test_long_password_truncation(self):
        """超过 72 字节的密码应被截断（bcrypt 限制）。"""
        long_password = "a" * 100
        hashed = get_password_hash(long_password)
        # 截断后仍能验证
        assert verify_password(long_password, hashed) is True
        # 前 72 字节相同的密码也能验证（因为被截断了）
        assert verify_password("a" * 72, hashed) is True

    def test_empty_password(self):
        """空密码应能哈希和验证。"""
        password = ""
        hashed = get_password_hash(password)
        assert verify_password(password, hashed) is True


class TestJWTCreation:
    """JWT 签发测试。"""

    def test_create_access_token(self):
        """签发 access token 应包含正确的 payload。"""
        data = {"sub": "user123:testuser", "role": "学员"}
        token = create_access_token(data=data)
        assert isinstance(token, str)
        assert len(token) > 0

        # 解码验证 payload
        payload = jwt.decode(token, "test_secret_key_for_unit_testing_only", algorithms=["HS256"])
        assert payload["sub"] == "user123:testuser"
        assert payload["role"] == "学员"
        assert payload["type"] == "access"
        assert "exp" in payload

    def test_create_refresh_token(self):
        """签发 refresh token 应包含 type=refresh。"""
        data = {"sub": "user123:testuser"}
        token = create_refresh_token(data=data)
        assert isinstance(token, str)

        payload = jwt.decode(token, "test_secret_key_for_unit_testing_only", algorithms=["HS256"])
        assert payload["type"] == "refresh"
        assert "exp" in payload

    def test_access_token_expiration(self):
        """access token 过期时间应约为 15 分钟后。"""
        data = {"sub": "user123:testuser"}
        before = datetime.now(UTC)
        token = create_access_token(data=data)
        after = datetime.now(UTC)

        payload = jwt.decode(token, "test_secret_key_for_unit_testing_only", algorithms=["HS256"])
        exp = datetime.fromtimestamp(payload["exp"], tz=UTC)

        # 过期时间应在 before+15min 和 after+15min 之间
        assert before + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES) <= exp <= after + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    def test_custom_expiration(self):
        """自定义过期时间应生效。"""
        data = {"sub": "user123:testuser"}
        custom_delta = timedelta(hours=2)
        token = create_access_token(data=data, expires_delta=custom_delta)

        payload = jwt.decode(token, "test_secret_key_for_unit_testing_only", algorithms=["HS256"])
        exp = datetime.fromtimestamp(payload["exp"], tz=UTC)
        now = datetime.now(UTC)

        # 应约为 2 小时后
        assert timedelta(hours=1.9) < (exp - now) < timedelta(hours=2.1)

    def test_token_not_modify_input(self):
        """签发 token 不应修改原始 data 字典。"""
        data = {"sub": "user123:testuser", "role": "学员"}
        original_data = data.copy()
        create_access_token(data=data)
        assert data == original_data  # 原始字典不应被修改


class TestJWTConfig:
    """JWT 配置默认值测试。"""

    def test_access_token_default_minutes(self):
        """access token 默认过期时间应为 15 分钟。"""
        assert ACCESS_TOKEN_EXPIRE_MINUTES == 15

    def test_refresh_token_default_days(self):
        """refresh token 默认过期时间应为 30 天。"""
        assert REFRESH_TOKEN_EXPIRE_DAYS == 30
