# ============================================================
# 配置管理模块单元测试
# 覆盖：get_env、get_env_int、get_env_bool、validate_config
# 运行：pytest tests/test_config.py -v
# ============================================================

import os
import sys
import pytest

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import (
    get_env,
    get_env_int,
    get_env_bool,
    validate_config,
    ConfigError,
    REQUIRED_ENV_VARS,
)


class TestGetEnv:
    """get_env 测试。"""

    def test_existing_var(self):
        """存在的环境变量应返回其值。"""
        os.environ["TEST_VAR_EXISTS"] = "hello"
        assert get_env("TEST_VAR_EXISTS") == "hello"
        del os.environ["TEST_VAR_EXISTS"]

    def test_missing_var_with_default(self):
        """不存在的环境变量应返回默认值。"""
        assert get_env("NONEXISTENT_VAR_12345", "default_value") == "default_value"

    def test_missing_var_no_default(self):
        """不存在且无默认值应返回 None。"""
        assert get_env("NONEXISTENT_VAR_12345") is None

    def test_empty_string_var(self):
        """空字符串环境变量应返回空字符串（而非 None）。"""
        os.environ["TEST_EMPTY_VAR"] = ""
        assert get_env("TEST_EMPTY_VAR") == ""
        del os.environ["TEST_EMPTY_VAR"]


class TestGetEnvInt:
    """get_env_int 测试。"""

    def test_valid_int(self):
        """有效整数字符串应返回对应整数。"""
        os.environ["TEST_INT_VAR"] = "42"
        assert get_env_int("TEST_INT_VAR", 0) == 42
        del os.environ["TEST_INT_VAR"]

    def test_missing_var_returns_default(self):
        """不存在的变量应返回默认值。"""
        assert get_env_int("NONEXISTENT_INT_VAR", 100) == 100

    def test_invalid_int_returns_default(self):
        """非整数字符串应返回默认值（不抛异常）。"""
        os.environ["TEST_INVALID_INT"] = "not_a_number"
        assert get_env_int("TEST_INVALID_INT", 50) == 50
        del os.environ["TEST_INVALID_INT"]

    def test_negative_int(self):
        """负整数应正确解析。"""
        os.environ["TEST_NEG_INT"] = "-10"
        assert get_env_int("TEST_NEG_INT", 0) == -10
        del os.environ["TEST_NEG_INT"]

    def test_zero_int(self):
        """0 应正确解析（不是默认值）。"""
        os.environ["TEST_ZERO_INT"] = "0"
        assert get_env_int("TEST_ZERO_INT", 999) == 0
        del os.environ["TEST_ZERO_INT"]


class TestGetEnvBool:
    """get_env_bool 测试。"""

    @pytest.mark.parametrize("value,expected", [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("YES", True),
        ("on", True),
        ("ON", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("NO", False),
        ("off", False),
        ("OFF", False),
    ])
    def test_various_bool_values(self, value, expected):
        """各种布尔值字符串应正确解析。"""
        os.environ["TEST_BOOL_VAR"] = value
        assert get_env_bool("TEST_BOOL_VAR", False) == expected
        del os.environ["TEST_BOOL_VAR"]

    def test_missing_var_returns_default(self):
        """不存在的变量应返回默认值。"""
        assert get_env_bool("NONEXISTENT_BOOL_VAR", True) is True
        assert get_env_bool("NONEXISTENT_BOOL_VAR", False) is False

    def test_invalid_bool_returns_default(self):
        """非布尔值字符串应返回默认值。"""
        os.environ["TEST_INVALID_BOOL"] = "maybe"
        assert get_env_bool("TEST_INVALID_BOOL", True) is False  # "maybe" 不在真值列表中
        del os.environ["TEST_INVALID_BOOL"]


class TestValidateConfig:
    """validate_config 测试。"""

    def test_all_required_present(self):
        """所有必填变量存在时应不抛异常。"""
        # 保存原始环境变量
        original_env = {}
        for key, _ in REQUIRED_ENV_VARS:
            original_env[key] = os.environ.get(key)
            os.environ[key] = "test_value"

        try:
            validate_config()  # 不应抛异常
        finally:
            # 恢复原始环境变量
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_missing_required_raises(self):
        """缺少必填变量时应抛出 ConfigError。"""
        # 保存并清除所有必填变量
        original_env = {}
        for key, _ in REQUIRED_ENV_VARS:
            original_env[key] = os.environ.get(key)
            os.environ.pop(key, None)

        try:
            with pytest.raises(ConfigError) as exc_info:
                validate_config()
            # 错误信息应包含缺失的变量名
            error_msg = str(exc_info.value)
            for key, _ in REQUIRED_ENV_VARS:
                assert key in error_msg
        finally:
            # 恢复原始环境变量
            for key, value in original_env.items():
                if value is not None:
                    os.environ[key] = value

    def test_empty_required_value_raises(self):
        """必填变量值为空字符串时应视为缺失。"""
        # 保存原始环境变量
        original_env = {}
        for key, _ in REQUIRED_ENV_VARS:
            original_env[key] = os.environ.get(key)
            os.environ[key] = "  "  # 空白字符串

        try:
            with pytest.raises(ConfigError):
                validate_config()
        finally:
            # 恢复原始环境变量
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class TestConfigError:
    """ConfigError 异常测试。"""

    def test_is_exception(self):
        """ConfigError 应是 Exception 的子类。"""
        assert issubclass(ConfigError, Exception)

    def test_can_raise_and_catch(self):
        """应能正常抛出和捕获。"""
        with pytest.raises(ConfigError):
            raise ConfigError("test error")
