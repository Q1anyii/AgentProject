# ============================================================
# 请求限流中间件
# 作用：对 /api/chat/ 等消耗 LLM 配额的接口进行速率限制
# 实现：基于 Redis 计数器（固定窗口），Redis 不可用时降级为内存限流
# 使用：在 main.py 中通过 app.add_middleware(RateLimitMiddleware) 注册
# ============================================================

import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

# 需要限流的路径前缀
RATE_LIMITED_PATHS = ["/api/chat/"]

# 限流配置：每个用户/IP 在时间窗口内最多请求次数
RATE_LIMIT_MAX_REQUESTS = 30  # 每窗口最多 30 次
RATE_LIMIT_WINDOW_SECONDS = 60  # 时间窗口 60 秒


class RateLimitMiddleware(BaseHTTPMiddleware):
    """基于 Redis 的固定窗口限流中间件。

    限流键：使用客户端 IP（完整实现可在鉴权后注入 user_id 到 request.state）。
    Redis 可用时用 Redis 计数器（支持多 worker 共享）；不可用时降级为内存限流。
    """

    def __init__(self, app):
        super().__init__(app)
        # 内存限流降级存储：key -> deque[timestamp]
        self._memory_store: Dict[str, Deque[float]] = defaultdict(deque)

    def _get_limit_key(self, request: Request) -> str:
        """获取限流键：使用客户端 IP。"""
        client_ip = request.client.host if request.client else "unknown"
        return f"rate_limit:{client_ip}"

    def _check_redis(self, key: str) -> bool:
        """使用 Redis 进行限流检查。

        Returns:
            True 表示允许通过，False 表示被限流
        """
        try:
            from service.cache_service import cache_service
            r = cache_service.redis
            if not r:
                return self._check_memory(key)

            # 固定窗口：INCR + EXPIRE
            count = r.incr(key)
            if count == 1:
                r.expire(key, RATE_LIMIT_WINDOW_SECONDS)

            if count > RATE_LIMIT_MAX_REQUESTS:
                return False
            return True
        except Exception as e:
            logger.warning(f"Redis 限流失败，降级为内存限流：{e}")
            return self._check_memory(key)

    def _check_memory(self, key: str) -> bool:
        """内存限流（降级方案，单进程有效）。

        使用滑动窗口：维护一个时间戳队列，移除窗口外的请求，检查当前数量。
        """
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW_SECONDS
        timestamps = self._memory_store[key]

        # 移除窗口外的时间戳
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            return False

        timestamps.append(now)
        return True

    async def dispatch(self, request: Request, call_next):
        """中间件主逻辑。"""
        path = request.url.path

        # 只对指定路径限流
        if not any(path.startswith(p) for p in RATE_LIMITED_PATHS):
            return await call_next(request)

        key = self._get_limit_key(request)
        allowed = self._check_redis(key)

        if not allowed:
            logger.warning(f"请求被限流 | path={path} | key={key}")
            return Response(
                content='{"ok": false, "detail": "请求过于频繁，请稍后再试"}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
            )

        return await call_next(request)
