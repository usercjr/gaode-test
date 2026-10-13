"""容错原语：熔断器与异步重试。"""
import asyncio
import random
import time
from typing import Awaitable, Callable, Optional, TypeVar

from core.errors import AppError, ErrorCode
from core.logging import get_logger

T = TypeVar("T")
log = get_logger("resilience")

# 单独引用，便于测试替换而不影响全局 asyncio.sleep
_sleep = asyncio.sleep
_now = time.monotonic


class CircuitOpenError(AppError):
    """熔断器打开时的拒绝。错误码沿用对应服务的“暂时不可用”（对用户而言可稍后重试），
    但重试逻辑必须立即放弃：冷却期内再试也一定被拒，退避等待只会拖慢失败（压测中实测多耗 1~1.5 秒）。"""

    def __init__(self, code: ErrorCode, breaker: str):
        super().__init__(code, detail=f"circuit open: {breaker}")


class CircuitBreaker:
    """连续失败达到阈值后断开，冷却期内直接拒绝；冷却结束放行一次试探请求（半开）。"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout_s: float = 30.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._half_open_in_flight = False

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return self.CLOSED
        if _now() - self._opened_at >= self.recovery_timeout_s:
            return self.HALF_OPEN
        return self.OPEN

    def allow(self) -> bool:
        state = self.state
        if state == self.CLOSED:
            return True
        if state == self.HALF_OPEN and not self._half_open_in_flight:
            self._half_open_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        if self._opened_at is not None:
            log.info("circuit closed", extra={"breaker": self.name})
        self._failures = 0
        self._opened_at = None
        self._half_open_in_flight = False

    def record_failure(self) -> None:
        self._half_open_in_flight = False
        self._failures += 1
        if self._opened_at is not None or self._failures >= self.failure_threshold:
            if self._opened_at is None:
                log.warning("circuit opened", extra={"breaker": self.name, "failures": self._failures})
            self._opened_at = _now()

    def snapshot(self) -> dict:
        return {"name": self.name, "state": self.state, "consecutive_failures": self._failures}


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay_s: float = 0.5,
    max_delay_s: float = 8.0,
    op: str = "op",
) -> T:
    """只重试 retryable 的 AppError；指数退避 + 抖动。"""
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except AppError as e:
            if not e.retryable or attempt == attempts or isinstance(e, CircuitOpenError):
                raise
            delay = min(max_delay_s, base_delay_s * 2 ** (attempt - 1)) * random.uniform(0.5, 1.5)
            log.warning(
                "retrying",
                extra={"op": op, "attempt": attempt, "code": e.code.value, "delay_s": round(delay, 2)},
            )
            await _sleep(delay)
    raise AppError(ErrorCode.INTERNAL, "unreachable")  # pragma: no cover
