# -*- coding: utf-8 -*-
"""
调用包裹：超时 / 重试 / 熔断 / 降级
--------------------------------
任何外部模型调用必须走这里，禁止裸调 LLM。
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

from config.settings import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 3
    recover_sec: float = 60.0
    failures: int = 0
    open_until: float = 0.0

    def allow(self) -> bool:
        return time.time() >= self.open_until

    def record_ok(self) -> None:
        self.failures = 0
        self.open_until = 0.0

    def record_fail(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open_until = time.time() + self.recover_sec
            logger.warning(
                "熔断打开: %s failures=%s recover=%.0fs",
                self.name,
                self.failures,
                self.recover_sec,
            )


@dataclass
class GuardResult:
    ok: bool
    value: Any
    degraded: bool = False
    error: str = ""
    source: str = "primary"


def run_with_timeout(fn: Callable[[], T], timeout_sec: float) -> T:
    # 线程超时：Windows 上可用；超时后函数可能仍在跑，但调用方不再等待
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(fn)
        try:
            return fut.result(timeout=timeout_sec)
        except FuturesTimeout as exc:
            raise TimeoutError(f"调用超时 {timeout_sec}s") from exc


def guarded_call(
    fn: Callable[[], T],
    *,
    fallback: Any,
    breaker: Optional[CircuitBreaker] = None,
    timeout_sec: Optional[float] = None,
    retries: int = 0,
    name: str = "call",
) -> GuardResult:
    """
    主路径失败 → 可选重试 → 熔断 → fallback。
    fallback 可以是值或零参 callable。
    """
    timeout_sec = timeout_sec if timeout_sec is not None else settings.llm_timeout_sec

    def _fallback() -> Any:
        return fallback() if callable(fallback) else fallback

    if breaker is not None and not breaker.allow():
        logger.warning("熔断中，直接降级: %s", name)
        return GuardResult(ok=True, value=_fallback(), degraded=True, source="circuit_open")

    last_err = ""
    attempts = 1 + max(0, retries)
    for i in range(attempts):
        try:
            value = run_with_timeout(fn, timeout_sec)
            if breaker is not None:
                breaker.record_ok()
            return GuardResult(ok=True, value=value, degraded=False, source="primary")
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            logger.warning("%s 第 %s 次失败: %s", name, i + 1, exc)
            if breaker is not None:
                breaker.record_fail()

    return GuardResult(
        ok=False,
        value=_fallback(),
        degraded=True,
        error=last_err,
        source="fallback",
    )
