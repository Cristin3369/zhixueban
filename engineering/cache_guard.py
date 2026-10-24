# -*- coding: utf-8 -*-
"""
缓存穿透 / 击穿 / 雪崩 防护
-------------------------
- 空值缓存：L1~L4 都未命中时缓存短 TTL 空结果，防恶意穿透
- 互斥锁：重建热点时只允许一个 worker 回源，其余短暂等待或读旧值
- 配合布隆过滤器使用（见 bloom_filter.py）
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional


class CacheGuard:
    """进程内互斥锁 + 空值标记辅助。分布式场景可换成 Redis SETNX。"""

    NULL_MARKER = "__NULL__"

    def __init__(self, lock_ttl_seconds: float = 5.0):
        self._locks: dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()
        self.lock_ttl = lock_ttl_seconds

    def _get_lock(self, key: str) -> threading.Lock:
        with self._meta_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def with_singleflight(
        self,
        key: str,
        loader: Callable[[], Any],
        waiter_sleep: float = 0.05,
        max_wait: float = 2.0,
    ) -> Any:
        """
        Singleflight：同一 key 并发回源时，只有一个执行 loader。
        其他线程自旋短暂等待（生产可改为订阅/条件变量）。
        """
        lock = self._get_lock(key)
        acquired = lock.acquire(blocking=False)
        if acquired:
            try:
                return loader()
            finally:
                lock.release()

        # 未抢到锁：等待持有者完成（简化版）
        deadline = time.time() + max_wait
        while time.time() < deadline:
            if lock.acquire(blocking=False):
                lock.release()
                break
            time.sleep(waiter_sleep)
        # 等待结束后由调用方自行再读缓存；这里返回 None 表示「请重试读缓存」
        return None

    @classmethod
    def is_null(cls, value: Any) -> bool:
        return value == cls.NULL_MARKER
