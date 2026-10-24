# -*- coding: utf-8 -*-
"""
简易布隆过滤器
-------------
放在 L1 查询前：若 key 一定不存在，直接拦截，避免无效请求打穿到 DB / LLM。
注意：布隆过滤器有假阳性、无假阴性——「说不在则一定不在」。
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, List


class BloomFilter:
    """
    纯 Python 位数组实现，依赖少、便于教学与本地演示。
    生产可替换为 RedisBloom / pybloom-live。
    """

    def __init__(self, expected_items: int = 100_000, false_positive_rate: float = 0.01):
        if expected_items <= 0:
            raise ValueError("expected_items 必须 > 0")
        # 经典公式：m = -n*ln(p)/(ln2)^2 ，k = m/n * ln2
        m = -expected_items * math.log(false_positive_rate) / (math.log(2) ** 2)
        self.size = max(int(m), 1024)
        k = (self.size / expected_items) * math.log(2)
        self.hash_count = max(int(k), 1)
        self.bit_array: List[int] = [0] * self.size
        self._count = 0

    def _hashes(self, item: str) -> Iterable[int]:
        """双哈希模拟多个独立哈希函数（Kirsch-Mitzenmacher）。"""
        raw = item.encode("utf-8")
        h1 = int(hashlib.md5(raw).hexdigest(), 16)
        h2 = int(hashlib.sha1(raw).hexdigest(), 16) or 1
        for i in range(self.hash_count):
            yield (h1 + i * h2) % self.size

    def add(self, item: str) -> None:
        for idx in self._hashes(item):
            self.bit_array[idx] = 1
        self._count += 1

    def might_contain(self, item: str) -> bool:
        """True = 可能存在；False = 一定不存在。"""
        return all(self.bit_array[idx] for idx in self._hashes(item))

    def __contains__(self, item: str) -> bool:
        return self.might_contain(item)

    @property
    def count_approx(self) -> int:
        return self._count
