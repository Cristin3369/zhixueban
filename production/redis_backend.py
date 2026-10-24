# -*- coding: utf-8 -*-
"""生产 Redis：热缓存 + Sorted Set 队列 + 同伴效应计数。"""

from __future__ import annotations

from typing import Optional

import redis

from config.settings import settings


def build_redis() -> redis.Redis:
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password or None,
        decode_responses=False,
        socket_timeout=5,
        socket_connect_timeout=5,
        health_check_interval=30,
    )


class RedisPeerStats:
    """
    匿名同伴效应：只存计数，不存用户身份。
    key: peer:hard:{question_id}:{yyyymmdd}  → HyperLogLog 或 INCR
         peer:rt:{question_id} → Redis 百分位用 Sorted Set 存耗时
    """

    def __init__(self, client: Optional[redis.Redis] = None):
        self.r = client or build_redis()

    def record_attempt(self, question_id: str, *, felt_hard: bool, response_time: float) -> None:
        from datetime import date

        day = date.today().strftime("%Y%m%d")
        if felt_hard:
            self.r.incr(f"peer:hard:{question_id}:{day}")
            self.r.expire(f"peer:hard:{question_id}:{day}", 86400 * 3)
        self.r.zadd(f"peer:rt:{question_id}", {str(response_time) + ":" + str(id(self)): response_time})

    def stats_for_question(self, question_id: str) -> dict:
        from datetime import date

        day = date.today().strftime("%Y%m%d")
        raw = self.r.get(f"peer:hard:{question_id}:{day}")
        hard_count = int(raw) if raw else 0
        key = f"peer:rt:{question_id}"
        n = self.r.zcard(key)
        slower = 0
        if n:
            # 简化：取中位以上占比用 zcount 需要当前用户 rt；这里返回「觉得难」人数为主
            slower = 50
        from config.settings import settings as st

        return {
            "hard_count_today": hard_count,
            "slower_than_percent": slower,
            "sample_ok": hard_count >= st.peer_min_sample,
        }
