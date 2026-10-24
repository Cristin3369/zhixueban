# -*- coding: utf-8 -*-
"""
复习任务队列（Redis Sorted Set）
------------------------------
Score = 到期时间戳；同到期时按任务类型优先级排序。
与多因子艾宾浩斯输出对接；支持精力差时顺延（防沉迷）。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class RedisZSetLike(Protocol):
    def zadd(self, name: str, mapping: Dict[Any, float]) -> Any: ...
    def zrangebyscore(self, name: str, min: Any, max: Any) -> List[Any]: ...
    def zrem(self, name: str, *values: Any) -> Any: ...


class InMemoryZSet:
    """无 Redis 时的内存 ZSET 模拟。"""

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, float]] = {}

    def zadd(self, name: str, mapping: Dict[Any, float]) -> int:
        bucket = self._data.setdefault(name, {})
        for member, score in mapping.items():
            key = member if isinstance(member, str) else json.dumps(member, ensure_ascii=False)
            bucket[key] = float(score)
        return len(mapping)

    def zrangebyscore(self, name: str, min_s: Any, max_s: Any) -> List[str]:
        bucket = self._data.get(name, {})
        lo = float("-inf") if min_s == "-inf" else float(min_s)
        hi = float("inf") if max_s == "+inf" else float(max_s)
        items = [(m, s) for m, s in bucket.items() if lo <= s <= hi]
        items.sort(key=lambda x: x[1])
        return [m for m, _ in items]

    def zrem(self, name: str, *values: Any) -> int:
        bucket = self._data.get(name, {})
        n = 0
        for v in values:
            key = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            if key in bucket:
                del bucket[key]
                n += 1
        return n


class StudyTaskQueue:
    """
    任务类型优先级（数值越小越高）：
    - high_freq_focus: 高频考点强化
    - chapter_review: 宏观章节
    - error_book: 错题本
    - due_review: 普通到期复习
    """

    TASK_PRIORITY = {
        "high_freq_focus": 0,
        "chapter_review": 1,
        "error_book": 2,
        "due_review": 3,
    }

    def __init__(self, redis_client: Optional[RedisZSetLike] = None, queue_name: str = "study:task_queue"):
        self.redis = redis_client or InMemoryZSet()
        self.queue = queue_name

    def add_task(
        self,
        student_id: str,
        task_type: str,
        due_timestamp: float,
        *,
        question_id: str = "",
        subject: str = "",
        session_id: str = "",
        meta: Optional[dict] = None,
    ) -> str:
        """
        全局队列写入，但必须打上 subject 标签。
        看板按科目聚合；会话内只消费本科目任务。
        """
        payload = {
            "student_id": student_id,
            "task_type": task_type,
            "due_timestamp": float(due_timestamp),
            "original_due": float(due_timestamp),
            "question_id": question_id,
            "subject": subject,
            "session_id": session_id,
            "meta": meta or {},
        }
        raw = json.dumps(payload, ensure_ascii=False)
        self.redis.zadd(self.queue, {raw: float(due_timestamp)})
        return raw

    def get_due_tasks(
        self,
        student_id: str,
        now: Optional[float] = None,
        *,
        subject: Optional[str] = None,
    ) -> List[dict]:
        now = now if now is not None else time.time()
        due = self.redis.zrangebyscore(self.queue, "-inf", now)
        student_tasks: List[dict] = []
        for item in due:
            raw = item.decode("utf-8") if isinstance(item, bytes) else item
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if payload.get("student_id") != student_id:
                continue
            if subject and payload.get("subject") and payload.get("subject") != subject:
                continue
            student_tasks.append(payload)
        student_tasks.sort(key=lambda x: self.TASK_PRIORITY.get(x.get("task_type", ""), 99))
        return student_tasks

    def get_next_task(self, student_id: str, *, subject: Optional[str] = None) -> Optional[dict]:
        tasks = self.get_due_tasks(student_id, subject=subject)
        return tasks[0] if tasks else None

    def count_by_subject(self, student_id: str, now: Optional[float] = None) -> Dict[str, Dict[str, int]]:
        """
        看板用：按科目统计到期任务类型数量。
        返回 {subject: {due_review: n, error_book: n, high_freq_focus: n, total: n}}
        """
        tasks = self.get_due_tasks(student_id, now=now)
        out: Dict[str, Dict[str, int]] = {}
        for t in tasks:
            subj = t.get("subject") or "未分科"
            bucket = out.setdefault(
                subj,
                {"due_review": 0, "error_book": 0, "high_freq_focus": 0, "chapter_review": 0, "total": 0},
            )
            typ = t.get("task_type", "due_review")
            if typ not in bucket:
                bucket[typ] = 0
            bucket[typ] += 1
            bucket["total"] += 1
        return out

    def defer_task(self, task_payload: dict, defer_seconds: float = 3600) -> dict:
        """顺延：高优任务占用时，或防沉迷触发时调用。"""
        old_raw = json.dumps(task_payload, ensure_ascii=False)
        try:
            self.redis.zrem(self.queue, old_raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("defer zrem failed: %s", exc)

        new_due = time.time() + defer_seconds
        task_payload = dict(task_payload)
        task_payload["due_timestamp"] = new_due
        self.redis.zadd(self.queue, {json.dumps(task_payload, ensure_ascii=False): new_due})
        logger.info("[队列] 任务 %s/%s 已顺延 %ss", task_payload.get("subject"), task_payload.get("task_type"), defer_seconds)
        return task_payload

    def defer_all_today(
        self,
        student_id: str,
        defer_seconds: float = 3600 * 24,
        *,
        subject: Optional[str] = None,
    ) -> int:
        """防沉迷：可按科目顺延，或全局顺延。"""
        tasks = self.get_due_tasks(student_id, subject=subject)
        for t in tasks:
            self.defer_task(t, defer_seconds=defer_seconds)
        return len(tasks)
