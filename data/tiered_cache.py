# -*- coding: utf-8 -*-
"""
四级分层数据存放
---------------
L1: 艾宾浩斯近期待学/复习 + 考试频率高需重点掌握（热）
L2: 已掌握 / 远期复习（温）
L3: 用户进页选定的科目与指定书本全量知识（会话/用户绑定）
L4: 根据科目扩展收集的其他教辅/书本知识（冷）

检索顺序：布隆 → L1 → L2 → L3 → L4 → 同构题生成 → 拒绝兜底
每一层都可配合混合检索+重排；层间置信度不够才下沉。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, Optional, Protocol

from config.settings import settings
from data.models import CacheLookupResult, QAPair, UserProfile
from engineering.bloom_filter import BloomFilter
from engineering.cache_guard import CacheGuard

logger = logging.getLogger(__name__)


class RedisLike(Protocol):
    def get(self, key: str) -> Optional[bytes]: ...
    def setex(self, key: str, time: int, value: Any) -> Any: ...
    def delete(self, *keys: str) -> Any: ...


class InMemoryKV:
    """无 Redis 时的内存替代，保证本地可跑。"""

    def __init__(self) -> None:
        self._store: Dict[str, tuple[float, str]] = {}  # key -> (expire_at, value)

    def get(self, key: str) -> Optional[bytes]:
        item = self._store.get(key)
        if not item:
            return None
        expire_at, value = item
        if expire_at and time.time() > expire_at:
            self._store.pop(key, None)
            return None
        return value.encode("utf-8")

    def setex(self, key: str, ttl: int, value: Any) -> None:
        raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        expire_at = time.time() + ttl if ttl else 0
        self._store[key] = (expire_at, raw)

    def delete(self, *keys: str) -> None:
        for k in keys:
            self._store.pop(k, None)


class TieredKnowledgeStore:
    """
    四级知识存放 + 防穿透。
    真实 DB / 向量库通过可选注入的 loader 回调对接。
    """

    def __init__(
        self,
        redis_client: Optional[RedisLike] = None,
        bloom: Optional[BloomFilter] = None,
        # 各层回源加载器（生产接 Milvus / PG / 对象存储）
        l2_loader: Optional[Callable[[str, str], Optional[dict]]] = None,
        l3_loader: Optional[Callable[[str, str], Optional[dict]]] = None,
        l4_loader: Optional[Callable[[str, str], Optional[dict]]] = None,
        # 同构题生成：输入知识点实体 dict，输出题面 dict
        isomorphic_generator: Optional[Callable[[dict], dict]] = None,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ):
        self.redis: RedisLike = redis_client or InMemoryKV()
        self.bloom = bloom or BloomFilter(expected_items=200_000, false_positive_rate=0.01)
        self.guard = CacheGuard()
        self.l2_loader = l2_loader
        self.l3_loader = l3_loader
        self.l4_loader = l4_loader
        self.isomorphic_generator = isomorphic_generator
        self.similarity_fn = similarity_fn or self._jaccard_sim
        self.confidence_threshold = settings.confidence_threshold
        self.isomorphic_sim_threshold = settings.isomorphic_sim_threshold

    # ------------------------------------------------------------------ #
    # Key 设计：带 user / track / version，硬隔离防串数据
    # ------------------------------------------------------------------ #
    @staticmethod
    def key_l1(user_id: str, qid: str) -> str:
        return f"L1:review:{user_id}:{qid}"

    @staticmethod
    def key_l2(user_id: str, qid: str) -> str:
        return f"L2:mastered:{user_id}:{qid}"

    @staticmethod
    def key_l3(user_id: str, qid: str) -> str:
        return f"L3:curriculum:{user_id}:{qid}"

    @staticmethod
    def key_l4(track: str, qid: str) -> str:
        return f"L4:extend:{track}:{qid}"

    @staticmethod
    def key_null(qid: str) -> str:
        return f"NULL:{qid}"

    @staticmethod
    def key_user_profile(user_id: str) -> str:
        return f"profile:{user_id}"

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #
    def register_known_id(self, question_id: str) -> None:
        """入库时写入布隆，供查询前拦截。"""
        self.bloom.add(question_id)

    def put_l1(self, user_id: str, qid: str, payload: dict) -> None:
        self.register_known_id(qid)
        self.redis.setex(self.key_l1(user_id, qid), settings.cache_ttl_l1, self._dumps(payload))

    def put_l2(self, user_id: str, qid: str, payload: dict) -> None:
        self.register_known_id(qid)
        self.redis.setex(self.key_l2(user_id, qid), settings.cache_ttl_l2, self._dumps(payload))

    def put_l3(self, user_id: str, qid: str, payload: dict) -> None:
        self.register_known_id(qid)
        self.redis.setex(self.key_l3(user_id, qid), settings.cache_ttl_l3, self._dumps(payload))

    def put_l4(self, track: str, qid: str, payload: dict) -> None:
        self.register_known_id(qid)
        self.redis.setex(self.key_l4(track, qid), settings.cache_ttl_l4, self._dumps(payload))

    def save_user_profile(self, profile: UserProfile) -> None:
        """用户进页选定科目书本 → 持久化，驱动 L3/L4 预热。"""
        self.redis.setex(
            self.key_user_profile(profile.user_id),
            settings.cache_ttl_l3,
            self._dumps(profile.__dict__),
        )

    def warm_l1_from_pairs(self, user_id: str, pairs: list[QAPair]) -> int:
        """把高频 / 近到期题目灌进 L1。"""
        n = 0
        for pair in pairs:
            payload = {
                "question": pair.question.to_dict(),
                "answer": pair.answer.to_dict(),
                "page_guide": pair.page_guide(),
            }
            self.put_l1(user_id, pair.question.question_id, payload)
            n += 1
        return n

    def mark_stale_by_version(self, qid: str, old_version: str) -> None:
        """
        数据血缘：新版教辅入库后，旧 version 缓存标脏。
        简化实现：删除各层同 qid 键（生产可写 stale 标志异步刷新）。
        """
        # 无法枚举所有 user，这里提供工具方法供运维按 user 调用
        logger.info("mark_stale qid=%s old_version=%s —— 请按用户维度调用 invalidate_user_question", qid, old_version)

    def invalidate_user_question(self, user_id: str, track: str, qid: str) -> None:
        self.redis.delete(
            self.key_l1(user_id, qid),
            self.key_l2(user_id, qid),
            self.key_l3(user_id, qid),
            self.key_l4(track, qid),
        )

    # ------------------------------------------------------------------ #
    # 级联读取（辅导 Agent 主路径）
    # ------------------------------------------------------------------ #
    def get_cascaded(
        self,
        question_id: str,
        user_id: str,
        exam_track: str,
        *,
        concept_node: Optional[dict] = None,
        min_confidence: Optional[float] = None,
    ) -> CacheLookupResult:
        """
        L1→L2→L3→L4；每层可视为「已排序结果的置信度门」。
        全未命中则尝试同构题；仍不够则拒绝生成。
        """
        trail: list[str] = []
        threshold = min_confidence if min_confidence is not None else self.confidence_threshold

        # 0) 空值短缓存：防穿透
        null_raw = self.redis.get(self.key_null(question_id))
        if null_raw is not None:
            return CacheLookupResult(
                source="NULL_CACHED",
                data=None,
                confidence=0.0,
                message="近期已确认知识库无此题，请稍后再试或换问法",
                layer_trail=["NULL"],
            )

        # 1) 布隆：一定不存在则直接拒绝（假阳性会落到后面层）
        if not self.bloom.might_contain(question_id) and concept_node is None:
            # 未知 id 且没有图谱实体可生成 → 仍允许走 L3/L4 loader（新题）
            trail.append("BLOOM_MISS")
        else:
            trail.append("BLOOM_OK")

        # 2) L1
        hit = self._get_json(self.key_l1(user_id, question_id))
        if hit:
            trail.append("L1")
            return CacheLookupResult(source="L1", data=hit, confidence=1.0, layer_trail=trail)

        # 3) L2
        hit = self._get_json(self.key_l2(user_id, question_id))
        if not hit and self.l2_loader:
            hit = self._singleflight_load(
                self.key_l2(user_id, question_id),
                lambda: self.l2_loader(user_id, question_id) if self.l2_loader else None,
                settings.cache_ttl_l2,
            )
        if hit:
            trail.append("L2")
            self.put_l1(user_id, question_id, hit)  # 回填热层
            return CacheLookupResult(source="L2", data=hit, confidence=0.95, layer_trail=trail)

        # 4) L3 用户科目书本
        hit = self._get_json(self.key_l3(user_id, question_id))
        if not hit and self.l3_loader:
            hit = self._singleflight_load(
                self.key_l3(user_id, question_id),
                lambda: self.l3_loader(user_id, question_id) if self.l3_loader else None,
                settings.cache_ttl_l3,
            )
        if hit:
            trail.append("L3")
            return CacheLookupResult(source="L3", data=hit, confidence=0.9, layer_trail=trail)

        # 5) L4 扩展教辅
        hit = self._get_json(self.key_l4(exam_track, question_id))
        if not hit and self.l4_loader:
            hit = self._singleflight_load(
                self.key_l4(exam_track, question_id),
                lambda: self.l4_loader(exam_track, question_id) if self.l4_loader else None,
                settings.cache_ttl_l4,
            )
        if hit:
            trail.append("L4")
            return CacheLookupResult(source="L4", data=hit, confidence=0.85, layer_trail=trail)

        # 6) 同构题生成（依赖图谱知识点实体）
        if concept_node and self.isomorphic_generator:
            trail.append("ISO_GEN")
            generated = self.isomorphic_generator(concept_node)
            gen_text = json.dumps(generated, ensure_ascii=False)
            definition = str(concept_node.get("definition", ""))
            sim = self.similarity_fn(gen_text, definition)
            if sim >= self.isomorphic_sim_threshold and sim >= threshold:
                return CacheLookupResult(
                    source="GENERATED",
                    data=generated,
                    confidence=sim,
                    message="基于知识点实体实时生成同构题",
                    layer_trail=trail,
                )
            trail.append(f"ISO_LOW_SIM:{sim:.3f}")

        # 7) 最终兜底：空值缓存 + 拒绝
        self.redis.setex(self.key_null(question_id), settings.cache_ttl_null, CacheGuard.NULL_MARKER)
        trail.append("FALLBACK")
        return CacheLookupResult(
            source="FALLBACK",
            data=None,
            confidence=0.0,
            message="当前知识库暂未收录该考点，建议翻阅指定教材或更换问法",
            layer_trail=trail,
        )

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    def _get_json(self, key: str) -> Optional[dict]:
        raw = self.redis.get(key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if CacheGuard.is_null(raw):
            return None
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {"value": data}
        except json.JSONDecodeError:
            return {"raw": raw}

    def _singleflight_load(
        self,
        cache_key: str,
        loader: Callable[[], Optional[dict]],
        ttl: int,
    ) -> Optional[dict]:
        def _load() -> Optional[dict]:
            data = loader()
            if data:
                self.redis.setex(cache_key, ttl, self._dumps(data))
            return data

        result = self.guard.with_singleflight(cache_key, _load)
        if result is None:
            # 抢锁失败：再读一次缓存
            return self._get_json(cache_key)
        return result

    @staticmethod
    def _dumps(payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _jaccard_sim(a: str, b: str) -> float:
        """无向量模型时的字符级 Jaccard，仅作本地兜底；生产请换 embedding cosine。"""
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)
