# -*- coding: utf-8 -*-
"""
时序知识图谱（Temporal Knowledge Graph）
--------------------------------------
四元组：(主体, 关系, 客体, 时间戳)
例：(学生, 学习了, 反向传播算法, 2026-09-01)

解决传统 RAG「撕碎再拼接」导致的时间线混乱：
检索时沿时间边回溯，按时间线性输出因果链。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class TemporalTriple:
    subject: str
    relation: str
    obj: str
    timestamp: float
    user_id: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    triple_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # 科目 / 会话隔离：检索必须带 filter，防跨科污染
    study_subject: str = ""  # 学习科目，避免与三元组 subject 字段混淆
    session_id: str = ""
    namespace: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class TemporalKnowledgeGraph:
    """
    内存版 TKG。生产可落 Neo4j（带 valid_from）或专用时序图库。
    """

    def __init__(self) -> None:
        self._triples: List[TemporalTriple] = []
        # 依赖边：obj 依赖 subject（学 B 前需学 A）——用于因果闭包
        self._depends: Dict[str, Set[str]] = {}

    def add(
        self,
        subject: str,
        relation: str,
        obj: str,
        *,
        user_id: str = "",
        timestamp: Optional[float] = None,
        meta: Optional[dict] = None,
        study_subject: str = "",
        session_id: str = "",
        namespace: str = "",
    ) -> TemporalTriple:
        t = TemporalTriple(
            subject=subject,
            relation=relation,
            obj=obj,
            timestamp=timestamp if timestamp is not None else time.time(),
            user_id=user_id,
            meta=meta or {},
            study_subject=study_subject,
            session_id=session_id,
            namespace=namespace,
        )
        self._triples.append(t)
        if relation in {"PREREQUISITE_FOR", "依赖", "depends_on"}:
            self._depends.setdefault(obj, set()).add(subject)
        return t

    def _match_scope(
        self,
        t: TemporalTriple,
        user_id: str,
        *,
        study_subject: str = "",
        session_id: str = "",
        namespace: str = "",
        allow_global: bool = True,
    ) -> bool:
        if user_id and t.user_id != user_id:
            return False
        if session_id and t.session_id and t.session_id != session_id:
            # 允许全局情绪类三元组
            if allow_global and t.study_subject == "__global__":
                return True
            return False
        if namespace and t.namespace and t.namespace != namespace:
            if allow_global and t.study_subject == "__global__":
                return True
            return False
        if study_subject and t.study_subject and t.study_subject not in {study_subject, "__global__"}:
            return False
        return True

    def timeline(
        self,
        user_id: str,
        *,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        limit: int = 100,
        study_subject: str = "",
        session_id: str = "",
        namespace: str = "",
    ) -> List[TemporalTriple]:
        """按时间排序的用户记忆时间线（可按科目/会话硬过滤）。"""
        items = [
            t
            for t in self._triples
            if self._match_scope(
                t, user_id, study_subject=study_subject, session_id=session_id, namespace=namespace
            )
        ]
        if start_ts is not None:
            items = [t for t in items if t.timestamp >= start_ts]
        if end_ts is not None:
            items = [t for t in items if t.timestamp <= end_ts]
        items.sort(key=lambda x: x.timestamp)
        return items[:limit]

    def search_similar(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        *,
        study_subject: str = "",
        session_id: str = "",
        namespace: str = "",
    ) -> List[TemporalTriple]:
        """粗粒度相似 + 科目/会话硬过滤（生产：向量检索 + metadata filter）。"""
        q = set(query)
        scored: List[Tuple[float, TemporalTriple]] = []
        for t in self._triples:
            if not self._match_scope(
                t, user_id, study_subject=study_subject, session_id=session_id, namespace=namespace
            ):
                continue
            blob = f"{t.subject}{t.relation}{t.obj}{t.meta}"
            score = len(q & set(blob)) / max(len(q), 1)
            scored.append((score, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_k]]

    def causal_closure(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        *,
        study_subject: str = "",
        session_id: str = "",
        namespace: str = "",
    ) -> List[TemporalTriple]:
        """
        因果闭包检索（限制在同一科目命名空间内）：
        1) Top-K 相关节点
        2) 沿依赖边反向补齐前置祖先
        3) 按时间戳线性输出
        """
        seeds = self.search_similar(
            user_id,
            query,
            top_k=top_k,
            study_subject=study_subject,
            session_id=session_id,
            namespace=namespace,
        )
        needed_concepts: Set[str] = set()
        for s in seeds:
            needed_concepts.add(s.obj)
            needed_concepts.add(s.subject)

        stack = list(needed_concepts)
        visited: Set[str] = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for parent in self._depends.get(node, set()):
                needed_concepts.add(parent)
                stack.append(parent)

        closure = [
            t
            for t in self._triples
            if self._match_scope(
                t, user_id, study_subject=study_subject, session_id=session_id, namespace=namespace
            )
            and (t.subject in needed_concepts or t.obj in needed_concepts)
        ]
        closure.sort(key=lambda x: x.timestamp)
        return closure

    def format_context(self, triples: List[TemporalTriple]) -> str:
        lines = []
        for t in triples:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(t.timestamp))
            lines.append(f"[{ts}] ({t.subject}) -{t.relation}-> ({t.obj})")
        return "\n".join(lines)
