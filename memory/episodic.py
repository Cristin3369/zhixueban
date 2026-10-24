# -*- coding: utf-8 -*-
"""
结构化情景记忆（Episodic Memory）
--------------------------------
每次交互存为带时间戳、参与者、思考过程（CoT）的结构化对象。

【科目隔离】每条 MemCell 必须带 subject + session_id + namespace。
检索 recent() 时强制按 session_id 或 subject 过滤，禁止跨科拼上下文。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class MemCell:
    """最小记忆单元（对话流原子）。"""

    cell_id: str
    user_id: str
    role: str  # user | tutor | companion | planner | system
    content: str
    timestamp: float
    cot: str = ""
    intent: str = ""
    entities: List[str] = field(default_factory=list)
    importance: float = 0.5
    version: int = 1
    # ----- 多科目隔离字段（必填落地）-----
    subject: str = ""  # 如「数学」；全局情绪可写 subject="__global__"
    session_id: str = ""
    namespace: str = ""  # user::math


@dataclass
class MemScene:
    """主题化场景：由多个 MemCell 巩固而成；同样绑定科目。"""

    scene_id: str
    user_id: str
    topic: str
    summary: str
    cell_ids: List[str]
    created_at: float
    updated_at: float
    keywords: List[str] = field(default_factory=list)
    subject: str = ""
    session_id: str = ""
    namespace: str = ""


class EpisodicMemoryStore:
    def __init__(self) -> None:
        self.cells: Dict[str, MemCell] = {}
        self.scenes: Dict[str, MemScene] = {}
        # 索引：user | session | namespace
        self._user_index: Dict[str, List[str]] = {}
        self._session_index: Dict[str, List[str]] = {}
        self._ns_index: Dict[str, List[str]] = {}

    def add_interaction(
        self,
        user_id: str,
        role: str,
        content: str,
        *,
        cot: str = "",
        intent: str = "",
        entities: Optional[List[str]] = None,
        importance: float = 0.5,
        timestamp: Optional[float] = None,
        subject: str = "",
        session_id: str = "",
        namespace: str = "",
    ) -> MemCell:
        cell = MemCell(
            cell_id=uuid.uuid4().hex[:12],
            user_id=user_id,
            role=role,
            content=content,
            timestamp=timestamp if timestamp is not None else time.time(),
            cot=cot,
            intent=intent,
            entities=entities or [],
            importance=importance,
            subject=subject,
            session_id=session_id,
            namespace=namespace,
        )
        self.cells[cell.cell_id] = cell
        self._user_index.setdefault(user_id, []).append(cell.cell_id)
        if session_id:
            self._session_index.setdefault(session_id, []).append(cell.cell_id)
        if namespace:
            self._ns_index.setdefault(namespace, []).append(cell.cell_id)
        return cell

    def recent(
        self,
        user_id: str,
        n: int = 10,
        *,
        session_id: Optional[str] = None,
        subject: Optional[str] = None,
        namespace: Optional[str] = None,
        include_global: bool = True,
    ) -> List[MemCell]:
        """
        默认按会话隔离取热记忆。
        include_global=True 时，额外允许 subject=__global__ 的情绪画像类记忆拼入
        （全局共享，但不拼其它科目的学习内容）。
        """
        if session_id:
            ids = list(self._session_index.get(session_id, []))
        elif namespace:
            ids = list(self._ns_index.get(namespace, []))
        else:
            ids = list(self._user_index.get(user_id, []))

        cells = [self.cells[i] for i in ids if i in self.cells]
        if subject and not session_id and not namespace:
            cells = [c for c in cells if c.subject == subject or (include_global and c.subject == "__global__")]
        elif include_global and (session_id or namespace):
            # 附带全局情绪 cell（同 user）
            global_cells = [
                self.cells[i]
                for i in self._user_index.get(user_id, [])
                if i in self.cells and self.cells[i].subject == "__global__"
            ]
            # 去重合并，全局放前面作画像提示
            seen = {c.cell_id for c in cells}
            for g in global_cells[-3:]:
                if g.cell_id not in seen:
                    cells.append(g)
        cells.sort(key=lambda c: c.timestamp)
        return cells[-n:]

    def retrieve_ranked(
        self,
        user_id: str,
        query: str,
        n: int = 8,
        *,
        session_id: Optional[str] = None,
        subject: Optional[str] = None,
        namespace: Optional[str] = None,
        now: Optional[float] = None,
    ) -> List[MemCell]:
        """冷记忆：相关×衰减×重要性，而不是只取最近 n 条。"""
        from memory.context_policy import rank_cells

        pool = self.recent(
            user_id,
            n=200,
            session_id=session_id,
            subject=subject,
            namespace=namespace,
            include_global=False,
        )
        ranked = rank_cells(pool, query, now=now or time.time(), top_n=n)
        return [c for _, c in ranked]

    def consolidate_scene(
        self,
        user_id: str,
        topic: str,
        cell_ids: List[str],
        summary: str,
        keywords: Optional[List[str]] = None,
        *,
        subject: str = "",
        session_id: str = "",
        namespace: str = "",
    ) -> MemScene:
        now = time.time()
        scene = MemScene(
            scene_id=uuid.uuid4().hex[:12],
            user_id=user_id,
            topic=topic,
            summary=summary,
            cell_ids=list(cell_ids),
            created_at=now,
            updated_at=now,
            keywords=keywords or [],
            subject=subject,
            session_id=session_id,
            namespace=namespace,
        )
        self.scenes[scene.scene_id] = scene
        return scene

    def missing_cell_ids(self, scene_id: str) -> List[str]:
        """情景骨架在、细节 cell 被压缩删除 → 部分记忆丢失。"""
        scene = self.scenes.get(scene_id)
        if not scene:
            return []
        return [cid for cid in scene.cell_ids if cid not in self.cells]

    def patch_missing_cell(
        self,
        *,
        user_id: str,
        content: str,
        subject: str = "",
        session_id: str = "",
        namespace: str = "",
        scene_id: str = "",
        importance: float = 0.95,
    ) -> MemCell:
        """用户确认补全后写入高权重事实，并挂回场景骨架。"""
        cell = self.add_interaction(
            user_id,
            "user",
            content,
            intent="memory_repair",
            importance=importance,
            subject=subject,
            session_id=session_id,
            namespace=namespace,
        )
        if scene_id and scene_id in self.scenes:
            scene = self.scenes[scene_id]
            # 用用户补全替换已丢失的占位 cell_id，避免反复追问同一空洞
            scene.cell_ids = [cid for cid in scene.cell_ids if cid in self.cells]
            if cell.cell_id not in scene.cell_ids:
                scene.cell_ids.append(cell.cell_id)
            scene.updated_at = time.time()
            scene.summary = (scene.summary + "；" + content[:40]).strip("；")
        return cell

    def detect_timeline_gaps(
        self,
        cells: List[MemCell],
        gap_hours: float = 36.0,
    ) -> List[tuple]:
        """
        记忆中断：相邻两条间隔过大（跨天未巩固 / 会话中途断线）。
        返回 [(prev_cell_id, next_cell_id, gap_hours), ...]
        """
        ordered = sorted(cells, key=lambda c: c.timestamp)
        gaps = []
        for a, b in zip(ordered, ordered[1:]):
            hours = (b.timestamp - a.timestamp) / 3600.0
            if hours >= gap_hours:
                gaps.append((a.cell_id, b.cell_id, round(hours, 1)))
        return gaps

    def reconstruct_from_scene(self, scene_id: str) -> str:
        scene = self.scenes.get(scene_id)
        if not scene:
            return ""
        parts = [
            f"主题: {scene.topic}",
            f"科目: {scene.subject or '未标注'}",
            f"摘要: {scene.summary}",
            "细节:",
        ]
        for cid in scene.cell_ids:
            cell = self.cells.get(cid)
            if cell:
                parts.append(f"- [{cell.role}] {cell.content}")
            else:
                parts.append(f"- [缺失片段 {cid}] （可由用户确认补全）")
        return "\n".join(parts)

    def decay_and_archive(
        self,
        user_id: str,
        older_than_days: float = 30,
        keep_importance: float = 0.7,
        *,
        namespace: Optional[str] = None,
    ) -> int:
        cutoff = time.time() - older_than_days * 86400
        removed = 0
        remain = []
        for cid in self._user_index.get(user_id, []):
            cell = self.cells.get(cid)
            if not cell:
                continue
            if namespace and cell.namespace != namespace:
                remain.append(cid)
                continue
            if cell.timestamp < cutoff and cell.importance < keep_importance:
                del self.cells[cid]
                removed += 1
            else:
                remain.append(cid)
        self._user_index[user_id] = remain
        return removed

    def to_dict_cell(self, cell: MemCell) -> dict:
        return asdict(cell)
