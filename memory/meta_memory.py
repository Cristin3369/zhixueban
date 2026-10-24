# -*- coding: utf-8 -*-
"""
元记忆 / 冲突裁决 / 自愈
-----------------------
常见故障 → 兜底（由 recover_memory 按优先级执行）：

| 问题              | 检测                         | 兜底                                      | 入口                         |
| 会话中断/跨天新开 | 当日 session 空、namespace 有 | 同科目 namespace 冷记忆回落               | recover_memory               |
| 时间线空洞        | 相邻 MemCell ≥36h            | 冷热拼接 + Scene 摘要                     | detect_timeline_gaps         |
| 部分细节丢失      | MemScene 骨架在、cell 缺失   | 场景摘要 + TKG 因果闭包 + 槽位问用户      | missing_cell_ids             |
| 图谱空洞          | 有对话无三元组               | 从 MemCell 回填 TKG                       | _bootstrap_tkg_from_cells    |
| 碎片化            | ≥6 轮无 MemScene             | 自动 consolidate                          | _auto_consolidate            |
| 语义冲突/漂移     | 新事实≠旧事实                | 时效覆盖+快照；高风险澄清不自动写         | ConflictResolver.upsert_fact |
| 逻辑断裂          | 有结论无 CoT                 | 离线反思抬权重待补全                      | offline_reflect              |
| 库膨胀            | 过期且低 importance          | decay 归档                                | decay_and_archive            |
| 以上都不够        | recovered 仍空               | 向用户确认，下一轮 ingest_user_repair     | ask_user / ingest_user_repair|
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from memory.episodic import EpisodicMemoryStore, MemCell
from memory.temporal_kg import TemporalKnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class MemoryHealthReport:
    ok: bool
    issues: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    ask_user: Optional[str] = None
    recovered_context: str = ""
    fallbacks_used: List[str] = field(default_factory=list)
    issue_types: List[str] = field(default_factory=list)


class ConflictResolver:
    """
    语义漂移与冲突仲裁：
    - 时效性优先：新覆盖旧，旧版保留快照
    - 明确性优先：用户显式声明 > 行为推断
    - 高风险矛盾：不自动覆盖，发起澄清
    """

    HIGH_RISK_KEYS = {"过敏", "疾病", "用药", "心理危机", "放弃考试"}

    def __init__(self) -> None:
        # key -> list of versioned facts
        self.facts: Dict[str, List[Dict[str, Any]]] = {}

    def upsert_fact(
        self,
        key: str,
        value: str,
        *,
        explicit: bool = False,
        user_id: str = "",
    ) -> Dict[str, Any]:
        versions = self.facts.setdefault(key, [])
        now = time.time()
        if versions:
            old = versions[-1]
            if old["value"] != value:
                # 高风险：要求澄清
                if any(r in key or r in value or r in old["value"] for r in self.HIGH_RISK_KEYS):
                    return {
                        "status": "need_clarify",
                        "message": (
                            f"您之前提到「{old['value']}」，现在变为「{value}」。"
                            f"需要我更新档案中的「{key}」吗？"
                        ),
                        "old": old,
                        "new_candidate": value,
                    }
                # 时效覆盖 + 快照
                versions.append(
                    {
                        "value": value,
                        "ts": now,
                        "explicit": explicit,
                        "user_id": user_id,
                        "supersedes": old.get("value"),
                    }
                )
                return {"status": "updated", "value": value}
            return {"status": "unchanged", "value": value}

        versions.append({"value": value, "ts": now, "explicit": explicit, "user_id": user_id})
        return {"status": "created", "value": value}

    def get_current(self, key: str) -> Optional[str]:
        vers = self.facts.get(key) or []
        return vers[-1]["value"] if vers else None


class MetaMemoryManager:
    """
    元记忆：Agent 对自身记忆状态的「上帝视角」。
    推理卡壳时检查关键记忆是否丢失并触发修复。
    """

    def __init__(
        self,
        episodic: Optional[EpisodicMemoryStore] = None,
        tkg: Optional[TemporalKnowledgeGraph] = None,
        resolver: Optional[ConflictResolver] = None,
    ):
        self.episodic = episodic or EpisodicMemoryStore()
        self.tkg = tkg or TemporalKnowledgeGraph()
        self.resolver = resolver or ConflictResolver()

    def recover_memory(
        self,
        user_id: str,
        goal: str,
        *,
        session_id: str = "",
        subject: str = "",
        namespace: str = "",
        user_utterance: str = "",
        pending_ask: str = "",
        last_active_at: float = 0.0,
    ) -> MemoryHealthReport:
        """
        记忆故障诊断 + 按优先级兜底（在线）。

        对应关系见各 issue_type：
        - empty：本会话无记忆 → 问用户（末端人工对齐）
        - interruption：时间线中断（相邻记录间隔过大）→ 冷热拼接 + 场景骨架
        - partial_loss：MemScene 有骨架但 cell 缺失 → TKG 补时间线 + 标缺失槽
        - tkg_gap：有对话无图谱 → 从 MemCell 回填三元组
        - fragment：有多轮无主题骨架 → 自动 consolidate
        - conflict：画像冲突 → ConflictResolver 澄清（不自动覆盖高风险）
        仍不够才 ask_user，用户下一轮回答走 ingest_user_repair。
        """
        if pending_ask and user_utterance:
            self.ingest_user_repair(
                user_id,
                user_utterance,
                subject=subject,
                session_id=session_id,
                namespace=namespace,
                topic=goal,
            )

        issues: List[str] = []
        actions: List[str] = []
        used: List[str] = []
        types: List[str] = []
        ask_user = None

        recent = self.episodic.recent(
            user_id,
            n=20,
            session_id=session_id or None,
            subject=subject or None,
            namespace=namespace or None,
        )
        # 跨天新 session_id：热记忆空，但同科目长期记忆还在 → 会话中断兜底
        if not recent and namespace:
            ns_recent = self.episodic.recent(
                user_id, n=20, namespace=namespace, include_global=True
            )
            if ns_recent:
                types.append("interruption")
                issues.append("会话中断：当日 session 为空，已回落到同科目 namespace")
                used.append("namespace_fallback")
                actions.append("用昨日同科 Scene/TKG 续写，不串其它科目")
                recent = ns_recent
                session_id = ""  # 后续重建不再卡死在新 session_id 上

        idle_h = (
            (time.time() - last_active_at) / 3600.0 if last_active_at else 0.0
        )
        if idle_h >= 36.0:
            types.append("interruption")
            issues.append(f"会话闲置 {idle_h:.1f}h 后恢复")
            used.append("resume_from_cold")
            actions.append("断点续学：注入上次结论，不假装无缝衔接")

        hotcold = self.build_hot_cold_context(
            user_id,
            goal,
            session_id=session_id,
            subject=subject,
            namespace=namespace,
        )
        recovered = "\n".join(x for x in [hotcold.get("hot"), hotcold.get("cold")] if x)

        # 1) 空记忆
        if not recent and not recovered.strip():
            types.append("empty")
            issues.append("无近期情景记忆且无法从图谱/场景重建")
            actions.append("ask_user")
            used.append("human_align")
            ask_user = (
                f"我这边还没有你在本科目关于「{goal[:40]}」的上下文，"
                "你能用一两句话回顾一下上次学到哪、结论是什么吗？"
            )
            return MemoryHealthReport(
                ok=False,
                issues=issues,
                actions=actions,
                ask_user=ask_user,
                recovered_context="",
                fallbacks_used=used,
                issue_types=types,
            )

        if not recent and recovered.strip():
            types.append("empty")
            issues.append("情景记忆空，已用场景/TKG 重建")
            used.append("reconstructive_recollect")
            actions.append("inject_cold_context")

        # 2) 时间线中断
        gaps = self.episodic.detect_timeline_gaps(recent, gap_hours=36.0)
        if gaps:
            types.append("interruption")
            issues.append(f"时间线中断 {len(gaps)} 处（间隔≥36h）")
            used.append("hot_cold_splice")
            actions.append("用冷记忆补中断两侧的前置结论")
            if hotcold.get("cold"):
                recovered = recovered or hotcold["cold"]

        # 3) 部分丢失：场景骨架在、cell 没了
        hole_scenes = []
        for s in self.episodic.scenes.values():
            if s.user_id != user_id:
                continue
            if subject and s.subject and s.subject != subject:
                continue
            missing = self.episodic.missing_cell_ids(s.scene_id)
            if missing:
                hole_scenes.append((s, missing))
        if hole_scenes:
            types.append("partial_loss")
            n_miss = sum(len(m) for _, m in hole_scenes)
            issues.append(f"部分记忆丢失：{len(hole_scenes)} 个场景缺 {n_miss} 条细节")
            used.append("scene_skeleton")
            used.append("tkg_causal_closure")
            actions.append("用 MemScene 摘要 + TKG 因果闭包填洞")
            for scene, _miss in hole_scenes[:2]:
                recovered += "\n" + self.episodic.reconstruct_from_scene(scene.scene_id)
            closure = self.tkg.causal_closure(
                user_id,
                goal,
                top_k=8,
                study_subject=subject,
                session_id=session_id,
                namespace=namespace,
            )
            if closure:
                recovered += "\n时间线补全:\n" + self.tkg.format_context(closure)
            # 细节仍缺 → 问用户补这一段，而不是整段失忆
            ask_user = (
                f"我记得我们讨论过「{hole_scenes[0][0].topic}」，"
                f"摘要还在（{hole_scenes[0][0].summary[:60]}），"
                "但中间几步推导丢失了。当时的关键结论你还记得吗？"
            )
            actions.append("ask_user_slot_fill")

        # 4) TKG 空洞：有对话无图
        closure = self.tkg.causal_closure(
            user_id,
            goal,
            top_k=5,
            study_subject=subject,
            session_id=session_id,
            namespace=namespace,
        )
        if recent and not closure:
            types.append("tkg_gap")
            issues.append("对话在、图谱缺，已从 MemCell 回填三元组")
            actions.append("bootstrap_tkg")
            used.append("bootstrap_tkg_from_cells")
            self._bootstrap_tkg_from_cells(user_id, recent)

        # 5) 碎片化：多轮无场景
        related_scenes = [
            s
            for s in self.episodic.scenes.values()
            if s.user_id == user_id
            and (not subject or s.subject in {"", subject})
            and (not session_id or s.session_id in {"", session_id})
        ]
        if recent and len(recent) >= 6 and not related_scenes:
            types.append("fragment")
            issues.append("多轮对话无主题骨架，已自动 consolidate")
            actions.append("auto_consolidate")
            used.append("memscene_consolidate")
            self._auto_consolidate(
                user_id, goal, recent, subject=subject, session_id=session_id, namespace=namespace
            )

        # 重建成功则不必整段追问（部分丢失除外，仍要槽位补全）
        reconstructed_ok = bool(recovered.strip()) and "partial_loss" not in types
        if reconstructed_ok:
            ask_user = None
            used.append("skip_ask_if_recovered")

        ok = "empty" not in types and "partial_loss" not in types
        if not ok and not ask_user:
            ask_user = (
                f"我注意到本科目讨论过「{goal[:40]}」，但上下文不完整。"
                "你能帮我回顾一下当时的结论吗？"
            )
            used.append("human_align")

        return MemoryHealthReport(
            ok=ok,
            issues=issues,
            actions=actions,
            ask_user=ask_user,
            recovered_context=recovered.strip(),
            fallbacks_used=used,
            issue_types=types,
        )

    def ingest_user_repair(
        self,
        user_id: str,
        utterance: str,
        *,
        subject: str = "",
        session_id: str = "",
        namespace: str = "",
        topic: str = "",
    ) -> MemCell:
        """人工对齐：用户补全写入高权重事实 + TKG，并挂回残缺场景。"""
        hole_id = ""
        for s in self.episodic.scenes.values():
            if s.user_id != user_id:
                continue
            if subject and s.subject and s.subject != subject:
                continue
            if self.episodic.missing_cell_ids(s.scene_id):
                hole_id = s.scene_id
                break
        cell = self.episodic.patch_missing_cell(
            user_id=user_id,
            content=utterance,
            subject=subject,
            session_id=session_id,
            namespace=namespace,
            scene_id=hole_id,
            importance=0.95,
        )
        self.tkg.add(
            user_id,
            "确认了",
            utterance[:80],
            user_id=user_id,
            study_subject=subject,
            session_id=session_id,
            namespace=namespace,
            meta={"source": "user_repair", "topic": topic},
        )
        self.resolver.upsert_fact(
            f"{user_id}:{subject}:last_conclusion",
            utterance[:200],
            explicit=True,
            user_id=user_id,
        )
        return cell

    def evaluate_consistency(
        self,
        user_id: str,
        goal: str,
        *,
        session_id: str = "",
        subject: str = "",
        namespace: str = "",
        user_utterance: str = "",
        pending_ask: str = "",
        last_active_at: float = 0.0,
    ) -> MemoryHealthReport:
        """兼容旧调用：走完整 recover 链路。"""
        return self.recover_memory(
            user_id,
            goal,
            session_id=session_id,
            subject=subject,
            namespace=namespace,
            user_utterance=user_utterance,
            pending_ask=pending_ask,
            last_active_at=last_active_at,
        )

    def reconstructive_recollect(
        self,
        user_id: str,
        topic: str,
        *,
        session_id: str = "",
        subject: str = "",
        namespace: str = "",
    ) -> str:
        """重建式回忆：严格限制在同科目 / 同会话命名空间。"""
        for scene in self.episodic.scenes.values():
            if scene.user_id != user_id:
                continue
            if subject and scene.subject and scene.subject != subject:
                continue
            if session_id and scene.session_id and scene.session_id != session_id:
                continue
            if topic in scene.topic or topic in scene.summary:
                base = self.episodic.reconstruct_from_scene(scene.scene_id)
                closure = self.tkg.causal_closure(
                    user_id,
                    topic,
                    top_k=5,
                    study_subject=subject,
                    session_id=session_id,
                    namespace=namespace,
                )
                return base + "\n\n时间线补全:\n" + self.tkg.format_context(closure)
        closure = self.tkg.causal_closure(
            user_id,
            topic,
            top_k=8,
            study_subject=subject,
            session_id=session_id,
            namespace=namespace,
        )
        if closure:
            return "时间线上下文:\n" + self.tkg.format_context(closure)
        return ""

    def offline_reflect(self, user_id: str, *, namespace: str = "") -> List[str]:
        notes: List[str] = []
        cells = self.episodic.recent(user_id, n=50, namespace=namespace or None)
        conclusions = [c for c in cells if "结论" in c.content or c.role == "tutor"]
        for c in conclusions:
            if not c.cot and c.importance >= 0.6:
                notes.append(f"逻辑断裂: cell={c.cell_id} 有结论但缺少 CoT，已标记待补全")
                c.importance = min(1.0, c.importance + 0.1)
        removed = self.episodic.decay_and_archive(user_id, namespace=namespace or None)
        if removed:
            notes.append(f"归档清理低价值记忆 {removed} 条，缓解库膨胀")
        return notes

    def build_hot_cold_context(
        self,
        user_id: str,
        query: str,
        hot_n: int = 10,
        *,
        session_id: str = "",
        subject: str = "",
        namespace: str = "",
    ) -> Dict[str, str]:
        """
        冷热分离 + 科目隔离：
        - hot：本会话近期轮次 + 少量全局情绪画像
        - cold：本科目 TKG/Scene 重建，绝不注入其它科目对话
        """
        hot = self.episodic.recent(
            user_id,
            n=hot_n,
            session_id=session_id or None,
            subject=subject or None,
            namespace=namespace or None,
            include_global=True,
        )
        hot_text = "\n".join(
            f"[{c.role}|{c.subject or 'global'}] {c.content}" for c in hot
        )
        from memory.context_policy import compress_cells

        ranked = self.episodic.retrieve_ranked(
            user_id,
            query,
            n=8,
            session_id=None,
            subject=subject or None,
            namespace=namespace or None,
        )
        cold_rank = compress_cells(ranked, max_chars=600)
        cold = self.reconstructive_recollect(
            user_id,
            query,
            session_id=session_id,
            subject=subject,
            namespace=namespace,
        )
        cold = "\n".join(x for x in [cold_rank, cold] if x)
        return {"hot": hot_text, "cold": cold}

    def _bootstrap_tkg_from_cells(self, user_id: str, cells: List[MemCell]) -> None:
        for c in cells:
            for ent in c.entities:
                self.tkg.add(
                    user_id,
                    "提到了",
                    ent,
                    user_id=user_id,
                    timestamp=c.timestamp,
                    study_subject=c.subject,
                    session_id=c.session_id,
                    namespace=c.namespace,
                )
            if c.intent:
                self.tkg.add(
                    user_id,
                    "意图",
                    c.intent,
                    user_id=user_id,
                    timestamp=c.timestamp,
                    study_subject=c.subject,
                    session_id=c.session_id,
                    namespace=c.namespace,
                )

    def _auto_consolidate(
        self,
        user_id: str,
        topic: str,
        cells: List[MemCell],
        *,
        subject: str = "",
        session_id: str = "",
        namespace: str = "",
    ) -> None:
        summary = "；".join(c.content[:40] for c in cells[-5:])
        self.episodic.consolidate_scene(
            user_id,
            topic=topic[:40] or "学习会话",
            cell_ids=[c.cell_id for c in cells[-8:]],
            summary=summary,
            keywords=list({e for c in cells for e in c.entities})[:10],
            subject=subject or (cells[-1].subject if cells else ""),
            session_id=session_id or (cells[-1].session_id if cells else ""),
            namespace=namespace or (cells[-1].namespace if cells else ""),
        )
