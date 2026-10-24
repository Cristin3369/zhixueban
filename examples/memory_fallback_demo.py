# -*- coding: utf-8 -*-
"""演示记忆中断 / 部分丢失时的兜底链路（不依赖外部服务）。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from memory.episodic import EpisodicMemoryStore
from memory.meta_memory import MetaMemoryManager
from memory.temporal_kg import TemporalKnowledgeGraph


def _show(title: str, report) -> None:
    print(f"\n== {title} ==")
    print("issue_types:", report.issue_types)
    print("fallbacks:", report.fallbacks_used)
    print("ok:", report.ok, "| ask_user:", bool(report.ask_user))
    if report.issues:
        print("issues:", "; ".join(report.issues))
    if report.ask_user:
        print("ask:", report.ask_user[:80])
    if report.recovered_context:
        print("recovered:", report.recovered_context[:120].replace("\n", " | "))


def main() -> None:
    epi = EpisodicMemoryStore()
    tkg = TemporalKnowledgeGraph()
    mm = MetaMemoryManager(episodic=epi, tkg=tkg)
    uid = "demo_user"
    ns = "demo_user::math"

    # 1) 完全空记忆 → 问用户
    r0 = mm.recover_memory(uid, "反向传播", subject="数学", session_id="sess_today", namespace=ns)
    _show("empty", r0)
    assert "empty" in r0.issue_types and r0.ask_user

    # 2) 昨日有记忆，今日新 session_id 为空 → namespace 回落
    yesterday = "sess_yesterday"
    c1 = epi.add_interaction(
        uid, "user", "昨天把反向传播链式法则过了一遍",
        intent="study", subject="数学", session_id=yesterday, namespace=ns, importance=0.8,
        timestamp=time.time() - 40 * 3600,
    )
    epi.add_interaction(
        uid, "tutor", "结论：先写局部梯度再连乘",
        intent="conclusion", subject="数学", session_id=yesterday, namespace=ns, importance=0.85,
        timestamp=time.time() - 39 * 3600,
    )
    tkg.add(uid, "学习了", "反向传播", user_id=uid, study_subject="数学",
            session_id=yesterday, namespace=ns, timestamp=time.time() - 39 * 3600)
    r1 = mm.recover_memory(uid, "反向传播", subject="数学", session_id="sess_today", namespace=ns)
    _show("interruption/namespace_fallback", r1)
    assert "interruption" in r1.issue_types
    assert "namespace_fallback" in r1.fallbacks_used
    assert r1.recovered_context

    # 3) 场景骨架在、细节 cell 被删 → 部分丢失 + 槽位追问
    scene = epi.consolidate_scene(
        uid, topic="反向传播", cell_ids=[c1.cell_id, "ghost_cell"],
        summary="链式法则结论还在", subject="数学", session_id=yesterday, namespace=ns,
    )
    assert epi.missing_cell_ids(scene.scene_id) == ["ghost_cell"]
    r2 = mm.recover_memory(uid, "反向传播", subject="数学", session_id=yesterday, namespace=ns)
    _show("partial_loss", r2)
    assert "partial_loss" in r2.issue_types and r2.ask_user

    # 4) 用户补全 → 高权重写回
    mm.ingest_user_repair(
        uid, "当时结论是局部梯度连乘，别漏偏导",
        subject="数学", session_id=yesterday, namespace=ns, topic="反向传播",
    )
    r3 = mm.recover_memory(
        uid, "反向传播", subject="数学", session_id=yesterday, namespace=ns,
        user_utterance="对，就是链式法则", pending_ask=r2.ask_user or "x",
    )
    _show("after_user_repair", r3)

    # 5) 高风险冲突不自动覆盖
    first = mm.resolver.upsert_fact("心理危机", "有过焦虑但能坚持", explicit=True, user_id=uid)
    clash = mm.resolver.upsert_fact("心理危机", "想放弃考试", explicit=True, user_id=uid)
    print("\n== conflict ==")
    print("first:", first["status"], "| clash:", clash["status"])
    assert clash["status"] == "need_clarify"

    print("\nmemory fallback demo ok")


if __name__ == "__main__":
    main()
