# -*- coding: utf-8 -*-
"""
反思机制
-------
在调度、计划、辅导等关键节点调用，检查输出是否合理，降低幻觉与不当加压。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.settings import settings


@dataclass
class ReflectionResult:
    approved: bool
    revised_action: Optional[str] = None  # 建议改走的 Agent / 模式
    notes: List[str] = field(default_factory=list)
    sanitized_payload: Dict[str, Any] = field(default_factory=dict)


class ReflectionEngine:
    """规则 + 可选 LLM 的轻量反思（默认规则，稳健、可测）。"""

    def reflect_routing(
        self,
        *,
        intent: str,
        emotion_intensity: float,
        targets: List[str],
        user_energy: float = 0.7,
    ) -> ReflectionResult:
        notes: List[str] = []
        targets = list(targets)
        payload: Dict[str, Any] = {"targets": targets}

        # 情绪崩溃：只允许陪伴，去掉计划/刷题
        if emotion_intensity >= settings.emotion_crisis_threshold:
            notes.append("情绪强度过高 → 仅 Companion，取消 Planner/强辅导")
            targets = ["Companion_Agent"]
            payload["mode"] = "comfort_only"
            return ReflectionResult(approved=True, revised_action="Companion_Agent", notes=notes, sanitized_payload=payload)

        # 精力极低：陪伴 + 轻量计划，禁止高压任务
        if user_energy <= settings.energy_rest_threshold:
            notes.append("精力过低 → Companion + 轻量 Planner，禁止高频加压")
            if "Companion_Agent" not in targets:
                targets.insert(0, "Companion_Agent")
            payload["mode"] = "light_plan"
            payload["targets"] = targets
            return ReflectionResult(approved=True, notes=notes, sanitized_payload=payload)

        # mixed：允许 Companion + Tutor 并行，但计划置后
        if intent == "mixed":
            notes.append("混合意图 → 先安抚再辅导，可并行调度")
            ordered = []
            for name in ["Companion_Agent", "Tutor_Agent", "Planner_Agent"]:
                if name in targets or name == "Companion_Agent":
                    if name not in ordered:
                        ordered.append(name)
            for t in targets:
                if t not in ordered:
                    ordered.append(t)
            payload["targets"] = ordered
            payload["mode"] = "comfort_then_tutor"

        return ReflectionResult(approved=True, notes=notes, sanitized_payload=payload)

    def reflect_plan(self, plan: Dict[str, Any], *, emotion_intensity: float, user_energy: float) -> ReflectionResult:
        notes = []
        plan = dict(plan)
        if emotion_intensity >= settings.emotion_crisis_threshold and plan.get("mode") != "rest_only":
            notes.append("计划与情绪状态冲突，强制 rest_only")
            plan = {"mode": "rest_only", "blocks": [], "message": "今日休息，不安排刷题"}
            return ReflectionResult(approved=False, notes=notes, sanitized_payload=plan)

        total_min = sum(b.get("minutes", 0) for b in plan.get("blocks") or [])
        if user_energy < 0.4 and total_min > 40:
            notes.append("计划过重，裁剪至 40 分钟内")
            # 简单裁剪
            new_blocks = []
            remain = 40
            for b in plan.get("blocks") or []:
                if remain <= 0:
                    break
                m = min(b.get("minutes", 0), remain)
                new_blocks.append({**b, "minutes": m})
                remain -= m
            plan["blocks"] = new_blocks
            plan["message"] = (plan.get("message") or "") + "（已按精力裁剪）"
        return ReflectionResult(approved=True, notes=notes, sanitized_payload=plan)

    def reflect_tutor_answer(self, answer_pack: Dict[str, Any]) -> ReflectionResult:
        notes = []
        pack = dict(answer_pack)
        if pack.get("status") != "success":
            notes.append("检索未达置信度或落地失败，保持拒绝，禁止补幻觉")
            pack["response"] = pack.get("response") or "当前知识库暂未收录该考点或相关数据，建议查阅指定教材对应页码，或换一种问法。"
            return ReflectionResult(approved=True, notes=notes, sanitized_payload=pack)
        text = pack.get("response") or ""
        if "【溯源】" not in text and "第" not in text:
            notes.append("成功回答缺少溯源，降级拒绝")
            pack["status"] = "fallback_triggered"
            pack["compose_mode"] = "refuse"
            pack["response"] = "当前知识库暂未收录该考点或相关数据，建议查阅指定教材对应页码，或换一种问法。"
            return ReflectionResult(approved=False, notes=notes, sanitized_payload=pack)
        if len(text) > 3000:
            notes.append("回答过长，截断以降低原文照抄风险")
            pack["response"] = text[:2400] + "\n……（已截断，请以教材页码为准）"
        if pack.get("compose_mode") == "template":
            notes.append("讲解走模板降级（LLM 不可用或未通过落地检查）")
        return ReflectionResult(approved=True, notes=notes, sanitized_payload=pack)
