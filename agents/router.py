# -*- coding: utf-8 -*-
"""
调度 Agent（Router / Dispatcher）
--------------------------------
识别：学科学习 / 情绪陪伴 / 制定计划 / 混合 / 语音倾诉等。
可同时调度多个下游 Agent，带等待与穿透兜底，关键路径强制反思。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from agents.reflection import ReflectionEngine
from config.settings import settings
from generation.envelope import CircuitBreaker, guarded_call

logger = logging.getLogger(__name__)


class IntentRouter(BaseModel):
    """强制结构化输出，避免自由文本解析翻车。"""

    intent: Literal[
        "study_help",
        "emotional_support",
        "plan_request",
        "mixed",
        "voice_companion",
    ] = Field(description="意图类别")
    core_topic: str = Field(description="核心主题/问题摘要")
    confidence: float = Field(description="0~1 置信度")
    emotion_intensity: float = Field(default=0.0, description="情绪强度 0~1")
    need_plan: bool = Field(default=False, description="是否需要制定/调整计划")
    agents_to_call: List[
        Literal["Tutor_Agent", "Companion_Agent", "Planner_Agent"]
    ] = Field(default_factory=list, description="建议并行/串行调用的 Agent 列表")


ROUTER_SYSTEM = """
你是考研/考公辅导系统的路由调度器。职责：识别意图、评估情绪强度、决定调用哪些 Agent。
意图：
- study_help：学科问题、真题、知识点
- emotional_support：焦虑、崩溃、倾诉
- plan_request：要复习计划、安排今日任务
- mixed：情绪 + 学习
- voice_companion：明确语音陪伴/边走边背
规则：
1. emotion_intensity>=0.85 时 agents_to_call 只能含 Companion_Agent
2. mixed 时至少包含 Companion_Agent 与 Tutor_Agent
3. plan_request 含 Planner_Agent；若情绪高先 Companion
4. 不要编造用户没说的科目细节
"""


class RuleBasedRouter:
    """
    无 LLM 时的规则路由，保证系统可启动。
    有 LLM 时用 LLMRouter 包装。
    """

    CRISIS_WORDS = ["崩溃", "不想活", "绝望", "放弃了", "焦虑发作", "哭不出来", "心跳好快"]
    PLAN_WORDS = ["计划", "安排", "今天学什么", "复习规划", "日程", "艾宾浩斯"]
    STUDY_WORDS = ["怎么解", "为什么", "公式", "真题", "知识点", "证明", "推导", "错题"]
    VOICE_WORDS = ["语音", "边走边背", "听", "朗读", "耳机"]

    def classify(self, user_input: str, *, user_energy: float = 0.7) -> Dict[str, Any]:
        """纯规则打标，不含反思。供与 LLM 路由做保守融合。"""
        text = user_input or ""
        emotion = 0.0
        if any(w in text for w in self.CRISIS_WORDS):
            emotion = 0.9
        elif any(w in text for w in ["难啊", "好累", "不想学", "焦虑", "压力"]):
            emotion = 0.55

        want_plan = any(w in text for w in self.PLAN_WORDS)
        want_study = any(w in text for w in self.STUDY_WORDS)
        want_voice = any(w in text for w in self.VOICE_WORDS)

        if emotion >= settings.emotion_crisis_threshold:
            intent = "emotional_support"
            agents = ["Companion_Agent"]
        elif want_voice and not want_study:
            intent = "voice_companion"
            agents = ["Companion_Agent"]
        elif want_plan and want_study:
            intent = "mixed"
            agents = ["Companion_Agent", "Tutor_Agent", "Planner_Agent"]
        elif want_plan:
            intent = "plan_request"
            agents = ["Planner_Agent"]
            if emotion >= 0.5:
                agents = ["Companion_Agent", "Planner_Agent"]
        elif emotion >= 0.5 and want_study:
            intent = "mixed"
            agents = ["Companion_Agent", "Tutor_Agent"]
        elif emotion >= 0.5 and not want_study:
            intent = "emotional_support"
            agents = ["Companion_Agent"]
        else:
            intent = "study_help"
            agents = ["Tutor_Agent"]

        return {
            "intent": intent,
            "core_topic": text[:80],
            "confidence": 0.7,
            "emotion_intensity": emotion,
            "need_plan": want_plan or intent == "plan_request",
            "agents_to_call": agents,
            "user_energy": user_energy,
        }

    def route(self, user_input: str, *, user_energy: float = 0.7) -> Dict[str, Any]:
        return self._with_reflection(self.classify(user_input, user_energy=user_energy))

    def _with_reflection(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        engine = ReflectionEngine()
        reflected = engine.reflect_routing(
            intent=raw["intent"],
            emotion_intensity=float(raw.get("emotion_intensity", 0)),
            targets=list(raw.get("agents_to_call") or []),
            user_energy=float(raw.get("user_energy", 0.7)),
        )
        targets = reflected.sanitized_payload.get("targets") or raw["agents_to_call"]
        # 主 target：兼容旧 startgraph 单路由
        primary = targets[0] if targets else "Companion_Agent"
        return {
            "target_agent": primary,
            "target_agents": targets,  # 支持多 Agent
            "payload": {
                "query": raw["core_topic"],
                "emotion_context": raw["core_topic"],
                "emotion_intensity": raw.get("emotion_intensity", 0),
                "need_plan": raw.get("need_plan", False),
                "mode": reflected.sanitized_payload.get("mode", "normal"),
            },
            "intent": raw["intent"],
            "confidence": raw.get("confidence", 0.7),
            "reflection_notes": reflected.notes,
        }


class RouterAgent:
    """
    优先尝试 LLM 结构化路由，失败则规则兜底（防穿透：路由失败不拖垮全站）。
    """

    def __init__(self, use_llm: bool = False):
        self.rules = RuleBasedRouter()
        self.use_llm = use_llm
        self._chain = None
        self._breaker = CircuitBreaker("router_llm")
        if use_llm:
            try:
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_openai import ChatOpenAI

                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", ROUTER_SYSTEM),
                        (
                            "human",
                            "学生输入：{user_input}\n用户精力(0-1)：{user_energy}\n请输出结构化意图。",
                        ),
                    ]
                )
                llm = ChatOpenAI(model=settings.llm_model, temperature=0)
                self._chain = prompt | llm.with_structured_output(IntentRouter)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM 路由不可用，使用规则路由: %s", exc)
                self._chain = None

    @staticmethod
    def _merge_rule_and_llm(rule_raw: dict, llm_raw: dict, user_input: str) -> dict:
        """
        企业路由原则：危机取并集里更保守的一边；LLM 低置信回规则；
        两边意图不一致时，情绪侧优先插入陪伴。
        """
        crisis = settings.emotion_crisis_threshold
        emotion = max(
            float(rule_raw.get("emotion_intensity") or 0),
            float(llm_raw.get("emotion_intensity") or 0),
        )
        if emotion >= crisis or any(w in (user_input or "") for w in RuleBasedRouter.CRISIS_WORDS):
            return {
                **rule_raw,
                "intent": "emotional_support",
                "emotion_intensity": max(emotion, 0.9),
                "agents_to_call": ["Companion_Agent"],
                "confidence": max(
                    float(rule_raw.get("confidence") or 0),
                    float(llm_raw.get("confidence") or 0),
                ),
            }

        llm_conf = float(llm_raw.get("confidence") or 0)
        if llm_conf < 0.55 or not llm_raw.get("agents_to_call"):
            return rule_raw

        agents: List[str] = []
        for src in (llm_raw.get("agents_to_call") or [], rule_raw.get("agents_to_call") or []):
            for a in src:
                if a not in agents:
                    agents.append(a)
        intent = llm_raw.get("intent") or rule_raw.get("intent")
        if rule_raw.get("intent") != llm_raw.get("intent"):
            pair = {rule_raw.get("intent"), llm_raw.get("intent")}
            if "emotional_support" in pair or (
                float(rule_raw.get("emotion_intensity") or 0) >= 0.5
                and llm_raw.get("intent") == "study_help"
            ):
                intent = "mixed"
                if "Companion_Agent" not in agents:
                    agents.insert(0, "Companion_Agent")
        return {
            "intent": intent,
            "core_topic": llm_raw.get("core_topic") or rule_raw.get("core_topic"),
            "confidence": llm_conf,
            "emotion_intensity": emotion,
            "need_plan": bool(rule_raw.get("need_plan") or llm_raw.get("need_plan")),
            "agents_to_call": agents or ["Tutor_Agent"],
            "user_energy": rule_raw.get("user_energy", 0.7),
        }

    def route(self, user_input: str, *, user_energy: float = 0.7) -> dict:
        rule_raw = self.rules.classify(user_input, user_energy=user_energy)
        try:
            if self._chain is not None:
                guard = guarded_call(
                    lambda: self._chain.invoke(
                        {"user_input": user_input, "user_energy": user_energy}
                    ),
                    fallback=None,
                    breaker=self._breaker,
                    timeout_sec=settings.router_llm_timeout_sec,
                    name="router_llm",
                )
                if guard.degraded or guard.value is None:
                    return self.rules._with_reflection(rule_raw)
                result: IntentRouter = guard.value
                llm_raw = result.model_dump()
                llm_raw["user_energy"] = user_energy
                merged = self._merge_rule_and_llm(rule_raw, llm_raw, user_input)
                return self.rules._with_reflection(merged)
        except Exception as exc:  # noqa: BLE001
            logger.exception("路由异常，Fallback 规则: %s", exc)
            fallback = self.rules._with_reflection(rule_raw)
            fallback["payload"]["error"] = str(exc)
            fallback["target_agent"] = fallback.get("target_agent") or "Companion_Agent"
            return fallback

        return self.rules._with_reflection(rule_raw)
