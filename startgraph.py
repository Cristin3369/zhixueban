# -*- coding: utf-8 -*-
"""
LangGraph 总编排
---------------
架构原则：一科一会话（Session Isolation）+ 全局调度看板 + 全局画像共享。

流程：
Router → MetaMemory(限本会话命名空间) → Dispatch(多 Agent) → 写回 本会话 记忆 / 全局情绪

打开 App：先看今日看板（跨科分配时间）；点击某科 → 带 session_id 进入专属对话。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.companion import CompanionAgent
from agents.planner import PlannerAgent
from agents.router import RouterAgent
from agents.tutor import TutorAgent
from config.settings import settings
from data.onboarding import onboard_user, resolve_exam_date
from data.session import SessionManager, SubjectSession
from data.study_queue import StudyTaskQueue
from data.tiered_cache import TieredKnowledgeStore
from memory.meta_memory import MetaMemoryManager
from retrieval.graph_rag import KnowledgeGraphRAG
from retrieval.hybrid_rag import AdvancedRAGEngine
from scheduling.calendar import ExamCalendarService
from scheduling.multi_factor_abhs import MultiFactorEbbinghausScheduler

logger = logging.getLogger(__name__)

# 向后兼容：onboard_user / resolve_exam_date 的实现已移到 data/onboarding.py，
# api/、examples/、tests/ 里的 `from startgraph import ...` 不用改。
__all__ = [
    "AgentState",
    "StudyAgentSystem",
    "SimplePipeline",
    "onboard_user",
    "resolve_exam_date",
]


def _merge_str(a: str, b: str) -> str:
    if not a:
        return b or ""
    if not b:
        return a
    return a.rstrip() + "\n\n---\n\n" + b.lstrip()

class AgentState(TypedDict, total=False):
    user_input: str
    user_id: str
    subject: str
    session_id: str
    namespace: str
    exam_track: str
    user_energy: float
    emotion_intensity: float
    intent_result: dict
    graph_rag_context: str
    next_review_ts: int
    plan: dict
    dashboard: dict
    memory_context: str
    memory_issue_types: List[str]
    memory_fallbacks: List[str]
    pending_memory_ask: str
    final_response: str
    agent_trace: List[str]
    reflection_notes: List[str]

class StudyAgentSystem:
    """考研/考公学习 Agent 系统门面（内置 SessionManager）。"""
    def __init__(
        self,
        *,
        use_llm_router: bool = False,
        force_memory_graph: bool = True,
    ):
        self.sessions = SessionManager()
        prod = settings.is_production
        stack = None
        if prod:
            from production.factory import build_production_stack

            # 生产环境栈包含：图谱检索、四级缓存、向量存储、BM25 索引、队列、LLM 客户端、语音识别客户端、语音合成客户端、重排模型、路由器。
            stack = build_production_stack()
            # 返回 ProductionStack：Redis、Milvus 向量库、Elasticsearch、重排模型、大模型、陪伴模型、Neo4j 图谱、同伴统计、语音合成、语音识别、任务队列、四级缓存。缺 Key 或 Neo4j 连不上会直接抛错。
            
            self.graph_rag = stack.graph_rag
            
            self.store = stack.store
            
            self.rag = AdvancedRAGEngine(
                vector_store=stack.vector_store,
                bm25_index=stack.bm25,
                llm_client=stack.llm,
                vlm_client=stack.llm,
                rerank_model=stack.reranker,
                graph_rag=self.graph_rag,
            )

            self.queue = stack.queue
            
            self.companion = CompanionAgent(
                peer=stack.peer,
                tts=stack.tts,
                asr=stack.asr,
                llm=stack.companion_llm,
            )
            
            self.router = RouterAgent(use_llm=True)
        else:
            self.graph_rag = KnowledgeGraphRAG(
                settings.neo4j_uri,
                settings.neo4j_user,
                settings.neo4j_password,
                force_memory=force_memory_graph,
            )
            self.store = TieredKnowledgeStore()
            
            self.rag = AdvancedRAGEngine(graph_rag=self.graph_rag)

            demo = "反向传播算法利用链式法则计算梯度。公式见高等数学与深度学习教材。"

            if hasattr(self.rag.vector_store, "add"):
                self.rag.vector_store.add(
                    demo, {"book_name": "深度学习花书", "page": 214, "subject": "数学"}
                )  # type: ignore
            if hasattr(self.rag.bm25_index, "add"):
                self.rag.bm25_index.add(
                    demo, {"book_name": "深度学习花书", "page": 214, "subject": "数学"}
                )  # type: ignore

            self.queue = StudyTaskQueue()

            self.companion = CompanionAgent()

            self.router = RouterAgent(use_llm=use_llm_router)

        self.calendar = ExamCalendarService()
        self.tutor = TutorAgent(
            rag_engine=self.rag, graph_rag=self.graph_rag, tiered_store=self.store
        )
        self.planner = PlannerAgent(
            scheduler=MultiFactorEbbinghausScheduler(),
            queue=self.queue,
            graph_rag=self.graph_rag,
            session_manager=self.sessions,
            calendar=self.calendar,
        )
        self.memory = MetaMemoryManager()
        from protocols.attach import attach_protocols

        attach_protocols(self)
        
    def _resolve_session(self, state: AgentState) -> SubjectSession:

        # 从 state 里提取 user_id 和 session_id，如果 session_id 不为空，则从 sessions 里获取会话，如果会话存在，则返回会话，否则创建会话。
        user_id = state.get("user_id") or "anonymous"
        sid = state.get("session_id") or ""
        if sid:
            sess = self.sessions.get_session(sid)
            if sess:
                return sess

    
        subject = state.get("subject") or state.get("exam_track") or "数学"
        # 如果 subject 为空，则从 state 里提取 exam_track，如果 exam_track 为空，则默认为「数学」。

        # 然后从 sessions 里获取或创建会话。
        return self.sessions.get_or_create_session(user_id, subject)
    def route_intent(self, state: AgentState) -> dict:
       
        
        logger.info("[Router] 解析意图… session=%s", state.get("session_id"))
        
        energy = float(state.get("user_energy", 0.7))
        
        profile = self.sessions.get_profile(state.get("user_id") or "anonymous")
        
        result = self.router.route(state.get("user_input", ""), user_energy=energy)
        
        notes = list(state.get("reflection_notes") or [])
        
        notes.extend(result.get("reflection_notes") or [])
        
        emotion = float(result.get("payload", {}).get("emotion_intensity", 0))

        # 如果全局画像存在，并且全局画像的情绪强度大于当前情绪，则更新当前情绪为全局画像的情绪强度。
        if profile and profile.emotion_intensity > emotion:
            emotion = profile.emotion_intensity

        if profile:
            energy = float(state.get("user_energy", profile.energy_level))
        return {
            "intent_result": result,
            "emotion_intensity": emotion,
            "user_energy": energy,
            "reflection_notes": notes,
            "agent_trace": list(state.get("agent_trace") or []) + ["Router"],
        }


    def check_memory(self, state: AgentState) -> dict:
        """元记忆：诊断中断/丢失并按优先级兜底，只读本科目命名空间。"""
        
        sess = self._resolve_session(state)
        user_id = sess.user_id
        topic = (state.get("intent_result") or {}).get("payload", {}).get("query", "")

        pending = str(sess.scratch.get("pending_memory_ask") or "")

        report = self.memory.recover_memory(
            user_id,
            topic or state.get("user_input", ""),
            session_id=sess.session_id,
            subject=sess.subject,
            namespace=sess.namespace,
            user_utterance=state.get("user_input", ""),
            pending_ask=pending,
            last_active_at=sess.last_active_at,
        )
        
        if pending:
            sess.scratch["pending_memory_ask"] = ""
        session_hot = sess.recent_context(8)
        
        memory_context = "\n".join(
            x for x in [session_hot, report.recovered_context] if x
        )
        
        extra = ""
        if report.ask_user:
            extra = "\n\n" + report.ask_user
            sess.scratch["pending_memory_ask"] = report.ask_user
        notes = self.memory.offline_reflect(user_id, namespace=sess.namespace)
        return {
            "session_id": sess.session_id,
            "subject": sess.subject,
            "namespace": sess.namespace,
            "memory_context": memory_context,
            "memory_issue_types": report.issue_types,
            "memory_fallbacks": report.fallbacks_used,
            "pending_memory_ask": report.ask_user or "",
            "final_response": (
                _merge_str(state.get("final_response", ""), extra)
                if extra
                else state.get("final_response", "")
            ),
            "agent_trace": list(state.get("agent_trace") or []) + ["MetaMemory"],
            "reflection_notes": list(state.get("reflection_notes") or [])
            + report.issues
            + notes,
        }

    def handle_companion_task(self, meta: dict) -> dict:
        """A2A Companion_Agent。危机话术仍在 Agent 内部，不经过生成模型。"""
        user_id = meta.get("user_id") or "anonymous"
        emotion = float(meta.get("emotion") or 0)
        energy = float(meta.get("energy") or 0.7)
        out = self.companion.comfort(
            meta.get("emotion_context") or meta.get("user_input") or "",
            emotion_intensity=emotion,
            voice_mode="voice" in (meta.get("intent") or ""),
            user_energy=energy,
            subject=meta.get("subject") or "",
            global_emotion_note=meta.get("global_note") or "",
        )
        texts = [out["response"]]
        patch = out.get("global_emotion_patch") or {}
        note = str(patch.get("note") or "")
        if note:
            verdict = self.memory.resolver.upsert_fact(
                f"{user_id}:emotion_note",
                note,
                explicit=True,
                user_id=user_id,
            )
            if verdict.get("status") == "need_clarify":
                texts.append(verdict["message"])
                note = ""
        self.sessions.update_global_emotion(
            user_id,
            emotion_intensity=float(patch.get("emotion_intensity", emotion)),
            energy_level=float(patch.get("energy_level", energy)),
            note=note,
        )
        self.memory.episodic.add_interaction(
            user_id,
            "companion",
            out["response"][:200],
            intent="emotion",
            subject="__global__",
            session_id="",
            namespace=f"{user_id}::__global__",
            importance=0.9,
        )
        return {
            "texts": texts,
            "trace": ["Companion"],
            "force_rest": bool(out.get("force_rest")),
        }

    def handle_tutor_task(self, meta: dict) -> dict:
        """A2A Tutor_Agent。题库和学完归档都走 MCP。"""
        user_id = meta.get("user_id") or "anonymous"
        subject = meta.get("subject") or ""
        query = meta.get("query") or meta.get("user_input") or ""
        gnote = meta.get("global_note") or ""
        preface = ""
        if gnote and subject and subject in gnote:
            preface = (
                f"我看你今天对「{subject}」状态不太好，"
                "我们先只讲一小段，能懂就收工。\n\n"
            )
        out = self.tutor.answer(
            query,
            user_id=user_id,
            exam_track=subject,
            memory_context=meta.get("memory_context") or "",
        )
        answer_text = out.get("response") or ""
        texts = [preface + answer_text]
        if out.get("status") == "success":
            learned = self.mcp.call_tool(
                "calendar.archive_learned",
                {
                    "user_id": user_id,
                    "subject": subject,
                    "content_summary": answer_text[:300],
                    "knowledge_points": [query[:40]],
                    "session_id": meta.get("session_id") or "",
                    "minutes": 15,
                },
            )
            texts.append(f"\n[日程] 已写入：{learned.get('title')}\n{learned.get('satisfaction_note')}")
        return {
            "texts": texts,
            "trace": ["Tutor"],
            "graph_ctx": out.get("graph_hint") or "",
        }

    def handle_planner_task(self, meta: dict) -> dict:
        """A2A Planner_Agent。看板写入日历时由计划 Agent 调用 MCP。"""
        from data.models import UserProfile

        user_id = meta.get("user_id") or "anonymous"
        subject = meta.get("subject") or ""
        energy = float(meta.get("energy") or 0.7)
        emotion = float(meta.get("emotion") or 0)
        dashboard = self.planner.build_today_dashboard(user_id)
        texts = [dashboard.get("text") or ""]
        gprofile = self.sessions.get_profile(user_id)
        profile = UserProfile(
            user_id=user_id,
            exam_track=subject,
            subjects=[subject] if subject else [],
            books=(gprofile.books_by_subject.get(subject, []) if gprofile else []),
            energy_level=energy,
            emotion_intensity=emotion,
        )
        minutes = next(
            (
                b["minutes"]
                for b in dashboard.get("blocks") or []
                if b.get("subject") == subject
            ),
            60,
        )
        plan = self.planner.make_plan(
            profile,
            focus_concept=meta.get("query") or "",
            subject=subject,
            available_minutes=minutes,
        )
        texts.append(self._format_plan(plan))
        next_ts = int(meta.get("next_review_ts") or 0)
        if not next_ts:
            sched = self.planner.apply_review_feedback(
                user_id,
                {},
                {
                    "is_correct": True,
                    "response_time": 30,
                    "difficulty": 3,
                    "user_energy": energy,
                    "emotion_intensity": emotion,
                    "exam_probability": 0.5,
                },
                subject=subject,
                session_id=meta.get("session_id") or "",
            )
            next_ts = int(sched.get("next_review_ts") or sched.get("next_review_timestamp") or 0)
        return {
            "texts": texts,
            "trace": ["Planner"],
            "plan": plan,
            "dashboard": dashboard,
            "next_review_ts": next_ts,
        }

    def dispatch_agents(self, state: AgentState) -> dict:
        
        intent = state.get("intent_result") or {}
        targets: List[str] = list(
            intent.get("target_agents") or [intent.get("target_agent") or "Companion_Agent"]
        )
        payload = intent.get("payload") or {}
        sess = self._resolve_session(state)
        user_id = sess.user_id
        energy = float(state.get("user_energy", 0.7))
        emotion = float(state.get("emotion_intensity") or payload.get("emotion_intensity") or 0)
        gprofile = self.sessions.get_profile(user_id)
        global_note = gprofile.emotion_note if gprofile else ""

        pieces: List[str] = []

        trace = list(state.get("agent_trace") or [])
        plan: dict = state.get("plan") or {}
        dashboard: dict = state.get("dashboard") or {}
        next_ts = int(state.get("next_review_ts") or 0)
        graph_ctx = state.get("graph_rag_context") or ""
        for name in targets:
            try:
                if name not in self.a2a.agents:
                    trace.append(f"Unknown:{name}")
                    continue
                data = self.a2a.send(
                    name,
                    {
                        "user_input": state.get("user_input", ""),
                        "query": payload.get("query") or state.get("user_input", ""),
                        "emotion_context": payload.get("emotion_context") or state.get("user_input", ""),
                        "intent": intent.get("intent") or "",
                        "session_id": sess.session_id,
                        "subject": sess.subject,
                        "namespace": sess.namespace,
                        "user_id": user_id,
                        "energy": energy,
                        "emotion": emotion,
                        "global_note": global_note,
                        "memory_context": state.get("memory_context") or "",
                        "next_review_ts": next_ts,
                    },
                )
                pieces.extend(data.get("texts") or [])
                trace.extend(data.get("trace") or [name])
                if data.get("graph_ctx"):
                    graph_ctx = data["graph_ctx"]
                if data.get("plan"):
                    plan = data["plan"]
                if data.get("dashboard"):
                    dashboard = data["dashboard"]
                if data.get("next_review_ts"):
                    next_ts = int(data["next_review_ts"])
                if data.get("force_rest"):
                    break
            except Exception as exc:  # noqa: BLE001
                logger.exception("Agent %s 失败: %s", name, exc)
                trace.append(f"Error:{name}")
        if not pieces:
            data = self.a2a.send(
                "Companion_Agent",
                {
                    "user_input": "系统繁忙",
                    "emotion_context": "系统繁忙",
                    "user_id": user_id,
                    "energy": energy,
                    "emotion": max(emotion, 0.3),
                    "subject": sess.subject,
                    "session_id": sess.session_id,
                },
            )
            pieces.extend(data.get("texts") or [])
            trace.append("FallbackCompanion")

        merged = "\n\n".join(p for p in pieces if p)
        sess.append_message("user", state.get("user_input", ""))
        sess.append_message("assistant", merged[:800])
        self.memory.episodic.add_interaction(
            user_id,
            "user",
            state.get("user_input", ""),
            intent=intent.get("intent", ""),
            subject=sess.subject,
            session_id=sess.session_id,
            namespace=sess.namespace,
        )
        self.memory.episodic.add_interaction(
            user_id,
            "system",
            merged[:500],
            intent="response",
            subject=sess.subject,
            session_id=sess.session_id,
            namespace=sess.namespace,
        )
        self.memory.tkg.add(
            user_id,
            "询问了",
            payload.get("query") or state.get("user_input", "")[:40],
            user_id=user_id,
            study_subject=sess.subject,
            session_id=sess.session_id,
            namespace=sess.namespace,
        )
        prev = state.get("final_response") or ""
        # 合并最终响应和片段，返回结果。
        return {    
            "final_response": _merge_str(prev, merged),
            "plan": plan,
            "dashboard": dashboard,
            "next_review_ts": next_ts,
            "graph_rag_context": graph_ctx,
            "session_id": sess.session_id,
            "subject": sess.subject,
            "namespace": sess.namespace,
            "agent_trace": trace,
        }

    @staticmethod
    def _format_plan(plan: dict) -> str:
        mode = plan.get("mode", "normal")
        msg = plan.get("message", "")
        subj = plan.get("subject") or ""
        lines = [f"【{subj} · 计划细化 · {mode}】{msg}"]
        for b in plan.get("blocks") or []:
            lines.append(
                f"- {b.get('type')}: {b.get('minutes')} 分钟 / {b.get('items')} 题"
            )
            if b.get("concepts"):
                lines.append(f"  前置补强: {', '.join(b['concepts'])}")
        if plan.get("deferred_tasks"):
            lines.append(f"（已顺延任务 {plan['deferred_tasks']} 项）")
        return "\n".join(lines)

    def build_graph(self, langgraph=None):
        try:
            from langgraph.graph import END, StateGraph

            workflow = StateGraph(AgentState)
            workflow.add_node("Router", self.route_intent)
            workflow.add_node("MetaMemory", self.check_memory)
            workflow.add_node("Dispatch", self.dispatch_agents)
            workflow.set_entry_point("Router")
            workflow.add_edge("Router", "MetaMemory")
            workflow.add_edge("MetaMemory", "Dispatch")
            workflow.add_edge("Dispatch", END)
            return workflow.compile()
        except Exception as exc:  # noqa: BLE001
            logger.warning("langgraph 不可用，使用简易流水线: %s", exc)
            return SimplePipeline(self)

    def invoke(self, user_input: str, **kwargs: Any) -> dict:
        user_id = kwargs.get("user_id", "anonymous")
        subject = kwargs.get("subject") or kwargs.get("exam_track") or "数学"
        session_id = kwargs.get("session_id") or ""
        if not session_id:
            session_id = self.sessions.get_or_create_session(user_id, subject).session_id
        g = self.sessions.get_profile(user_id)
        energy = float(kwargs.get("user_energy", g.energy_level if g else 0.7))
        emotion = float(kwargs.get("emotion_intensity", g.emotion_intensity if g else 0))
        app = self.build_graph()
        state: AgentState = {
            "user_input": user_input,
            "user_id": user_id,
            "subject": subject,
            "session_id": session_id,
            "exam_track": subject,
            "user_energy": energy,
            "emotion_intensity": emotion,
            "final_response": "",
            "agent_trace": [],
            "reflection_notes": [],
        }
        return app.invoke(state)

    def open_dashboard(self, user_id: str, **kwargs: Any) -> dict:
        """App 首页：全局今日看板 + 倒计时 + 日程同步。"""
        return self.planner.build_today_dashboard(
            user_id, available_minutes=kwargs.get("available_minutes")
        )

    def open_agenda(self, user_id: str, *, day: str = "") -> dict:
        """日历页：倒计时 + 当日日程 + 今日/本周满足感。"""
        g = self.sessions.get_profile(user_id)
        exam_date = g.target_exam_date if g else ""
        label = g.exam_track if g else "考试"
        return self.calendar.build_agenda_view(
            user_id,
            exam_date=exam_date,
            day=day or None,
            exam_label=label or "考试",
        )

    def complete_calendar_event(self, user_id: str, event_id: str, note: str = "") -> dict:
        ev = self.calendar.complete_event(user_id, event_id, note=note)
        if not ev:
            return {"ok": False, "message": "事件不存在"}
        digest = self.calendar.satisfaction_digest(user_id)
        return {
            "ok": True,
            "event": ev.to_dict(),
            "satisfaction": digest.to_dict(),
            "celebration": digest.celebration,
        }

    def archive_learning(
        self,
        user_id: str,
        *,
        subject: str,
        summary: str,
        knowledge_points: Optional[List[str]] = None,
        session_id: str = "",
        minutes: int = 15,
    ) -> dict:
        """手动把学过的内容整理进日程（错题本整理、笔记总结也可调）。"""
        ev = self.calendar.archive_learned(
            user_id,
            subject=subject,
            content_summary=summary,
            knowledge_points=knowledge_points,
            session_id=session_id,
            minutes=minutes,
        )
        return {
            "ok": True,
            "event": ev.to_dict(),
            "satisfaction": self.calendar.satisfaction_digest(user_id).to_dict(),
        }

    def open_named_window(self, user_id: str, title: str) -> dict:
        """新建对话：窗口名即科目专属会话。"""
        sess = self.sessions.open_named_window(user_id, title)
        g = self.sessions.get_profile(user_id)
        return {
            "ok": True,
            "session_id": sess.session_id,
            "subject": sess.subject,
            "namespace": sess.namespace,
            "windows": self.sessions.list_windows(user_id),
            "profile": g.to_dict() if g else {},
            "hint": f"已打开「{sess.subject}」专属会话，这一窗只记这一科。",
            "messages": [
                {"role": m.get("role"), "content": m.get("content") or ""}
                for m in sess.messages
            ],
        }

    def list_windows(self, user_id: str) -> dict:
        g = self.sessions.get_profile(user_id)
        return {
            "windows": self.sessions.list_windows(user_id),
            "profile": g.to_dict() if g else {},
        }

    def switch_subject(self, user_id: str, subject: str) -> dict:
        """看板点击某科 → 返回专属 session。"""
        sess = self.sessions.switch_subject(user_id, subject)
        g = self.sessions.get_profile(user_id)
        return {
            "session_id": sess.session_id,
            "subject": sess.subject,
            "namespace": sess.namespace,
            "global_energy": g.energy_level if g else 0.7,
            "global_emotion": g.emotion_intensity if g else 0.0,
            "emotion_note": g.emotion_note if g else "",
            "hint": f"已进入「{subject}」专属会话，历史不会与其它科目串台",
            "messages": [
                {"role": m.get("role"), "content": m.get("content") or ""}
                for m in (sess.messages or [])
            ],
        }


class SimplePipeline:
    def __init__(self, system: StudyAgentSystem):
        self.system = system

    def invoke(self, state: dict) -> dict:
        s: AgentState = dict(state)  # type: ignore
        s.update(self.system.route_intent(s))
        s.update(self.system.check_memory(s))
        s.update(self.system.dispatch_agents(s))
        return s


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    system = StudyAgentSystem(force_memory_graph=True)
    onboard_user(
        system,
        "u001",
        "考研",
        ["数学", "英语", "政治", "专业课"],
        {
            "数学": ["高等数学同济版"],
            "英语": ["黄皮书"],
            "政治": ["精讲精练"],
            "专业课": ["学校指定教材"],
        },
        target_exam_date="2026-12-20",
    )
    print(system.open_dashboard("u001")["text"])
    print("--- 日历页 ---")
    agenda = system.open_agenda("u001")
    print(agenda.get("countdown"))
    print("今日事件数:", len(agenda.get("events") or []))
    print(system.switch_subject("u001", "数学"))
    result = system.invoke(
        "反向传播算法的链式法则到底是怎么推导的？",
        user_id="u001",
        subject="数学",
        user_energy=0.8,
    )
    print("=== 数学会话回复 ===")
    print(result.get("final_response"))
    print("session:", result.get("session_id"), "trace:", result.get("agent_trace"))
    print("满足感:", system.calendar.satisfaction_digest("u001").celebration)
