# -*- coding: utf-8 -*-
"""
制定计划 Agent
-------------
- 单科会话内：多因子艾宾浩斯 + GraphRAG 前置补强
- 全局：今日看板跨科目分配时间（对话仍隔离）
任务入队必须带 subject，供看板聚合与会话过滤。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.reflection import ReflectionEngine
from data.models import ReviewFeedback, StudentSpacedRecord, UserProfile
from data.session import GlobalUserProfile, SessionManager
from data.study_queue import StudyTaskQueue
from retrieval.graph_rag import KnowledgeGraphRAG
from scheduling.calendar import CountdownInfo, ExamCalendarService, SatisfactionDigest
from scheduling.dashboard import DailyDashboardScheduler
from scheduling.multi_factor_abhs import MultiFactorEbbinghausScheduler


class PlannerAgent:
    def __init__(
        self,
        scheduler: Optional[MultiFactorEbbinghausScheduler] = None,
        queue: Optional[StudyTaskQueue] = None,
        graph_rag: Optional[KnowledgeGraphRAG] = None,
        session_manager: Optional[SessionManager] = None,
        calendar: Optional[ExamCalendarService] = None,
    ):
        self.scheduler = scheduler or MultiFactorEbbinghausScheduler()
        self.queue = queue or StudyTaskQueue()
        self.graph_rag = graph_rag or KnowledgeGraphRAG(force_memory=True)
        self.sessions = session_manager or SessionManager()
        self.dashboard = DailyDashboardScheduler()
        self.calendar = calendar or ExamCalendarService()
        self.reflection = ReflectionEngine()
        self.mcp = None  # 由 protocols.attach 注入；日历与图谱走 MCP

    def make_plan(
        self,
        profile: UserProfile,
        *,
        due_count: int = 10,
        high_freq_count: int = 5,
        error_book_count: int = 5,
        focus_concept: str = "",
        available_minutes: int = 120,
        subject: str = "",
    ) -> Dict[str, Any]:
        """单科会话内的当日块计划（被看板某一科点击后细化）。"""
        raw = self.scheduler.build_daily_plan(
            due_count=due_count,
            high_freq_count=high_freq_count,
            error_book_count=error_book_count,
            user_energy=profile.energy_level,
            emotion_intensity=profile.emotion_intensity,
            available_minutes=available_minutes,
        )

        graph_hint = {}
        if focus_concept:
            if self.mcp is not None:
                graph_hint = self.mcp.call_tool("graph.plan", {"concept": focus_concept}) or {}
            else:
                graph_hint = self.graph_rag.traverse_for_plan(focus_concept)
            if graph_hint.get("must_review_first") and raw.get("mode") == "normal":
                raw.setdefault("blocks", []).insert(
                    0,
                    {
                        "type": "prerequisite_patch",
                        "minutes": 15,
                        "items": len(graph_hint["must_review_first"]),
                        "concepts": graph_hint["must_review_first"],
                    },
                )
                raw["message"] = (raw.get("message") or "") + "；已按图谱补前置基础。"

        reflected = self.reflection.reflect_plan(
            raw,
            emotion_intensity=profile.emotion_intensity,
            user_energy=profile.energy_level,
        )
        plan = reflected.sanitized_payload
        plan["reflection_notes"] = reflected.notes
        plan["graph_hint"] = graph_hint
        plan["subjects"] = profile.subjects
        plan["books"] = profile.books
        plan["subject"] = subject or (profile.subjects[0] if profile.subjects else "")

        if plan.get("mode") in {"rest_only", "light"} or profile.emotion_intensity >= 0.85:
            deferred = self.queue.defer_all_today(
                profile.user_id,
                defer_seconds=3600 * 24,
                subject=subject or None,
            )
            plan["deferred_tasks"] = deferred

        return plan

    def build_today_dashboard(
        self,
        user_id: str,
        *,
        available_minutes: Optional[int] = None,
        sync_calendar: bool = True,
    ) -> Dict[str, Any]:
        """
        全局看板：打开 App 首页用。
        同步写入当日日程；附带考试倒计时与满足感摘要。
        """
        profile = self.sessions.get_profile(user_id)
        if not profile:
            profile = GlobalUserProfile(user_id=user_id)
            self.sessions.upsert_profile(profile)
        if profile.exam_track == "考研" and not profile.target_exam_date:
            from scheduling.calendar import next_kaoyan_exam_date

            profile.target_exam_date = next_kaoyan_exam_date().isoformat()
        if available_minutes is not None:
            profile.daily_available_minutes = available_minutes

        counts = self.queue.count_by_subject(user_id)
        due_by = {s: v.get("total", 0) for s, v in counts.items()}
        error_by = {s: v.get("error_book", 0) for s, v in counts.items()}
        hf_by = {s: v.get("high_freq_focus", 0) for s, v in counts.items()}

        # 临近考试：自动压缩看板总时长（考前一周轻量）
        countdown = None
        if profile.target_exam_date:
            countdown = self._countdown(profile.target_exam_date, profile.exam_track or "考试")
            if countdown and countdown.days_left <= 7:
                profile.daily_available_minutes = min(profile.daily_available_minutes, 120)
            elif countdown and countdown.days_left <= 30:
                # 冲刺期略增高频权重感：看板分钟可保持，阶段提示交给日历
                pass

        dash = self.dashboard.build(
            profile,
            session_manager=self.sessions,
            due_by_subject=due_by,
            error_by_subject=error_by,
            high_freq_by_subject=hf_by,
        )
        dash["text"] = DailyDashboardScheduler.format_text(dash)

        if sync_calendar and dash.get("mode") != "rest_only":
            dash["calendar_synced"] = self._sync_dashboard(user_id, dash)
        else:
            dash["calendar_synced"] = 0

        if countdown:
            dash["countdown"] = countdown.to_dict()
            dash["phase_hint"] = self._phase_hint(countdown.days_left)
            dash["text"] = (
                f"[倒计时] {countdown.message}\n"
                f"阶段：{countdown.phase_title} — {countdown.phase_tip}\n\n"
                + dash["text"]
            )

        digest = self._satisfaction(user_id)
        dash["satisfaction"] = digest.to_dict()
        if digest.done_count:
            dash["text"] += f"\n\n[成就] {digest.celebration}"
        return dash

    def apply_review_feedback(
        self,
        user_id: str,
        record: StudentSpacedRecord | dict,
        feedback: ReviewFeedback | dict,
        *,
        question_id: str = "",
        subject: str = "",
        session_id: str = "",
    ) -> Dict[str, Any]:
        result = self.scheduler.calculate_next_review(record, feedback)
        if not result.get("force_rest"):
            self.queue.add_task(
                user_id,
                result.get("suggested_task_type", "due_review"),
                result["next_review_timestamp"],
                question_id=question_id,
                subject=subject,
                session_id=session_id,
                meta={"factor_breakdown": result.get("factor_breakdown")},
            )
            # 同步到日历：下次复习日
            from datetime import datetime

            due_day = datetime.fromtimestamp(result["next_review_timestamp"]).date().isoformat()
            self.calendar.add_review_event(
                user_id,
                subject=subject or "未分科",
                session_id=session_id,
                question_id=question_id,
                due_day=due_day,
                title=f"{subject or '复习'} · 艾宾浩斯到期",
            )
            # 若本次答对，把内容沉淀进「已学归档」增强满足感
            fb = feedback if isinstance(feedback, dict) else feedback.__dict__
            if fb.get("is_correct", True):
                self._archive_learned(
                    user_id,
                    subject=subject or "未分科",
                    content_summary=f"复习完成 question={question_id or '今日任务'}",
                    knowledge_points=[question_id] if question_id else [],
                    session_id=session_id,
                    minutes=max(5, int(float(fb.get("response_time", 30)) / 60) or 5),
                )
        else:
            self.queue.defer_all_today(user_id, subject=subject or None)
        result["subject"] = subject
        result["session_id"] = session_id
        result["satisfaction"] = self._satisfaction(user_id).to_dict()
        return result

    def _countdown(self, exam_date: str, exam_label: str) -> Optional[CountdownInfo]:
        if self.mcp is None:
            return self.calendar.countdown(exam_date, exam_label=exam_label)
        raw = self.mcp.call_tool(
            "calendar.countdown",
            {"exam_date": exam_date, "exam_label": exam_label},
        )
        return CountdownInfo(**raw) if raw else None

    def _phase_hint(self, days_left: int) -> Dict[str, Any]:
        if self.mcp is None:
            return self.calendar.phase_plan_hint(days_left)
        return self.mcp.call_tool("calendar.phase_hint", {"days_left": days_left}) or {}

    def _sync_dashboard(self, user_id: str, dash: Dict[str, Any]) -> int:
        if self.mcp is None:
            return len(self.calendar.sync_from_dashboard(user_id, dash))
        synced = self.mcp.call_tool(
            "calendar.sync_dashboard",
            {"user_id": user_id, "dashboard": dash},
        )
        return int((synced or {}).get("count") or 0)

    def _satisfaction(self, user_id: str) -> SatisfactionDigest:
        if self.mcp is None:
            return self.calendar.satisfaction_digest(user_id, period="today")
        raw = self.mcp.call_tool(
            "calendar.satisfaction",
            {"user_id": user_id, "period": "today"},
        )
        return SatisfactionDigest(**raw)

    def _archive_learned(self, user_id: str, **kwargs: Any) -> None:
        if self.mcp is None:
            self.calendar.archive_learned(user_id, **kwargs)
            return
        self.mcp.call_tool("calendar.archive_learned", {"user_id": user_id, **kwargs})
