# -*- coding: utf-8 -*-
"""
考试日历 · 倒计时 · 学习日程表
-----------------------------
目标：
1. 根据考试日倒计时，自动划分备考阶段（基础/强化/模考/冲刺）；
2. 把「今日看板」与艾宾浩斯任务落到可勾选的日程事件；
3. 把用户真正学过的内容整理进日程「已完成」归档，形成满足感与连续打卡。

注意：日程是【全局视图】，事件仍带 subject / session_id，点击可回到对应科目会话。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class PrepPhase(str, Enum):
    """距考试天数对应的备考阶段（可按考种微调）。"""

    FOUNDATION = "foundation"  # 基础夯实
    INTENSIFY = "intensify"  # 强化提高
    MOCK = "mock"  # 模考查漏
    SPRINT = "sprint"  # 冲刺背诵
    EXAM_WEEK = "exam_week"  # 考前一周
    EXAM_DAY = "exam_day"
    AFTER = "after"


class EventStatus(str, Enum):
    PLANNED = "planned"
    DONE = "done"
    SKIPPED = "skipped"
    DEFERRED = "deferred"


class EventKind(str, Enum):
    STUDY = "study"  # 计划学习块
    REVIEW = "review"  # 艾宾浩斯复习
    LEARNED = "learned"  # 已学内容归档（满足感来源）
    MILESTONE = "milestone"  # 里程碑庆祝
    REST = "rest"
    MOCK_EXAM = "mock_exam"


PHASE_COPY = {
    PrepPhase.FOUNDATION: ("基础夯实期", "重理解与体系，少盲目刷难题"),
    PrepPhase.INTENSIFY: ("强化提高期", "高频考点 + 错题闭环"),
    PrepPhase.MOCK: ("模考查漏期", "套卷节奏，专攻薄弱模块"),
    PrepPhase.SPRINT: ("冲刺背诵期", "政治/作文/公式清单反复过"),
    PrepPhase.EXAM_WEEK: ("考前一周", "只做轻量回顾与作息调整"),
    PrepPhase.EXAM_DAY: ("考试日", "稳住节奏，相信积累"),
    PrepPhase.AFTER: ("考后", "好好休息，等待成绩"),
}


def next_kaoyan_exam_date(today: Optional[date] = None) -> date:
    """全国硕士研究生初试：当年 12 月第三个星期六。已考完则取下一年。"""
    today = today or date.today()

    def third_saturday(year: int) -> date:
        first = date(year, 12, 1)
        offset = (5 - first.weekday()) % 7
        return date(year, 12, 1 + offset + 14)

    exam = third_saturday(today.year)
    if today > exam + timedelta(days=1):
        exam = third_saturday(today.year + 1)
    return exam


def parse_exam_date(s: str) -> Optional[date]:
    """支持 YYYY-MM-DD / YYYYMMDD / YYYY/MM/DD。"""
    if not s:
        return None
    s = s.strip().replace("/", "-")
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s[:10] if fmt == "%Y-%m-%d" else s[:8], fmt).date()
        except ValueError:
            continue
    return None


def days_until(exam: date, today: Optional[date] = None) -> int:
    today = today or date.today()
    return (exam - today).days


def phase_for_days(days_left: int) -> PrepPhase:
    if days_left < 0:
        return PrepPhase.AFTER
    if days_left == 0:
        return PrepPhase.EXAM_DAY
    if days_left <= 7:
        return PrepPhase.EXAM_WEEK
    if days_left <= 30:
        return PrepPhase.SPRINT
    if days_left <= 60:
        return PrepPhase.MOCK
    if days_left <= 120:
        return PrepPhase.INTENSIFY
    return PrepPhase.FOUNDATION


@dataclass
class CountdownInfo:
    exam_date: str
    days_left: int
    hours_left: float
    phase: str
    phase_title: str
    phase_tip: str
    urgency: str  # calm | normal | high | critical
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CalendarEvent:
    """日程表单条事件。"""

    event_id: str
    user_id: str
    title: str
    day: str  # YYYY-MM-DD
    kind: str = EventKind.STUDY.value
    subject: str = ""
    session_id: str = ""
    minutes: int = 30
    status: str = EventStatus.PLANNED.value
    source: str = ""  # dashboard | ebbinghaus | manual | learned_auto
    summary: str = ""  # 学了什么（满足感文案素材）
    knowledge_points: List[str] = field(default_factory=list)
    satisfaction_note: str = ""  # 完成后展示的鼓励语
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SatisfactionDigest:
    """满足感摘要：周报 / 今日成就。"""

    period: str  # today | week
    done_count: int
    minutes_studied: int
    subjects_touched: List[str]
    highlights: List[str]
    streak_days: int
    celebration: str

    def to_dict(self) -> dict:
        return asdict(self)


class ExamCalendarService:
    """
    日历服务（内存实现；生产可接 Google Calendar / 钉钉 / 自建 PG）。
    """

    def __init__(self) -> None:
        # user_id -> events
        self._events: Dict[str, List[CalendarEvent]] = {}
        # user_id -> 连续打卡日期集合 YYYY-MM-DD
        self._checkin_days: Dict[str, set] = {}

    # ---------- 倒计时 ---------- #
    def countdown(
        self,
        exam_date_str: str,
        *,
        now: Optional[datetime] = None,
        exam_label: str = "考试",
    ) -> Optional[CountdownInfo]:
        exam = parse_exam_date(exam_date_str)
        if not exam:
            return None
        now = now or datetime.now()
        exam_dt = datetime.combine(exam, datetime.min.time().replace(hour=9))
        delta = exam_dt - now
        days_left = days_until(exam, now.date())
        hours_left = max(0.0, delta.total_seconds() / 3600.0)
        phase = phase_for_days(days_left)
        title, tip = PHASE_COPY[phase]

        if days_left < 0:
            urgency, msg = "calm", f"{exam_label}已结束，辛苦了。"
        elif days_left == 0:
            urgency, msg = "critical", f"今天就是{exam_label}！稳住，你准备过的都在。"
        elif days_left <= 7:
            urgency, msg = "critical", f"距{exam_label}还有 {days_left} 天，进入轻量冲刺与作息管理。"
        elif days_left <= 30:
            urgency, msg = "high", f"距{exam_label}还有 {days_left} 天，请盯紧高频与错题。"
        elif days_left <= 90:
            urgency, msg = "normal", f"距{exam_label}还有 {days_left} 天，按阶段推进即可。"
        else:
            urgency, msg = "calm", f"距{exam_label}还有 {days_left} 天，先把基础打牢。"

        return CountdownInfo(
            exam_date=exam.isoformat(),
            days_left=days_left,
            hours_left=round(hours_left, 1),
            phase=phase.value,
            phase_title=title,
            phase_tip=tip,
            urgency=urgency,
            message=msg,
        )

    def phase_plan_hint(self, days_left: int) -> Dict[str, Any]:
        """按阶段给出日程侧重点（给看板/计划 Agent 参考）。"""
        phase = phase_for_days(days_left)
        title, tip = PHASE_COPY[phase]
        focus = {
            PrepPhase.FOUNDATION: ["体系梳理", "基础题", "前置知识补漏"],
            PrepPhase.INTENSIFY: ["高频考点", "错题本", "限时训练"],
            PrepPhase.MOCK: ["整卷模考", "薄弱模块专攻", "复盘错因"],
            PrepPhase.SPRINT: ["清单背诵", "公式卡片", "作文框架"],
            PrepPhase.EXAM_WEEK: ["轻量回顾", "作息调整", "心态安抚"],
            PrepPhase.EXAM_DAY: ["证件检查", "放松"],
            PrepPhase.AFTER: ["休息"],
        }[phase]
        return {"phase": phase.value, "title": title, "tip": tip, "focus": focus}

    # ---------- 日程 CRUD ---------- #
    def add_event(self, event: CalendarEvent) -> CalendarEvent:
        self._events.setdefault(event.user_id, []).append(event)
        return event

    def list_events(
        self,
        user_id: str,
        *,
        day: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        subject: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[CalendarEvent]:
        items = list(self._events.get(user_id, []))
        if day:
            items = [e for e in items if e.day == day]
        if start:
            items = [e for e in items if e.day >= start]
        if end:
            items = [e for e in items if e.day <= end]
        if subject:
            items = [e for e in items if e.subject == subject]
        if status:
            items = [e for e in items if e.status == status]
        items.sort(key=lambda e: (e.day, e.created_at))
        return items

    def get_event(self, user_id: str, event_id: str) -> Optional[CalendarEvent]:
        for e in self._events.get(user_id, []):
            if e.event_id == event_id:
                return e
        return None

    # ---------- 从看板 / 学习内容生成日程 ---------- #
    def sync_from_dashboard(
        self,
        user_id: str,
        dashboard: Dict[str, Any],
        *,
        day: Optional[str] = None,
    ) -> List[CalendarEvent]:
        """
        把今日看板 blocks 写成当天「计划学习」事件。
        同一天同科目同 title 不重复插入（幂等）。
        """
        day = day or date.today().isoformat()
        existing = {(e.subject, e.title, e.day) for e in self.list_events(user_id, day=day)}
        created: List[CalendarEvent] = []
        for b in dashboard.get("blocks") or []:
            title = f"{b.get('subject')} · {b.get('task_summary') or '学习块'}"
            key = (b.get("subject", ""), title, day)
            if key in existing:
                continue
            ev = CalendarEvent(
                event_id=uuid.uuid4().hex[:12],
                user_id=user_id,
                title=title,
                day=day,
                kind=EventKind.STUDY.value,
                subject=b.get("subject", ""),
                session_id=b.get("session_id", ""),
                minutes=int(b.get("minutes") or 30),
                source="dashboard",
                summary=b.get("task_summary", ""),
                meta={"order": b.get("order", 0)},
            )
            self.add_event(ev)
            created.append(ev)
        return created

    def add_review_event(
        self,
        user_id: str,
        *,
        subject: str,
        session_id: str = "",
        question_id: str = "",
        due_day: Optional[str] = None,
        minutes: int = 20,
        title: str = "",
    ) -> CalendarEvent:
        """艾宾浩斯到期 → 日程「复习」事件。"""
        due_day = due_day or date.today().isoformat()
        ev = CalendarEvent(
            event_id=uuid.uuid4().hex[:12],
            user_id=user_id,
            title=title or f"{subject} · 到期复习",
            day=due_day,
            kind=EventKind.REVIEW.value,
            subject=subject,
            session_id=session_id,
            minutes=minutes,
            source="ebbinghaus",
            summary=f"question_id={question_id}" if question_id else "",
            meta={"question_id": question_id},
        )
        return self.add_event(ev)

    def archive_learned(
        self,
        user_id: str,
        *,
        subject: str,
        content_summary: str,
        knowledge_points: Optional[List[str]] = None,
        session_id: str = "",
        minutes: int = 0,
        day: Optional[str] = None,
    ) -> CalendarEvent:
        """
        把用户刚学过的内容整理进日程「已完成归档」。
        这是满足感的核心：看得见「我今天学了什么」。
        """
        day = day or date.today().isoformat()
        kps = knowledge_points or []
        kp_text = "、".join(kps[:5]) if kps else content_summary[:40]
        note = self._satisfaction_line(subject, kp_text)
        ev = CalendarEvent(
            event_id=uuid.uuid4().hex[:12],
            user_id=user_id,
            title=f"[已掌握] {subject} · {kp_text[:24]}",
            day=day,
            kind=EventKind.LEARNED.value,
            subject=subject,
            session_id=session_id,
            minutes=minutes,
            status=EventStatus.DONE.value,
            source="learned_auto",
            summary=content_summary[:500],
            knowledge_points=kps,
            satisfaction_note=note,
            completed_at=time.time(),
        )
        self.add_event(ev)
        self._mark_checkin(user_id, day)
        # 每累计 5 个 learned 插一条里程碑
        learned_today = [
            e
            for e in self.list_events(user_id, day=day, status=EventStatus.DONE.value)
            if e.kind == EventKind.LEARNED.value
        ]
        if len(learned_today) > 0 and len(learned_today) % 5 == 0:
            self.add_event(
                CalendarEvent(
                    event_id=uuid.uuid4().hex[:12],
                    user_id=user_id,
                    title=f"[里程碑] 今日已沉淀 {len(learned_today)} 个知识点",
                    day=day,
                    kind=EventKind.MILESTONE.value,
                    status=EventStatus.DONE.value,
                    source="auto_milestone",
                    satisfaction_note="进度被看见了，这就是在变强。",
                    completed_at=time.time(),
                )
            )
        return ev

    def complete_event(self, user_id: str, event_id: str, *, note: str = "") -> Optional[CalendarEvent]:
        """用户勾选完成计划块 → 打卡 + 满足感文案。"""
        ev = self.get_event(user_id, event_id)
        if not ev:
            return None
        ev.status = EventStatus.DONE.value
        ev.completed_at = time.time()
        if note:
            ev.summary = (ev.summary + " | " + note).strip(" |")
        if not ev.satisfaction_note:
            ev.satisfaction_note = self._satisfaction_line(ev.subject, ev.title)
        self._mark_checkin(user_id, ev.day)
        return ev

    # ---------- 满足感 / 打卡 ---------- #
    def streak_days(self, user_id: str, *, today: Optional[date] = None) -> int:
        today = today or date.today()
        days = self._checkin_days.get(user_id) or set()
        streak = 0
        cur = today
        while cur.isoformat() in days:
            streak += 1
            cur -= timedelta(days=1)
        return streak

    def satisfaction_digest(
        self,
        user_id: str,
        *,
        period: str = "today",
        today: Optional[date] = None,
    ) -> SatisfactionDigest:
        today = today or date.today()
        if period == "week":
            start = (today - timedelta(days=6)).isoformat()
            end = today.isoformat()
            events = self.list_events(user_id, start=start, end=end, status=EventStatus.DONE.value)
        else:
            events = self.list_events(
                user_id, day=today.isoformat(), status=EventStatus.DONE.value
            )

        subjects = sorted({e.subject for e in events if e.subject})
        minutes = sum(e.minutes for e in events)
        highlights = []
        for e in events:
            if e.kind in {EventKind.LEARNED.value, EventKind.MILESTONE.value}:
                highlights.append(e.title)
            elif e.satisfaction_note:
                highlights.append(e.satisfaction_note)
        highlights = highlights[:8]
        streak = self.streak_days(user_id, today=today)
        celebration = self._celebration(len(events), minutes, streak, subjects)
        return SatisfactionDigest(
            period=period,
            done_count=len(events),
            minutes_studied=minutes,
            subjects_touched=subjects,
            highlights=highlights,
            streak_days=streak,
            celebration=celebration,
        )

    def build_agenda_view(
        self,
        user_id: str,
        *,
        exam_date: str = "",
        day: Optional[str] = None,
        exam_label: str = "考研",
    ) -> Dict[str, Any]:
        """
        前端「日历页」一次拉取：倒计时 + 当日日程 + 满足感摘要 + 阶段提示。
        """
        day = day or date.today().isoformat()
        cd = self.countdown(exam_date, exam_label=exam_label) if exam_date else None
        events = [e.to_dict() for e in self.list_events(user_id, day=day)]
        digest = self.satisfaction_digest(user_id, period="today")
        week = self.satisfaction_digest(user_id, period="week")
        phase_hint = {}
        if cd:
            phase_hint = self.phase_plan_hint(cd.days_left)
        return {
            "day": day,
            "countdown": cd.to_dict() if cd else None,
            "phase_hint": phase_hint,
            "events": events,
            "today_satisfaction": digest.to_dict(),
            "week_satisfaction": week.to_dict(),
            "ui_hint": "勾选完成或学完知识点会自动归档，点亮连续打卡与里程碑",
        }

    def plan_week_skeleton(
        self,
        user_id: str,
        subjects: List[str],
        *,
        exam_date: str = "",
        start: Optional[date] = None,
        minutes_per_day: int = 240,
    ) -> List[CalendarEvent]:
        """
        根据倒计时阶段，预生成未来 7 天空壳日程（再由看板精细填充）。
        """
        start = start or date.today()
        cd = self.countdown(exam_date) if exam_date else None
        days_left = cd.days_left if cd else 180
        hint = self.phase_plan_hint(days_left)
        created: List[CalendarEvent] = []
        if not subjects:
            return created
        per = max(20, minutes_per_day // max(len(subjects), 1))
        for i in range(7):
            d = (start + timedelta(days=i)).isoformat()
            focus = hint["focus"][i % len(hint["focus"])]
            for j, subj in enumerate(subjects):
                title = f"{subj} · {hint['title']} · {focus}"
                # 避免重复
                if any(e.day == d and e.subject == subj and e.source == "week_skeleton" for e in self.list_events(user_id, day=d)):
                    continue
                ev = CalendarEvent(
                    event_id=uuid.uuid4().hex[:12],
                    user_id=user_id,
                    title=title,
                    day=d,
                    kind=EventKind.STUDY.value if hint["phase"] != PrepPhase.EXAM_WEEK.value else EventKind.REST.value,
                    subject=subj,
                    minutes=per if hint["phase"] != PrepPhase.EXAM_DAY.value else 15,
                    source="week_skeleton",
                    summary=hint["tip"],
                    meta={"phase": hint["phase"], "order": j},
                )
                self.add_event(ev)
                created.append(ev)
        return created

    # ---------- 内部 ---------- #
    def _mark_checkin(self, user_id: str, day: str) -> None:
        self._checkin_days.setdefault(user_id, set()).add(day)

    @staticmethod
    def _satisfaction_line(subject: str, tip: str) -> str:
        subject = subject or "今天"
        return f"太好了！「{subject}」又推进了一格：{tip[:36]}。积累正在变成实力。"

    @staticmethod
    def _celebration(done: int, minutes: int, streak: int, subjects: List[str]) -> str:
        if done == 0:
            return "今天还是白纸，完成一小块就会点亮日程，给自己一个可见的进度。"
        subj = "、".join(subjects) if subjects else "多科"
        base = f"今日完成 {done} 项、约 {minutes} 分钟，覆盖 {subj}。"
        if streak >= 7:
            return base + f" 已连续打卡 {streak} 天，你的日程表就是勋章墙。"
        if streak >= 3:
            return base + f" 连续 {streak} 天在线，节奏很漂亮。"
        return base + " 每勾掉一项，都是实实在在的进步。"
