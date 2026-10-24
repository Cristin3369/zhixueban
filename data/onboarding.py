# -*- coding: utf-8 -*-
"""用户注册登记：登录、选科目、填教材、算考试日。

原本写在 startgraph.py 里，和 LangGraph 编排混在一起。抽出来后：
  - startgraph.py 只做门面转发，`from startgraph import onboard_user` 仍然可用
  - 本模块不反向 import startgraph（只在 TYPE_CHECKING 里取类型），避免循环依赖
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List

from data.books import format_book, group_books, normalize_books
from data.models import UserProfile
from data.session import GlobalUserProfile
from scheduling.calendar import next_kaoyan_exam_date

if TYPE_CHECKING:
    from startgraph import StudyAgentSystem

logger = logging.getLogger(__name__)


def resolve_exam_date(exam_track: str, target_exam_date: str = "") -> str:
    """考研按全国初试日历自动算；其它考种才用手填日期。"""
    if (target_exam_date or "").strip():
        return target_exam_date.strip()
    if exam_track == "考研":
        return next_kaoyan_exam_date().isoformat()
    return ""


def onboard_user(
    system: StudyAgentSystem,
    user_id: str,
    exam_track: str,
    subjects: List[str] | None = None,
    books: List[str] | Dict[str, List[str]] | None = None,
    *,
    school: str = "",
    major: str = "",
    books_note: str = "",
    daily_available_minutes: int = 240,
    subject_weights: Dict[str, float] | None = None,
    target_exam_date: str = "",
    keep_windows: bool = True,
) -> GlobalUserProfile:
    """
    登录登记：名字、考研/考公、学校、专业、教材。
    科目窗口不在登录时写死，由用户「新建对话」起名后再开专属会话。
    考研考试日按全国统考自动计算。
    """
    existing = system.sessions.get_profile(user_id)
    # None=未传则沿用已有窗口；[]=登录时明确还没开窗，不要把旧四科带回来
    if subjects is None:
        subjects = list(existing.subjects) if (keep_windows and existing) else []
    else:
        subjects = list(subjects)
    books_note = (books_note or "").strip()
    target_exam_date = resolve_exam_date(exam_track, target_exam_date)
    book_entries = normalize_books(books, books_note)
    if not book_entries and existing and keep_windows:
        book_entries = list(existing.books or [])
    books_by_subject = group_books(book_entries)
    books_note = "；".join(format_book(b) for b in book_entries)

    if existing and keep_windows:
        school = school or existing.school
        major = major or existing.major
        if not subject_weights:
            subject_weights = existing.subject_weights

    profile = GlobalUserProfile(
        user_id=user_id,
        exam_track=exam_track,
        school=school,
        major=major,
        subjects=list(subjects),
        books_by_subject=books_by_subject,
        books=book_entries,
        books_note=books_note,
        daily_available_minutes=daily_available_minutes,
        subject_weights=subject_weights or {},
        target_exam_date=target_exam_date,
    )
    system.sessions.upsert_profile(profile)
    catalog = list(dict.fromkeys(
        ([books_note] if books_note else [])
        + [b for bs in books_by_subject.values() for b in bs]
    ))
    system.store.save_user_profile(
        UserProfile(
            user_id=user_id,
            exam_track=exam_track,
            school=school,
            major=major,
            subjects=subjects,
            books=catalog,
            target_exam_date=target_exam_date,
        )
    )
    sessions = system.sessions.ensure_sessions_for_subjects(user_id, subjects) if subjects else []
    for book in catalog:
        qid = f"catalog:{book}"
        system.store.put_l4(exam_track, qid, {"book": book, "note": "扩展教辅目录占位"})
        system.store.register_known_id(qid)

    if target_exam_date and subjects:
        system.calendar.plan_week_skeleton(
            user_id,
            subjects,
            exam_date=target_exam_date,
            minutes_per_day=daily_available_minutes,
        )

    logger.info(
        "onboard user=%s school=%s major=%s windows=%s exam=%s",
        user_id,
        school,
        major,
        [s.session_id for s in sessions],
        target_exam_date,
    )
    return profile
