# -*- coding: utf-8 -*-
"""
多科目会话管理（强烈建议：一科一会话）
------------------------------------
结论：考研 4 科不要塞进同一个对话框。

为什么必须隔离？
1. 上下文污染：英语语境串进政治 / 数学，引发幻觉
2. Token 爆炸与注意力稀释：四科历史挤爆窗口
3. 艾宾浩斯队列错乱：错题无法打准科目标签

本模块落地三层：
A. 会话级隔离 Session：session_id = user_{id}_{subject_slug}_{date}
B. 长期记忆隔离 Memory Namespace：检索强制 subject / session_id 过滤
C. 全局共享画像 User Profile：情绪/精力跨会话可读，辅导可据此降压

时间怎么排？
- 对话隔离，但调度引擎全局统一（任务打 subject 标签进 Redis 队列）
- 前端「今日学习看板」按科目块展示；点击某科 → 唤起该科专属 session
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def slugify_subject(subject: str) -> str:
    """
    窗口名 → session / 命名空间短标识。
    按用户起的全名隔离：英语一 和 英语 不是同一个窗。
    """
    s = (subject or "").strip()
    if not s:
        return "unknown"
    if re.search(r"[\u4e00-\u9fff]", s):
        return "subj_" + "".join(f"{ord(c):x}" for c in s[:24])
    safe = re.sub(r"[^\w]+", "_", s.lower(), flags=re.UNICODE).strip("_")
    return safe or "unknown"


def make_session_id(user_id: str, subject: str, day: Optional[str] = None) -> str:
    """
    标准 session_id：user_{uid}_{subject_slug}_{YYYYMMDD}
    同一天同科复用同一会话，跨天自动新开（也可改为长期固定会话）。
    """
    day = day or datetime.now().strftime("%Y%m%d")
    return f"user_{user_id}_{slugify_subject(subject)}_{day}"


def memory_namespace(user_id: str, subject: str) -> str:
    """
    长期记忆 / 向量库命名空间。
    检索时必须：WHERE user_id=? AND subject=?（或 namespace=?）
    """
    return f"{user_id}::{slugify_subject(subject)}"


@dataclass
class GlobalUserProfile:
    """
    全局共享画像（跨 4 个科目会话可读可写）。
    陪伴 Agent 写入的焦虑状态，数学会话打开时立刻可见。
    """

    user_id: str
    exam_track: str = "考研"  # 考研 / 考公
    school: str = ""
    major: str = ""
    # 会话窗口名（一窗一科），由用户新建对话时写入，不写死四科
    subjects: List[str] = field(default_factory=list)
    # 科目 → 指定教材列表
    books_by_subject: Dict[str, List[str]] = field(default_factory=dict)
    books: List[Dict[str, str]] = field(default_factory=list)  # {title, edition, kind}
    books_note: str = ""
    target_exam_date: str = ""
    energy_level: float = 0.7
    emotion_intensity: float = 0.0
    emotion_note: str = ""  # 如「不想学数学」
    preferred_mode: str = "text"
    # 今日总可用学习分钟（看板分配用）
    daily_available_minutes: int = 240
    # 科目权重：可按弱科加大（默认均分）
    subject_weights: Dict[str, float] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SubjectSession:
    """
    单科目独立会话。
    对话历史、临时工具变量、短期上下文只挂在本 session 下；
    其它科目的 session 绝对读不到（防串台）。
    """

    session_id: str
    user_id: str
    subject: str
    subject_slug: str
    namespace: str  # 长期记忆命名空间
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    # 本会话短期消息（热数据）；生产可落 Redis List
    messages: List[Dict[str, Any]] = field(default_factory=list)
    # 本会话临时变量（如当前章节、正在讲的题号）
    scratch: Dict[str, Any] = field(default_factory=dict)
    active: bool = True

    def append_message(self, role: str, content: str, **meta: Any) -> None:
        self.messages.append(
            {
                "role": role,
                "content": content,
                "ts": time.time(),
                "meta": meta,
            }
        )
        self.last_active_at = time.time()
        # 防止单会话无限膨胀：只保留最近 N 轮在 RAM
        if len(self.messages) > 40:
            self.messages = self.messages[-40:]

    def recent_context(self, n: int = 10) -> str:
        """仅本科目会话的热上下文，绝不混入其它科。"""
        lines = []
        for m in self.messages[-n:]:
            lines.append(f"[{m['role']}] {m['content']}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class DashboardBlock:
    """今日看板上的一个科目时间块。"""

    subject: str
    session_id: str
    minutes: int
    task_summary: str
    due_count: int = 0
    error_count: int = 0
    high_freq_count: int = 0
    order: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class SessionManager:
    """
    会话注册表：一用户多科目多会话。
    - open_subject_session / get_or_create
    - 列出某用户今日所有科目会话
    - 切换科目 = 切换 session_id（前端看板点击）
    """

    def __init__(self) -> None:
        # session_id -> SubjectSession
        self._sessions: Dict[str, SubjectSession] = {}
        # user_id -> GlobalUserProfile
        self._profiles: Dict[str, GlobalUserProfile] = {}
        # user_id -> {subject_slug -> session_id}（当日活跃指针）
        self._active: Dict[str, Dict[str, str]] = {}

    # ---------- 全局画像 ---------- #
    def upsert_profile(self, profile: GlobalUserProfile) -> GlobalUserProfile:
        profile.touch()
        # 未设权重则均分
        if profile.subjects and not profile.subject_weights:
            w = 1.0 / len(profile.subjects)
            profile.subject_weights = {s: w for s in profile.subjects}
        self._profiles[profile.user_id] = profile
        return profile

    def get_profile(self, user_id: str) -> Optional[GlobalUserProfile]:
        return self._profiles.get(user_id)

    def update_global_emotion(
        self,
        user_id: str,
        *,
        emotion_intensity: float,
        energy_level: Optional[float] = None,
        note: str = "",
    ) -> GlobalUserProfile:
        """陪伴 Agent 写入的情绪，供所有科目会话读取。"""
        profile = self._profiles.get(user_id) or GlobalUserProfile(user_id=user_id)
        profile.emotion_intensity = float(emotion_intensity)
        if energy_level is not None:
            profile.energy_level = float(energy_level)
        if note:
            profile.emotion_note = note
        profile.touch()
        self._profiles[user_id] = profile
        return profile

    # ---------- 科目会话 ---------- #
    def get_or_create_session(self, user_id: str, subject: str) -> SubjectSession:
        sid = make_session_id(user_id, subject)
        if sid in self._sessions:
            # 不刷新 last_active：中断检测要看「上次真正说话」而不是「又打开了会话」
            return self._sessions[sid]
        slug = slugify_subject(subject)
        sess = SubjectSession(
            session_id=sid,
            user_id=user_id,
            subject=subject,
            subject_slug=slug,
            namespace=memory_namespace(user_id, subject),
        )
        self._sessions[sid] = sess
        self._active.setdefault(user_id, {})[slug] = sid
        return sess

    def get_session(self, session_id: str) -> Optional[SubjectSession]:
        return self._sessions.get(session_id)

    def list_user_sessions(self, user_id: str, *, only_today: bool = True) -> List[SubjectSession]:
        today = datetime.now().strftime("%Y%m%d")
        out = []
        for s in self._sessions.values():
            if s.user_id != user_id:
                continue
            if only_today and not s.session_id.endswith(today):
                continue
            out.append(s)
        out.sort(key=lambda x: x.subject)
        return out

    def ensure_sessions_for_subjects(self, user_id: str, subjects: List[str]) -> List[SubjectSession]:
        return [self.get_or_create_session(user_id, subj) for subj in subjects]

    def open_named_window(self, user_id: str, title: str) -> SubjectSession:
        """新建对话窗口：名字就是这一科的专属会话。"""
        name = (title or "").strip()
        if not name:
            raise ValueError("窗口要有名字")
        profile = self._profiles.get(user_id) or GlobalUserProfile(user_id=user_id)
        if name not in profile.subjects:
            profile.subjects.append(name)
            if profile.books_note:
                profile.books_by_subject.setdefault(name, [profile.books_note])
            n = len(profile.subjects)
            profile.subject_weights = {s: 1.0 / n for s in profile.subjects}
        self._profiles[user_id] = profile
        return self.get_or_create_session(user_id, name)

    def list_windows(self, user_id: str) -> List[Dict[str, Any]]:
        profile = self._profiles.get(user_id)
        names: List[str] = list(profile.subjects) if profile else []
        for s in self._sessions.values():
            if s.user_id == user_id and s.subject not in names:
                names.append(s.subject)
        out = []
        for name in names:
            sess = self.get_or_create_session(user_id, name)
            out.append(
                {
                    "subject": name,
                    "session_id": sess.session_id,
                    "namespace": sess.namespace,
                }
            )
        return out

    def switch_subject(self, user_id: str, subject: str) -> SubjectSession:
        """前端点击某窗口 → 唤起/创建该科会话。"""
        return self.get_or_create_session(user_id, subject)
