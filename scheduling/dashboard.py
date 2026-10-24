# -*- coding: utf-8 -*-
"""
今日学习看板（全局时间分配）
--------------------------
对话按科目隔离，但「今天学多久、先学哪科」由全局看板统筹。

输入：全局画像（科目权重、精力、情绪）+ 各科到期/错题/高频计数
输出：有序 DashboardBlock 列表（含 session_id，供前端一键唤起会话）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from config.settings import settings
from data.session import (
    DashboardBlock,
    GlobalUserProfile,
    SessionManager,
    make_session_id,
)


class DailyDashboardScheduler:
    """
    把全日可用分钟按权重切到各科，并叠加上「到期任务压力」。
    情绪崩溃 → 空看板（只休息）；
    精力低 → 总分钟打折 + 优先弱科轻量块。
    """

    def build(
        self,
        profile: GlobalUserProfile,
        *,
        session_manager: SessionManager,
        due_by_subject: Optional[Dict[str, int]] = None,
        error_by_subject: Optional[Dict[str, int]] = None,
        high_freq_by_subject: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        due_by_subject = due_by_subject or {}
        error_by_subject = error_by_subject or {}
        high_freq_by_subject = high_freq_by_subject or {}
        subjects = list(profile.subjects) or list(
            set(list(due_by_subject) + list(error_by_subject) + list(high_freq_by_subject))
        )

        if profile.emotion_intensity >= settings.emotion_crisis_threshold:
            return {
                "mode": "rest_only",
                "message": "情绪高压：今日看板清空，只休息安抚，不切换刷题会话",
                "blocks": [],
                "total_minutes": 0,
            }

        budget = int(profile.daily_available_minutes)
        if profile.energy_level <= settings.energy_rest_threshold:
            budget = max(30, int(budget * 0.35))
            mode = "light"
            message = "精力偏低：已压缩总时长，建议只点开最弱一科轻量会话"
        elif profile.energy_level < 0.5:
            budget = max(45, int(budget * 0.7))
            mode = "reduced"
            message = "精力一般：已按比例缩短各科块"
        else:
            mode = "normal"
            message = "已按科目权重 + 到期压力生成今日看板"

        if not subjects:
            return {
                "mode": mode,
                "message": "还没有对话窗口。去对话页给窗口起个名字，这一窗就专门问这一科。",
                "blocks": [],
                "total_minutes": 0,
            }

        # 确保每科有独立 session
        session_manager.ensure_sessions_for_subjects(profile.user_id, subjects)

        weights = dict(profile.subject_weights or {})
        for s in subjects:
            weights.setdefault(s, 1.0)
        # 到期/错题压力微调权重（错题多的科多分一点时间）
        for s in subjects:
            pressure = (
                due_by_subject.get(s, 0) * 0.05
                + error_by_subject.get(s, 0) * 0.08
                + high_freq_by_subject.get(s, 0) * 0.04
            )
            weights[s] = weights[s] * (1.0 + min(pressure, 1.5))

        total_w = sum(weights[s] for s in subjects) or 1.0
        blocks: List[DashboardBlock] = []
        # 按压力排序：错题+到期多的先学（间隔切换也更符合大脑热身）
        ordered = sorted(
            subjects,
            key=lambda s: (
                -(error_by_subject.get(s, 0) + due_by_subject.get(s, 0)),
                s,
            ),
        )
        allocated = 0
        for i, subj in enumerate(ordered):
            if i == len(ordered) - 1:
                minutes = max(15, budget - allocated)
            else:
                minutes = max(15, int(budget * (weights[subj] / total_w)))
                allocated += minutes
            due_c = due_by_subject.get(subj, 0)
            err_c = error_by_subject.get(subj, 0)
            hf_c = high_freq_by_subject.get(subj, 0)
            summary_parts = []
            if hf_c:
                summary_parts.append(f"高频{hf_c}")
            if err_c:
                summary_parts.append(f"错题{err_c}")
            if due_c:
                summary_parts.append(f"到期{due_c}")
            if not summary_parts:
                summary_parts.append("巩固/预习")
            sess = session_manager.get_or_create_session(profile.user_id, subj)
            blocks.append(
                DashboardBlock(
                    subject=subj,
                    session_id=sess.session_id,
                    minutes=minutes,
                    task_summary=" + ".join(summary_parts),
                    due_count=due_c,
                    error_count=err_c,
                    high_freq_count=hf_c,
                    order=i,
                )
            )

        return {
            "mode": mode,
            "message": message,
            "blocks": [b.to_dict() for b in blocks],
            "total_minutes": sum(b.minutes for b in blocks),
            "emotion_intensity": profile.emotion_intensity,
            "energy_level": profile.energy_level,
            "hint": "点击某一科 → 前端用对应 session_id 打开专属会话，勿混聊",
        }

    @staticmethod
    def format_text(dashboard: Dict[str, Any]) -> str:
        lines = [
            f"【今日学习看板 · {dashboard.get('mode')}】{dashboard.get('message', '')}",
            f"合计约 {dashboard.get('total_minutes', 0)} 分钟",
        ]
        for b in dashboard.get("blocks") or []:
            lines.append(
                f"{b.get('order', 0) + 1}. {b.get('subject')} "
                f"{b.get('minutes')} 分钟 — {b.get('task_summary')} "
                f"(session={b.get('session_id')})"
            )
        if dashboard.get("hint"):
            lines.append(dashboard["hint"])
        return "\n".join(lines)
