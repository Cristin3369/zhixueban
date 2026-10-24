# -*- coding: utf-8 -*-
"""
多因子自适应艾宾浩斯调度
----------------------
在原「反应时间 + 主观难度」双因子基础上，扩展为：
1. 答题时间 response_time
2. 用户自评难度 subjective_difficulty
3. 考试出现概率 exam_probability
4. 题目客观难度 objective_difficulty
5. 用户当日精力 user_energy
6. 情绪强度 emotion_intensity（过高则不加压，交由陪伴/防沉迷）
7. 可选：自我保持评分 retention_self_score

输出：下次复习时间戳 + 更新后的 SM-2 状态 + 是否触发休息。
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

from config.settings import settings
from data.models import ReviewFeedback, StudentSpacedRecord


class MultiFactorEbbinghausScheduler:
    """多因子间隔重复调度器（计划 Agent 的核心算法）。"""

    def __init__(
        self,
        base_intervals: Optional[list[int]] = None,
        energy_rest_threshold: Optional[float] = None,
        emotion_crisis_threshold: Optional[float] = None,
    ):
        self.BASE_INTERVALS = base_intervals or list(settings.base_intervals_days)
        self.energy_rest_threshold = (
            energy_rest_threshold
            if energy_rest_threshold is not None
            else settings.energy_rest_threshold
        )
        self.emotion_crisis_threshold = (
            emotion_crisis_threshold
            if emotion_crisis_threshold is not None
            else settings.emotion_crisis_threshold
        )

    def calculate_next_review(
        self,
        student_record: StudentSpacedRecord | Dict[str, Any],
        feedback: ReviewFeedback | Dict[str, Any],
        *,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        计算下一次复习到期时间。
        返回字段统一使用 next_review_timestamp（兼容旧 next_review_ts 别名）。
        """
        now = now if now is not None else time.time()
        record = self._as_record(student_record)
        fb = self._as_feedback(feedback)

        # ----- 防沉迷 / 情绪危机：不加压，顺延并标记 -----
        if fb.emotion_intensity >= self.emotion_crisis_threshold:
            return self._rest_result(
                record,
                now,
                reason="emotion_crisis",
                message="检测到情绪高压，今日只休息安抚，复习计划已顺延",
                defer_days=1,
            )
        if fb.user_energy <= self.energy_rest_threshold:
            return self._rest_result(
                record,
                now,
                reason="low_energy",
                message="今日精力不足，已触发防沉迷：计划平滑顺延",
                defer_days=1,
            )

        # ----- SM-2 / 艾宾浩斯基础间隔 -----
        last_interval = record.last_interval
        consecutive_correct = record.consecutive_correct
        ef = record.ef

        if not fb.is_correct:
            next_interval_days = 1
            consecutive_correct = 0
        else:
            consecutive_correct += 1
            if consecutive_correct <= len(self.BASE_INTERVALS):
                next_interval_days = self.BASE_INTERVALS[consecutive_correct - 1]
            else:
                next_interval_days = max(1, round(last_interval * ef))

        # ----- 多因子修正（惩罚缩短间隔 / 奖励拉长）-----
        factors = self._compute_factor_penalties(fb)
        # 综合系数限制在 [0.45, 1.35]，避免任务爆炸或拖得过久
        adjust = min(1.35, max(0.45, 1.0 - factors["total_penalty"] + factors["total_bonus"]))
        final_interval_days = max(1, math.ceil(next_interval_days * adjust))

        # 高频考点：适当加密复习（间隔再打九折）
        if fb.exam_probability >= 0.7 and fb.is_correct:
            final_interval_days = max(1, math.ceil(final_interval_days * 0.9))

        next_ts = int(now + final_interval_days * 86400)

        # EF 更新：正确略升，错误下降；客观很难且答对则小幅奖励
        ef_delta = 0.1 if fb.is_correct else -0.2
        if fb.is_correct and fb.objective_difficulty >= 4:
            ef_delta += 0.05
        new_ef = max(1.3, min(2.8, ef + ef_delta))

        updated = StudentSpacedRecord(
            last_interval=final_interval_days,
            consecutive_correct=consecutive_correct,
            ef=new_ef,
            total_reviews=record.total_reviews + 1,
            last_review_ts=now,
        )

        result = {
            "next_review_timestamp": next_ts,
            "next_review_ts": next_ts,  # 兼容旧 LangGraph 状态字段
            "final_interval_days": final_interval_days,
            "adjust_factor": adjust,
            "factor_breakdown": factors,
            "force_rest": False,
            "rest_reason": None,
            "message": "",
            "updated_state": updated.__dict__,
            "suggested_task_type": self._suggest_task_type(fb),
        }
        return result

    def build_daily_plan(
        self,
        *,
        due_count: int,
        high_freq_count: int,
        error_book_count: int,
        user_energy: float,
        emotion_intensity: float,
        available_minutes: int = 120,
    ) -> Dict[str, Any]:
        """
        计划 Agent：根据多因子与容量生成当日计划草案（再经反思模块校验）。
        """
        if emotion_intensity >= self.emotion_crisis_threshold:
            return {
                "mode": "rest_only",
                "blocks": [],
                "message": "情绪接近崩溃，今日仅陪伴休息，不安排刷题",
            }
        if user_energy <= self.energy_rest_threshold:
            return {
                "mode": "light",
                "blocks": [
                    {"type": "light_review", "minutes": min(25, available_minutes), "items": min(5, due_count)}
                ],
                "message": "精力偏低，仅保留轻量复习，其余顺延周末",
            }

        # 按精力分配分钟：高频 > 错题 > 到期复习
        energy_scale = 0.5 + 0.5 * user_energy
        budget = int(available_minutes * energy_scale)
        blocks = []
        remain = budget
        for typ, cnt, unit_min in [
            ("high_freq_focus", high_freq_count, 8),
            ("error_book", error_book_count, 10),
            ("due_review", due_count, 6),
        ]:
            if remain <= 0 or cnt <= 0:
                continue
            n = min(cnt, max(1, remain // unit_min))
            minutes = n * unit_min
            blocks.append({"type": typ, "minutes": minutes, "items": n})
            remain -= minutes

        return {
            "mode": "normal",
            "blocks": blocks,
            "budget_minutes": budget,
            "message": "已按艾宾浩斯到期 + 高频 + 错题本生成计划",
        }

    # ------------------------------------------------------------------ #
    def _compute_factor_penalties(self, fb: ReviewFeedback) -> Dict[str, float]:
        # A. 反应时间：基准 20s，越慢惩罚越大
        time_penalty = max(0.0, (fb.response_time - 20.0) / 100.0)
        # B. 主观难度
        subj_penalty = max(0.0, (fb.subjective_difficulty - 3) * 0.2)
        # C. 客观难度：很难且答错 → 额外缩短；很难但答对 → 下面用 bonus
        obj_penalty = 0.0
        if not fb.is_correct and fb.objective_difficulty >= 4:
            obj_penalty = 0.1
        # D. 考试概率高且不熟（答错或自评难）→ 加密
        exam_penalty = 0.0
        if fb.exam_probability >= 0.6 and (not fb.is_correct or fb.subjective_difficulty >= 4):
            exam_penalty = 0.15 * fb.exam_probability
        # E. 精力中等偏低：略微缩短，避免虚假「学会了」
        energy_penalty = max(0.0, (0.5 - fb.user_energy) * 0.2) if fb.user_energy < 0.5 else 0.0
        # F. 情绪有波动但未崩溃：小惩罚
        emotion_penalty = 0.05 if 0.5 <= fb.emotion_intensity < self.emotion_crisis_threshold else 0.0

        total_penalty = time_penalty + subj_penalty + obj_penalty + exam_penalty + energy_penalty + emotion_penalty

        # Bonus：精力好 + 答对 + 自评不难
        total_bonus = 0.0
        if fb.is_correct and fb.user_energy >= 0.75 and fb.subjective_difficulty <= 2:
            total_bonus += 0.15
        if fb.retention_self_score is not None and fb.retention_self_score >= 4 and fb.is_correct:
            total_bonus += 0.05

        return {
            "time_penalty": time_penalty,
            "subjective_penalty": subj_penalty,
            "objective_penalty": obj_penalty,
            "exam_penalty": exam_penalty,
            "energy_penalty": energy_penalty,
            "emotion_penalty": emotion_penalty,
            "total_penalty": total_penalty,
            "total_bonus": total_bonus,
        }

    def _suggest_task_type(self, fb: ReviewFeedback) -> str:
        if not fb.is_correct:
            return "error_book"
        if fb.exam_probability >= 0.7:
            return "high_freq_focus"
        return "due_review"

    def _rest_result(
        self,
        record: StudentSpacedRecord,
        now: float,
        *,
        reason: str,
        message: str,
        defer_days: int,
    ) -> Dict[str, Any]:
        next_ts = int(now + defer_days * 86400)
        return {
            "next_review_timestamp": next_ts,
            "next_review_ts": next_ts,
            "final_interval_days": defer_days,
            "adjust_factor": 1.0,
            "factor_breakdown": {},
            "force_rest": True,
            "rest_reason": reason,
            "message": message,
            "updated_state": record.__dict__,
            "suggested_task_type": "due_review",
        }

    @staticmethod
    def _as_record(obj: StudentSpacedRecord | Dict[str, Any]) -> StudentSpacedRecord:
        if isinstance(obj, StudentSpacedRecord):
            return obj
        return StudentSpacedRecord(
            last_interval=int(obj.get("last_interval", 0)),
            consecutive_correct=int(obj.get("consecutive_correct", 0)),
            ef=float(obj.get("ef", 2.5)),
            total_reviews=int(obj.get("total_reviews", 0)),
            last_review_ts=float(obj.get("last_review_ts", 0)),
        )

    @staticmethod
    def _as_feedback(obj: ReviewFeedback | Dict[str, Any]) -> ReviewFeedback:
        if isinstance(obj, ReviewFeedback):
            return obj
        return ReviewFeedback(
            is_correct=bool(obj.get("is_correct", True)),
            response_time=float(obj.get("response_time", obj.get("response_time_sec", 30))),
            subjective_difficulty=int(obj.get("subjective_difficulty", obj.get("difficulty", 3))),
            objective_difficulty=float(obj.get("objective_difficulty", 3)),
            exam_probability=float(obj.get("exam_probability", 0.5)),
            user_energy=float(obj.get("user_energy", obj.get("energy", 0.7))),
            emotion_intensity=float(obj.get("emotion_intensity", 0.0)),
            retention_self_score=obj.get("retention_self_score"),
        )


# 兼容旧类名
AdaptiveEbbinghausScheduler = MultiFactorEbbinghausScheduler
