# -*- coding: utf-8 -*-
"""
日历 / 倒计时 / 日程满足感 演示
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from startgraph import StudyAgentSystem, onboard_user


def main() -> None:
    system = StudyAgentSystem(force_memory_graph=True)
    uid = "cal_demo"

    print("========== 1. 登记考试日 + 4 科 ==========")
    onboard_user(
        system,
        uid,
        "考研",
        ["数学", "英语", "政治", "专业课"],
        {"数学": ["同济高数"], "英语": ["黄皮书"], "政治": ["精讲精练"], "专业课": ["指定教材"]},
        target_exam_date="2026-12-20",
        daily_available_minutes=240,
    )

    print("\n========== 2. 倒计时与阶段 ==========")
    agenda = system.open_agenda(uid)
    print(agenda["countdown"])
    print("阶段提示:", agenda["phase_hint"])

    print("\n========== 3. 看板 → 自动同步今日日程 ==========")
    dash = system.open_dashboard(uid)
    print(dash["text"][:500])
    print("同步事件数:", dash.get("calendar_synced"))

    print("\n========== 4. 辅导成功 → 学过内容进日程归档 ==========")
    # 先灌一道 L1 题保证 tutor status=success
    from data.processor import ExamDataProcessor

    p = ExamDataProcessor()
    pairs = p.annotate_pages_batch(
        [
            {
                "stem": "泰勒公式的麦克劳林展开是什么？",
                "answer": "f(0)+f'(0)x+...",
                "book_name": "同济高数",
                "q_page": 120,
                "a_page": 355,
                "subject": "高等数学",
                "exam_track": "数学",
                "knowledge_points": ["泰勒公式"],
                "explanation": "代入 x=0 处导数。",
            }
        ]
    )
    system.store.warm_l1_from_pairs(uid, pairs)
    qid = pairs[0].question.question_id
    sw = system.switch_subject(uid, "数学")
    # 直接走 tutor 成功路径并归档
    out = system.tutor.answer(
        pairs[0].question.stem, user_id=uid, exam_track="数学", question_id=qid
    )
    print("tutor status:", out.get("status"))
    if out.get("status") == "success":
        system.archive_learning(
            uid,
            subject="数学",
            summary=out.get("response", "")[:200],
            knowledge_points=["泰勒公式"],
            session_id=sw["session_id"],
            minutes=20,
        )

    print("\n========== 5. 勾选完成一块计划 ==========")
    events = system.calendar.list_events(uid, day=agenda["day"])
    planned = [e for e in events if e.status == "planned"]
    if planned:
        done = system.complete_calendar_event(uid, planned[0].event_id, note="已完成该块")
        print(done.get("celebration"))

    print("\n========== 6. 今日满足感 ==========")
    agenda2 = system.open_agenda(uid)
    print(agenda2["today_satisfaction"])
    print("今日事件样例:")
    for e in (agenda2.get("events") or [])[:5]:
        print(f"  [{e['status']}] {e['kind']} {e['title']}")


if __name__ == "__main__":
    main()
