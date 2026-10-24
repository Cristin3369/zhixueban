# -*- coding: utf-8 -*-
"""
多科目「一科一会话」演示
----------------------
验证：
1. onboard 开 4 个独立 session
2. 今日看板分配时间并绑定 session_id
3. 数学会话问答不会出现在英语记忆里
4. 陪伴写入的全局情绪，切到数学会话仍可感知
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from startgraph import StudyAgentSystem, onboard_user


def main() -> None:
    system = StudyAgentSystem(force_memory_graph=True)
    uid = "kaoyan_demo"

    print("========== 1. 进页：4 科各自开会话 ==========")
    profile = onboard_user(
        system,
        uid,
        "考研",
        ["数学", "英语", "政治", "专业课"],
        {
            "数学": ["高等数学同济七版"],
            "英语": ["黄皮书"],
            "政治": ["精讲精练"],
            "专业课": ["学校指定教材"],
        },
        daily_available_minutes=240,
        # 弱科可加大权重，例如数学 1.4
        subject_weights={"数学": 1.4, "英语": 1.0, "政治": 0.8, "专业课": 1.0},
    )
    for s in system.sessions.list_user_sessions(uid):
        print(f"  - {s.subject}: {s.session_id} | ns={s.namespace}")

    print("\n========== 2. 往全局队列塞各科任务（模拟艾宾浩斯到期） ==========")
    import time

    now = time.time() - 10
    system.queue.add_task(uid, "error_book", now, subject="数学", question_id="m1")
    system.queue.add_task(uid, "due_review", now, subject="英语", question_id="e1")
    system.queue.add_task(uid, "high_freq_focus", now, subject="政治", question_id="p1")
    system.queue.add_task(uid, "due_review", now, subject="专业课", question_id="z1")

    print("\n========== 3. 今日看板（时间分配，不混对话） ==========")
    dash = system.open_dashboard(uid)
    print(dash["text"])

    print("\n========== 4. 点开数学会话做题 ==========")
    sw = system.switch_subject(uid, "数学")
    print(sw)
    math_res = system.invoke(
        "泰勒公式怎么用？",
        user_id=uid,
        subject="数学",
        session_id=sw["session_id"],
    )
    print("数学 session:", math_res.get("session_id"))
    print("数学回复片段:", (math_res.get("final_response") or "")[:180])

    print("\n========== 5. 切到英语会话：应看不到数学题干历史 ==========")
    sw_en = system.switch_subject(uid, "英语")
    en_sess = system.sessions.get_session(sw_en["session_id"])
    math_sess = system.sessions.get_session(sw["session_id"])
    print("英语会话消息条数:", len(en_sess.messages) if en_sess else 0)
    print("数学会话消息条数:", len(math_sess.messages) if math_sess else 0)
    en_mem = system.memory.episodic.recent(
        uid, n=20, session_id=sw_en["session_id"], include_global=False
    )
    leaked = [c for c in en_mem if "泰勒" in c.content]
    print("英语命名空间是否串入「泰勒」:", bool(leaked), "(期望 False)")

    print("\n========== 6. 倾诉焦虑 → 写入全局画像 → 再进数学会话可感知 ==========")
    system.invoke(
        "我今天极度焦虑，不想学数学了...",
        user_id=uid,
        subject="英语",  # 哪怕在英语会话倾诉，也写全局
        session_id=sw_en["session_id"],
    )
    g = system.sessions.get_profile(uid)
    print("全局情绪:", g.emotion_intensity if g else None, "note:", g.emotion_note if g else None)
    sw2 = system.switch_subject(uid, "数学")
    print("切回数学时带上的全局提示:", sw2.get("emotion_note"))


if __name__ == "__main__":
    main()
