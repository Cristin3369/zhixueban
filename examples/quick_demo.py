# -*- coding: utf-8 -*-
"""
端到端演示（建议先看本文件注释再读各模块）
----------------------------------------
场景：学生进页选科目书本 → 高频题进 L1 → 提问辅导 → 情绪崩溃只安抚 → 多因子反馈排计划
无需 Redis / Neo4j / 付费 LLM。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.models import UserProfile
from data.processor import ExamDataProcessor
from data.tiered_cache import TieredKnowledgeStore
from startgraph import StudyAgentSystem, onboard_user


def main() -> None:
    print("========== 1. 用户进页：登记科目与书本（写入 L3 画像） ==========")
    system = StudyAgentSystem(force_memory_graph=True)
    # StudyAgentSystem(force_memory_graph=True) 会走内存版知识图谱和内存 KV，数据量大时更推荐
    profile = onboard_user(
        system,
        user_id="demo_user",
        exam_track="考研数学",
        subjects=["高等数学"],
        books=["高等数学（同济七版）"],
    )
    print("画像:", profile)
    # profile 是全局画像，里面包含用户信息、学习目标、学习习惯、学习进度、学习成果等



    print("\n========== 2. 数据处理：题答绑定 + 页码 + 频率 ==========")
    processor = ExamDataProcessor()
    # ExamDataProcessor 是数据处理器，用于处理考试数据，包括题答绑定、页码、频率等
    # annotate_pages_batch 是数据处理器的方法，用于处理考试数据，包括题答绑定、页码、频率等，输入是一个列表，输出是一个列表

    pairs = processor.annotate_pages_batch(
        [
            {
                "stem": "求函数 f(x)=e^x 在 x=0 处的泰勒展开到二阶。",
                "answer": "1 + x + x^2/2 + o(x^2)",
                "book_name": "高等数学（同济七版）",
                "q_page": 120,
                "a_page": 355,
                "subject": "高等数学",
                "exam_track": "考研数学",
                "chapter": "微分中值定理与导数应用",
                "difficulty": 3,
                "knowledge_points": ["泰勒公式"],
                "senior_tips": "近十年选择/填空高频，注意拉格朗日余项写法",
                "explanation": "代入麦克劳林公式即可。",
            }
        ]
    )
    # 模拟该考点在历年真题中出现 5 次
    fp = processor._fingerprint(pairs[0].question)
    # _fingerprint 指纹模拟。指纹是「科目 + 排序后的知识点 + 去空白题干」的哈希，同构题可以归到同一条计数上。
    for year in range(2019, 2024):
        processor.ingest_past_paper_occurrence(fp, year, "考研数学一")
    # ingest_past_paper_occurrence 历年真题计数。

    pairs = processor.apply_frequency_to_pairs(pairs)
    # apply_frequency_to_pairs 频率归一化。
   
    print("翻书指引:", pairs[0].page_guide())
    # page_guide() 打印的就是「题目见……；答案见……」。
    print("频率归一化:", pairs[0].question.exam_frequency_norm)



    print("\n========== 3. 高频题灌入 L1 热缓存 ==========")
    store: TieredKnowledgeStore = system.store
    # TieredKnowledgeStore 是知识图谱，用于存储知识图谱，包括知识图谱、知识图谱的嵌入、知识图谱的相似度等
    n = store.warm_l1_from_pairs("demo_user", pairs)
    # warm_l1_from_pairs 把这道题的题干、答案和翻书指引写进 L1，并登记进布隆过滤器。返回的 n 就是写入题数，这里是 1
    print(f"已写入 L1 {n} 题")

    qid = pairs[0].question.question_id
    # 图谱绑定，便于同构题路径
    system.graph_rag.link_question_concept(qid, "链式法则")
    # link_question_concept(qid, "链式法则") 只在内存图谱里记下「这道题考链式法则」。演示里题干其实是泰勒展开，这个绑定是为了后面缓存未命中时能按知识点出同构题，并不是在改题面。

    print("\n========== 4. 辅导：按题号走四级缓存 ==========")
    ans = system.tutor.answer(
        pairs[0].question.stem,
        user_id="demo_user",
        exam_track="考研数学",
        question_id=qid,
    )
    # tutor.answer 带了 question_id，所以先走四级缓存，不先做向量检索：
    # 用题号到图谱里找概念（刚才绑的「链式法则」）。
    # get_cascaded：空值缓存没有 → 布隆说这个 id 可能存在 → L1 命中，置信度 1.0。
    # 用缓存里的解析、学长学姐提示和页码拼成回答。strategy 会是 tiered:L1。
    print(ans.get("response", "")[:500])
    print("strategy:", ans.get("strategy"))

    print("\n========== 5. 情绪崩溃：调度应只走陪伴 ==========")
    crisis = system.invoke(
        "我真的快崩溃了，背了十遍还是记不住，感觉考研没希望了...",
        user_id="demo_user",
        exam_track="考研数学",
        user_energy=0.3,
    )
    print(crisis.get("final_response", "")[:400])
    print("trace:", crisis.get("agent_trace"))


    print("\n========== 6. 多因子反馈 → 下次复习 ==========")
    fb = system.planner.apply_review_feedback(
        "demo_user",
        {"last_interval": 2, "consecutive_correct": 2, "ef": 2.5},
        {
            "is_correct": True,
            "response_time": 45,
            "subjective_difficulty": 4,
            "objective_difficulty": 3,
            "exam_probability": pairs[0].question.exam_frequency_norm,
            "user_energy": 0.7,
            "emotion_intensity": 0.2,
        },
        question_id=qid,
    )
    print("下次复习间隔(天):", fb["final_interval_days"])
    print("因子:", fb["factor_breakdown"])


if __name__ == "__main__":
    main()
