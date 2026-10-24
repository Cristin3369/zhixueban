# -*- coding: utf-8 -*-
"""辅导改写包裹 + 路由保守融合（不依赖外部 LLM）。"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.router import RouterAgent
from generation.policy import finetune_summary
from generation.tutor_composer import TutorComposer
from retrieval.hybrid_rag import AdvancedRAGEngine


class DumpingLLM:
    """模拟「把资料原文吐回去」的不守规矩模型。"""

    def generate(self, prompt: str) -> str:
        return prompt  # 必触发照抄拦截


def main() -> None:
    print(finetune_summary())
    print("\n-- compose: dump LLM degrades to template --")
    dumping = TutorComposer(llm=DumpingLLM())
    pack = dumping.compose(
        "反向传播的链式法则怎么推？",
        [
            {
                "text": "反向传播利用链式法则把输出误差逐层回传到各参数。",
                "score": 0.91,
                "metadata": {"book_name": "深度学习花书", "page": 214},
            }
        ],
    )
    print("status:", pack["status"], "compose_mode:", pack.get("compose_mode"))
    print(pack["response"][:280])
    assert pack["status"] == "success"
    assert pack.get("compose_mode") == "template"
    assert "【溯源】" in pack["response"]
    assert "参考资料" not in pack["response"]

    print("\n-- rag path still wraps --")
    rag = AdvancedRAGEngine()
    if hasattr(rag.vector_store, "add"):
        rag.vector_store.add(
            "反向传播利用链式法则把输出误差逐层回传到各参数。",
            {"book_name": "深度学习花书", "page": 214, "subject": "数学"},
        )
    if hasattr(rag.bm25_index, "add"):
        rag.bm25_index.add(
            "反向传播利用链式法则把输出误差逐层回传到各参数。",
            {"book_name": "深度学习花书", "page": 214, "subject": "数学"},
        )
    rag_pack = rag.generate_answer_with_fallback("反向传播的链式法则怎么推？")
    print("rag status:", rag_pack["status"], rag_pack.get("compose_mode"), rag_pack.get("strategy"))

    print("\n-- refuse: no evidence --")
    empty = TutorComposer().compose("量子引力考研真题第888页", [])
    print(empty["status"], empty["compose_mode"])
    assert empty["status"] == "fallback_triggered"

    print("\n-- router: crisis conservative --")
    r = RouterAgent(use_llm=False)
    crisis = r.route("我崩溃了不想活，泰勒公式怎么证")
    print("targets:", crisis["target_agents"], "intent:", crisis["intent"])
    assert crisis["target_agents"] == ["Companion_Agent"]

    study = r.route("反向传播链式法则怎么推导")
    print("study targets:", study["target_agents"])
    assert "Tutor_Agent" in study["target_agents"]
    print("\ngeneration envelope demo ok")


if __name__ == "__main__":
    main()
