# -*- coding: utf-8 -*-
"""
辅导生成器：检索证据 → 改写讲解（不是把原题答案直接吐给用户）
--------------------------------------------------------------
主路径：通用 LLM 按固定四段改写。
降级：模板组句（不依赖模型）。
再失败：Learn to Refuse。
该 LLM 不微调；考点以外挂证据为准。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from config.settings import settings
from generation.envelope import CircuitBreaker, guarded_call
from generation.grounding import Evidence, check_answer, refuse_pack
from retrieval.query_rewrite import REFUSE_MESSAGE, should_refuse

logger = logging.getLogger(__name__)

TUTOR_COMPOSE_SYSTEM = """你是考研/考公辅导讲解器，不是知识库。
必须只根据「参考资料」组织语言，禁止补充资料里没有的定理、年份、页码、结论。
输出且只输出四段：
【考点】一句话点名知识点
【思路】用自己的话讲推导/解法步骤，连续照抄原文不得超过 40 字
【易错】1~2 条；资料没有就写「资料未单列易错点，对照原书例题」
【溯源】必须出现资料中的书名与页码
资料不够就只输出：资料不足
"""


class TutorComposer:
    def __init__(self, llm: Any = None, breaker: Optional[CircuitBreaker] = None):
        self.llm = llm
        self.breaker = breaker or CircuitBreaker("tutor_compose")

    def compose(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        *,
        memory_context: str = "",
        strategy: str = "hybrid",
        attempts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        attempts = list(attempts or [])
        if not docs or should_refuse(float(docs[0].get("score") or 0), settings.confidence_threshold):
            return refuse_pack("exhausted" if not docs else strategy, attempts)

        evidences = [Evidence.from_doc(d) for d in docs if d.get("text")]
        if not evidences:
            return refuse_pack("empty_evidence", attempts)

        template = self._template(query, evidences)
        llm_text = ""
        compose_mode = "template"

        if self.llm is not None:
            guard = guarded_call(
                lambda: self._llm_compose(query, evidences, memory_context),
                fallback="",
                breaker=self.breaker,
                timeout_sec=settings.tutor_compose_timeout_sec,
                retries=1,
                name="tutor_compose_llm",
            )
            llm_text = (guard.value or "").strip()
            if llm_text and not guard.degraded:
                grounded = check_answer(llm_text, evidences, allow_long_quote=False)
                if grounded.ok:
                    text = grounded.patched_text
                    compose_mode = "llm"
                    return self._success(text, docs, strategy, attempts, compose_mode)
                attempts.append("grounding_fail:" + ",".join(grounded.reasons[:2]))
                logger.info("LLM 讲解未通过落地检查，降级模板: %s", grounded.reasons)

        # 模板组句：允许使用资料字段，但仍截断、强制溯源
        grounded_t = check_answer(template, evidences, allow_long_quote=True)
        text = grounded_t.patched_text or template
        if memory_context:
            text += "\n\n（已参考你的学习时间线摘要，避免重复讲已掌握内容）"
        return self._success(text, docs, strategy, attempts, compose_mode)

    def compose_from_cache(
        self,
        query: str,
        question: dict,
        answer: dict,
        page_guide: str,
        *,
        layer: str,
        confidence: float,
        layer_trail: list,
        memory_context: str = "",
    ) -> Dict[str, Any]:
        """
        缓存命中也不能原文倾销题干+答案。
        把结构化字段当成证据，再走同一套改写/模板。
        """
        book = ""
        page = "?"
        if question.get("book_ref"):
            br = question["book_ref"]
            book = br.get("book_name", "") or ""
            page = str(br.get("page", "?"))
        evidence_text = "\n".join(
            x
            for x in [
                question.get("knowledge_point") or query,
                answer.get("explanation") or "",
                answer.get("answer_text") or "",
                question.get("senior_tips") or "",
                page_guide,
            ]
            if x
        )
        doc = {
            "text": evidence_text[:1200],
            "score": max(confidence, settings.confidence_threshold),
            "metadata": {
                "book_name": book or "指定教材",
                "page": page,
                "layer": layer,
                "page_guide": page_guide,
            },
        }
        pack = self.compose(
            query or question.get("stem") or "",
            [doc],
            memory_context=memory_context,
            strategy=f"tiered:{layer}",
            attempts=[f"cache:{layer}"],
        )
        pack["layer_trail"] = layer_trail
        pack["top_score"] = confidence
        return pack

    def _llm_compose(self, query: str, evidences: List[Evidence], memory_context: str) -> str:
        blocks = []
        for i, ev in enumerate(evidences[:3], start=1):
            snippet = ev.text[: settings.evidence_snippet_chars]
            blocks.append(f"[资料{i} | {ev.cite_line()} | score={ev.score:.2f}]\n{snippet}")
        mem = f"\n学生已学摘要（只用于避免重复，不能当新证据）：{memory_context[:300]}" if memory_context else ""
        prompt = (
            f"{TUTOR_COMPOSE_SYSTEM}\n\n参考资料：\n"
            + "\n\n".join(blocks)
            + f"\n\n学生问题：{query}{mem}\n"
        )
        if hasattr(self.llm, "generate"):
            return self.llm.generate(prompt)
        raise RuntimeError("llm 无 generate")

    @staticmethod
    def _template(query: str, evidences: List[Evidence]) -> str:
        ev = evidences[0]
        body = re.sub(r"\s+", " ", ev.text).strip()
        # 截成讲解语气，避免把整段题库原文甩出去
        idea = body[:180] + ("…" if len(body) > 180 else "")
        kp = query.strip()[:40] or ev.metadata.get("knowledge_point") or "相关考点"
        cites = "；".join(dict.fromkeys(e.cite_line() for e in evidences[:3]))
        return (
            f"【考点】{kp}\n"
            f"【思路】结合资料：{idea}\n"
            f"【易错】对照原书例题核对该公式的适用条件，不要只背结论。\n"
            f"【溯源】本题解析基于 {cites}，建议结合原书例题理解。"
        )

    @staticmethod
    def _success(
        text: str,
        docs: List[Dict[str, Any]],
        strategy: str,
        attempts: List[str],
        compose_mode: str,
    ) -> Dict[str, Any]:
        return {
            "status": "success",
            "response": text,
            "sources": [d.get("metadata") for d in docs],
            "strategy": strategy,
            "attempts": attempts,
            "top_score": docs[0].get("score"),
            "compose_mode": compose_mode,
        }


# 给同构题一条固定包裹，避免像「原题」
def wrap_isomorphic(gen: dict) -> str:
    name = gen.get("knowledge_point") or "该知识点"
    return (
        f"【考点】{name}（同构练习，非原题原文）\n"
        f"【思路】{gen.get('stem')}\n"
        f"【易错】不要把生成题当成真题页码去对答案。\n"
        f"【溯源】知识库暂无原题，已按图谱定义生成练习；请回指定教材核对该概念。"
        f"\n参考定义：{(gen.get('answer') or '')[:160]}"
    )
