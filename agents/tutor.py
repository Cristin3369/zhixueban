# -*- coding: utf-8 -*-
"""
辅导 Agent
---------
询问某题时：
1. 混合检索 + Rerank，按 L1→L2→L3→L4 级联
2. 置信度不足则改写/降维/GraphRAG
3. 再不行 → 知识点实体同构题
4. 仍低 → Learn to Refuse
命中后必须走 TutorComposer：LLM 改写四段讲解 → 落地检查失败则模板组句 → 再失败拒答。
讲解模型不微调；禁止把题库原文直接当回复。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from agents.reflection import ReflectionEngine
from data.tiered_cache import TieredKnowledgeStore
from generation.envelope import CircuitBreaker
from generation.tutor_composer import TutorComposer, wrap_isomorphic
from retrieval.graph_rag import KnowledgeGraphRAG
from retrieval.hybrid_rag import AdvancedRAGEngine
from retrieval.query_rewrite import REFUSE_MESSAGE

logger = logging.getLogger(__name__)


class TutorAgent:
    def __init__(
        self,
        rag_engine: Optional[AdvancedRAGEngine] = None,
        graph_rag: Optional[KnowledgeGraphRAG] = None,
        tiered_store: Optional[TieredKnowledgeStore] = None,
    ):
        self.graph_rag = graph_rag or KnowledgeGraphRAG(force_memory=True)
        self.rag = rag_engine or AdvancedRAGEngine(graph_rag=self.graph_rag)
        self.store = tiered_store or TieredKnowledgeStore(
            isomorphic_generator=self._gen_isomorphic,
        )
        # 把图谱挂到同构生成路径
        if self.store.isomorphic_generator is None:
            self.store.isomorphic_generator = self._gen_isomorphic
        self.reflection = ReflectionEngine()
        self.composer = TutorComposer(
            llm=getattr(self.rag, "llm", None),
            breaker=CircuitBreaker("tutor_agent"),
        )
        self.rag._composer = self.composer  # 检索生成与缓存讲解共用同一包裹
        self.mcp = None  # 由 protocols.attach 注入；题库与图谱走 MCP

    def _gen_isomorphic(self, concept_node: dict) -> dict:
        name = concept_node.get("current_concept") or concept_node.get("name") or "知识点"
        definition = concept_node.get("definition", "")
        formula = concept_node.get("formula", "")
        return {
            "stem": f"【同构练习】请用自己的话解释「{name}」，并说明公式 {formula} 的含义。",
            "answer": definition,
            "knowledge_point": name,
            "generated": True,
        }

    def answer(
        self,
        query: str,
        *,
        user_id: str = "anonymous",
        exam_track: str = "考研数学",
        question_id: Optional[str] = None,
        memory_context: str = "",
    ) -> Dict[str, Any]:
        """
        主入口。question_id 可空：无 id 时走语义检索；有 id 时先走四级缓存。
        """
        layer_pack = None
        if question_id:
            concept = self.graph_rag.get_concept_by_question_id(question_id)
            if concept is None:
                # 尝试用 query 对齐实体，便于同构生成
                concept = self.graph_rag.get_knowledge_card(query)
            layer_pack = self.store.get_cascaded(
                question_id,
                user_id,
                exam_track,
                concept_node=concept,
            )
            if layer_pack.source in {"L1", "L2", "L3", "L4"} and layer_pack.data:
                data = layer_pack.data if isinstance(layer_pack.data, dict) else {}
                page_guide = data.get("page_guide") or ""
                q = data.get("question") or {}
                a = data.get("answer") or {}
                pack = self.composer.compose_from_cache(
                    query,
                    q,
                    a,
                    page_guide,
                    layer=layer_pack.source,
                    confidence=layer_pack.confidence,
                    layer_trail=layer_pack.layer_trail,
                    memory_context=memory_context,
                )
                return self.reflection.reflect_tutor_answer(pack).sanitized_payload

            if layer_pack.source == "GENERATED" and layer_pack.data:
                gen = layer_pack.data
                pack = {
                    "status": "success",
                    "response": wrap_isomorphic(gen),
                    "sources": [{"layer": "GENERATED"}],
                    "strategy": "isomorphic",
                    "top_score": layer_pack.confidence,
                    "layer_trail": layer_pack.layer_trail,
                    "compose_mode": "isomorphic",
                }
                return self.reflection.reflect_tutor_answer(pack).sanitized_payload

            if layer_pack.source in {"FALLBACK", "NULL_CACHED"}:
                # 继续尝试语义检索，而不是立刻放弃（query 可能仍能命中）
                logger.info("题号缓存未命中，转语义检索: %s", layer_pack.layer_trail)

        if self.mcp is not None:
            pack = self.mcp.call_tool(
                "kb.answer",
                {"query": query, "memory_context": memory_context},
            )
            if pack.get("status") == "success":
                graph_prompt = self.mcp.call_tool("graph.prompt", {"query": query}) or ""
                pack["graph_hint"] = str(graph_prompt)[:500]
        else:
            pack = self.rag.generate_answer_with_fallback(query, memory_context=memory_context)
            if pack.get("status") == "success":
                graph_prompt = self.graph_rag.build_rag_prompt(query, user_question=query)
                pack["graph_hint"] = graph_prompt[:500]
        if pack.get("status") != "success":
            pack["response"] = pack.get("response") or REFUSE_MESSAGE

        reflected = self.reflection.reflect_tutor_answer(pack)
        return reflected.sanitized_payload
