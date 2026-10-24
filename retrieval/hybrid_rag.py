# -*- coding: utf-8 -*-
"""
混合检索引擎
-----------
向量检索（语义） + BM25（关键词）→ RRF 融合 → Cross-Encoder Rerank → Top-3
配合四级缓存层与 GraphRAG；置信度不足则改写/降维/换策略，最终 Learn to Refuse。
图表公式切片逻辑保留并增强。
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from config.settings import settings
from data.models import RetrievalHit
from retrieval.query_rewrite import (
    HeuristicQueryRewriter,
    should_refuse,
)

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]: ...


class BM25Index(Protocol):
    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]: ...


class Reranker(Protocol):
    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]: ...


class VLMClient(Protocol):
    def generate_caption(self, image_url: str) -> str: ...


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...
    def rewrite_query(self, query: str) -> str: ...


# -------------------- 本地可运行 Stub -------------------- #
class StubVectorStore:
    def __init__(self, corpus: Optional[List[Dict[str, Any]]] = None):
        self.corpus = corpus or []

    def add(self, text: str, metadata: Optional[dict] = None) -> None:
        self.corpus.append({"text": text, "metadata": metadata or {}, "score": 0.0})

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        scored = []
        q = set(query)
        for doc in self.corpus:
            s = self._overlap(q, doc["text"])
            scored.append({**doc, "score": s})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _overlap(qset: set, text: str) -> float:
        t = set(text)
        if not qset or not t:
            return 0.0
        return len(qset & t) / max(len(qset), 1)


class StubBM25:
    """极简 BM25 风格：TF * IDF 近似，足够演示专有名词召回。"""

    def __init__(self, corpus: Optional[List[Dict[str, Any]]] = None):
        self.corpus = corpus or []

    def add(self, text: str, metadata: Optional[dict] = None) -> None:
        self.corpus.append({"text": text, "metadata": metadata or {}})

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        terms = self._tokenize(query)
        if not terms:
            return []
        N = max(len(self.corpus), 1)
        df = {t: sum(1 for d in self.corpus if t in d["text"]) for t in terms}
        results = []
        for doc in self.corpus:
            score = 0.0
            for t in terms:
                tf = doc["text"].count(t)
                if tf <= 0:
                    continue
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * (tf * 2.2) / (tf + 1.2)
            results.append({**doc, "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9]+|\d+", text)


class StubReranker:
    """无 BGE-Reranker 时：用 query-doc 字符重合度重排。生产替换为 Cross-Encoder。"""

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        q = set(query)
        rescored = []
        for d in docs:
            overlap = len(q & set(d.get("text", ""))) / max(len(q), 1)
            # 结合原 score
            score = 0.6 * float(d.get("score", 0)) + 0.4 * overlap
            # 归一到约 0~1
            score = min(1.0, score if score <= 1 else score / (score + 1))
            rescored.append({**d, "score": score})
        rescored.sort(key=lambda x: x["score"], reverse=True)
        return rescored[:top_k]


class StubVLM:
    def generate_caption(self, image_url: str) -> str:
        return f"[VLM摘要]{image_url}"


class StubLLM:
    def __init__(self, rewriter: Optional[HeuristicQueryRewriter] = None):
        self.rewriter = rewriter or HeuristicQueryRewriter()

    def generate(self, prompt: str) -> str:
        # 演示：截取资料摘要
        return "【演示回答】请结合召回资料理解该考点；完整版请接入真实 LLM。\n" + prompt[:400]

    def rewrite_query(self, query: str) -> str:
        return self.rewriter.rewrite_query(query)


class AdvancedRAGEngine:
    """混合检索 + 多级兜底的辅导检索引擎。"""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        llm_client: Optional[LLMClient] = None,
        vlm_client: Optional[VLMClient] = None,
        rerank_model: Optional[Reranker] = None,
        graph_rag: Any = None,
        confidence_threshold: Optional[float] = None,
    ):
        self.vector_store = vector_store or StubVectorStore()
        self.bm25_index = bm25_index or StubBM25()
        self.rewriter = HeuristicQueryRewriter()
        self.llm = llm_client or StubLLM(self.rewriter)
        self.vlm = vlm_client or StubVLM()
        self.rerank_model = rerank_model or StubReranker()
        self.graph_rag = graph_rag
        self.CONFIDENCE_THRESHOLD = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.confidence_threshold
        )
        self.recall_k = settings.hybrid_recall_top_k
        self.rerank_k = settings.hybrid_rerank_top_k

    # ---------- 文档切片 ---------- #
    def process_complex_document(self, raw_elements: List[Dict]) -> List[Dict]:
        chunks: List[Dict] = []
        for element in raw_elements:
            etype = element.get("type", "text")
            if etype == "image":
                caption = self.vlm.generate_caption(element.get("url", ""))
                chunks.append(
                    {
                        "text": f"[图表摘要] {caption}",
                        "metadata": {"img_url": element.get("url"), "type": "image", **element.get("book_meta", {})},
                    }
                )
            elif etype == "table":
                header = element.get("header", [])
                for row in element.get("rows", []):
                    chunks.append(
                        {
                            "text": f"[表格数据] 表头: {header} | 数据: {row}",
                            "metadata": {"type": "table", **element.get("book_meta", {})},
                        }
                    )
            elif etype == "formula":
                chunks.append(
                    {
                        "text": f"[数学公式] {element.get('latex', '')} | 解释: {element.get('context', '')}",
                        "metadata": {"type": "formula", "latex": element.get("latex", ""), **element.get("book_meta", {})},
                    }
                )
            else:
                chunks.append(
                    {
                        "text": element.get("text", ""),
                        "metadata": {"type": "text", **element.get("book_meta", {})},
                    }
                )
        # 同步写入 stub 索引，便于本地演示
        for c in chunks:
            if hasattr(self.vector_store, "add"):
                self.vector_store.add(c["text"], c.get("metadata"))  # type: ignore[attr-defined]
            if hasattr(self.bm25_index, "add"):
                self.bm25_index.add(c["text"], c.get("metadata"))  # type: ignore[attr-defined]
        return chunks

    # ---------- RRF ---------- #
    @staticmethod
    def reciprocal_rank_fusion(
        *result_lists: Sequence[Dict[str, Any]],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion：score = Σ 1/(k + rank)。
        用 text 作为去重键（生产可用 chunk_id）。
        """
        fused: Dict[str, Dict[str, Any]] = {}
        for results in result_lists:
            for rank, doc in enumerate(results, start=1):
                key = doc.get("id") or doc.get("text", "")[:200]
                if key not in fused:
                    fused[key] = {
                        "text": doc.get("text", ""),
                        "metadata": dict(doc.get("metadata") or {}),
                        "score": 0.0,
                    }
                fused[key]["score"] += 1.0 / (k + rank)
                # 合并 metadata
                fused[key]["metadata"].update(doc.get("metadata") or {})
        out = list(fused.values())
        out.sort(key=lambda x: x["score"], reverse=True)
        return out

    # ---------- 混合检索 ---------- #
    def hybrid_search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.search_by_strategy(query, "hybrid_rerank", top_k=top_k)

    def search_by_strategy(
        self,
        query: str,
        strategy: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        消融用：vector / bm25 / hybrid_rrf / hybrid_rerank。
        线上默认 hybrid_rerank。
        """
        top_k = top_k or self.recall_k
        vec = self.vector_store.search(query, top_k=top_k)
        bm = self.bm25_index.search(query, top_k=top_k)
        if strategy == "vector":
            return vec[: self.rerank_k]
        if strategy == "bm25":
            return bm[: self.rerank_k]
        fused = self.reciprocal_rank_fusion(vec, bm)
        if strategy == "hybrid_rrf":
            return fused[: self.rerank_k]
        return self.rerank_model.rerank(query, fused, top_k=self.rerank_k)

    def search_with_fallbacks(self, query: str) -> Dict[str, Any]:
        """
        完整兜底：混合检索 → 改写多视角 → 关键词降维 → BM25 only → GraphRAG。
        """
        attempts: List[str] = []
        docs = self.hybrid_search(query)
        attempts.append("hybrid")
        if docs and not should_refuse(docs[0]["score"], self.CONFIDENCE_THRESHOLD):
            return {"docs": docs, "strategy": "hybrid", "attempts": attempts}

        # 查询改写多变体
        variants = self.rewriter.expand_variants(query)
        best: List[Dict[str, Any]] = docs or []
        for v in variants[1:4]:
            attempts.append(f"rewrite:{v[:20]}")
            cand = self.hybrid_search(v)
            if cand and (not best or cand[0]["score"] > best[0]["score"]):
                best = cand
            if best and not should_refuse(best[0]["score"], self.CONFIDENCE_THRESHOLD):
                return {"docs": best, "strategy": "query_rewrite", "attempts": attempts}

        # 降维关键词 + 纯 BM25
        keywords = self.rewriter.generalize_keywords(query)
        attempts.append(f"bm25:{keywords}")
        bm25_only = self.bm25_index.search(keywords, top_k=self.recall_k)
        bm25_reranked = self.rerank_model.rerank(query, bm25_only, top_k=self.rerank_k)
        if bm25_reranked and not should_refuse(bm25_reranked[0]["score"], self.CONFIDENCE_THRESHOLD):
            return {"docs": bm25_reranked, "strategy": "bm25_only", "attempts": attempts}

        # GraphRAG 实体多跳
        if self.graph_rag:
            attempts.append("graphrag")
            card = self.graph_rag.get_knowledge_card(keywords or query)
            if card:
                text = (
                    f"{card.get('current_concept')}: {card.get('definition')} "
                    f"公式:{card.get('formula')} 前置:{card.get('prerequisites')}"
                )
                graph_doc = {"text": text, "score": 0.72, "metadata": {"source": "graphrag", **card}}
                if not should_refuse(graph_doc["score"], self.CONFIDENCE_THRESHOLD):
                    return {"docs": [graph_doc], "strategy": "graphrag", "attempts": attempts}

        return {"docs": best, "strategy": "exhausted", "attempts": attempts}

    def generate_answer_with_fallback(
        self,
        query: str,
        *,
        copyright_book_hint: str = "",
        memory_context: str = "",
    ) -> Dict[str, Any]:
        """
        检索后必须改写：通用 LLM 组织语言 + 落地检查；失败则模板组句；再失败则拒答。
        此 LLM 不微调，不得把题库原文直接展示给用户。
        """
        from generation.tutor_composer import TutorComposer

        pack = self.search_with_fallbacks(query)
        docs = pack["docs"]
        if copyright_book_hint:
            for d in docs:
                meta = d.setdefault("metadata", {})
                meta.setdefault("book_name", copyright_book_hint)
        composer = getattr(self, "_composer", None)
        if composer is None:
            composer = TutorComposer(llm=self.llm)
            self._composer = composer
        return composer.compose(
            query,
            docs,
            memory_context=memory_context,
            strategy=pack["strategy"],
            attempts=pack.get("attempts") or [],
        )

    def hits_to_models(self, docs: List[Dict[str, Any]], layer: str = "") -> List[RetrievalHit]:
        return [
            RetrievalHit(
                text=d.get("text", ""),
                score=float(d.get("score", 0)),
                metadata=d.get("metadata") or {},
                layer=layer,
            )
            for d in docs
        ]
