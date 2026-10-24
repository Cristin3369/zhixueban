# -*- coding: utf-8 -*-
"""
讲解落地检查：引用、照抄、无依据发挥
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from config.settings import settings
from generation.claims import unsupported_claims
from retrieval.query_rewrite import REFUSE_MESSAGE

LEAK_PATTERNS = (
    "根据我的训练",
    "作为人工智能",
    "我是 ChatGPT",
    "我是 DeepSeek",
    "我不知道教材",
    "网上普遍认为",
)

CITE_RE = re.compile(r"(第\s*\d+\s*页)|溯源|《.+》")


@dataclass
class Evidence:
    text: str
    book: str = ""
    page: str = "?"
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    @classmethod
    def from_doc(cls, doc: Dict[str, Any]) -> "Evidence":
        meta = dict(doc.get("metadata") or {})
        return cls(
            text=str(doc.get("text") or ""),
            book=str(meta.get("book_name") or meta.get("book") or "指定教材"),
            page=str(meta.get("page", "?")),
            metadata=meta,
            score=float(doc.get("score") or 0.0),
        )

    def cite_line(self) -> str:
        return f"《{self.book}》第{self.page}页"


@dataclass
class GroundingReport:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    patched_text: str = ""
    unsupported_claims: List[str] = field(default_factory=list)


def _longest_overlap(answer: str, evidence: str, min_len: int) -> int:
    """连续字符重合长度；用于拦截大段原文照抄。"""
    if not answer or not evidence:
        return 0
    a, e = answer.replace("\n", ""), evidence.replace("\n", "")
    best = 0
    # 滑窗：控制复杂度，证据截断
    e = e[:1200]
    a = a[:2000]
    step = max(8, min_len // 2)
    for i in range(0, max(len(e) - min_len + 1, 1), step):
        chunk = e[i : i + min_len]
        if chunk and chunk in a:
            # 尝试向两侧扩展
            L = min_len
            while i + L < len(e) and e[i : i + L + 1] in a:
                L += 1
            best = max(best, L)
    return best


def check_answer(
    answer: str,
    evidences: List[Evidence],
    *,
    allow_long_quote: bool = False,
) -> GroundingReport:
    reasons: List[str] = []
    text = (answer or "").strip()
    if not text:
        return GroundingReport(ok=False, reasons=["空回答"])

    if any(p in text for p in LEAK_PATTERNS):
        reasons.append("模型越权引用训练知识")

    if "资料不足" in text and len(text) < 80:
        return GroundingReport(ok=False, reasons=["模型自报资料不足"])

    joined = "\n".join(ev.text for ev in evidences)
    max_quote = settings.max_quote_chars
    if not allow_long_quote:
        overlap = _longest_overlap(text, joined, max_quote)
        if overlap >= max_quote:
            reasons.append(f"连续照抄原文 {overlap} 字，超过 {max_quote}")

    if not CITE_RE.search(text):
        reasons.append("缺少教材页码溯源")

    bad_claims = unsupported_claims(text, [ev.text for ev in evidences])
    if not allow_long_quote and len(bad_claims) >= 2:
        reasons.append(f"claim 级无证据 {len(bad_claims)} 条")

    if reasons:
        return GroundingReport(
            ok=False,
            reasons=reasons,
            patched_text=text,
            unsupported_claims=bad_claims,
        )

    # 补一条标准溯源（有引用格式也再钉死页码，防模型写了「见教材」却不写页）
    cites = "；".join(dict.fromkeys(ev.cite_line() for ev in evidences[:3]))
    if cites and cites not in text:
        text = text.rstrip() + f"\n【溯源】本题解析基于 {cites}，建议结合原书例题理解。"
    return GroundingReport(ok=True, patched_text=text, unsupported_claims=bad_claims)


def refuse_pack(strategy: str, attempts: List[str] | None = None) -> Dict[str, Any]:
    return {
        "status": "fallback_triggered",
        "response": REFUSE_MESSAGE,
        "sources": [],
        "strategy": strategy,
        "attempts": attempts or [],
        "compose_mode": "refuse",
    }
