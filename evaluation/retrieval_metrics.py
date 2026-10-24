# -*- coding: utf-8 -*-
"""检索指标：Recall@K / NDCG@K / MRR。"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Set


def _hit(doc: dict, relevant_substrings: Sequence[str], relevant_ids: Set[str]) -> bool:
    did = str(doc.get("id") or "")
    if did and did in relevant_ids:
        return True
    text = str(doc.get("text") or "")
    return any(s and s in text for s in relevant_substrings)


def recall_at_k(ranked: Sequence[dict], relevant_substrings: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_substrings and not relevant_ids:
        return 0.0
    got = any(_hit(d, relevant_substrings, relevant_ids) for d in ranked[:k])
    return 1.0 if got else 0.0


def mrr(ranked: Sequence[dict], relevant_substrings: Sequence[str], relevant_ids: Set[str]) -> float:
    for i, d in enumerate(ranked, start=1):
        if _hit(d, relevant_substrings, relevant_ids):
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[dict], relevant_substrings: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    dcg = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        rel = 1.0 if _hit(d, relevant_substrings, relevant_ids) else 0.0
        dcg += rel / math.log2(i + 1)
    # 二元相关：理想是第一条就命中
    idcg = 1.0 / math.log2(2)
    return dcg / idcg if idcg else 0.0


def aggregate(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = rows[0].keys()
    n = len(rows)
    return {k: round(sum(r[k] for r in rows) / n, 4) for k in keys} | {"n": float(n)}
