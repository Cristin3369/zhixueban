# -*- coding: utf-8 -*-
"""把讲解拆成 claim，逐条看是否能在证据里找到支撑。"""

from __future__ import annotations

import re
from typing import List, Sequence

_SPLIT = re.compile(r"[。！？\n]+")
_SKIP = ("【考点】", "【思路】", "【易错】", "【溯源】", "资料不足")


def split_claims(answer: str) -> List[str]:
    raw = []
    for line in _SPLIT.split(answer or ""):
        t = line.strip()
        for prefix in ("【考点】", "【思路】", "【易错】", "【溯源】"):
            if t.startswith(prefix):
                t = t[len(prefix) :].strip()
        if len(t) < 8:
            continue
        if t in _SKIP:
            continue
        raw.append(t)
    return raw[:12]


def claim_supported(claim: str, evidence: str, min_hits: int = 2) -> bool:
    """
    轻量 Faithfulness：claim 中长度≥2 的中文/英文片段，至少 min_hits 个出现在证据里。
    不是 NLI，只挡住「整句完全漂在证据外」。
    """
    blob = evidence or ""
    if not claim or not blob:
        return False
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", claim)
    if not tokens:
        return claim[:8] in blob
    hits = sum(1 for tok in tokens if tok in blob)
    return hits >= min(min_hits, max(1, len(tokens) // 3))


def unsupported_claims(answer: str, evidences: Sequence[str]) -> List[str]:
    blob = "\n".join(evidences)
    return [c for c in split_claims(answer) if not claim_supported(c, blob)]
