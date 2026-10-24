# -*- coding: utf-8 -*-
"""
冷记忆打分：相关度 × 时间衰减 × 重要性
------------------------------------
不把「记忆」写成产品功能，而写成窗口装填策略。
Milvus 不在这里：向量库属于题库切片，不属于会话记忆。
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

from memory.episodic import MemCell


def lexical_overlap(query: str, text: str) -> float:
    q = set(query or "")
    t = set(text or "")
    if not q or not t:
        return 0.0
    return len(q & t) / max(len(q), 1)


def time_decay(age_hours: float, half_life_hours: float = 72.0) -> float:
    """指数衰减：每过一个半衰期，权重减半。"""
    if half_life_hours <= 0:
        return 1.0
    return math.pow(0.5, max(age_hours, 0.0) / half_life_hours)


def memory_score(
    cell: MemCell,
    query: str,
    *,
    now: float,
    half_life_hours: float = 72.0,
) -> float:
    age_h = max(0.0, (now - cell.timestamp) / 3600.0)
    rel = lexical_overlap(query, cell.content)
    return rel * time_decay(age_h, half_life_hours) * float(cell.importance or 0.0)


def rank_cells(
    cells: Sequence[MemCell],
    query: str,
    *,
    now: float,
    top_n: int = 8,
    half_life_hours: float = 72.0,
    min_score: float = 1e-6,
) -> List[Tuple[float, MemCell]]:
    scored = [
        (memory_score(c, query, now=now, half_life_hours=half_life_hours), c)
        for c in cells
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(s, c) for s, c in scored[:top_n] if s >= min_score]


def compress_cells(cells: Iterable[MemCell], max_chars: int = 600) -> str:
    """滑动截断：按已排序顺序拼到字符上限，近似「摘要压缩进窗口」。"""
    parts: List[str] = []
    used = 0
    for c in cells:
        piece = f"[{c.role}] {c.content.strip()}"
        if used + len(piece) + 1 > max_chars:
            remain = max_chars - used - 1
            if remain > 20:
                parts.append(piece[:remain] + "…")
            break
        parts.append(piece)
        used += len(piece) + 1
    return "\n".join(parts)
