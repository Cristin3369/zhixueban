# -*- coding: utf-8 -*-
"""
查询改写与检索策略切换
---------------------
兜底链路：
1. Query Rewrite：抽实体、消歧、同义扩展 → 多视角重搜
2. 搜索词降维：长句 → 核心关键词
3. 策略切换：向量失败 → BM25；关系型 → GraphRAG
4. Learn to Refuse：低于阈值则标准化拒绝
"""

from __future__ import annotations

import re
from typing import List, Protocol


class LLMRewriteClient(Protocol):
    def rewrite_query(self, query: str) -> str: ...


class HeuristicQueryRewriter:
    """无 LLM 时的规则改写，保证链路完整。"""

    # 简易同义/考公考研常见别名
    SYNONYMS = {
        "反向传播": ["BP算法", "误差反向传播", "backpropagation"],
        "马原": ["马克思主义基本原理", "马克思主义原理"],
        "行测": ["行政职业能力测验"],
    }

    def rewrite_query(self, query: str) -> str:
        variants = self.expand_variants(query)
        # 主改写：取「最短关键词视角」
        return variants[0] if variants else query

    def expand_variants(self, query: str) -> List[str]:
        q = query.strip()
        variants = [q]
        # 降维：去掉语气词
        core = re.sub(r"(到底|怎么|如何|请问|麻烦|啊|呢|吗|呀)+", "", q)
        core = re.sub(r"\s+", "", core)
        if core and core != q:
            variants.append(core)
        # 同义扩展
        for k, syns in self.SYNONYMS.items():
            if k in q:
                for s in syns:
                    variants.append(q.replace(k, s))
        # 实体粗提：连续中文 2~8 字片段
        entities = re.findall(r"[\u4e00-\u9fff]{2,8}", q)
        for e in entities[:3]:
            if e not in variants:
                variants.append(e)
        # 去重保序
        seen = set()
        out = []
        for v in variants:
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def generalize_keywords(self, query: str, max_terms: int = 5) -> str:
        """放弃精确长句，提取核心词回退检索。"""
        terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_\-]+|\d+", query)
        return " ".join(terms[:max_terms])


REFUSE_MESSAGE = "当前知识库暂未收录该考点或相关数据，建议查阅指定教材对应页码，或换一种问法。"


def should_refuse(top_score: float, threshold: float) -> bool:
    return top_score < threshold
