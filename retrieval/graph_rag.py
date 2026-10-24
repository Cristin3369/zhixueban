# -*- coding: utf-8 -*-
"""
GraphRAG（Neo4j）
---------------
- 多跳前置/后置：解决跨章节逻辑关联
- 题目 → 知识点实体
- 易混淆边（可由错题行为离线沉淀）
- 同构题生成所需的定义/公式卡片
无 Neo4j 时自动降级到内存图谱，保证演示可跑。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class InMemoryConceptGraph:
    """本地演示用的迷你知识图谱。"""

    def __init__(self) -> None:
        # name -> node
        self.nodes: Dict[str, Dict[str, Any]] = {
            "反向传播算法": {
                "name": "反向传播算法",
                "definition": "利用链式法则从损失向参数反向传播梯度的训练方法。",
                "formula": r"\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a}\frac{\partial a}{\partial z}\frac{\partial z}{\partial w}",
            },
            "链式法则": {
                "name": "链式法则",
                "definition": "复合函数求导法则，是反向传播的数学基础。",
                "formula": r"(f\circ g)' = (f'\circ g)\cdot g'",
            },
            "梯度下降": {
                "name": "梯度下降",
                "definition": "沿损失函数梯度反方向更新参数以最小化损失。",
                "formula": r"w_{t+1}=w_t-\eta\nabla L(w_t)",
            },
        }
        # (from, to) PREREQUISITE_FOR: from 是 to 的前置
        self.prereq_edges = [
            ("链式法则", "反向传播算法"),
            ("反向传播算法", "梯度下降"),
        ]
        self.confusable: List[tuple[str, str]] = [("梯度下降", "反向传播算法")]
        self.question_to_concept: Dict[str, str] = {}

    def get_card(self, concept_name: str, hops: int = 1) -> Optional[dict]:
        if concept_name not in self.nodes:
            # 模糊：包含匹配
            for name in self.nodes:
                if concept_name in name or name in concept_name:
                    concept_name = name
                    break
            else:
                return None
        current = self.nodes[concept_name]
        pres = [a for a, b in self.prereq_edges if b == concept_name]
        posts = [b for a, b in self.prereq_edges if a == concept_name]
        # 简单多跳：再扩一层前置的前置
        if hops >= 2:
            extra = []
            for p in list(pres):
                extra.extend([a for a, b in self.prereq_edges if b == p])
            pres = list(dict.fromkeys(pres + extra))
        conf = []
        for a, b in self.confusable:
            if a == concept_name:
                conf.append(b)
            elif b == concept_name:
                conf.append(a)
        return {
            "current_concept": current["name"],
            "definition": current["definition"],
            "formula": current.get("formula", ""),
            "prerequisites": pres,
            "extensions": posts,
            "confusable": conf,
        }


class KnowledgeGraphRAG:
    """
    GraphRAG 主类：优先 Neo4j，失败则 InMemory。
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        *,
        force_memory: bool = False,
    ):
        self._memory = InMemoryConceptGraph()
        self.driver = None
        if not force_memory:
            try:
                from neo4j import GraphDatabase  # type: ignore

                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                # 轻量探测
                with self.driver.session() as session:
                    session.run("RETURN 1")
                logger.info("Neo4j 已连接: %s", uri)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Neo4j 不可用，使用内存图谱: %s", exc)
                self.driver = None

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def get_knowledge_card(self, concept_name: str, hops: int = 2) -> Optional[dict]:
        """多跳前置/后置知识卡片。"""
        if not self.driver:
            return self._memory.get_card(concept_name, hops=hops)

        # 可变跳数：用路径长度限制
        cypher = """
        MATCH (current:Concept)
        WHERE current.name = $concept_name
           OR current.name CONTAINS $concept_name
        WITH current LIMIT 1
        OPTIONAL MATCH pre_path = (pre:Concept)-[:PREREQUISITE_FOR*1..$hops]->(current)
        OPTIONAL MATCH post_path = (current)-[:PREREQUISITE_FOR*1..$hops]->(post:Concept)
        OPTIONAL MATCH (current)-[:CONFUSABLE_WITH]-(conf:Concept)
        RETURN
            current.name AS current_concept,
            current.definition AS definition,
            current.formula AS formula,
            collect(DISTINCT pre.name) AS prerequisites,
            collect(DISTINCT post.name) AS extensions,
            collect(DISTINCT conf.name) AS confusable
        """
        try:
            with self.driver.session() as session:
                # neo4j 参数 hops 在部分版本不能直接用于变长；降级固定 2 跳写法
                cypher_fixed = """
                MATCH (current:Concept)
                WHERE current.name = $concept_name OR current.name CONTAINS $concept_name
                WITH current LIMIT 1
                OPTIONAL MATCH (pre:Concept)-[:PREREQUISITE_FOR*1..2]->(current)
                OPTIONAL MATCH (current)-[:PREREQUISITE_FOR*1..2]->(post:Concept)
                OPTIONAL MATCH (current)-[:CONFUSABLE_WITH]-(conf:Concept)
                RETURN
                    current.name AS current_concept,
                    current.definition AS definition,
                    current.formula AS formula,
                    collect(DISTINCT pre.name) AS prerequisites,
                    collect(DISTINCT post.name) AS extensions,
                    collect(DISTINCT conf.name) AS confusable
                """
                result = session.run(cypher_fixed, concept_name=concept_name).single()
                if not result:
                    return self._memory.get_card(concept_name, hops=hops)
                return {
                    "current_concept": result["current_concept"],
                    "definition": result["definition"] or "",
                    "formula": result["formula"] or "",
                    "prerequisites": [x for x in result["prerequisites"] if x],
                    "extensions": [x for x in result["extensions"] if x],
                    "confusable": [x for x in result["confusable"] if x],
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cypher 失败，回落内存图: %s", exc)
            return self._memory.get_card(concept_name, hops=hops)

    def get_concept_by_question_id(self, question_id: str) -> Optional[dict]:
        """题目 ID → 知识点实体（同构题生成入口）。"""
        if not self.driver:
            name = self._memory.question_to_concept.get(question_id)
            if not name:
                return None
            return self._memory.get_card(name)

        cypher = """
        MATCH (q:Question {id: $qid})-[:TESTS]->(c:Concept)
        RETURN c.name AS name, c.definition AS definition, c.formula AS formula
        LIMIT 1
        """
        try:
            with self.driver.session() as session:
                row = session.run(cypher, qid=question_id).single()
                if not row:
                    return None
                return {
                    "current_concept": row["name"],
                    "definition": row["definition"] or "",
                    "formula": row["formula"] or "",
                    "prerequisites": [],
                    "extensions": [],
                    "confusable": [],
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_concept_by_question_id 失败: %s", exc)
            return None

    def link_question_concept(self, question_id: str, concept_name: str) -> None:
        """本地演示：绑定题与知识点。"""
        self._memory.question_to_concept[question_id] = concept_name

    def add_confusable_edge(self, a: str, b: str) -> None:
        """隐性知识：大量学生混淆 A/B 时沉淀边。"""
        self._memory.confusable.append((a, b))
        if not self.driver:
            return
        cypher = """
        MERGE (x:Concept {name: $a})
        MERGE (y:Concept {name: $b})
        MERGE (x)-[:CONFUSABLE_WITH]->(y)
        """
        try:
            with self.driver.session() as session:
                session.run(cypher, a=a, b=b)
        except Exception as exc:  # noqa: BLE001
            logger.warning("写入易混淆边失败: %s", exc)

    def traverse_for_plan(self, concept_name: str) -> Dict[str, Any]:
        """
        跨章节前置/后置推理，供计划 Agent 决定「先补哪块基础」。
        """
        card = self.get_knowledge_card(concept_name, hops=2) or {}
        return {
            "focus": card.get("current_concept", concept_name),
            "must_review_first": card.get("prerequisites", []),
            "next_unlock": card.get("extensions", []),
            "watch_confusable": card.get("confusable", []),
            "definition": card.get("definition", ""),
            "formula": card.get("formula", ""),
        }

    def build_rag_prompt(self, concept_name: str, user_question: str = "") -> str:
        card = self.get_knowledge_card(concept_name)
        if not card:
            return (
                f"知识图谱中未找到关于「{concept_name}」的相关记录。"
                "请触发检索兜底或拒绝生成。"
            )
        q = user_question or concept_name
        return f"""
你是考研/考公辅导 AI。请严格依据知识图谱卡片解答，禁止编造卡片外结论。

=== 知识图谱上下文 ===
1. 核心概念：{card['current_concept']}
2. 定义：{card['definition']}
3. 公式：{card.get('formula') or '无'}
4. 前置基础：{', '.join(card['prerequisites']) if card['prerequisites'] else '无'}
5. 后置延伸：{', '.join(card['extensions']) if card['extensions'] else '无'}
6. 易混淆点：{', '.join(card.get('confusable') or []) or '无'}
======================

学生问题：{q}

输出要求：
1. 先通俗解释定义与公式；
2. 若有前置，提醒回顾；
3. 末尾用「知识点解析」风格总结，并提示回课本对照；
4. 资料不足时明确说「资料不足」，不要发挥。
""".strip()
