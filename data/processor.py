# -*- coding: utf-8 -*-
"""
数据处理流水线
-------------
目标：
1. 把参考答案和题标注上「哪本书第几页」，并做成可对读的 QAPair；
2. 识别图表 / 公式，生成可检索的文字摘要（配合 VLM）；
3. 写入客观难度；
4. 根据历年真题统计每道题出现频率，归一化后供调度重点掌握。
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Protocol

from data.models import (
    AnswerItem,
    BookRef,
    ContentType,
    MediaAsset,
    QAPair,
    QuestionItem,
)


class VLMClient(Protocol):
    """多模态客户端协议：生产环境接 GPT-4o / Qwen-VL 等。"""

    def generate_caption(self, image_url: str) -> str: ...


class StubVLM:
    """无外网时的占位实现，保证管线可本地跑通。"""

    def generate_caption(self, image_url: str) -> str:
        return f"图表占位摘要（请接入真实 VLM）: {image_url}"


class ExamDataProcessor:
    """
    将原始教研材料（真题、讲义、答案）规范化为可检索、可溯源的结构化数据。
    """

    def __init__(self, vlm: Optional[VLMClient] = None):
        self.vlm = vlm or StubVLM()
        # 历年出现计数：question_fingerprint -> count
        self._frequency_counter: Counter = Counter()

    # ------------------------------------------------------------------ #
    # 题答配对与页码标注
    # ------------------------------------------------------------------ #
    def build_qa_pair(
        self,
        stem: str,
        answer_text: str,
        *,
        book_name: str,
        question_page: int,
        answer_page: int,
        subject: str,
        exam_track: str,
        chapter: str = "",
        edition: str = "2026",
        difficulty_objective: float = 3.0,
        knowledge_points: Optional[List[str]] = None,
        senior_tips: str = "",
        question_id: Optional[str] = None,
        explanation: str = "",
        teacher: str = "",
    ) -> QAPair:
        """
        强制绑定题与答，并分别标注书页。
        question_id 缺省时用题干哈希，保证幂等入库。
        """
        qid = question_id or self._stable_id(book_name, question_page, stem)
        q_ref = BookRef(
            book_name=book_name,
            page=question_page,
            chapter=chapter,
            edition=edition,
            teacher=teacher,
        )
        a_ref = BookRef(
            book_name=book_name,
            page=answer_page,
            chapter=chapter,
            edition=edition,
            teacher=teacher,
        )
        question = QuestionItem(
            question_id=qid,
            stem=stem.strip(),
            book_ref=q_ref,
            subject=subject,
            exam_track=exam_track,
            difficulty_objective=float(difficulty_objective),
            knowledge_points=knowledge_points or [],
            senior_tips=senior_tips,
            version=edition,
        )
        answer = AnswerItem(
            question_id=qid,
            answer_text=answer_text.strip(),
            book_ref=a_ref,
            explanation=explanation,
            version=edition,
        )
        return QAPair(question=question, answer=answer)

    def annotate_pages_batch(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[QAPair]:
        """
        批量标注。每行期望字段：
        stem, answer, book_name, q_page, a_page, subject, exam_track,
        可选 chapter/difficulty/knowledge_points/senior_tips/...
        """
        pairs: List[QAPair] = []
        for row in rows:
            pairs.append(
                self.build_qa_pair(
                    stem=row["stem"],
                    answer_text=row["answer"],
                    book_name=row["book_name"],
                    question_page=int(row["q_page"]),
                    answer_page=int(row.get("a_page", row["q_page"])),
                    subject=row["subject"],
                    exam_track=row["exam_track"],
                    chapter=row.get("chapter", ""),
                    edition=row.get("edition", "2026"),
                    difficulty_objective=float(row.get("difficulty", 3)),
                    knowledge_points=row.get("knowledge_points"),
                    senior_tips=row.get("senior_tips", ""),
                    question_id=row.get("question_id"),
                    explanation=row.get("explanation", ""),
                    teacher=row.get("teacher", ""),
                )
            )
        return pairs

    # ------------------------------------------------------------------ #
    # 图表 / 公式识别与切片
    # ------------------------------------------------------------------ #
    def process_multimodal_elements(
        self,
        raw_elements: List[Dict[str, Any]],
        book_ref: BookRef,
    ) -> List[Dict[str, Any]]:
        """
        处理含图片、公式、表格的原始文档流，输出可入库的 chunk 列表。
        注意：公式保留 LaTeX；图片用 VLM 摘要进向量，原图 URL 进 metadata 供前端展示。
        """
        chunks: List[Dict[str, Any]] = []
        for element in raw_elements:
            etype = element.get("type", "text")
            meta_base = {
                "book_name": book_ref.book_name,
                "page": book_ref.page,
                "chapter": book_ref.chapter,
                "edition": book_ref.edition,
                "type": etype,
            }

            if etype == ContentType.IMAGE.value or etype == "image":
                url = element.get("url", "")
                caption = self.vlm.generate_caption(url)
                chunks.append(
                    {
                        "text": f"[图表摘要] {caption}",
                        "metadata": {**meta_base, "img_url": url},
                        "media": MediaAsset(
                            content_type=ContentType.IMAGE,
                            url=url,
                            caption=caption,
                        ),
                    }
                )
            elif etype == ContentType.TABLE.value or etype == "table":
                header = element.get("header", [])
                for row in element.get("rows", []):
                    chunks.append(
                        {
                            "text": f"[表格数据] 表头: {header} | 数据: {row}",
                            "metadata": {**meta_base, "type": "table"},
                            "media": MediaAsset(
                                content_type=ContentType.TABLE,
                                header=list(header),
                                rows=[list(map(str, row))],
                            ),
                        }
                    )
            elif etype == ContentType.FORMULA.value or etype == "formula":
                latex = element.get("latex", "")
                context = element.get("context", "")
                chunks.append(
                    {
                        "text": f"[数学公式] {latex} | 解释: {context}",
                        "metadata": {**meta_base, "type": "formula", "latex": latex},
                        "media": MediaAsset(
                            content_type=ContentType.FORMULA,
                            latex=latex,
                            caption=context,
                        ),
                    }
                )
            else:
                text = element.get("text", "")
                if text.strip():
                    chunks.append(
                        {
                            "text": text,
                            "metadata": {**meta_base, "type": "text"},
                        }
                    )
        return chunks

    def extract_formulas_from_text(self, text: str) -> List[str]:
        """简易启发式：抓取 $...$ 与 \\[...\\] 中的公式，供后续专门建索引。"""
        patterns = [
            r"\$\$(.+?)\$\$",
            r"\$(.+?)\$",
            r"\\\[(.+?)\\\]",
            r"\\\((.+?)\\\)",
        ]
        found: List[str] = []
        for pat in patterns:
            found.extend(re.findall(pat, text, flags=re.DOTALL))
        return [f.strip() for f in found if f.strip()]

    # ------------------------------------------------------------------ #
    # 历年真题频率
    # ------------------------------------------------------------------ #
    def ingest_past_paper_occurrence(
        self,
        question_fingerprint: str,
        year: int,
        paper_name: str = "",
    ) -> None:
        """
        录入「某年真题出现过该题/同构题」一次。
        fingerprint 建议用知识点+题型+关键数字归一化后的哈希，而不是裸题干。
        """
        _ = (year, paper_name)  # 预留审计字段，可扩展写入明细表
        self._frequency_counter[question_fingerprint] += 1

    def apply_frequency_to_pairs(self, pairs: List[QAPair]) -> List[QAPair]:
        """
        把计数写回 QuestionItem，并做 min-max 归一化到 exam_frequency_norm。
        频率高 → 进 L1 重点掌握池的权重更大。
        """
        if not pairs:
            return pairs

        # 若外部已有计数，优先用；否则按 fingerprint 查 counter
        raw_counts: List[int] = []
        for p in pairs:
            fp = self._fingerprint(p.question)
            count = self._frequency_counter.get(fp, p.question.exam_frequency)
            p.question.exam_frequency = int(count)
            raw_counts.append(int(count))

        max_c = max(raw_counts) if raw_counts else 0
        min_c = min(raw_counts) if raw_counts else 0
        # 只有一道题时 min==max，不能用差值归一化，否则会得到 0
        if max_c == min_c:
            for p in pairs:
                p.question.exam_frequency_norm = 1.0 if p.question.exam_frequency > 0 else 0.0
        else:
            span = max_c - min_c
            for p in pairs:
                p.question.exam_frequency_norm = (p.question.exam_frequency - min_c) / span
        return pairs

    def rank_high_frequency(
        self,
        pairs: List[QAPair],
        top_n: int = 50,
        min_norm: float = 0.6,
    ) -> List[QAPair]:
        """筛选高频题，默认供 L1 预热。"""
        filtered = [p for p in pairs if p.question.exam_frequency_norm >= min_norm]
        filtered.sort(key=lambda x: x.question.exam_frequency, reverse=True)
        return filtered[:top_n]

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _stable_id(book_name: str, page: int, stem: str) -> str:
        raw = f"{book_name}|{page}|{stem.strip()}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _fingerprint(q: QuestionItem) -> str:
        """同构题归并用：知识点排序 + 题干去空白后哈希。"""
        kps = ",".join(sorted(q.knowledge_points))
        stem_norm = re.sub(r"\s+", "", q.stem)
        raw = f"{q.subject}|{kps}|{stem_norm}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()
