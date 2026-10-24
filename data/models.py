# -*- coding: utf-8 -*-
"""
领域数据模型
-----------
核心约定：
1. 每道题 / 答案必须带「书名 + 页码」，方便学生翻课本；
2. 题与答案通过 question_id 一对一绑定；
3. 图表 / 公式有独立类型，避免被切成无语义碎片；
4. 难度、历年出现频率用于多因子调度与重点标注。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional
import time


class ContentType(str, Enum):
    """切片内容类型：检索与展示策略会 accordingly 分支。"""

    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    FORMULA = "formula"
    AUDIO = "audio"  # 语音陪伴场景的 ASR 文本


class ExamTrack(str, Enum):
    """考试赛道，用于 L3 科目隔离，防止跨科目记忆污染。"""

    KAOYAN_MATH = "考研数学"
    KAOYAN_ENGLISH = "考研英语"
    KAOYAN_POLITICS = "考研政治"
    KAOGONG_XINGCE = "考公行测"
    KAOGONG_SHENLUN = "考公申论"
    OTHER = "其他"


@dataclass
class BookRef:
    """课本溯源：哪本书第几页（商业侧也用于版权标注）。"""

    book_name: str
    page: int
    chapter: str = ""
    edition: str = ""  # 如「2026 大纲版」，配合数据血缘/版本控制
    teacher: str = ""  # 名师讲义作者，可空

    def cite(self) -> str:
        """生成给学生看的溯源短句。"""
        ch = f"「{self.chapter}」" if self.chapter else ""
        return f"《{self.book_name}》{ch}第 {self.page} 页"


@dataclass
class MediaAsset:
    """图表 / 公式等多模态资产。"""

    content_type: ContentType
    # 公式用 latex；图表用 url 或本地 path；表格用结构化 rows
    latex: str = ""
    url: str = ""
    caption: str = ""  # VLM 生成的文字摘要，便于向量检索
    header: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)


@dataclass
class QuestionItem:
    """
    题目实体：与 AnswerItem 通过 question_id 严格对应。
    frequency: 历年真题出现频次（归一化到 0~1 也可，这里用原始次数 + 归一化字段）。
    """

    question_id: str
    stem: str  # 题干
    book_ref: BookRef
    subject: str
    exam_track: str
    difficulty_objective: float = 3.0  # 题本身客观难度 1~5（教研标注）
    exam_frequency: int = 0  # 历年出现次数
    exam_frequency_norm: float = 0.0  # 归一化频率 0~1，调度用
    knowledge_points: List[str] = field(default_factory=list)  # 关联知识点实体名
    media: List[MediaAsset] = field(default_factory=list)
    # 学长学姐说：易出难度提示
    senior_tips: str = ""
    tags: List[str] = field(default_factory=list)
    version: str = "2026"  # 大纲/教材版本，缓存失效用
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerItem:
    """参考答案：必须与题目同书页可对读。"""

    question_id: str
    answer_text: str
    book_ref: BookRef  # 答案所在页（可能与题干页不同，如答案集）
    explanation: str = ""
    media: List[MediaAsset] = field(default_factory=list)
    version: str = "2026"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QAPair:
    """题答绑定单元，缓存与检索的基本单元。"""

    question: QuestionItem
    answer: AnswerItem

    def page_guide(self) -> str:
        """方便学生翻课本的双页提示。"""
        q = self.question.book_ref.cite()
        a = self.answer.book_ref.cite()
        return f"题目见 {q}；答案见 {a}"


@dataclass
class UserProfile:
    """
    用户进页时选定的科目与书本 → 写入 L3。
    energy_level / emotion 会进入多因子调度与防沉迷。
    """

    user_id: str
    exam_track: str
    school: str = ""
    major: str = ""
    subjects: List[str] = field(default_factory=list)
    books: List[str] = field(default_factory=list)
    target_exam_date: str = ""  # ISO 日期
    energy_level: float = 0.7  # 0~1，今日精力
    emotion_intensity: float = 0.0  # 0~1，情绪强度
    preferred_mode: str = "text"  # text | voice


@dataclass
class ReviewFeedback:
    """
    多因子反馈：替代原先「双因子」。
    调度 Agent / 计划 Agent 都会消费此结构。
    """

    is_correct: bool = True
    response_time: float = 30.0  # 答题用时（秒）
    subjective_difficulty: int = 3  # 用户自评难度 1~5
    objective_difficulty: float = 3.0  # 题目本身难度
    exam_probability: float = 0.5  # 考试出现概率（由 frequency_norm 等推得）
    user_energy: float = 0.7  # 今日状态/精力
    emotion_intensity: float = 0.0  # 情绪因子
    retention_self_score: Optional[int] = None  # 可选：自我感觉记住程度 1~5


@dataclass
class StudentSpacedRecord:
    """艾宾浩斯 / SM-2 持久化状态。"""

    last_interval: int = 0
    consecutive_correct: int = 0
    ef: float = 2.5
    total_reviews: int = 0
    last_review_ts: float = 0.0


@dataclass
class CacheLookupResult:
    """四级缓存统一返回结构。"""

    source: str  # L1 | L2 | L3 | L4 | GENERATED | FALLBACK | NULL_CACHED
    data: Any = None
    confidence: float = 1.0
    message: str = ""
    layer_trail: List[str] = field(default_factory=list)  # 走过的层，便于调试


@dataclass
class RetrievalHit:
    """混合检索单条命中。"""

    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    layer: str = ""  # 来自哪一层缓存/库
