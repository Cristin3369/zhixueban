# -*- coding: utf-8 -*-
"""
全局配置中心（上线优先）
----------------------
APP_ENV=production 时：必须连上 Redis / Neo4j / Milvus / ES / LLM，缺服务直接失败。
敏感信息只从环境变量读取，禁止把 Key 写进代码。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return float(raw)


@dataclass
class Settings:
    """考研/考公学习 Agent 系统的统一配置。"""

    # production | staging | local（线上 Conda 环境写成 production）
    app_env: str = field(default_factory=lambda: _env("APP_ENV", "local"))

    # ---------- 在线 LLM（DeepSeek / 通义兼容 OpenAI 协议）----------
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "deepseek-chat"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.2))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", _env("DEEPSEEK_API_KEY", "")))
    openai_base_url: str = field(
        default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.deepseek.com")
    )
    llm_timeout_sec: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_SEC", 60.0))

    # 陪伴 Agent：QLoRA+DPO 后的 7B，经 vLLM 提供 OpenAI 兼容口
    companion_model: str = field(default_factory=lambda: _env("COMPANION_MODEL", "zhixueban-companion-7b-dpo"))
    companion_base_url: str = field(default_factory=lambda: _env("COMPANION_BASE_URL", ""))
    companion_api_key: str = field(default_factory=lambda: _env("COMPANION_API_KEY", "EMPTY"))

    # ---------- Neo4j ----------
    neo4j_uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: _env("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "password"))

    # ---------- Redis ----------
    redis_host: str = field(default_factory=lambda: _env("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: _env_int("REDIS_PORT", 6379))
    redis_db: int = field(default_factory=lambda: _env_int("REDIS_DB", 0))
    redis_password: str = field(default_factory=lambda: _env("REDIS_PASSWORD", ""))

    # ---------- Milvus ----------
    milvus_uri: str = field(default_factory=lambda: _env("MILVUS_URI", "http://127.0.0.1:19530"))
    milvus_collection: str = field(default_factory=lambda: _env("MILVUS_COLLECTION", "zhixueban_chunks"))
    milvus_dim: int = field(default_factory=lambda: _env_int("MILVUS_DIM", 1024))  # bge-m3

    # ---------- Elasticsearch BM25 ----------
    es_url: str = field(default_factory=lambda: _env("ES_URL", "http://127.0.0.1:9200"))
    es_index: str = field(default_factory=lambda: _env("ES_INDEX", "zhixueban_chunks"))
    es_user: str = field(default_factory=lambda: _env("ES_USER", ""))
    es_password: str = field(default_factory=lambda: _env("ES_PASSWORD", ""))

    # ---------- 向量 / 重排微服务（GPU）----------
    embed_url: str = field(default_factory=lambda: _env("EMBED_URL", "http://127.0.0.1:8081"))
    rerank_url: str = field(default_factory=lambda: _env("RERANK_URL", "http://127.0.0.1:8082"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-m3"))
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))

    # ---------- ASR / TTS ----------
    asr_provider: str = field(default_factory=lambda: _env("ASR_PROVIDER", "dashscope"))
    tts_provider: str = field(default_factory=lambda: _env("TTS_PROVIDER", "dashscope"))
    dashscope_api_key: str = field(default_factory=lambda: _env("DASHSCOPE_API_KEY", ""))
    asr_model: str = field(default_factory=lambda: _env("ASR_MODEL", "paraformer-realtime-v2"))
    tts_model: str = field(default_factory=lambda: _env("TTS_MODEL", "cosyvoice-v1"))
    tts_voice: str = field(default_factory=lambda: _env("TTS_VOICE", "longxiaochun"))

    # ---------- 四级 TTL ----------
    cache_ttl_l1: int = 3600 * 6
    cache_ttl_l2: int = 3600 * 24 * 7
    cache_ttl_l3: int = 3600 * 24
    cache_ttl_l4: int = 3600 * 24 * 30
    cache_ttl_null: int = 60

    # ---------- 检索 ----------
    hybrid_recall_top_k: int = 20
    hybrid_rerank_top_k: int = 3
    confidence_threshold: float = field(default_factory=lambda: _env_float("CONFIDENCE_THRESHOLD", 0.60))
    isomorphic_sim_threshold: float = 0.75
    tutor_compose_timeout_sec: float = field(
        default_factory=lambda: _env_float("TUTOR_COMPOSE_TIMEOUT_SEC", 20.0)
    )
    router_llm_timeout_sec: float = field(
        default_factory=lambda: _env_float("ROUTER_LLM_TIMEOUT_SEC", 8.0)
    )
    companion_timeout_sec: float = field(
        default_factory=lambda: _env_float("COMPANION_TIMEOUT_SEC", 12.0)
    )
    max_quote_chars: int = field(default_factory=lambda: _env_int("MAX_QUOTE_CHARS", 40))
    evidence_snippet_chars: int = field(
        default_factory=lambda: _env_int("EVIDENCE_SNIPPET_CHARS", 400)
    )

    # ---------- 艾宾浩斯 ----------
    base_intervals_days: List[int] = field(default_factory=lambda: [1, 2, 4, 7, 15, 30])
    energy_rest_threshold: float = 0.25
    emotion_crisis_threshold: float = 0.85

    peer_min_sample: int = 30

    api_host: str = field(default_factory=lambda: _env("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 8000))

    exam_tracks: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "考研数学一": ["高等数学", "线性代数", "概率论与数理统计"],
            "考研英语一": ["英语真题", "词汇", "阅读理解"],
            "考研政治": ["马原", "毛中特", "史纲", "思修"],
            "考公行测": ["数量关系", "判断推理", "资料分析", "言语理解"],
            "考公申论": ["归纳概括", "综合分析", "提出对策", "大作文"],
        }
    )

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"production", "prod", "staging"}


settings = Settings()
