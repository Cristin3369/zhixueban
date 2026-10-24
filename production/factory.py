# -*- coding: utf-8 -*-
"""
生产装配工厂
-----------
APP_ENV=production 时：Redis / Neo4j / Milvus / ES / LLM 缺一不可，启动失败即暴露配置问题。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ProductionStack:
    redis: Any
    vector_store: Any
    bm25: Any
    reranker: Any
    llm: Any
    companion_llm: Any
    graph_rag: Any
    peer: Any
    tts: Any
    asr: Any
    queue: Any
    store: Any


def build_production_stack() -> ProductionStack:
    if not settings.is_production:
        raise RuntimeError("build_production_stack 仅用于 production/staging")

    missing = []
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY 或 DEEPSEEK_API_KEY")
    if missing:
        raise RuntimeError("生产环境缺少: " + ", ".join(missing))

    from data.study_queue import StudyTaskQueue
    from data.tiered_cache import TieredKnowledgeStore
    from production.asr_tts import DashScopeASR, DashScopeTTS
    from production.es_bm25 import ElasticsearchBM25
    from production.http_clients import BGERerankerClient, OpenAICompatLLM
    from production.milvus_store import MilvusVectorStore
    from production.redis_backend import RedisPeerStats, build_redis
    from retrieval.graph_rag import KnowledgeGraphRAG

    rds = build_redis()
    rds.ping()

    graph = KnowledgeGraphRAG(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        force_memory=False,
    )
    if graph.driver is None:
        raise RuntimeError("Neo4j 连接失败，生产禁止落入内存图谱。")

    vector = MilvusVectorStore()
    bm25 = ElasticsearchBM25()
    rerank = BGERerankerClient()
    llm = OpenAICompatLLM()

    companion_llm = None
    if settings.companion_base_url:
        companion_llm = OpenAICompatLLM(
            model=settings.companion_model,
            base_url=settings.companion_base_url,
            api_key=settings.companion_api_key,
            temperature=0.7,
        )

    peer = RedisPeerStats(rds)
    asr = tts = None
    if settings.dashscope_api_key:
        tts = DashScopeTTS()
        asr = DashScopeASR()
    else:
        logger.warning("未配置 DASHSCOPE_API_KEY，语音陪伴不可用，文本陪伴仍可用。")

    queue = StudyTaskQueue(redis_client=rds)
    store = TieredKnowledgeStore(redis_client=rds)

    logger.info("生产栈已装配：Milvus + ES + BGE-Rerank + Neo4j + Redis + DeepSeek")
    return ProductionStack(
        redis=rds,
        vector_store=vector,
        bm25=bm25,
        reranker=rerank,
        llm=llm,
        companion_llm=companion_llm,
        graph_rag=graph,
        peer=peer,
        tts=tts,
        asr=asr,
        queue=queue,
        store=store,
    )
