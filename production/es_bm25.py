# -*- coding: utf-8 -*-
"""Elasticsearch BM25：专有名词 / 公式名 / 题号精确召回。"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch

from config.settings import settings

logger = logging.getLogger(__name__)


class ElasticsearchBM25:
    def __init__(self, *, url: Optional[str] = None, index: Optional[str] = None):
        self.url = url or settings.es_url
        self.index = index or settings.es_index
        kwargs: Dict[str, Any] = {"hosts": [self.url]}
        if settings.es_user:
            kwargs["basic_auth"] = (settings.es_user, settings.es_password)
        self.client = Elasticsearch(**kwargs)
        self._ensure_index()

    def _ensure_index(self) -> None:
        if self.client.indices.exists(index=self.index):
            return
        mapping_text = {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"}
        mappings = {
            "properties": {
                "text": mapping_text,
                "subject": {"type": "keyword"},
                "user_id": {"type": "keyword"},
                "book_name": {"type": "keyword"},
                "page": {"type": "integer"},
                "chunk_type": {"type": "keyword"},
            }
        }
        try:
            self.client.indices.create(
                index=self.index,
                mappings=mappings,
                settings={"number_of_shards": 1, "number_of_replicas": 1},
            )
        except Exception:
            mappings["properties"]["text"] = {"type": "text"}
            self.client.indices.create(
                index=self.index,
                mappings=mappings,
                settings={"number_of_shards": 1, "number_of_replicas": 1},
            )
            logger.warning("ES 无 IK 分词，已回落 standard analyzer")
        logger.info("已创建 ES index %s", self.index)

    def add(self, text: str, metadata: Optional[dict] = None) -> str:
        meta = metadata or {}
        cid = str(meta.get("id") or uuid.uuid4().hex)
        self.client.index(
            index=self.index,
            id=cid,
            document={
                "text": text,
                "subject": meta.get("subject", ""),
                "user_id": meta.get("user_id", ""),
                "book_name": meta.get("book_name", ""),
                "page": int(meta.get("page") or 0),
                "chunk_type": meta.get("type", "text"),
            },
            refresh=True,
        )
        return cid

    def search(
        self,
        query: str,
        top_k: int = 20,
        *,
        subject: str = "",
        user_id: str = "",
    ) -> List[Dict[str, Any]]:
        filters = []
        if subject:
            filters.append({"term": {"subject": subject}})
        if user_id:
            filters.append({"term": {"user_id": user_id}})
        body: Dict[str, Any] = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [{"match": {"text": {"query": query, "operator": "or"}}}],
                    "filter": filters,
                }
            },
        }
        resp = self.client.search(index=self.index, query=body["query"], size=top_k)
        out = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            out.append(
                {
                    "id": hit["_id"],
                    "text": src.get("text", ""),
                    "score": float(hit["_score"] or 0),
                    "metadata": {
                        "subject": src.get("subject"),
                        "user_id": src.get("user_id"),
                        "book_name": src.get("book_name"),
                        "page": src.get("page"),
                        "type": src.get("chunk_type"),
                    },
                }
            )
        return out
