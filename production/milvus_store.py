# -*- coding: utf-8 -*-
"""Milvus 向量库：科目/用户 metadata 硬过滤，防跨科污染。"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from config.settings import settings
from production.http_clients import EmbedClient

logger = logging.getLogger(__name__)


class MilvusVectorStore:
    def __init__(
        self,
        *,
        uri: Optional[str] = None,
        collection: Optional[str] = None,
        dim: Optional[int] = None,
        embedder: Optional[EmbedClient] = None,
    ):
        from pymilvus import Collection, connections, utility
        from pymilvus import CollectionSchema, DataType, FieldSchema

        self.uri = uri or settings.milvus_uri
        self.collection_name = collection or settings.milvus_collection
        self.dim = dim or settings.milvus_dim
        self.embedder = embedder or EmbedClient()
        connections.connect(alias="default", uri=self.uri)

        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8000),
                FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="book_name", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="page", dtype=DataType.INT64),
                FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=32),
            ]
            schema = CollectionSchema(fields, description="zhixueban hybrid chunks")
            col = Collection(self.collection_name, schema)
            col.create_index(
                "embedding",
                {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}},
            )
            logger.info("已创建 Milvus collection %s", self.collection_name)
        self.col = Collection(self.collection_name)
        self.col.load()

    def add(self, text: str, metadata: Optional[dict] = None) -> str:
        meta = metadata or {}
        vec = self.embedder.embed([text])[0]
        cid = str(meta.get("id") or uuid.uuid4().hex)
        self.col.insert(
            [
                [cid],
                [vec],
                [text[:8000]],
                [str(meta.get("subject", ""))[:64]],
                [str(meta.get("user_id", ""))[:64]],
                [str(meta.get("book_name", ""))[:256]],
                [int(meta.get("page") or 0)],
                [str(meta.get("type", "text"))[:32]],
            ]
        )
        self.col.flush()
        return cid

    def search(
        self,
        query: str,
        top_k: int = 20,
        *,
        subject: str = "",
        user_id: str = "",
    ) -> List[Dict[str, Any]]:
        qv = self.embedder.embed([query])[0]
        expr_parts = []
        if subject:
            expr_parts.append(f'subject == "{subject}"')
        if user_id:
            expr_parts.append(f'user_id == "{user_id}"')
        expr = " && ".join(expr_parts) if expr_parts else None
        hits = self.col.search(
            data=[qv],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            expr=expr,
            output_fields=["text", "subject", "user_id", "book_name", "page", "chunk_type"],
        )
        out: List[Dict[str, Any]] = []
        for hit in hits[0]:
            entity = hit.entity
            out.append(
                {
                    "id": hit.id,
                    "text": entity.get("text"),
                    "score": float(hit.score),
                    "metadata": {
                        "subject": entity.get("subject"),
                        "user_id": entity.get("user_id"),
                        "book_name": entity.get("book_name"),
                        "page": entity.get("page"),
                        "type": entity.get("chunk_type"),
                    },
                }
            )
        return out
