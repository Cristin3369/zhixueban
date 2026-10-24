# -*- coding: utf-8 -*-
"""HTTP Cross-Encoder 重排：BGE-Reranker-v2-m3。"""

from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="zhixueban-reranker")
_model = None


def get_model():
    global _model
    if _model is None:
        from FlagEmbedding import FlagReranker

        name = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        _model = FlagReranker(name, use_fp16=True)
    return _model


class RerankReq(BaseModel):
    query: str
    documents: List[str]


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/rerank")
def rerank(req: RerankReq):
    model = get_model()
    pairs = [[req.query, d] for d in req.documents]
    scores = model.compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]
    return {"scores": [float(s) for s in scores]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.reranker:app", host="0.0.0.0", port=8082)
