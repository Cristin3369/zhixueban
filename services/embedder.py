# -*- coding: utf-8 -*-
"""HTTP 嵌入微服务：BGE-M3，供入库与检索共用。"""

from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="zhixueban-embedder")
_model = None


def get_model():
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel

        name = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
        _model = BGEM3FlagModel(name, use_fp16=True)
    return _model


class EmbedReq(BaseModel):
    texts: List[str]


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/embed")
def embed(req: EmbedReq):
    model = get_model()
    out = model.encode(req.texts, return_dense=True)
    dense = out["dense_vecs"]
    vectors = [v.tolist() if hasattr(v, "tolist") else list(v) for v in dense]
    return {"vectors": vectors, "dim": len(vectors[0]) if vectors else 0}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.embedder:app", host="0.0.0.0", port=8081)
