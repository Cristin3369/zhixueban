# -*- coding: utf-8 -*-
"""
检索四路消融。结果写入 evaluation/results/retrieval_ablation.json。
标注 Query 太少时只作链路演示，禁止把指标写进简历。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.retrieval_metrics import aggregate, mrr, ndcg_at_k, recall_at_k
from retrieval.hybrid_rag import AdvancedRAGEngine

STRATEGIES = ("bm25", "vector", "hybrid_rrf", "hybrid_rerank")


def load_qrels(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def seed_demo_corpus(engine: AdvancedRAGEngine) -> None:
    samples = [
        ("反向传播利用链式法则把输出误差逐层回传到参数。", {"book_name": "花书", "page": 214}),
        ("泰勒公式把函数在点处展开成多项式加余项。", {"book_name": "高数", "page": 88}),
        ("矛盾特殊性要求具体问题具体分析。", {"book_name": "马原", "page": 12}),
    ]
    for text, meta in samples:
        if hasattr(engine.vector_store, "add"):
            engine.vector_store.add(text, meta)
        if hasattr(engine.bm25_index, "add"):
            engine.bm25_index.add(text, meta)


def run(engine: AdvancedRAGEngine | None = None, qrels_path: str = "") -> dict:
    qrels = load_qrels(Path(qrels_path or _ROOT / "evaluation" / "data" / "retrieval_qrels.jsonl"))
    engine = engine or AdvancedRAGEngine()
    seed_demo_corpus(engine)
    per_strategy: dict[str, list] = {s: [] for s in STRATEGIES}
    k = 3
    for row in qrels:
        q = row["query"]
        subs = row.get("relevant_substrings") or []
        ids = set(row.get("relevant_ids") or [])
        for s in STRATEGIES:
            ranked = engine.search_by_strategy(q, s)
            per_strategy[s].append(
                {
                    "recall@3": recall_at_k(ranked, subs, ids, k),
                    "ndcg@3": ndcg_at_k(ranked, subs, ids, k),
                    "mrr": mrr(ranked, subs, ids),
                }
            )
    summary = {s: aggregate(rows) for s, rows in per_strategy.items()}
    payload = {
        "metrics": summary,
        "n_queries": len(qrels),
        "note": "n_queries<50 时方差大，不要把本文件数字写进简历。换成自建标注集后再对外。",
    }
    out = _ROOT / "evaluation" / "results" / "retrieval_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
