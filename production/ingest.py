# -*- coding: utf-8 -*-
"""把结构化题答/切片双写 Milvus + Elasticsearch。上线入库入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from production.es_bm25 import ElasticsearchBM25
from production.milvus_store import MilvusVectorStore


def ingest_jsonl(path: Path) -> int:
    """
    每行 JSON：
    {"text": "...", "subject": "数学", "book_name": "同济高数", "page": 120, "type": "text", "user_id": ""}
    """
    vec = MilvusVectorStore()
    bm25 = ElasticsearchBM25()
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.pop("text")
            meta = row
            vec.add(text, meta)
            bm25.add(text, meta)
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    args = p.parse_args()
    n = ingest_jsonl(Path(args.jsonl))
    print(f"ingested {n} chunks into Milvus + ES")


if __name__ == "__main__":
    main()
