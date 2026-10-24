# 知识切片

这里是入库切片（jsonl）的存放目录，每行一条：

```json
{"text": "…", "subject": "数学", "book_name": "…", "page": 1}
```

字段说明：`text` 正文、`subject` 科目、`book_name` 书名、`page` 页码；可选还有
`chunk_type`、`knowledge_point`、`image_paths`、`user_id` 等，完整 schema 见
`production/milvus_store.py` 与 `production/es_bm25.py`。

灌库示例：

```bash
python -m production.ingest --jsonl data/kb/你的切片.jsonl
```

> 有版权的教材 / 真题原文不要放进仓库（`data/kb/ch1/` 已被 .gitignore 排除）。
> 本目录只有这份说明会随仓库提交，切片数据请自行准备。
