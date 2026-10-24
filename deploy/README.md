# 生产上线（Conda）

这套栈在本机或云主机用 Conda 环境跑，不经过 Docker。中间件（Redis / Neo4j / Milvus / Elasticsearch）各自安装后监听 localhost。

## 组件

| 组件 | 作用 |
|------|------|
| FastAPI | 业务编排、SSE |
| Redis | L1–L4 缓存、复习 ZSET、同伴计数 |
| Neo4j | GraphRAG 前置/后置/易混淆 |
| Milvus + BGE-M3 | 语义向量 |
| Elasticsearch | BM25 关键词 |
| BGE-Reranker 服务 | Top-20 → Top-3 |
| DeepSeek | 辅导生成 / 路由 |
| vLLM 7B DPO | 陪伴生成 |
| DashScope | ASR/TTS |

## 用 Docker 起中间件（可选）

不想在本机逐个安装 Redis / Neo4j / Milvus / Elasticsearch 时，用根目录的 compose 一键拉起。
Python 代码仍然用 Conda 跑，两者不冲突。

```bash
docker compose up -d          # 首次会拉镜像，需几分钟
docker compose ps             # 等 redis / neo4j / milvus / elasticsearch 都 healthy
```

端口与 `config/settings.py` 的默认值一致，`deploy/.env` 里的地址不用改。

Neo4j 默认密码是 `password`（与 `settings.py` 一致，开箱即连）。上生产改强密码：

```powershell
# PowerShell；改完记得同步 deploy/.env 的 NEO4J_PASSWORD
$env:NEO4J_PASSWORD = "yourStrongPass"
docker compose up -d neo4j
```

四个中间件全开约占 4–6 GB 内存。只调试某一块可单独起，例如 `docker compose up -d redis neo4j`。

停止用 `docker compose down`（数据保留在 named volumes）；连数据一起删用 `docker compose down -v`。
Neo4j 浏览器在 [http://localhost:7474](http://localhost:7474)。

MinIO 走 `registry.cncfstack.com/quay.io/minio/minio`（quay.io 的国内镜像代理），因为 Milvus 官方示例里的
`RELEASE.2023-03-20T20-16-18Z` 已从 Docker Hub 下架。代理站若失效，换回 `minio/minio:latest` 即可——注意
2025 版 minio 只认 `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`，旧的 `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` 已移除。

官方 Elasticsearch 镜像不含 IK 中文分词，`production/es_bm25.py` 会自动回落到 standard 分词（日志里
「ES 无 IK 分词」属正常）。要中文分词质量，需自行制作带 `analysis-ik` 插件的镜像。

## 创建环境

在项目根目录，首次先建环境（环境名 `abhs`），之后只需激活：

```bash
conda env create -f environment.yml
conda activate abhs
pip install -r requirements-prod.txt
```

GPU 嵌入 / 重排另装：

```bash
pip install -r requirements-gpu.txt
```

## 环境变量

```bash
cp deploy/.env.example deploy/.env
```

填写 `DEEPSEEK_API_KEY`、`NEO4J_PASSWORD` 等。`deploy/.env` 已被 gitignore。线上必须：

```
APP_ENV=production
DEEPSEEK_API_KEY=...
```

中间件地址默认是本机（`localhost` / `127.0.0.1`）。未配齐时 API **拒绝启动**（不再静默落到内存 Stub）。

加载方式：把 `deploy/.env` 里的键写进当前 shell，或放到进程环境里再启动。

PowerShell 示例：

```powershell
conda activate abhs
Get-Content deploy/.env | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -notmatch '=') { return }
  $k, $v = $_.Split('=', 2)
  Set-Item -Path "Env:$($k.Trim())" -Value $v.Trim()
}
```

## 启动 API

先确认 Redis、Neo4j、Milvus、Elasticsearch 已在本机监听。需要嵌入 / 重排时，另开终端：

```bash
python -m services.embedder
python -m services.reranker
```

然后：

```bash
python -m api.sse_server
```

浏览器打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。

## 灌库

把教材/真题做成 jsonl（`text, subject, book_name, page`）：

```bash
python -m production.ingest --jsonl data/kb/你的切片.jsonl
```

Neo4j 考点用 Cypher 导入 `Concept` / `PREREQUISITE_FOR` / `CONFUSABLE_WITH` / `Question-[:TESTS]->Concept`。
