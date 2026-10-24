# 智学伴 · 考研 / 考公学习 Agent

按艾宾浩斯间隔安排复习，并提供制定计划、辅导答疑、情绪陪伴。考点来自检索，不写进模型权重。一科一会话，学习内容不跨科。

本地不连 Redis、Neo4j、Milvus 也能跑通演示。生产环境缺任一依赖会直接启动失败。

## 架构

```text
FastAPI / SSE
    │
    ▼
LangGraph：Router → MetaMemory → Dispatch
    Dispatch 用 A2A message/send 调用下面三个 Agent，不直接调它们的函数
    ├─ 辅导   通过 MCP 调 kb.answer / graph.prompt，再改写；学完用 calendar.archive_learned 归档
    ├─ 计划   通过 MCP 调日历倒计时、日程同步、图谱前置
    └─ 陪伴   规则模板；危机只走可审计话术。配置 COMPANION_BASE_URL 后才调用外部模型
```

会话号是 `user_{uid}_{subject}_{日期}`。检索和记忆都带 `subject` 与 `namespace`。情绪和精力可以跨科共享。

Milvus 只存题库切片。会话记忆在情景记忆里，不进向量库。

## 目录

```text
agents/           路由、辅导、计划、陪伴、反思
api/              FastAPI + SSE
config/           环境变量配置
data/             注册登录、会话、四级缓存、复习队列
data/kb/          知识切片（自备数据，格式与灌库见 data/kb/README.md）
retrieval/        混合检索、GraphRAG、查询改写
memory/           情景记忆、时序图谱、冷热上下文
scheduling/       多因子艾宾浩斯、日历与看板
generation/       生成包裹、溯源校验、微调边界
protocols/        MCP 工具服务与 A2A Agent 互调
production/       Milvus、Elasticsearch、Redis、语音
deploy/           Conda 生产说明与环境变量样例
evaluation/       检索消融
examples/         可直接运行的演示
```

## 本地运行

首次创建环境（Python 3.11，环境名 `abhs`）：

```bash
conda env create -f environment.yml
conda activate abhs
python startgraph.py
```

已有 Python 3.11 环境则只需 `pip install -r requirements.txt`。

启动 API（同时打开页面）：

```bash
conda activate abhs
python -m api.sse_server
```

浏览器打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。登记科目后进入今日看板，点某一科打开专属会话；日程页可打卡。页面在 `web/`，由 FastAPI 直接提供，不需要另起前端。

常用接口：

| 方法 | 路径 | 作用 |
|------|------|------|
| POST | `/onboard` | 登记科目、书本、考试日 |
| POST | `/dashboard` | 今日看板与倒计时 |
| POST | `/agenda` | 日历与完成打卡 |
| POST | `/chat/stream` | SSE 对话，请求带 `session_id` |
| POST | `/feedback` | 复习反馈，更新间隔 |

演示：

```bash
python examples/session_demo.py
python examples/calendar_demo.py
python examples/memory_fallback_demo.py
python examples/generation_envelope_demo.py
```

协议入口（API 启动后）：

| 方法 | 路径 | 作用 |
|------|------|------|
| POST | `/mcp` | MCP JSON-RPC：`tools/list`、`tools/call` |
| GET | `/a2a/{agent}/.well-known/agent-card.json` | 辅导 / 陪伴 / 计划的 Agent Card |
| POST | `/a2a/{agent}` | A2A `message/send` |

## 生产部署

用 Conda 跑 Python 侧，不把应用打包进镜像。中间件（Redis、Neo4j、Milvus、Elasticsearch）可用根目录 `docker-compose.yml` 一键起，也可各自本机安装；BGE-M3、BGE-Reranker、DeepSeek、vLLM、DashScope 走外部服务。说明见 [deploy/README.md](deploy/README.md)。

```bash
conda env create -f environment.yml   # 首次
conda activate abhs
pip install -r requirements-prod.txt
cp deploy/.env.example deploy/.env
# 填写密钥；中间件地址默认 localhost。deploy/.env 已被 .gitignore 排除
python -m api.sse_server
python -m production.ingest --jsonl data/kb/你的切片.jsonl
```

## 仓库里没有的东西

克隆后可以直接跑本地演示，不需要再准备题库或模型。

- 没有教材、真题全文。`python startgraph.py` 启动时只写入一句示例。生产入库要自己准备 `text / subject / book_name / page` 的 jsonl，有版权的书不要放进仓库。
- 没有微调权重。陪伴在未配置 `COMPANION_BASE_URL` 时走规则模板，危机话术不经过生成模型。
- `evaluation/results/retrieval_ablation.json` 只有 3 条查询，不能当作检索效果。
