# -*- coding: utf-8 -*-
"""
智学伴 × 算法岗：只融进项目、能答辩的方向（按优先级）
------------------------------------------------
P0 必须做成可跑、可出数（没数就别写简历）。
P1 绑定陪伴 7B 推理，数字只能来自 bench 脚本。
P2 面试口述 / 与现有模块轻绑定，禁止虚构指标。
X  不硬融：和本项目权威模型无关，或会变成假算法。
"""

from __future__ import annotations

from typing import Dict, List

INTEGRATION_RANK: List[Dict[str, str]] = [
    {
        "rank": "P0-1",
        "theme": "评测体系",
        "fit": "最高",
        "why": "算法岗看你怎么证明变好。陪伴已有 pairwise+五维量表；缺检索消融和讲解可靠性。",
        "how": "evaluation/ 跑四路检索 Recall/NDCG/MRR；讲解用 Faithfulness（claim 有证据）+ 拒答率。数字只写 results/*.json。",
        "resume_ok_after": "隔离测试集 n、winrate、拒答率、NDCG 消融表",
    },
    {
        "rank": "P0-2",
        "theme": "数据与偏好闭环",
        "fit": "最高",
        "why": "DPO 上限在 chosen/rejected 质量和泄漏控制，不在 Agent 个数。",
        "how": "线上 Bad Case（幻觉/该拒未拒/鸡汤）经 ingest_badcase 进待标注池，人工确认后再进 SFT/DPO；prepare_pairs 做训练/评测隔离。",
        "resume_ok_after": "场景配比、最少对数、train/eval 去重方式（仍不要写 5 万除非真有）",
    },
    {
        "rank": "P0-3",
        "theme": "检索 ablation",
        "fit": "最高",
        "why": "混合检索已经在链路里，算法差异在对比实验，不在再叠一个 HyDE 名词。",
        "how": "同一 qrels：BM25 / 向量 / RRF / RRF+Rerank。HyDE 未做就不写。",
        "resume_ok_after": "四列指标表，n=标注 Query 数",
    },
    {
        "rank": "P0-4",
        "theme": "拒答与证据约束",
        "fit": "很高",
        "why": "辅导不微调、只改写，可靠性就是算法贡献：无证据不说。",
        "how": "置信度门控 + 照抄拦截 + claim 级和证据重合；危机陪伴不走生成。",
        "resume_ok_after": "阈值、该拒未拒/过度拒答在评测集上的计数，不要编幻觉率百分比",
    },
    {
        "rank": "P1-1",
        "theme": "推理与部署（vLLM）",
        "fit": "高，但必须先测",
        "why": "陪伴 7B 本来就要 vLLM；PagedAttention/量化是原理题，简历必须有 TTFT/TPOT/P99。",
        "how": "serving/bench_vllm.py 打真实服务。没跑过禁止写「P99 从 X 到 Y」。AWQ/GPTQ 作为可选量化实验，与 QLoRA 训练量化分开讲。",
        "resume_ok_after": "serving/results/latency.json 里的数 + 卡型、max_model_len、并发",
    },
    {
        "rank": "P1-2",
        "theme": "上下文管理算法（记忆改写法）",
        "fit": "高",
        "why": "不要写「长期记忆产品」，写 score=相关×时间衰减×重要性，再摘要压缩进窗口。",
        "how": "memory/context_policy.py 给冷记忆排序；Scene 摘要当压缩。不要把 Milvus 说成记忆库（Milvus 是题库切片）。不要写 KV Cache 丢 token 降 40%。",
        "resume_ok_after": "公式 + 窗口 token 上限；连贯性分数需单独盲评才可写",
    },
    {
        "rank": "P2-1",
        "theme": "训练稳定与实验管理",
        "fit": "中高",
        "why": "yaml 里已有 lr/beta/accum；算法岗问 loss 降指标不涨、灾难性遗忘。",
        "how": "checkpoint 目录约定 + metrics.json；有 GPU 再加 WandB。DeepSpeed ZeRO 只在真 OOM 时加，不要先写进简历。",
        "resume_ok_after": "数据版本（jsonl 条数与哈希）、超参来自 dpo_qlora.yaml",
    },
    {
        "rank": "P2-2",
        "theme": "文档结构化（图表公式切片）",
        "fit": "中，且已有雏形",
        "why": "hybrid_rag.process_complex_document 已按 image/table/formula 切片。",
        "how": "写成解析噪声会打坏 RAG，切片是为召回而非 OCR 产品。没有版面评测集就不要写准确率。",
        "resume_ok_after": "切片类型与 metadata（latex/表头）；解析评测另建才写数",
    },
    {
        "rank": "P2-3",
        "theme": "Transformer 基础（绑定 Qwen2.5-7B）",
        "fit": "面试口述，不进业务代码",
        "why": "基座是 Decoder-only + GQA + RoPE + RMSNorm，用来解释 QLoRA 打哪些投影、长上下文外推。",
        "how": "简历一句绑定陪伴基座即可。禁止堆 MoE/ALiBi（本基座不是）。",
        "resume_ok_after": "无实验数字",
    },
    {
        "rank": "X",
        "theme": "不硬融",
        "fit": "不要为算法而算法",
        "why": "开放 ReAct、投机解码、动态 KV 淘汰 40%、5 万苏格拉底、HyDE、意图 F1=92%、DeepSpeed 未用就写 ZeRO-2。",
        "how": "路由保持规则+结构化 LLM；任务图保持固定边。投机解码/FlashAttention 留在原理题。",
        "resume_ok_after": "无",
    },
]


def format_rank() -> str:
    lines = []
    for row in INTEGRATION_RANK:
        lines.append(f"{row['rank']} {row['theme']}（匹配：{row['fit']}）")
        lines.append(f"  为何：{row['why']}")
        lines.append(f"  融入：{row['how']}")
    return "\n".join(lines)
