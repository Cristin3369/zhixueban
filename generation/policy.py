# -*- coding: utf-8 -*-
"""
微调边界（企业级默认：能不调就不调）
--------------------------------
考点必须来自检索/题库，不能进模型权重。
只调「说话方式」和「安全边界」，不调「知识」。
"""

from __future__ import annotations

from typing import Dict, List

# 面试/评审可直接对照此表
FINE_TUNE_MATRIX: List[Dict[str, str]] = [
    {
        "component": "Companion_Agent",
        "model": "7B（vLLM）",
        "finetune": "要：QLoRA SFT + DPO",
        "why": "短句、去鸡汤、崩溃只休息、不催刷题。风格与安全策略适合进权重。",
        "knowledge": "禁止把考点/原题写进训练语料。",
    },
    {
        "component": "Tutor 生成模型（讲解改写）",
        "model": "在线通用 LLM（DeepSeek 等）",
        "finetune": "不要",
        "why": "只负责把检索证据改写成「知识点解析」，知识以外挂为准。微调会背训练集、撞版权、过期考点难热更新。",
        "knowledge": "题干/答案/解析只作 prompt 证据，且有引用与照抄拦截。",
    },
    {
        "component": "Router_Agent",
        "model": "规则 + 可选结构化 LLM / 后期可换轻量分类器",
        "finetune": "不要调大模型。高并发时可训一个小分类头，不是 LLM SFT。",
        "why": "路由要低延迟、可审计、危机宁可误判成陪伴。LLM 路由失败必须回规则。",
        "knowledge": "无。",
    },
    {
        "component": "Planner_Agent",
        "model": "无生成模型（公式+反思）",
        "finetune": "不要",
        "why": "间隔、精力裁剪、防沉迷是规则，不是文风问题。",
        "knowledge": "队列与日历即长期状态。",
    },
    {
        "component": "Query Rewrite / Rerank",
        "model": "规则改写 + BGE-Reranker",
        "finetune": "不要调生成模型；Rerank 用现成 Cross-Encoder。",
        "why": "改写失败回规则变体；重排是排序不是生成。",
        "knowledge": "无。",
    },
]


def finetune_summary() -> str:
    lines = ["组件 | 是否微调 | 原因"]
    for row in FINE_TUNE_MATRIX:
        lines.append(f"- {row['component']}：{row['finetune']}。{row['why']}")
    return "\n".join(lines)
