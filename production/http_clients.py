# -*- coding: utf-8 -*-
"""生产级 HTTP 客户端（嵌入 / 重排 / LLM）。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from config.settings import settings
from retrieval.query_rewrite import HeuristicQueryRewriter

logger = logging.getLogger(__name__)


class EmbedClient:
    def __init__(self, base_url: Optional[str] = None, timeout: float = 60.0):
        self.base_url = (base_url or settings.embed_url).rstrip("/")
        self.timeout = timeout

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/embed", json={"texts": texts})
            r.raise_for_status()
            return r.json()["vectors"]


class BGERerankerClient:
    """调用 GPU 侧重排服务，接口对齐 AdvancedRAGEngine.rerank_model。"""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 60.0):
        self.base_url = (base_url or settings.rerank_url).rstrip("/")
        self.timeout = timeout

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        if not docs:
            return []
        texts = [d.get("text", "") for d in docs]
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(
                f"{self.base_url}/rerank",
                json={"query": query, "documents": texts},
            )
            r.raise_for_status()
            scores = r.json()["scores"]
        out = []
        for d, s in zip(docs, scores):
            out.append({**d, "score": float(s)})
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:top_k]


class OpenAICompatLLM:
    """DeepSeek / vLLM / 通义：OpenAI Chat Completions。"""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ):
        self.model = model or settings.llm_model
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.api_key = api_key or settings.openai_api_key
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.timeout = timeout if timeout is not None else settings.llm_timeout_sec
        self.rewriter = HeuristicQueryRewriter()

    def _chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        if not self.api_key and "localhost" not in self.base_url and "companion" not in self.base_url:
            raise RuntimeError("OPENAI_API_KEY / DEEPSEEK_API_KEY 未配置，生产环境禁止空 Key 调用。")
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions" if url.endswith("/v1") else f"{url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key or 'EMPTY'}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"]

    def generate(self, prompt: str) -> str:
        return self._chat([{"role": "user", "content": prompt}])

    def rewrite_query(self, query: str) -> str:
        try:
            text = self._chat(
                [
                    {
                        "role": "system",
                        "content": "你改写考研/考公检索问句：抽出考点实体、补全歧义、给一个更利于检索的短句。只输出改写结果。",
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
            )
            return (text or "").strip() or self.rewriter.rewrite_query(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM rewrite 失败，回落规则: %s", exc)
            return self.rewriter.rewrite_query(query)

    def generate_caption(self, image_url: str) -> str:
        """无多模态时的占位；生产可换 Qwen-VL。"""
        return self.generate(f"请用一句话描述这张考研资料图（仅据 URL 猜测用途，不确定就说图表）：{image_url}")
