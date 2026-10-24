# -*- coding: utf-8 -*-
"""
打陪伴 vLLM：TTFT / TPOT / 端到端。没服务就退出码 2，不编数字。
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx


def one_call(url: str, model: str, api_key: str, prompt: str) -> dict:
    chat = url.rstrip("/")
    if not chat.endswith("/chat/completions"):
        chat = f"{chat}/chat/completions" if chat.endswith("/v1") else f"{chat}/v1/chat/completions"
    t0 = time.perf_counter()
    first_token_at = None
    text = []
    with httpx.stream(
        "POST",
        chat,
        headers={"Authorization": f"Bearer {api_key or 'EMPTY'}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": 0.7,
        },
        timeout=120.0,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = ((obj.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
            if delta and first_token_at is None:
                first_token_at = time.perf_counter()
            if delta:
                text.append(delta)
    t1 = time.perf_counter()
    n_out = max(len("".join(text)), 1)
    ttft_ms = ((first_token_at or t1) - t0) * 1000
    tot_ms = (t1 - t0) * 1000
    tpot_ms = (tot_ms - ttft_ms) / max(n_out, 1)
    return {"ttft_ms": ttft_ms, "e2e_ms": tot_ms, "tpot_ms": tpot_ms, "out_chars": n_out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="")
    ap.add_argument("--model", default="zhixueban-companion-7b-dpo")
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="serving/results/latency.json")
    args = ap.parse_args()
    if not args.base_url:
        print("未提供 --base_url，跳过。不要把未测的 P99 写进简历。")
        raise SystemExit(2)

    prompt = "我连续错了八道线代，有点想放弃，你怎么说？"
    rows = [one_call(args.base_url, args.model, args.api_key, prompt) for _ in range(args.n)]
    e2e = sorted(r["e2e_ms"] for r in rows)
    p99_idx = min(len(e2e) - 1, max(0, int(round(0.99 * (len(e2e) - 1)))))
    metrics = {
        "n": len(rows),
        "ttft_ms_mean": round(statistics.mean(r["ttft_ms"] for r in rows), 1),
        "tpot_ms_mean": round(statistics.mean(r["tpot_ms"] for r in rows), 2),
        "e2e_ms_mean": round(statistics.mean(e2e), 1),
        "e2e_ms_p99_sample": round(e2e[p99_idx], 1),
        "note": "样本 P99 在 n 很小时不稳定；对外请注明卡型、max_model_len、并发。",
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metrics": metrics, "runs": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
