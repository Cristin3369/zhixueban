# -*- coding: utf-8 -*-
"""
A2A（Agent2Agent）的最小实现。
-----------------------------
每个 Agent 一张 Agent Card。调用方只发 message/send，不直接调用对方的 Python 方法。
任务状态：submitted → working → completed / failed。
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List


Handler = Callable[[Dict[str, Any]], Dict[str, Any]]


class A2AHub:
    def __init__(self) -> None:
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.sent: List[str] = []

    def register(self, key: str, card: Dict[str, Any], handler: Handler) -> None:
        card = dict(card)
        card.setdefault("name", key)
        card.setdefault("version", "1.0.0")
        card.setdefault("capabilities", {"streaming": False})
        card["url"] = f"/a2a/{key}"
        self.agents[key] = {"card": card, "handler": handler}

    def agent_card(self, key: str) -> Dict[str, Any]:
        if key not in self.agents:
            raise KeyError(key)
        return self.agents[key]["card"]

    def handle(self, key: str, request: Dict[str, Any]) -> Dict[str, Any]:
        req_id = request.get("id")
        if key not in self.agents:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32001, "message": f"unknown agent: {key}"},
            }
        method = request.get("method") or ""
        if method != "message/send":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"unsupported method: {method}"},
            }
        params = request.get("params") or {}
        message = params.get("message") or {}
        metadata = params.get("metadata") or {}
        task_id = uuid.uuid4().hex[:12]
        context_id = message.get("contextId") or metadata.get("session_id") or ""
        try:
            data = self.agents[key]["handler"](metadata)
        except Exception as exc:  # noqa: BLE001
            task = {
                "id": task_id,
                "contextId": context_id,
                "status": {"state": "failed", "message": str(exc)},
                "artifacts": [],
            }
            return {"jsonrpc": "2.0", "id": req_id, "result": task}
        texts = [str(x) for x in (data.get("texts") or []) if x]
        task = {
            "id": task_id,
            "contextId": context_id,
            "status": {"state": "completed"},
            "artifacts": [
                {
                    "artifactId": uuid.uuid4().hex[:8],
                    "parts": [
                        {"kind": "text", "text": "\n\n".join(texts)},
                        {"kind": "data", "data": data},
                    ],
                }
            ],
        }
        return {"jsonrpc": "2.0", "id": req_id, "result": task}

    def send(self, key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """编排器唯一的跨 Agent 入口。"""
        self.sent.append(key)
        text = str(metadata.get("user_input") or metadata.get("query") or "")
        request = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex[:8],
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": uuid.uuid4().hex[:12],
                    "contextId": metadata.get("session_id") or "",
                    "parts": [{"kind": "text", "text": text}],
                },
                "metadata": metadata,
            },
        }
        resp = self.handle(key, request)
        if "error" in resp:
            raise RuntimeError(resp["error"]["message"])
        task = resp["result"]
        if task["status"]["state"] != "completed":
            raise RuntimeError(task["status"].get("message") or "a2a task failed")
        parts = (task.get("artifacts") or [{}])[0].get("parts") or []
        data = {}
        for part in parts:
            if part.get("kind") == "data":
                data = part.get("data") or {}
        data["task_id"] = task["id"]
        data["a2a_state"] = task["status"]["state"]
        return data
