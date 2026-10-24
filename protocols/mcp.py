# -*- coding: utf-8 -*-
"""
MCP（Model Context Protocol）的最小实现。
------------------------------------------
只实现本项目用到的 JSON-RPC 方法：initialize、tools/list、tools/call。
工具是日历、题库检索、图谱卡片。Agent 不直接 import 这些服务。
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List


ToolFn = Callable[[Dict[str, Any]], Any]


class McpServer:
    def __init__(self, name: str = "zhixueban-tools") -> None:
        self.name = name
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.calls: List[str] = []

    def add_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        fn: ToolFn,
    ) -> None:
        self.tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "fn": fn,
        }

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        req_id = request.get("id")
        method = request.get("method") or ""
        params = request.get("params") or {}
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": self.name, "version": "1.0.0"},
                    "capabilities": {"tools": {}},
                }
            elif method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": t["name"],
                            "description": t["description"],
                            "inputSchema": t["inputSchema"],
                        }
                        for t in self.tools.values()
                    ]
                }
            elif method == "tools/call":
                name = params.get("name") or ""
                if name not in self.tools:
                    raise KeyError(f"unknown tool: {name}")
                self.calls.append(name)
                raw = self.tools[name]["fn"](params.get("arguments") or {})
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(raw, ensure_ascii=False, default=str),
                        }
                    ],
                    "isError": False,
                }
            else:
                raise ValueError(f"unknown method: {method}")
        except Exception as exc:  # noqa: BLE001
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(exc)},
            }
        return {"jsonrpc": "2.0", "id": req_id, "result": result}


class McpClient:
    """进程内客户端。请求体就是 MCP 的 JSON-RPC，不绕过工具表。"""

    def __init__(self, server: McpServer) -> None:
        self.server = server
        self._seq = 0

    def _rpc(self, method: str, params: Dict[str, Any] | None = None) -> Any:
        self._seq += 1
        resp = self.server.handle(
            {"jsonrpc": "2.0", "id": self._seq, "method": method, "params": params or {}}
        )
        if "error" in resp:
            raise RuntimeError(resp["error"]["message"])
        return resp["result"]

    def list_tools(self) -> List[dict]:
        return list(self._rpc("tools/list").get("tools") or [])

    def call_tool(self, name: str, arguments: Dict[str, Any] | None = None) -> Any:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments or {}})
        text = ((result.get("content") or [{}])[0]).get("text") or "null"
        return json.loads(text)
