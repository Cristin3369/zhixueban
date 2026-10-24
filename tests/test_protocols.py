# -*- coding: utf-8 -*-
"""调度必须走 A2A，题库和日历必须走 MCP。"""

from __future__ import annotations

import unittest

from startgraph import StudyAgentSystem


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = StudyAgentSystem(force_memory_graph=True)

    def test_mcp_lists_calendar_and_kb(self) -> None:
        resp = self.system.mcp_server.handle(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )
        names = {t["name"] for t in resp["result"]["tools"]}
        self.assertIn("kb.answer", names)
        self.assertIn("calendar.archive_learned", names)
        self.assertIn("graph.plan", names)

    def test_dispatch_uses_a2a_and_tutor_uses_mcp(self) -> None:
        result = self.system.invoke("反向传播的链式法则是什么？", user_id="u1", subject="数学")
        self.assertIn("Tutor_Agent", self.system.a2a.sent)
        self.assertIn("kb.answer", self.system.mcp_server.calls)
        self.assertTrue(result.get("final_response"))

    def test_companion_a2a_card_and_message(self) -> None:
        card = self.system.a2a.agent_card("Companion_Agent")
        self.assertIn("skills", card)
        task = self.system.a2a.handle(
            "Companion_Agent",
            {
                "jsonrpc": "2.0",
                "id": "c1",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "messageId": "m1",
                        "contextId": "s1",
                        "parts": [{"kind": "text", "text": "我快崩溃了"}],
                    },
                    "metadata": {
                        "user_input": "我快崩溃了",
                        "emotion_context": "我快崩溃了",
                        "emotion": 0.95,
                        "user_id": "u1",
                        "energy": 0.4,
                        "subject": "数学",
                    },
                },
            },
        )
        self.assertEqual(task["result"]["status"]["state"], "completed")
        text = task["result"]["artifacts"][0]["parts"][0]["text"]
        self.assertIn("停下来", text)


if __name__ == "__main__":
    unittest.main()
