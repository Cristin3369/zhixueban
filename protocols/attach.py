# -*- coding: utf-8 -*-
"""把日历、题库、图谱挂成 MCP 工具，把三个 Agent 挂成 A2A 服务。"""

from __future__ import annotations

from typing import Any, Dict

from protocols.a2a import A2AHub
from protocols.mcp import McpClient, McpServer


def _obj(schema_props: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "object", "properties": schema_props}


def build_mcp(system: Any) -> McpServer:
    server = McpServer()

    def kb_answer(args: Dict[str, Any]) -> Any:
        return system.rag.generate_answer_with_fallback(
            args.get("query") or "",
            memory_context=args.get("memory_context") or "",
        )

    def graph_prompt(args: Dict[str, Any]) -> Any:
        query = args.get("query") or ""
        return system.graph_rag.build_rag_prompt(query, user_question=query)

    def graph_plan(args: Dict[str, Any]) -> Any:
        return system.graph_rag.traverse_for_plan(args.get("concept") or "")

    def archive(args: Dict[str, Any]) -> Any:
        ev = system.calendar.archive_learned(
            args.get("user_id") or "",
            subject=args.get("subject") or "",
            content_summary=args.get("content_summary") or "",
            knowledge_points=args.get("knowledge_points") or [],
            session_id=args.get("session_id") or "",
            minutes=int(args.get("minutes") or 15),
        )
        return ev.to_dict()

    def countdown(args: Dict[str, Any]) -> Any:
        cd = system.calendar.countdown(
            args.get("exam_date") or "",
            exam_label=args.get("exam_label") or "考试",
        )
        return cd.to_dict() if cd else None

    def phase_hint(args: Dict[str, Any]) -> Any:
        return system.calendar.phase_plan_hint(int(args.get("days_left") or 0))

    def sync_dashboard(args: Dict[str, Any]) -> Any:
        created = system.calendar.sync_from_dashboard(
            args.get("user_id") or "",
            args.get("dashboard") or {},
        )
        return {"count": len(created)}

    def satisfaction(args: Dict[str, Any]) -> Any:
        digest = system.calendar.satisfaction_digest(
            args.get("user_id") or "",
            period=args.get("period") or "today",
        )
        return digest.to_dict()

    server.add_tool(
        "kb.answer",
        "题库混合检索并生成带来源的讲解包",
        _obj({"query": {"type": "string"}, "memory_context": {"type": "string"}}),
        kb_answer,
    )
    server.add_tool(
        "graph.prompt",
        "按知识点取图谱卡片，拼进辅导提示",
        _obj({"query": {"type": "string"}}),
        graph_prompt,
    )
    server.add_tool(
        "graph.plan",
        "计划前查看前置知识点",
        _obj({"concept": {"type": "string"}}),
        graph_plan,
    )
    server.add_tool(
        "calendar.archive_learned",
        "把刚学过的内容归档进日程",
        _obj(
            {
                "user_id": {"type": "string"},
                "subject": {"type": "string"},
                "content_summary": {"type": "string"},
                "knowledge_points": {"type": "array"},
                "session_id": {"type": "string"},
                "minutes": {"type": "integer"},
            }
        ),
        archive,
    )
    server.add_tool(
        "calendar.countdown",
        "考试倒计时与备考阶段",
        _obj({"exam_date": {"type": "string"}, "exam_label": {"type": "string"}}),
        countdown,
    )
    server.add_tool(
        "calendar.phase_hint",
        "按剩余天数给阶段提示",
        _obj({"days_left": {"type": "integer"}}),
        phase_hint,
    )
    server.add_tool(
        "calendar.sync_dashboard",
        "把今日看板写入日程",
        _obj({"user_id": {"type": "string"}, "dashboard": {"type": "object"}}),
        sync_dashboard,
    )
    server.add_tool(
        "calendar.satisfaction",
        "今日完成与满足感摘要",
        _obj({"user_id": {"type": "string"}, "period": {"type": "string"}}),
        satisfaction,
    )
    return server


def _card(name: str, description: str, skill_id: str, skill_name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "protocolVersion": "0.2.0",
        "skills": [
            {
                "id": skill_id,
                "name": skill_name,
                "description": description,
                "tags": ["zhixueban"],
            }
        ],
    }


def attach_protocols(system: Any) -> None:
    server = build_mcp(system)
    client = McpClient(server)
    hub = A2AHub()
    hub.register(
        "Tutor_Agent",
        _card("智学伴辅导", "按题库证据讲解，并写回学习日程", "answer", "辅导答疑"),
        system.handle_tutor_task,
    )
    hub.register(
        "Companion_Agent",
        _card("智学伴陪伴", "情绪安抚；危机只休息，不排计划", "comfort", "情绪陪伴"),
        system.handle_companion_task,
    )
    hub.register(
        "Planner_Agent",
        _card("智学伴计划", "按精力和到期压力排今日复习", "plan", "制定计划"),
        system.handle_planner_task,
    )
    system.mcp_server = server
    system.mcp = client
    system.a2a = hub
    system.tutor.mcp = client
    system.planner.mcp = client
