# -*- coding: utf-8 -*-
"""
FastAPI + SSE 流式接口
---------------------
复杂链路（四级检索 + GraphRAG + Rerank + LLM）串行 TTFT 可能很长，
用 Server-Sent Events 尽早把首包推给前端，掩盖后端延迟。

同时提供：
- 用户进页科目书本登记（L3）
- 复习反馈写入多因子调度
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

# 将项目根加入 path（python -m api.sse_server）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import settings

logger = logging.getLogger(__name__)

# 延迟导入，避免文档工具扫到时强依赖 fastapi
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False
    FastAPI = object  # type: ignore


if HAS_FASTAPI:

    class ChatRequest(BaseModel):
        user_id: str = "anonymous"
        message: str
        # 一窗一科：窗口名即 subject
        subject: str = ""
        session_id: str = ""
        exam_track: str = ""  # 兼容旧字段；空则用 subject
        user_energy: float = Field(0.7, ge=0, le=1)
        emotion_intensity: float = Field(0.0, ge=0, le=1)

    class OnboardRequest(BaseModel):
        user_id: str
        exam_track: str = "考研"
        school: str = ""
        major: str = ""
        books_note: str = ""
        subjects: List[str] = Field(default_factory=list)
        books: Any = Field(default_factory=list)
        daily_available_minutes: int = 240
        subject_weights: Dict[str, float] = Field(default_factory=dict)
        target_exam_date: str = ""  # 考研可空，后端按全国统考自动算

    class SwitchSubjectRequest(BaseModel):
        user_id: str
        subject: str

    class NewWindowRequest(BaseModel):
        user_id: str
        title: str

    class DashboardRequest(BaseModel):
        user_id: str
        available_minutes: Optional[int] = None

    class AgendaRequest(BaseModel):
        user_id: str
        day: str = ""  # YYYY-MM-DD，空=今天

    class CompleteEventRequest(BaseModel):
        user_id: str
        event_id: str
        note: str = ""

    class ArchiveLearnedRequest(BaseModel):
        user_id: str
        subject: str
        summary: str
        knowledge_points: List[str] = Field(default_factory=list)
        session_id: str = ""
        minutes: int = 15

    class FeedbackRequest(BaseModel):
        user_id: str
        question_id: str = ""
        subject: str = ""
        session_id: str = ""
        is_correct: bool = True
        response_time: float = 30
        subjective_difficulty: int = 3
        objective_difficulty: float = 3
        exam_probability: float = 0.5
        user_energy: float = 0.7
        emotion_intensity: float = 0.0
        student_record: Dict[str, Any] = Field(default_factory=dict)

    def create_app() -> FastAPI:
        from startgraph import StudyAgentSystem, onboard_user

        app = FastAPI(title="智学伴", version="1.1.0")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        system = StudyAgentSystem(force_memory_graph=True)
        web_dir = _ROOT / "web"
        if web_dir.is_dir():
            app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/")
        def home():
            index = web_dir / "index.html"
            if not index.is_file():
                raise HTTPException(404, "未找到 web/index.html")
            return FileResponse(index)

        @app.get("/health")
        def health() -> dict:
            return {
                "ok": True,
                "session_model": "one_subject_one_session",
                "protocols": ["mcp", "a2a"],
            }

        @app.post("/mcp")
        async def mcp_rpc(request: dict) -> dict:
            return system.mcp_server.handle(request)

        @app.get("/a2a/{agent}/.well-known/agent-card.json")
        def agent_card(agent: str) -> dict:
            try:
                return system.a2a.agent_card(agent)
            except KeyError:
                raise HTTPException(status_code=404, detail=f"unknown agent: {agent}")

        @app.post("/a2a/{agent}")
        async def a2a_rpc(agent: str, request: dict) -> dict:
            return system.a2a.handle(agent, request)

        @app.post("/onboard")
        def onboard(req: OnboardRequest) -> dict:
            """登录：名字、考研/考公、学校、专业、教材。考研日自动计算。"""
            profile = onboard_user(
                system,
                req.user_id,
                req.exam_track,
                req.subjects if req.subjects is not None else None,
                req.books,
                school=req.school,
                major=req.major,
                books_note=req.books_note,
                daily_available_minutes=req.daily_available_minutes,
                subject_weights=req.subject_weights or None,
                target_exam_date=req.target_exam_date,
            )
            countdown = None
            if profile.target_exam_date:
                cd = system.calendar.countdown(
                    profile.target_exam_date, exam_label=req.exam_track
                )
                countdown = cd.to_dict() if cd else None
            return {
                "ok": True,
                "profile": profile.to_dict(),
                "windows": system.sessions.list_windows(req.user_id),
                "countdown": countdown,
            }

        @app.post("/profile")
        def update_profile(req: OnboardRequest) -> dict:
            """个人页改考研、学校、专业、教材；保留已有对话窗口。"""
            return onboard(req)

        @app.get("/sessions")
        def list_sessions(user_id: str) -> dict:
            return system.list_windows(user_id)

        @app.post("/session/new")
        def new_window(req: NewWindowRequest) -> dict:
            title = (req.title or "").strip()
            if not title:
                raise HTTPException(400, "给窗口起个名字再开")
            return system.open_named_window(req.user_id, title)

        @app.post("/dashboard")
        def dashboard(req: DashboardRequest) -> dict:
            """今日学习看板：跨科分配时间 + 倒计时 + 同步日程。"""
            return system.open_dashboard(
                req.user_id, available_minutes=req.available_minutes
            )

        @app.post("/agenda")
        def agenda(req: AgendaRequest) -> dict:
            """日历页：倒计时、当日事件、满足感摘要。"""
            return system.open_agenda(req.user_id, day=req.day)

        @app.post("/calendar/complete")
        def complete_event(req: CompleteEventRequest) -> dict:
            """勾选完成日程项 → 打卡与满足感。"""
            return system.complete_calendar_event(req.user_id, req.event_id, req.note)

        @app.post("/calendar/archive")
        def archive_learned(req: ArchiveLearnedRequest) -> dict:
            """把学过的内容整理进日程归档。"""
            return system.archive_learning(
                req.user_id,
                subject=req.subject,
                summary=req.summary,
                knowledge_points=req.knowledge_points,
                session_id=req.session_id,
                minutes=req.minutes,
            )

        @app.post("/session/switch")
        def switch_subject(req: SwitchSubjectRequest) -> dict:
            return system.switch_subject(req.user_id, req.subject)

        @app.post("/feedback")
        def feedback(req: FeedbackRequest) -> dict:
            result = system.planner.apply_review_feedback(
                req.user_id,
                req.student_record,
                {
                    "is_correct": req.is_correct,
                    "response_time": req.response_time,
                    "subjective_difficulty": req.subjective_difficulty,
                    "objective_difficulty": req.objective_difficulty,
                    "exam_probability": req.exam_probability,
                    "user_energy": req.user_energy,
                    "emotion_intensity": req.emotion_intensity,
                },
                question_id=req.question_id,
                subject=req.subject,
                session_id=req.session_id,
            )
            return result

        def _invoke_chat(req: ChatRequest) -> dict:
            subject = req.subject or req.exam_track
            if not subject:
                raise HTTPException(400, "先给对话窗口起名字")
            return system.invoke(
                req.message,
                user_id=req.user_id,
                subject=subject,
                session_id=req.session_id,
                exam_track=subject,
                user_energy=req.user_energy,
                emotion_intensity=req.emotion_intensity,
            )

        async def _sse_generator(req: ChatRequest) -> AsyncGenerator[bytes, None]:
            yield _sse("status", {"stage": "routing", "subject": req.subject})
            await asyncio.sleep(0)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: _invoke_chat(req))
            yield _sse(
                "status",
                {
                    "stage": "generating",
                    "trace": result.get("agent_trace"),
                    "session_id": result.get("session_id"),
                },
            )
            text = result.get("final_response") or ""
            chunks = [p for p in text.split("\n\n") if p.strip()] or [text]
            for i, chunk in enumerate(chunks):
                yield _sse("token", {"index": i, "text": chunk})
                await asyncio.sleep(0.01)
            yield _sse(
                "done",
                {
                    "next_review_ts": result.get("next_review_ts"),
                    "plan": result.get("plan"),
                    "dashboard": result.get("dashboard"),
                    "session_id": result.get("session_id"),
                    "subject": result.get("subject"),
                    "namespace": result.get("namespace"),
                    "reflection_notes": result.get("reflection_notes"),
                },
            )

        @app.post("/chat/stream")
        async def chat_stream(req: ChatRequest) -> StreamingResponse:
            if not req.message.strip():
                raise HTTPException(400, "message 不能为空")
            return StreamingResponse(_sse_generator(req), media_type="text/event-stream")

        @app.post("/chat")
        def chat(req: ChatRequest) -> dict:
            return _invoke_chat(req)

        return app

    def _sse(event: str, data: dict) -> bytes:
        payload = json.dumps(data, ensure_ascii=False)
        return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")

    app = create_app()

else:
    app = None  # type: ignore


if __name__ == "__main__":
    if not HAS_FASTAPI:
        raise SystemExit("缺少 fastapi / uvicorn，请先 pip install -r requirements.txt")
    import uvicorn

    uvicorn.run(
        "api.sse_server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
