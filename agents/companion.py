# -*- coding: utf-8 -*-
"""
陪伴 Agent
---------
- 文本安抚 + 可选语音（ASR/TTS 接口预留）
- 同伴效应：匿名群体难度/用时分位，增强说服力
- 情绪崩溃：只休息，不推计划
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Protocol

from config.settings import settings
from generation.envelope import CircuitBreaker, guarded_call


class PeerStatsProvider(Protocol):
    def stats_for_question(self, question_id: str) -> Dict[str, Any]: ...


class InMemoryPeerStats:
    """演示用同伴数据。生产接数仓：今日觉得难的人数、用时分位等。"""

    def stats_for_question(self, question_id: str) -> Dict[str, Any]:
        seed = sum(ord(c) for c in (question_id or "x"))
        rng = random.Random(seed)
        hard_count = rng.randint(80, 800)
        pct_slower = rng.randint(55, 92)
        return {
            "hard_count_today": hard_count,
            "slower_than_percent": pct_slower,
            "sample_ok": hard_count >= settings.peer_min_sample,
        }


class TTSClient(Protocol):
    def synthesize(self, text: str) -> bytes: ...


class ASRClient(Protocol):
    def transcribe(self, audio: bytes) -> str: ...


class StubTTS:
    def synthesize(self, text: str) -> bytes:
        return b""


class StubASR:
    def transcribe(self, audio: bytes) -> str:
        return ""


COMPANION_SYSTEM = """你是考研/考公陪伴学伴，说话短、口语、真诚，禁止鸡汤堆砌，禁止催促刷题。
规则：
1. 若学生情绪崩溃（不想活、崩溃、放弃），只安抚与建议休息，绝不布置学习任务。
2. 若只是焦虑但仍想学，先共情再给一个很小的下一步。
3. 不要假装自己是心理医生；危机会话引导寻求现实支持。
4. 若提供了同伴统计，可引用，但不得编造数字。
"""


class CompanionAgent:
    def __init__(
        self,
        peer: Optional[PeerStatsProvider] = None,
        tts: Optional[TTSClient] = None,
        asr: Optional[ASRClient] = None,
        llm: Any = None,
    ):
        self.peer = peer or InMemoryPeerStats()
        self.tts = tts or StubTTS()
        self.asr = asr or StubASR()
        self.llm = llm  # QLoRA+DPO 后的陪伴模型；空则规则模板
        self._breaker = CircuitBreaker("companion_llm")

    def comfort(
        self,
        emotion_context: str,
        *,
        emotion_intensity: float = 0.5,
        question_id: str = "",
        voice_mode: bool = False,
        user_energy: float = 0.7,
        subject: str = "",
        global_emotion_note: str = "",
    ) -> Dict[str, Any]:
        peer_line = ""
        if question_id:
            st = self.peer.stats_for_question(question_id)
            if st.get("sample_ok"):
                peer_line = (
                    f"今天有 {st['hard_count_today']} 位同学也觉得这道题很难，"
                    f"你的反应时间超过了约 {st['slower_than_percent']}% 的人，这很正常，我们慢慢来。"
                )

        if emotion_intensity >= settings.emotion_crisis_threshold:
            mode = "crisis_rest"
        elif emotion_intensity >= 0.5:
            mode = "comfort_light_plan"
        else:
            mode = "gentle"

        text = self._rule_text(mode, emotion_context, peer_line)
        # 危机话术必须确定、可审计，禁止再走生成模型
        if mode != "crisis_rest" and self.llm is not None:
            guard = guarded_call(
                lambda: self._llm_text(
                    emotion_context=emotion_context,
                    emotion_intensity=emotion_intensity,
                    user_energy=user_energy,
                    subject=subject,
                    peer_line=peer_line,
                    global_emotion_note=global_emotion_note,
                    mode=mode,
                ),
                fallback=text,
                breaker=self._breaker,
                timeout_sec=settings.companion_timeout_sec,
                name="companion_llm",
            )
            if guard.value:
                text = guard.value

        # 从其它会话带来的全局画像提示（例如「今天不想学数学」）
        if global_emotion_note and subject:
            text += f"\n（我记得你提到过：{global_emotion_note}。在「{subject}」这边我们也会放慢节奏。）"

        if voice_mode:
            text = self._to_spoken(text)
            audio = self.tts.synthesize(text)
        else:
            audio = b""

        return {
            "response": text,
            "mode": mode,
            "voice_mode": voice_mode,
            "audio_bytes_len": len(audio),
            "peer_attached": bool(peer_line),
            "force_rest": emotion_intensity >= settings.emotion_crisis_threshold
            or user_energy <= settings.energy_rest_threshold,
            "global_emotion_patch": {
                "emotion_intensity": emotion_intensity,
                "energy_level": user_energy,
                "note": emotion_context[:120],
            },
        }

    def from_audio(self, audio: bytes, **kwargs: Any) -> Dict[str, Any]:
        text = self.asr.transcribe(audio) or kwargs.get("fallback_text", "（语音未识别）")
        return self.comfort(text, voice_mode=True, **kwargs)

    @staticmethod
    def _rule_text(mode: str, emotion_context: str, peer_line: str) -> str:
        if mode == "crisis_rest":
            return (
                "我听到你现在非常难受。先停下来，不用刷题、不用看计划。"
                "喝口水，做三次深呼吸。你不是一个人在撑。"
                f"{(' ' + peer_line) if peer_line else ''}"
                "等情绪缓一点，我们再聊下一步。今天的任务我帮你按暂停了。"
            )
        if mode == "comfort_light_plan":
            return (
                f"我理解你因为「{emotion_context}」感到压力。"
                f"{peer_line or '难受的时候放慢一点，并不等于落后。'}"
                "我们先只做一件很小的事，做完就休息。"
            )
        return (
            f"收到，关于「{emotion_context}」，我在这儿陪着你。"
            f"{peer_line}"
            "准备好了就继续，不着急。"
        )

    def _llm_text(
        self,
        *,
        emotion_context: str,
        emotion_intensity: float,
        user_energy: float,
        subject: str,
        peer_line: str,
        global_emotion_note: str,
        mode: str,
    ) -> str:
        user = (
            f"情绪强度={emotion_intensity:.2f} 精力={user_energy:.2f} 科目={subject or '未指定'} 模式={mode}\n"
            f"学生原话：{emotion_context}\n"
            f"同伴统计（可引用，禁止编造）：{peer_line or '无'}\n"
            f"全局画像备注：{global_emotion_note or '无'}\n"
            "请直接给出对学生说的话。"
        )
        return (self.llm._chat(  # noqa: SLF001
            [{"role": "system", "content": COMPANION_SYSTEM}, {"role": "user", "content": user}]
        ) or "").strip()

    @staticmethod
    def _to_spoken(text: str) -> str:
        text = text.replace("\n", "。")
        if len(text) > 120:
            text = text[:117] + "…"
        return text
