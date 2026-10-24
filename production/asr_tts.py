# -*- coding: utf-8 -*-
"""生产 ASR / TTS：默认阿里云 DashScope，可换讯飞。"""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class DashScopeTTS:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.dashscope_api_key
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY 未配置，生产 TTS 无法启动。")

    def synthesize(self, text: str) -> bytes:
        import dashscope
        from dashscope.audio.tts_v2 import SpeechSynthesizer

        dashscope.api_key = self.api_key
        synth = SpeechSynthesizer(model=settings.tts_model, voice=settings.tts_voice)
        audio = synth.call(text)
        if audio is None:
            logger.error("TTS 返回空音频")
            return b""
        return audio if isinstance(audio, bytes) else bytes(audio)


class DashScopeASR:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.dashscope_api_key
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY 未配置，生产 ASR 无法启动。")

    def transcribe(self, audio: bytes) -> str:
        """短音频识别；长音频请改用录音文件识别 API。"""
        import tempfile
        from pathlib import Path

        import dashscope
        from dashscope.audio.asr import Recognition

        dashscope.api_key = self.api_key
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            path = f.name
        rec = Recognition(model=settings.asr_model, format="wav", sample_rate=16000)
        result = rec.call(path)
        Path(path).unlink(missing_ok=True)
        if not result or not getattr(result, "output", None):
            return ""
        # dashscope 版本差异：尽量取出文本
        out = result.output
        if isinstance(out, dict):
            return str(out.get("text") or out.get("sentence", "") or "")
        return str(out)
