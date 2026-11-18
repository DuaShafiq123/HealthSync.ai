"""
transcribe.py - audio file -> text, using Whisper (faster-whisper, runs on the CPU, no PyTorch).

The model is set by SCRIBE_WHISPER_MODEL. Default "large-v3-turbo" handles Urdu mixed with
English medical words. For pure Urdu try "kingabzpro/whisper-large-v3-urdu-ct2" (Hugging Face,
Apache-2.0). The model is downloaded once and cached (in Docker: the hf-cache volume).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .config import Settings


@dataclass
class Transcript:
    text: str
    confidence: float      # 0..1, rough: mean token probability of the spoken segments
    language: str
    duration_s: float


class Transcriber(Protocol):
    def transcribe(self, audio_path: str) -> Transcript: ...


class WhisperTranscriber:
    def __init__(self, settings: Settings):
        from faster_whisper import WhisperModel   # optional dependency, only needed here
        device = settings.whisper_device
        compute = "int8" if device in ("cpu", "auto") else "float16"
        self.model = WhisperModel(settings.whisper_model, device=device, compute_type=compute)
        self.language = settings.whisper_language or None
        self.prompt = settings.whisper_prompt

    def transcribe(self, audio_path: str) -> Transcript:
        segments, info = self.model.transcribe(
            audio_path, language=self.language, vad_filter=True, beam_size=5,
            # A short, neutral Urdu-script prompt keeps the output in Urdu script. Tested 18 Nov 2026:
            # an English prompt full of medical words made large-v3-turbo write Hindi (Devanagari)
            # script, and a long Urdu prompt with medical words leaked its words into the transcript.
            initial_prompt=self.prompt or None,
            # do not feed earlier text back in: stops repeated sentences on long recordings
            condition_on_previous_text=False)
        texts, weights, probs = [], [], []
        for s in segments:
            t = s.text.strip()
            if not t:
                continue
            texts.append(t)
            length = max(s.end - s.start, 0.1)
            spoken = 1.0 - min(max(s.no_speech_prob, 0.0), 1.0)
            weights.append(length)
            probs.append(math.exp(s.avg_logprob) * spoken)
        conf = sum(p * w for p, w in zip(probs, weights, strict=True)) / sum(weights) if weights else 0.0
        return Transcript(text="\n".join(texts), confidence=round(conf, 3),
                          language=info.language, duration_s=round(info.duration, 1))


class FakeTranscriber:
    """Returns a fixed code-mixed Urdu consultation. Used by tests and for UI work without a model."""
    TEXT = ("Doctor: Assalam o alaikum, batayein kya masla hai?\n"
            "Patient: Doctor sahab teen hafte se thakawat hai aur khare hone pe chakkar aata hai.\n"
            "Doctor: Seene mein dard ya saans phoolna?\n"
            "Patient: Nahi, seene mein dard nahi.\n"
            "Doctor: BP 100/60 hai. CBC karwa lein. Tablet ferrous sulphate 200 mg din mein do baar, ek mahina. "
            "Do hafte baad report ke saath aayein.")

    def transcribe(self, audio_path: str) -> Transcript:
        return Transcript(text=self.TEXT, confidence=0.9, language="ur", duration_s=42.0)


def make_transcriber(settings: Settings) -> Transcriber:
    if settings.transcriber == "fake":
        return FakeTranscriber()
    if settings.transcriber == "whisper":
        return WhisperTranscriber(settings)
    raise ValueError(f"SCRIBE_TRANSCRIBER must be 'whisper' or 'fake', got {settings.transcriber!r}")
