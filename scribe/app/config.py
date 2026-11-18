"""
config.py - every setting comes from an environment variable, so Docker can change it
without touching the code. Defaults are safe for a laptop demo.
"""
import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    data_dir: str = field(default_factory=lambda: os.environ.get("SCRIBE_DATA_DIR", "data"))
    # "whisper" (real speech-to-text) or "fake" (tests and UI work without a model)
    transcriber: str = field(default_factory=lambda: os.environ.get("SCRIBE_TRANSCRIBER", "whisper"))
    whisper_model: str = field(default_factory=lambda: os.environ.get("SCRIBE_WHISPER_MODEL", "large-v3-turbo"))
    whisper_language: str = field(default_factory=lambda: os.environ.get("SCRIBE_WHISPER_LANGUAGE", "ur"))
    whisper_device: str = field(default_factory=lambda: os.environ.get("SCRIBE_WHISPER_DEVICE", "auto"))
    # "claude" (real note writing) or "fake"
    note_writer: str = field(default_factory=lambda: os.environ.get("SCRIBE_NOTE_WRITER", "claude"))
    claude_model: str = field(default_factory=lambda: os.environ.get("SCRIBE_CLAUDE_MODEL", "claude-opus-5-5"))
    # Where approved notes go: "fhir" (POST a FHIR transaction bundle), "webhook" (POST our JSON), "none"
    erp_mode: str = field(default_factory=lambda: os.environ.get("SCRIBE_ERP_MODE", "fhir"))
    erp_url: str = field(default_factory=lambda: os.environ.get("SCRIBE_ERP_URL", "http://localhost:8091/fhir"))
    erp_token: str = field(default_factory=lambda: os.environ.get("SCRIBE_ERP_TOKEN", ""))
    # Audio is deleted as soon as it is transcribed, unless this is switched on
    keep_audio: bool = field(default_factory=lambda: _bool("SCRIBE_KEEP_AUDIO", False))
    # Load the speech model at start-up instead of on the first recording (on in docker-compose)
    preload: bool = field(default_factory=lambda: _bool("SCRIBE_PRELOAD", False))
    max_audio_mb: int = field(default_factory=lambda: int(os.environ.get("SCRIBE_MAX_AUDIO_MB", "100")))
    hospital_system: str = field(default_factory=lambda: os.environ.get("SCRIBE_MRN_SYSTEM", "urn:healthsync:mrn"))
    access_code: str = field(default_factory=lambda: os.environ.get("SCRIBE_ACCESS_CODE", ""))
    timezone: str = field(default_factory=lambda: os.environ.get("SCRIBE_TIMEZONE", "Asia/Karachi"))


def get_settings() -> Settings:
    return Settings()
