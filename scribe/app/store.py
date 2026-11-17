"""
store.py - keeps consultation sessions in one SQLite file (data/scribe.db).

One row per consultation. Audio files live next to it in data/audio/ and are deleted after
transcription unless SCRIBE_KEEP_AUDIO is on.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

from .models import ConsultNote, Patient, SessionView

COLUMNS = ["id", "status", "patient", "doctor", "department", "created_at", "consent_at",
           "audio_path", "transcript", "transcript_confidence", "note", "approved_at",
           "erp_reference", "error"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.audio_dir = os.path.join(data_dir, "audio")
        os.makedirs(self.audio_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(os.path.join(data_dir, "scribe.db"), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, patient TEXT NOT NULL,
                doctor TEXT NOT NULL, department TEXT, created_at TEXT NOT NULL,
                consent_at TEXT NOT NULL, audio_path TEXT, transcript TEXT,
                transcript_confidence REAL, note TEXT, approved_at TEXT,
                erp_reference TEXT, error TEXT)""")
        self._db.commit()

    # ---------------- write ----------------
    def create(self, patient: Patient, doctor: str, department: Optional[str]) -> str:
        sid = uuid.uuid4().hex[:12]
        with self._lock:
            self._db.execute(
                "INSERT INTO sessions (id, status, patient, doctor, department, created_at, consent_at)"
                " VALUES (?, 'waiting_for_audio', ?, ?, ?, ?, ?)",
                (sid, patient.model_dump_json(), doctor, department, now_iso(), now_iso()))
            self._db.commit()
        return sid

    def update(self, sid: str, **fields) -> None:
        bad = set(fields) - set(COLUMNS)
        if bad:
            raise ValueError(f"unknown columns: {bad}")
        if "note" in fields and isinstance(fields["note"], ConsultNote):
            fields["note"] = fields["note"].model_dump_json()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE sessions SET {cols} WHERE id = ?", (*fields.values(), sid))
            self._db.commit()

    def audio_path_for(self, sid: str, suffix: str) -> str:
        return os.path.join(self.audio_dir, f"{sid}{suffix}")

    # ---------------- read ----------------
    def raw(self, sid: str) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._db.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()

    def get(self, sid: str) -> Optional[SessionView]:
        r = self.raw(sid)
        return None if r is None else _view(r)

    def list(self, limit: int = 50) -> list[SessionView]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [_view(r) for r in rows]


def _view(r: sqlite3.Row) -> SessionView:
    return SessionView(
        id=r["id"], status=r["status"], patient=Patient.model_validate_json(r["patient"]),
        doctor=r["doctor"], department=r["department"], created_at=r["created_at"],
        transcript=r["transcript"], transcript_confidence=r["transcript_confidence"],
        note=ConsultNote.model_validate(json.loads(r["note"])) if r["note"] else None,
        approved_at=r["approved_at"], erp_reference=r["erp_reference"], error=r["error"])
