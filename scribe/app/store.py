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
           "erp_reference", "error",
           # added later; created by _migrate on older databases
           "draft_note", "edit_stats", "audio_at", "ready_at", "sent_at"]
LATER_COLUMNS = {"draft_note": "TEXT", "edit_stats": "TEXT", "audio_at": "TEXT",
                 "ready_at": "TEXT", "sent_at": "TEXT"}


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
        have = {r["name"] for r in self._db.execute("PRAGMA table_info(sessions)")}
        for col, kind in LATER_COLUMNS.items():
            if col not in have:
                self._db.execute(f"ALTER TABLE sessions ADD COLUMN {col} {kind}")
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
        for k in ("note", "draft_note"):
            if isinstance(fields.get(k), ConsultNote):
                fields[k] = fields[k].model_dump_json()
        if isinstance(fields.get("edit_stats"), dict):
            fields["edit_stats"] = json.dumps(fields["edit_stats"])
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE sessions SET {cols} WHERE id = ?", (*fields.values(), sid))
            self._db.commit()

    def transition(self, sid: str, allowed_from: tuple[str, ...], **fields) -> bool:
        """Change a session only if its status is still one of allowed_from, in ONE statement.
        Returns False if another request got there first (double approve, upload race, ...)."""
        bad = set(fields) - set(COLUMNS)
        if bad:
            raise ValueError(f"unknown columns: {bad}")
        for k in ("note", "draft_note"):
            if isinstance(fields.get(k), ConsultNote):
                fields[k] = fields[k].model_dump_json()
        if isinstance(fields.get("edit_stats"), dict):
            fields["edit_stats"] = json.dumps(fields["edit_stats"])
        cols = ", ".join(f"{k} = ?" for k in fields)
        marks = ", ".join("?" for _ in allowed_from)
        with self._lock:
            n = self._db.execute(f"UPDATE sessions SET {cols} WHERE id = ? AND status IN ({marks})",
                                 (*fields.values(), sid, *allowed_from)).rowcount
            self._db.commit()
        return n == 1

    def recover_interrupted(self) -> int:
        """After a restart, work that was in progress is gone. Mark it so the doctor can retry.
        The recording is still on disk (it is deleted only after a transcript is saved)."""
        with self._lock:
            n = self._db.execute(
                "UPDATE sessions SET status = 'failed', error = 'Interrupted by a restart. Press Try again.'"
                " WHERE status IN ('transcribing', 'drafting')").rowcount
            n += self._db.execute(
                "UPDATE sessions SET status = 'send_failed',"
                " error = 'Interrupted while sending. Approve again to resend.' WHERE status = 'sending'").rowcount
            self._db.commit()
        return n

    def audio_path_for(self, sid: str, suffix: str) -> str:
        return os.path.join(self.audio_dir, f"{sid}{suffix}")

    # ---------------- read ----------------
    def raw(self, sid: str) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._db.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()

    def get(self, sid: str) -> Optional[SessionView]:
        r = self.raw(sid)
        return None if r is None else _view(r)

    def list(self, limit: int = 50, mrn: Optional[str] = None) -> list[SessionView]:
        sql, args = "SELECT * FROM sessions", []
        if mrn:
            sql += " WHERE json_extract(patient, '$.mrn') = ?"
            args.append(mrn)
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY created_at DESC, rowid DESC LIMIT ?",
                                    (*args, limit)).fetchall()
        return [_view(r) for r in rows]

    def rows(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute("SELECT * FROM sessions").fetchall()


def _view(r: sqlite3.Row) -> SessionView:
    return SessionView(
        id=r["id"], status=r["status"], patient=Patient.model_validate_json(r["patient"]),
        doctor=r["doctor"], department=r["department"], created_at=r["created_at"],
        transcript=r["transcript"], transcript_confidence=r["transcript_confidence"],
        note=ConsultNote.model_validate(json.loads(r["note"])) if r["note"] else None,
        approved_at=r["approved_at"], erp_reference=r["erp_reference"], error=r["error"],
        ready_at=r["ready_at"], sent_at=r["sent_at"],
        edit_stats=json.loads(r["edit_stats"]) if r["edit_stats"] else None)
