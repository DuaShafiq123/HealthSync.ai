"""
main.py - the scribe web service.

Flow for one consultation:
  1. POST /api/sessions                 doctor enters patient + confirms consent
  2. POST /api/sessions/{id}/audio      browser uploads the recording when the doctor presses Stop
     -> background: speech-to-text -> Claude drafts the note -> status "review"
  3. PUT  /api/sessions/{id}/note       doctor edits the draft
  4. POST /api/sessions/{id}/approve    doctor approves -> FHIR bundle -> hospital system
Run:  uvicorn app.main:app --port 8090   (see README for Docker)
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .config import Settings, get_settings
from .erp import ErpClient, ErpError
from .fhir import build_bundle
from .models import ConsultNote, SessionCreate, SessionView
from .note import NoteError, NoteWriter, make_note_writer
from .store import Store, now_iso
from .transcribe import Transcriber, make_transcriber

log = logging.getLogger("scribe")
STATIC = os.path.join(os.path.dirname(__file__), "static")
LOW_CONFIDENCE = 0.35         # below this, the doctor is warned that the audio was hard to hear
AUDIO_TYPES = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/wav": ".wav", "audio/x-wav": ".wav",
               "audio/mpeg": ".mp3", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a"}
EDITABLE = {"review", "send_failed"}


def create_app(settings: Optional[Settings] = None, transcriber: Optional[Transcriber] = None,
               note_writer: Optional[NoteWriter] = None, erp: Optional[ErpClient] = None) -> FastAPI:
    settings = settings or get_settings()
    store = Store(settings.data_dir)
    erp = erp or ErpClient(settings)
    app = FastAPI(title="HealthSync Consultation Scribe")

    # The speech model is big: load it once, on first use, not at start-up.
    lazy: dict = {"transcriber": transcriber, "note_writer": note_writer}
    lock = threading.Lock()

    def get(name: str):
        with lock:
            if lazy[name] is None:
                lazy[name] = make_transcriber(settings) if name == "transcriber" else make_note_writer(settings)
            return lazy[name]

    if n := store.recover_interrupted():
        log.warning("%d consultation(s) were interrupted by a restart and can be retried", n)

    if settings.preload:
        # Download and load the speech model when the container starts, so the first
        # patient of the day does not wait for it.
        def preload():
            try:
                get("transcriber")
                log.info("speech model ready")
            except Exception:
                log.exception("speech model preload failed; it will be retried on the first recording")
        threading.Thread(target=preload, daemon=True).start()

    def must_get(sid: str) -> SessionView:
        s = store.get(sid)
        if s is None:
            raise HTTPException(404, "No such consultation")
        return s

    # ---------------- background work ----------------
    def process(sid: str) -> None:
        row = store.raw(sid)
        try:
            transcript, conf = row["transcript"], row["transcript_confidence"]
            if transcript is None:
                store.update(sid, status="transcribing", error=None)
                t = get("transcriber").transcribe(row["audio_path"])
                transcript, conf = t.text, t.confidence
                store.update(sid, transcript=transcript, transcript_confidence=conf)
                if not settings.keep_audio and row["audio_path"] and os.path.exists(row["audio_path"]):
                    os.remove(row["audio_path"])
                    store.update(sid, audio_path=None)
            if not transcript.strip():
                store.update(sid, status="failed", error="No speech was heard in the recording")
                return
            store.update(sid, status="drafting")
            s = store.get(sid)
            context = (f"department: {s.department or 'general OPD'}; "
                       f"patient age: {s.patient.age if s.patient.age is not None else 'not given'}; "
                       f"sex: {s.patient.sex or 'not given'}")
            note = get("note_writer").write(transcript, context)
            if conf is not None and conf < LOW_CONFIDENCE:
                note.uncertain.insert(0, f"Speech recognition confidence was low ({conf:.2f}). "
                                         "Check the whole note against what was said.")
            store.update(sid, note=note, status="review")
        except NoteError as e:
            store.update(sid, status="failed", error=str(e))
        except Exception as e:                       # keep the session, show the error, allow retry
            log.exception("processing %s failed", sid)
            store.update(sid, status="failed", error=f"{type(e).__name__}: {e}")

    # ---------------- routes ----------------
    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "transcriber": settings.transcriber, "note_writer": settings.note_writer,
                "erp_mode": settings.erp_mode, "speech_model_ready": lazy["transcriber"] is not None}

    @app.post("/api/sessions", status_code=201)
    def create_session(body: SessionCreate) -> SessionView:
        if not body.consent:
            raise HTTPException(400, "Recording needs the consent of the patient and the doctor")
        sid = store.create(body.patient, body.doctor.strip(), body.department)
        return store.get(sid)

    @app.get("/api/sessions")
    def list_sessions(limit: int = 50) -> list[SessionView]:
        return store.list(min(max(limit, 1), 500))

    @app.get("/api/sessions/{sid}")
    def get_session(sid: str) -> SessionView:
        return must_get(sid)

    @app.post("/api/sessions/{sid}/audio", status_code=202)
    def upload_audio(sid: str, background: BackgroundTasks, audio: UploadFile = File(...)) -> SessionView:
        s = must_get(sid)
        if s.status != "waiting_for_audio":
            raise HTTPException(409, f"This consultation already has a recording (status: {s.status})")
        ctype = (audio.content_type or "").split(";")[0].strip()
        suffix = AUDIO_TYPES.get(ctype) or os.path.splitext(audio.filename or "")[1].lower()
        if suffix not in set(AUDIO_TYPES.values()):
            raise HTTPException(415, f"Unsupported audio type {ctype or audio.filename!r}")
        path = store.audio_path_for(sid, suffix)
        limit, size = settings.max_audio_mb * 1024 * 1024, 0
        with open(path, "wb") as f:
            while chunk := audio.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    f.close()
                    os.remove(path)
                    raise HTTPException(413, f"Recording is larger than {settings.max_audio_mb} MB")
                f.write(chunk)
        if size == 0:
            os.remove(path)
            raise HTTPException(400, "The recording is empty")
        store.update(sid, audio_path=path, status="transcribing")
        background.add_task(process, sid)
        return store.get(sid)

    @app.post("/api/sessions/{sid}/retry", status_code=202)
    def retry(sid: str, background: BackgroundTasks) -> SessionView:
        s = must_get(sid)
        row = store.raw(sid)
        if s.status != "failed":
            raise HTTPException(409, "Only a failed consultation can be retried")
        if row["transcript"] is None and not (row["audio_path"] and os.path.exists(row["audio_path"])):
            raise HTTPException(409, "The recording is gone; please record again")
        store.update(sid, status="transcribing" if row["transcript"] is None else "drafting", error=None)
        background.add_task(process, sid)
        return store.get(sid)

    @app.put("/api/sessions/{sid}/note")
    def edit_note(sid: str, note: ConsultNote) -> SessionView:
        s = must_get(sid)
        if s.status not in EDITABLE:
            raise HTTPException(409, f"The note cannot be edited now (status: {s.status})")
        store.update(sid, note=note)
        return store.get(sid)

    def bundle_for(s: SessionView, approved_at: str) -> dict:
        return build_bundle(patient=s.patient, doctor=s.doctor, department=s.department, note=s.note,
                            encounter_start=s.created_at, approved_at=approved_at,
                            mrn_system=settings.hospital_system, session_id=s.id, tz=settings.timezone)

    @app.get("/api/sessions/{sid}/fhir")
    def preview_fhir(sid: str) -> dict:
        s = must_get(sid)
        if s.note is None:
            raise HTTPException(409, "There is no note yet")
        return bundle_for(s, s.approved_at or now_iso())

    @app.post("/api/sessions/{sid}/approve")
    def approve(sid: str) -> SessionView:
        s = must_get(sid)
        if s.status not in EDITABLE or s.note is None:
            raise HTTPException(409, f"Only a reviewed note can be approved (status: {s.status})")
        approved_at = s.approved_at or now_iso()
        store.update(sid, status="sending", approved_at=approved_at, error=None)
        s = store.get(sid)
        payload = {"session": s.id, "patient": s.patient.model_dump(), "doctor": s.doctor,
                   "department": s.department, "approved_at": approved_at,
                   "note": s.note.model_dump(exclude={"dialogue"})}
        try:
            ref = erp.send(bundle_for(s, approved_at), payload)
            store.update(sid, status="sent", erp_reference=ref)
        except ErpError as e:
            store.update(sid, status="send_failed", error=str(e))
        return store.get(sid)

    return app


app = create_app() if os.environ.get("SCRIBE_NO_APP") != "1" else None
