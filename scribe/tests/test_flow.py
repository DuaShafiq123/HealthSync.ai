"""End to end: doctor starts a visit -> recording -> draft note -> edit -> approve -> hospital ERP."""
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from app.models import ConsultNote
from app.note import NoteError
from app.store import Store
from app.transcribe import Transcript
from conftest import PATIENT, make_scribe, start_visit, upload


def test_consent_is_required(scribe):
    r = start_visit(scribe, consent=False)
    assert r.status_code == 400
    assert "consent" in r.json()["detail"].lower()


def test_full_visit_reaches_erp_and_creates_followup(scribe, erp, tmp_path):
    s = start_visit(scribe).json()
    assert s["status"] == "waiting_for_audio"

    r = upload(scribe, s["id"])
    assert r.status_code == 202
    s = scribe.get(f"/api/sessions/{s['id']}").json()
    assert s["status"] == "review"
    assert "chakkar" in s["transcript"]
    assert s["note"]["follow_up"] == {"needed": True, "in_days": 14, "reason": "Review CBC report"}
    # audio is deleted after transcription by default
    assert os.listdir(tmp_path / "scribe" / "audio") == []

    note = s["note"]
    note["advice"].append("Eat iron-rich food")
    assert scribe.put(f"/api/sessions/{s['id']}/note", json=note).status_code == 200

    s = scribe.post(f"/api/sessions/{s['id']}/approve").json()
    assert s["status"] == "sent", s["error"]
    assert s["erp_reference"].startswith("Encounter/")

    fu = erp.get("/api/followups").json()
    assert len(fu) == 1
    due = (datetime.now(ZoneInfo("Asia/Karachi")).date() + timedelta(days=14)).isoformat()
    assert fu[0].pop("whatsapp_url").startswith("https://wa.me/923001234567?text=")
    assert "Dr. Sana Malik" in fu[0].pop("reminder_ur")
    assert fu[0] | {"id": None} == {"id": None, "due_date": due, "state": "upcoming",
                                    "reason": "Review CBC report", "patient": "Ayesha Khan",
                                    "mrn": "MR-1001", "phone": "0300-1234567",
                                    "doctor": "Dr. Sana Malik", "days_overdue": 0}
    comp = erp.get("/fhir/Composition").json()["entry"][0]["resource"]
    assert "Eat iron-rich food" in comp["text"]["div"]
    a = erp.get("/api/analytics").json()
    assert a["visits_total"] == 1 and a["top_medications"] == [{"name": "Ferrous sulphate", "count": 1}]
    assert a["top_tests"] == [{"name": "CBC", "count": 1}]


def test_same_mrn_reuses_the_patient_in_the_erp(scribe, erp):
    for _ in range(2):
        sid = start_visit(scribe).json()["id"]
        upload(scribe, sid)
        assert scribe.post(f"/api/sessions/{sid}/approve").json()["status"] == "sent"
    assert erp.get("/fhir/Patient").json()["total"] == 1
    assert erp.get("/fhir/Practitioner").json()["total"] == 1
    assert erp.get("/fhir/Encounter").json()["total"] == 2


def test_patient_identifiers_never_reach_the_note_writer(tmp_path, erp):
    seen = []

    class Spy:
        def write(self, transcript, context):
            seen.append(transcript + context)
            return ConsultNote(chief_complaint="x")

    c = make_scribe(tmp_path, erp, note_writer=Spy())
    sid = start_visit(c).json()["id"]
    upload(c, sid)
    assert len(seen) == 1
    for secret in (PATIENT["name"], PATIENT["mrn"], PATIENT["phone"]):
        assert secret not in seen[0]
    assert "34" in seen[0] and "Medicine OPD" in seen[0]


def test_wrong_or_empty_audio_is_rejected(scribe):
    sid = start_visit(scribe).json()["id"]
    assert upload(scribe, sid, ctype="text/plain", name="x.txt").status_code == 415
    assert upload(scribe, sid, data=b"").status_code == 400
    assert scribe.get(f"/api/sessions/{sid}").json()["status"] == "waiting_for_audio"
    assert upload(scribe, sid).status_code == 202
    assert upload(scribe, sid).status_code == 409          # second recording for the same visit


def test_audio_size_limit(tmp_path, erp):
    c = make_scribe(tmp_path, erp, max_audio_mb=1)
    sid = start_visit(c).json()["id"]
    assert upload(c, sid, data=b"x" * (1024 * 1024 + 1)).status_code == 413
    assert os.listdir(tmp_path / "scribe" / "audio") == []


def test_low_confidence_speech_warns_the_doctor(tmp_path, erp):
    class Mumbled:
        def transcribe(self, path):
            return Transcript(text="kuch samajh nahi aaya", confidence=0.2, language="ur", duration_s=10)

    c = make_scribe(tmp_path, erp, transcriber=Mumbled())
    sid = start_visit(c).json()["id"]
    upload(c, sid)
    s = c.get(f"/api/sessions/{sid}").json()
    assert s["status"] == "review"
    assert s["note"]["uncertain"][0].startswith("Speech recognition confidence was low (0.20)")


def test_silence_fails_with_a_clear_message(tmp_path, erp):
    class Silent:
        def transcribe(self, path):
            return Transcript(text="  ", confidence=0.0, language="ur", duration_s=5)

    c = make_scribe(tmp_path, erp, transcriber=Silent())
    sid = start_visit(c).json()["id"]
    upload(c, sid)
    s = c.get(f"/api/sessions/{sid}").json()
    assert (s["status"], s["error"]) == ("failed", "No speech was heard in the recording. Please record again.")


def test_note_failure_keeps_transcript_and_retry_works(tmp_path, erp):
    calls = []

    class Flaky:
        def write(self, transcript, context):
            calls.append(1)
            if len(calls) == 1:
                raise NoteError("Claude API rate limit reached, try again in a minute")
            return ConsultNote(chief_complaint="Fatigue")

    c = make_scribe(tmp_path, erp, note_writer=Flaky())
    sid = start_visit(c).json()["id"]
    upload(c, sid)
    s = c.get(f"/api/sessions/{sid}").json()
    assert s["status"] == "failed" and "rate limit" in s["error"]
    assert s["transcript"]                               # not lost, so no re-recording is needed
    assert c.post(f"/api/sessions/{sid}/retry").status_code == 202
    assert c.get(f"/api/sessions/{sid}").json()["note"]["chief_complaint"] == "Fatigue"


def test_erp_down_then_back(tmp_path, erp):
    class Down(httpx.Client):
        def post(self, *a, **k):
            raise httpx.ConnectError("connection refused")

    c = make_scribe(tmp_path, Down())
    sid = start_visit(c).json()["id"]
    upload(c, sid)
    s = c.post(f"/api/sessions/{sid}/approve").json()
    assert s["status"] == "send_failed" and "Could not reach" in s["error"]
    first_approval = s["approved_at"]

    c2 = make_scribe(tmp_path, erp)                       # same data folder, ERP reachable again
    s = c2.post(f"/api/sessions/{sid}/approve").json()
    assert s["status"] == "sent"
    assert s["approved_at"] == first_approval


def test_note_is_locked_after_sending(scribe):
    sid = start_visit(scribe).json()["id"]
    upload(scribe, sid)
    note = scribe.get(f"/api/sessions/{sid}").json()["note"]
    scribe.post(f"/api/sessions/{sid}/approve")
    assert scribe.put(f"/api/sessions/{sid}/note", json=note).status_code == 409
    assert scribe.post(f"/api/sessions/{sid}/approve").status_code == 409


def test_cannot_approve_before_a_note_exists(scribe):
    sid = start_visit(scribe).json()["id"]
    assert scribe.post(f"/api/sessions/{sid}/approve").status_code == 409
    assert scribe.get("/api/sessions/nope").status_code == 404


def test_webhook_mode_sends_our_json(tmp_path):
    got = []

    class Hook(httpx.Client):
        def post(self, url, json=None, headers=None):
            got.append((url, json, headers))
            return httpx.Response(200, json={"id": "ERP-77"})

    c = make_scribe(tmp_path, Hook(), erp_mode="webhook", erp_token="s3cret")
    sid = start_visit(c).json()["id"]
    upload(c, sid)
    s = c.post(f"/api/sessions/{sid}/approve").json()
    assert (s["status"], s["erp_reference"]) == ("sent", "ERP-77")
    url, body, headers = got[0]
    assert body["patient"]["mrn"] == "MR-1001" and "dialogue" not in body["note"]
    assert headers["Authorization"] == "Bearer s3cret"


def test_visit_interrupted_by_restart_can_be_retried(tmp_path, erp):
    c = make_scribe(tmp_path, erp)
    sid = start_visit(c).json()["id"]
    # the state a crash leaves behind: recording saved, transcription never finished
    store = Store(str(tmp_path / "scribe"))
    audio = store.audio_path_for(sid, ".webm")
    with open(audio, "wb") as f:
        f.write(b"fake")
    store.update(sid, audio_path=audio, status="transcribing")

    c2 = make_scribe(tmp_path, erp)                       # restart
    s = c2.get(f"/api/sessions/{sid}").json()
    assert (s["status"], s["error"]) == ("failed", "Interrupted by a restart. Press Try again.")
    assert c2.post(f"/api/sessions/{sid}/retry").status_code == 202
    assert c2.get(f"/api/sessions/{sid}").json()["status"] == "review"


def test_interrupted_send_can_be_resent(tmp_path, erp):
    c = make_scribe(tmp_path, erp)
    sid = start_visit(c).json()["id"]
    upload(c, sid)
    Store(str(tmp_path / "scribe")).update(sid, status="sending")
    c2 = make_scribe(tmp_path, erp)
    assert c2.get(f"/api/sessions/{sid}").json()["status"] == "send_failed"
    assert c2.post(f"/api/sessions/{sid}/approve").json()["status"] == "sent"
