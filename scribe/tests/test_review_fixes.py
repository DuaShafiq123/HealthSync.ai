"""Regression tests for the issues found in the 18 Nov code review."""
import threading
import time

import httpx
from fastapi.testclient import TestClient

from app.fhir import build_bundle, doctor_id
from app.models import ConsultNote, FollowUp, Patient
from app.store import Store
from app.transcribe import Transcript
from conftest import PATIENT, TODAY, make_scribe, start_visit, upload
from erp_mock.main import create_app as create_erp


def ready_visit(c, patient=None, doctor="Dr. Sana Malik"):
    r = c.post("/api/sessions", json={"patient": patient or PATIENT, "doctor": doctor, "consent": True})
    sid = r.json()["id"]
    upload(c, sid)
    return sid


def test_two_approvals_at_once_send_only_once(tmp_path, erp):
    sends = []

    class SlowErp(httpx.Client):
        def post(self, url, json=None, headers=None):
            sends.append(1)
            time.sleep(0.3)
            return erp.post("/fhir", json=json)

    a = make_scribe(tmp_path, SlowErp())
    b = make_scribe(tmp_path, SlowErp())               # a second tab / server worker
    sid = ready_visit(a)
    codes = []
    threads = [threading.Thread(target=lambda c=c: codes.append(c.post(f"/api/sessions/{sid}/approve").status_code))
               for c in (a, b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(codes) == [200, 409]
    assert len(sends) == 1
    assert erp.get("/fhir/Encounter").json()["total"] == 1


def test_resend_after_lost_reply_creates_no_duplicates(tmp_path, erp):
    class CommitsThenTimesOut(httpx.Client):
        def post(self, url, json=None, headers=None):
            erp.post("/fhir", json=json)               # the hospital system saved it...
            raise httpx.ReadTimeout("no reply")        # ...but the answer never arrived

    c = make_scribe(tmp_path, CommitsThenTimesOut())
    sid = ready_visit(c)
    assert c.post(f"/api/sessions/{sid}/approve").json()["status"] == "send_failed"
    c2 = make_scribe(tmp_path, erp)
    assert c2.post(f"/api/sessions/{sid}/approve").json()["status"] == "sent"
    for rtype in ("Encounter", "Composition", "Condition", "MedicationRequest", "ServiceRequest", "Appointment"):
        assert erp.get(f"/fhir/{rtype}").json()["total"] == 1, rtype
    assert len(erp.get("/api/followups").json()) == 1


def test_unexpected_error_while_sending_does_not_leave_visit_stuck(tmp_path):
    class Broken(httpx.Client):
        def post(self, *a, **k):
            raise RuntimeError("boom")

    c = make_scribe(tmp_path, Broken())
    sid = ready_visit(c)
    s = c.post(f"/api/sessions/{sid}/approve").json()
    assert s["status"] == "send_failed" and "boom" in s["error"]


def test_webhook_answering_with_a_list_counts_as_sent(tmp_path):
    class ListReply(httpx.Client):
        def post(self, *a, **k):
            return httpx.Response(200, json=["ok"])

    c = make_scribe(tmp_path, ListReply(), erp_mode="webhook")
    sid = ready_visit(c)
    s = c.post(f"/api/sessions/{sid}/approve").json()
    assert (s["status"], s["erp_reference"]) == ("sent", "accepted")


def test_urdu_named_doctors_stay_separate(scribe, erp):
    assert doctor_id("ڈاکٹر علی") != doctor_id("ڈاکٹر احمد")
    for i, dr in enumerate(["ڈاکٹر علی", "ڈاکٹر احمد"]):
        sid = ready_visit(scribe, patient={**PATIENT, "mrn": f"U{i}"}, doctor=dr)
        assert scribe.post(f"/api/sessions/{sid}/approve").json()["status"] == "sent"
    names = sorted(p["resource"]["name"][0]["text"] for p in erp.get("/fhir/Practitioner").json()["entry"])
    assert names == sorted(["ڈاکٹر علی", "ڈاکٹر احمد"])


def test_mrn_is_trimmed_and_escaped(scribe):
    s = start_visit(scribe, patient={**PATIENT, "mrn": "  MR-1001 ", "name": " Ayesha Khan "}).json()
    assert (s["patient"]["mrn"], s["patient"]["name"]) == ("MR-1001", "Ayesha Khan")
    b = build_bundle(patient=Patient(name="X", mrn="A|1,2"), doctor="D", department=None, note=ConsultNote(),
                     encounter_start="2026-11-18T09:00:00+00:00", approved_at="2026-11-18T09:00:00+00:00",
                     mrn_system="urn:m", session_id="s")
    assert b["entry"][0]["request"]["ifNoneExist"] == "identifier=urn:m|A%5C%7C1%5C%2C2"


def test_returning_patient_gets_new_phone(scribe, erp):
    first = ready_visit(scribe, patient={**PATIENT, "phone": None})
    scribe.post(f"/api/sessions/{first}/approve")
    second = ready_visit(scribe, patient={**PATIENT, "phone": "0333-7654321"})
    scribe.post(f"/api/sessions/{second}/approve")
    patients = erp.get("/fhir/Patient").json()
    assert patients["total"] == 1
    assert patients["entry"][0]["resource"]["telecom"] == [{"system": "phone", "value": "0333-7654321"}]


def test_silent_recording_can_be_recorded_again(tmp_path, erp):
    calls = []

    class SilentThenSpeech:
        def transcribe(self, path):
            calls.append(path)
            text = "" if len(calls) == 1 else "Patient: sar dard hai"
            return Transcript(text=text, confidence=0.9, language="ur", duration_s=5)

    c = make_scribe(tmp_path, erp, transcriber=SilentThenSpeech())
    sid = start_visit(c).json()["id"]
    upload(c, sid)
    s = c.get(f"/api/sessions/{sid}").json()
    assert s["status"] == "failed" and "record again" in s["error"]
    assert c.post(f"/api/sessions/{sid}/retry").status_code == 409     # nothing to retry
    assert upload(c, sid).status_code == 202                            # a new recording is accepted
    s = c.get(f"/api/sessions/{sid}").json()
    assert s["status"] == "review" and s["transcript"] == "Patient: sar dard hai"


def test_late_background_work_cannot_overwrite_a_sent_note(tmp_path, erp):
    c = make_scribe(tmp_path, erp)
    sid = ready_visit(c)
    c.post(f"/api/sessions/{sid}/approve")
    store = Store(str(tmp_path / "scribe"))
    assert store.transition(sid, ("drafting",), status="review", note=ConsultNote()) is False
    assert c.get(f"/api/sessions/{sid}").json()["status"] == "sent"


def test_erp_rolls_back_a_half_bad_bundle(erp):
    good = {"fullUrl": "urn:uuid:1", "resource": {"resourceType": "Patient", "name": [{"text": "A"}]},
            "request": {"method": "POST", "url": "Patient"}}
    bad = {"fullUrl": "urn:uuid:2", "resource": "not an object", "request": {"method": "POST"}}
    assert erp.post("/fhir", json={"resourceType": "Bundle", "type": "transaction",
                                   "entry": [good, bad]}).status_code == 400
    ok = erp.post("/fhir", json={"resourceType": "Bundle", "type": "transaction", "entry": [good]})
    assert ok.status_code == 200
    assert erp.get("/fhir/Patient").json()["total"] == 1       # the half-written row was undone


def test_visits_are_counted_on_the_hospital_date(tmp_path):
    erp = TestClient(create_erp(str(tmp_path / "e.db"), today=lambda: TODAY))
    b = build_bundle(patient=Patient(name="N", mrn="N1"), doctor="D", department=None, note=ConsultNote(),
                     encounter_start="2026-11-17T20:30:00+00:00",   # 01:30 on 18 Nov in Karachi
                     approved_at="2026-11-17T20:40:00+00:00", mrn_system="m", session_id="tz")
    erp.post("/fhir", json=b)
    days = {d["date"]: d["visits"] for d in erp.get("/api/analytics").json()["visits_per_day"]}
    assert days["2026-11-18"] == 1 and days["2026-11-17"] == 0


def test_same_day_follow_up_is_written_in_the_note():
    b = build_bundle(patient=Patient(name="N", mrn="N1"), doctor="D", department=None,
                     note=ConsultNote(follow_up=FollowUp(needed=True, in_days=0, reason="review today")),
                     encounter_start="2026-11-18T09:00:00+00:00", approved_at="2026-11-18T09:10:00+00:00",
                     mrn_system="m", session_id="z")
    comp = next(e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "Composition")
    assert "in 0 days: review today" in comp["text"]["div"]
