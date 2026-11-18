"""Edit tracking (AI draft vs approved note), scribe stats, previous visits, old-database upgrade."""
import sqlite3

from app.edits import compare
from app.models import ConsultNote, Diagnosis, FollowUp, Medication, Symptom
from app.note import FakeNoteWriter
from app.store import Store
from app.transcribe import FakeTranscriber
from conftest import PATIENT, start_visit, upload


def draft():
    return FakeNoteWriter().write(FakeTranscriber.TEXT, "")


def test_unchanged_note():
    e = compare(draft(), draft())
    assert e["unchanged"] is True
    assert (e["items_added"], e["items_removed"], e["items_modified"]) == (0, 0, 0)


def test_added_removed_modified_are_counted_per_section():
    d = draft()
    f = d.model_copy(deep=True)
    f.medications = [Medication(name="Ferrous Sulphate", dose="200 mg", frequency="twice daily",
                                duration="2 months"),                 # modified (duration), case ignored
                     Medication(name="Vitamin C", dose="500 mg")]    # added: AI missed it
    f.tests_ordered = []                                              # removed: AI added a test not ordered
    f.assessment = [Diagnosis(name="Iron deficiency anaemia", certainty="confirmed", icd10="D50.9")]
    f.history = d.history + " Heavy periods."
    f.follow_up = FollowUp(needed=True, in_days=30)
    e = compare(d, f)
    assert e["lists"]["medications"]["added"] == ["vitamin c"]
    assert e["lists"]["medications"]["modified"] == ["ferrous sulphate"]
    assert e["lists"]["tests_ordered"]["removed"] == ["cbc"]
    assert e["lists"]["assessment"]["modified"] == ["iron deficiency anaemia"]
    assert e["text"]["history"]["changed"] and 0.5 < e["text"]["history"]["similarity"] < 1
    assert e["text"]["chief_complaint"] == {"changed": False, "similarity": 1.0}
    assert e["follow_up_changed"] is True
    assert (e["items_added"], e["items_removed"], e["items_modified"]) == (1, 1, 2)
    assert e["unchanged"] is False


def test_whitespace_and_case_only_edits_do_not_count():
    d = draft()
    f = d.model_copy(deep=True)
    f.chief_complaint = "  " + d.chief_complaint.upper() + " "
    f.symptoms = [Symptom(name=s.name.lower(), present=s.present, duration=s.duration, detail=s.detail)
                  for s in d.symptoms]
    assert compare(d, f)["unchanged"] is True


def test_approval_records_edits_and_stats(scribe):
    # visit 1 approved as drafted, visit 2 with one medicine added
    for add in (False, True):
        sid = start_visit(scribe).json()["id"]
        upload(scribe, sid)
        note = scribe.get(f"/api/sessions/{sid}").json()["note"]
        if add:
            note["medications"].append({"name": "Vitamin C"})
            scribe.put(f"/api/sessions/{sid}/note", json=note)
        s = scribe.post(f"/api/sessions/{sid}/approve").json()
        assert s["status"] == "sent" and s["sent_at"] and s["ready_at"]
        assert s["edit_stats"]["unchanged"] is (not add)
    st = scribe.get("/api/stats").json()
    assert st["consultations"] == 2 and st["by_status"] == {"sent": 2}
    assert st["approved_unchanged_pct"] == 50.0
    assert st["mean_items_added_per_note"] == 0.5
    assert st["sections"]["medications"] == {"draft_items": 2, "added": 1, "removed": 0, "modified": 0}
    assert st["median_seconds_recording_to_draft"] is not None


def test_stats_on_empty_scribe(scribe):
    st = scribe.get("/api/stats").json()
    assert st["consultations"] == 0 and st["approved_unchanged_pct"] is None


def test_previous_visits_by_mrn(scribe):
    a = start_visit(scribe).json()["id"]
    start_visit(scribe, patient={**PATIENT, "mrn": "MR-OTHER"})
    b = start_visit(scribe).json()["id"]
    got = [s["id"] for s in scribe.get("/api/sessions", params={"mrn": "MR-1001"}).json()]
    assert got == [b, a]                                  # newest first, other patients excluded
    assert scribe.get("/api/sessions", params={"mrn": "nobody"}).json() == []


def test_old_database_is_upgraded(tmp_path):
    d = tmp_path / "old"
    d.mkdir()
    db = sqlite3.connect(d / "scribe.db")
    db.execute("""CREATE TABLE sessions (id TEXT PRIMARY KEY, status TEXT NOT NULL, patient TEXT NOT NULL,
        doctor TEXT NOT NULL, department TEXT, created_at TEXT NOT NULL, consent_at TEXT NOT NULL,
        audio_path TEXT, transcript TEXT, transcript_confidence REAL, note TEXT, approved_at TEXT,
        erp_reference TEXT, error TEXT)""")
    db.execute("INSERT INTO sessions (id, status, patient, doctor, created_at, consent_at) VALUES "
               "('x', 'sent', ?, 'Dr. A', '2026-11-01T09:00:00+00:00', '2026-11-01T09:00:00+00:00')",
               ('{"name": "P", "mrn": "M"}',))
    db.commit()
    db.close()
    s = Store(str(d))
    v = s.get("x")
    assert v.status == "sent" and v.edit_stats is None
    s.update("x", draft_note=ConsultNote(), edit_stats={"unchanged": True})
    assert s.get("x").edit_stats == {"unchanged": True}
