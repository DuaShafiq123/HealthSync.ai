"""FHIR bundle shape, and the ERP mock's follow-up tracker and analytics."""
from datetime import date

from app.fhir import build_bundle
from app.models import ConsultNote, Diagnosis, FollowUp, Medication, Patient
from app.note import FakeNoteWriter
from app.transcribe import FakeTranscriber

P = Patient(name="Ali <script>", mrn="MR-9", sex="male", age=60)


def bundle(note=None, today=date(2026, 11, 18)):
    note = note or FakeNoteWriter().write(FakeTranscriber.TEXT, "")
    return build_bundle(patient=P, doctor="Dr. Omar", department=None, note=note,
                        encounter_start="2026-11-18T09:00:00+00:00", approved_at="2026-11-18T09:20:00+00:00",
                        mrn_system="urn:hosp:mrn", session_id="abc", today=today)


def no_empty(obj, path="$"):
    """FHIR does not allow empty arrays, empty objects or empty strings."""
    if isinstance(obj, dict):
        assert obj, f"empty object at {path}"
        for k, v in obj.items():
            no_empty(v, f"{path}.{k}")
    elif isinstance(obj, list):
        assert obj, f"empty array at {path}"
        for i, v in enumerate(obj):
            no_empty(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        assert obj.strip(), f"empty string at {path}"


def test_bundle_contents():
    b = bundle()
    types = [e["resource"]["resourceType"] for e in b["entry"]]
    assert types == ["Patient", "Practitioner", "Encounter", "Composition", "Condition",
                     "MedicationRequest", "ServiceRequest", "Appointment"]
    assert b["entry"][0]["request"]["ifNoneExist"] == "identifier=urn:hosp:mrn|MR-9"
    appt = b["entry"][-1]["resource"]
    assert appt["requestedPeriod"] == [{"start": "2026-12-02", "end": "2026-12-02"}]
    cond = b["entry"][4]["resource"]
    assert cond["code"]["coding"][0]["code"] == "D50.9"
    assert cond["verificationStatus"]["coding"][0]["code"] == "provisional"
    no_empty(b)


def test_bundle_escapes_text_and_handles_minimal_note():
    note = ConsultNote(chief_complaint="pain <b>& fever</b>",
                       assessment=[Diagnosis(name="Dengue", certainty="ruled_out")],
                       medications=[Medication(name="Paracetamol")], red_flags=["Bleeding gums"])
    b = bundle(note)
    no_empty(b)
    comp = next(e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "Composition")
    assert "&lt;b&gt;&amp; fever" in comp["text"]["div"] and "<b>&" not in comp["text"]["div"]
    cond = next(e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "Condition")
    assert cond["verificationStatus"]["coding"][0]["code"] == "refuted"
    assert "Appointment" not in [e["resource"]["resourceType"] for e in b["entry"]]
    flag = next(e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "Flag")
    assert flag["code"]["text"] == "Bleeding gums"


def test_followup_date_uses_hospital_local_date():
    note = ConsultNote(follow_up=FollowUp(needed=True, in_days=14))
    # 20:30 UTC on 18 Nov is 01:30 on 19 Nov in Karachi
    b = build_bundle(patient=P, doctor="Dr. Omar", department=None, note=note,
                     encounter_start="2026-11-18T20:00:00+00:00", approved_at="2026-11-18T20:30:00+00:00",
                     mrn_system="m", session_id="s")
    appt = next(e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "Appointment")
    assert appt["requestedPeriod"][0]["start"] == "2026-12-03"


def test_followup_without_days_is_not_scheduled():
    note = ConsultNote(follow_up=FollowUp(needed=True, in_days=None, reason="if worse"))
    assert "Appointment" not in [e["resource"]["resourceType"] for e in bundle(note)["entry"]]


def _visit(erp, today, days, mrn):
    note = ConsultNote(chief_complaint="x", follow_up=FollowUp(needed=True, in_days=days, reason="r"),
                       assessment=[Diagnosis(name="Hypertension", certainty="confirmed")])
    b = build_bundle(patient=Patient(name=f"P{mrn}", mrn=mrn), doctor="Dr. Omar", department=None,
                     note=note, encounter_start=f"{today}T09:00:00+00:00",
                     approved_at=f"{today}T09:10:00+00:00", mrn_system="urn:m", session_id=mrn,
                     today=today)
    r = erp.post("/fhir", json=b)
    assert r.status_code == 200, r.text


def test_followup_states_and_adherence(erp):
    # erp "today" is 2026-11-18 (conftest)
    _visit(erp, date(2026, 11, 1), 7, "A")      # due 11-08 -> overdue 10 days
    _visit(erp, date(2026, 11, 1), 10, "B")     # due 11-11 -> overdue, then marked done
    _visit(erp, date(2026, 11, 17), 2, "C")     # due 11-19 -> due (within 2 days)
    _visit(erp, date(2026, 11, 18), 30, "D")    # due 12-18 -> upcoming
    fu = erp.get("/api/followups").json()
    assert [(f["mrn"], f["state"]) for f in fu] == [("A", "overdue"), ("B", "overdue"),
                                                    ("C", "due"), ("D", "upcoming")]
    assert fu[0]["days_overdue"] == 10
    b_id = fu[1]["id"]
    assert erp.post(f"/api/followups/{b_id}/done").status_code == 200
    assert [f["mrn"] for f in erp.get("/api/followups?state=overdue").json()] == ["A"]
    a = erp.get("/api/analytics").json()
    assert a["followups"] == {"overdue": 1, "due": 1, "upcoming": 1, "done": 1, "cancelled": 0}
    assert a["followup_adherence_pct"] == 50.0
    assert a["top_diagnoses"] == [{"name": "Hypertension", "count": 4}]
    assert a["visits_per_doctor"] == [{"doctor": "Dr. Omar", "visits": 4}]
    days = {d["date"]: d["visits"] for d in a["visits_per_day"]}
    assert days["2026-11-01"] == 2 and days["2026-11-18"] == 1 and len(days) == 30


def test_erp_rejects_bad_input(erp):
    assert erp.post("/fhir", json={"resourceType": "Patient"}).status_code == 400
    assert erp.post("/fhir", content=b"not json").status_code == 400
    assert erp.post("/api/followups/999/done").status_code == 404
    assert erp.get("/fhir/Patient/999").status_code == 404


def test_reminders_and_whatsapp_numbers(erp):
    from erp_mock.main import pk_whatsapp_number
    assert [pk_whatsapp_number(p) for p in ("0300-1234567", "+92 300 1234567", "00923001234567",
                                            "042-35761234", "", None)] == \
        ["923001234567"] * 3 + [None, None, None]
    note = ConsultNote(follow_up=FollowUp(needed=True, in_days=3, reason="BP check"))
    b = build_bundle(patient=Patient(name="Ayesha", mrn="R1", phone="0300-1234567"), doctor="Dr. Omar",
                     department=None, note=note, encounter_start="2026-11-01T09:00:00+00:00",
                     approved_at="2026-11-01T09:10:00+00:00", mrn_system="m", session_id="r")
    erp.post("/fhir", json=b)
    f = erp.get("/api/followups").json()[0]
    assert f["state"] == "overdue"
    assert f["reminder_ur"].startswith("السلام علیکم Ayesha") and "04/11/2026" in f["reminder_ur"]
    assert f["whatsapp_url"].startswith("https://wa.me/923001234567?text=%D8%A7")
    erp.post(f"/api/followups/{f['id']}/done")
    f = erp.get("/api/followups").json()[0]
    assert (f["state"], f["reminder_ur"], f["whatsapp_url"]) == ("done", None, None)
