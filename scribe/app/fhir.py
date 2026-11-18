"""
fhir.py - approved consultation -> FHIR R4 transaction Bundle.

FHIR is the standard most hospital systems (HMS / ERP / EMR) can import. One consultation becomes:
  Patient (matched on MRN, created only if new)   Practitioner (the doctor)
  Encounter (the visit)                            Composition (the full note, readable text)
  Condition       per diagnosis                    MedicationRequest  per medicine
  ServiceRequest  per test ordered                 Flag               per red flag
  Appointment     (status "proposed") when a follow-up was asked for
"""
from __future__ import annotations

import hashlib
import html
import re
import unicodedata
import uuid
from urllib.parse import quote
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from .models import ConsultNote, Patient

ICD10 = "http://hl7.org/fhir/sid/icd-10"
DOCTOR_SYSTEM = "urn:healthsync:doctor"
SESSION_SYSTEM = "urn:healthsync:scribe-session"
VERIFICATION = {"confirmed": "confirmed", "suspected": "provisional", "ruled_out": "refuted"}


def _urn() -> str:
    return f"urn:uuid:{uuid.uuid4()}"


def doctor_id(name: str) -> str:
    """Stable id for a doctor. Latin letters are kept for readability; a hash of the full name
    keeps Urdu-script names apart (they would all become 'unknown' otherwise)."""
    norm = " ".join(unicodedata.normalize("NFC", name).lower().split())
    slug = re.sub(r"[^a-z0-9]+", "-", norm).strip("-")
    digest = hashlib.sha256(norm.encode()).hexdigest()[:10]
    return f"{slug}-{digest}" if slug else digest


def _token(system: str, value: str) -> str:
    """identifier search token for ifNoneExist: FHIR-escape , | $ \\ in the value, then URL-encode."""
    escaped = re.sub(r"([\\,|$])", r"\\\1", value)
    return f"identifier={quote(system, safe=':/')}|{quote(escaped, safe='')}"


def _entry(resource: dict, ident: Optional[tuple[str, str]] = None) -> dict:
    """POST a resource. With ident, it gets that identifier and is created only if no resource
    with the same identifier exists, so sending the same consultation twice creates nothing new."""
    req = {"method": "POST", "url": resource["resourceType"]}
    if ident:
        system, value = ident
        idt = {"system": system, "value": value}
        # Composition.identifier is a single Identifier in R4; elsewhere it is a list
        resource["identifier"] = idt if resource["resourceType"] == "Composition" else [idt]
        req["ifNoneExist"] = _token(system, value)
    return {"fullUrl": _urn(), "resource": resource, "request": req}


def _div(rows: list[tuple[str, str]]) -> dict:
    body = "".join(f"<p><b>{html.escape(k)}:</b> {html.escape(v)}</p>" for k, v in rows if v)
    return {"status": "generated",
            "div": f'<div xmlns="http://www.w3.org/1999/xhtml">{body or "<p>(empty)</p>"}</div>'}


def _dose_text(m) -> str:
    parts = [m.dose, m.frequency, f"for {m.duration}" if m.duration else None, m.instructions]
    return ", ".join(p for p in parts if p)


def build_bundle(*, patient: Patient, doctor: str, department: Optional[str], note: ConsultNote,
                 encounter_start: str, approved_at: str, mrn_system: str,
                 session_id: str, today: Optional[date] = None, tz: str = "Asia/Karachi") -> dict:
    # Follow-up dates count from the visit's date in the hospital's own time zone, not UTC:
    # a visit at 1 am in Karachi is still "yesterday" in UTC.
    today = today or datetime.fromisoformat(approved_at).astimezone(ZoneInfo(tz)).date()
    sid = lambda kind: (SESSION_SYSTEM, f"{session_id}/{kind}")  # noqa: E731
    p = _entry({
        "resourceType": "Patient",
        "name": [{"text": patient.name}],
        **({"gender": patient.sex if patient.sex != "other" else "other"} if patient.sex else {}),
        **({"telecom": [{"system": "phone", "value": patient.phone}]} if patient.phone else {}),
        **({"extension": [{"url": "urn:healthsync:age-at-visit", "valueInteger": patient.age}]}
           if patient.age is not None else {}),
    }, (mrn_system, patient.mrn))
    dr = _entry({
        "resourceType": "Practitioner",
        "name": [{"text": doctor}],
    }, (DOCTOR_SYSTEM, doctor_id(doctor)))
    pat_ref, dr_ref = {"reference": p["fullUrl"]}, {"reference": dr["fullUrl"]}

    enc = _entry({
        "resourceType": "Encounter",
        "status": "finished",
        "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB",
                  "display": "ambulatory"},
        "subject": pat_ref,
        "participant": [{"individual": dr_ref}],
        "period": {"start": encounter_start, "end": approved_at},
        **({"serviceType": {"text": department}} if department else {}),
        **({"reasonCode": [{"text": note.chief_complaint}]} if note.chief_complaint else {}),
    }, sid("encounter"))
    enc_ref = {"reference": enc["fullUrl"]}
    common = {"subject": pat_ref, "encounter": enc_ref}
    entries = [p, dr, enc]

    for i, d in enumerate(note.assessment):
        code = {"text": d.name}
        if d.icd10:
            code["coding"] = [{"system": ICD10, "code": d.icd10, "display": d.name}]
        entries.append(_entry({
            "resourceType": "Condition", **common, "code": code, "recordedDate": approved_at,
            "verificationStatus": {"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                "code": VERIFICATION[d.certainty]}]},
            "category": [{"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                "code": "encounter-diagnosis"}]}],
        }, sid(f"condition/{i}")))

    for i, m in enumerate(note.medications):
        entries.append(_entry({
            "resourceType": "MedicationRequest", **common, "status": "active", "intent": "order",
            "medicationCodeableConcept": {"text": m.name}, "authoredOn": approved_at,
            "requester": dr_ref, "dosageInstruction": [{"text": _dose_text(m) or m.name}],
        }, sid(f"medication/{i}")))

    for i, t in enumerate(note.tests_ordered):
        entries.append(_entry({
            "resourceType": "ServiceRequest", **common, "status": "active", "intent": "order",
            "code": {"text": t}, "authoredOn": approved_at, "requester": dr_ref,
        }, sid(f"test/{i}")))

    for i, f in enumerate(note.red_flags):
        entries.append(_entry({
            "resourceType": "Flag", "status": "active", "subject": pat_ref, "encounter": enc_ref,
            "category": [{"text": "red-flag"}], "code": {"text": f}, "author": dr_ref,
        }, sid(f"flag/{i}")))

    fu = note.follow_up
    if fu.needed and fu.in_days is not None:
        due = (today + timedelta(days=fu.in_days)).isoformat()
        entries.append(_entry({
            "resourceType": "Appointment", "status": "proposed",
            "description": fu.reason or "Follow-up visit",
            "requestedPeriod": [{"start": due, "end": due}],
            "participant": [{"actor": pat_ref, "status": "needs-action"},
                            {"actor": dr_ref, "status": "accepted"}],
            "extension": [{"url": "urn:healthsync:encounter", "valueReference": enc_ref}],
        }, sid("follow-up")))

    symptoms = "; ".join(
        f"{s.name}{'' if s.present else ' (denied)'}"
        f"{f', {s.duration}' if s.duration else ''}{f', {s.detail}' if s.detail else ''}"
        for s in note.symptoms)
    sections = [
        ("Chief complaint", note.chief_complaint), ("History", note.history), ("Symptoms", symptoms),
        ("Examination", note.examination),
        ("Assessment", "; ".join(f"{d.name} ({d.certainty})" for d in note.assessment)),
        ("Medications", "; ".join(f"{m.name} {_dose_text(m)}".strip() for m in note.medications)),
        ("Tests ordered", "; ".join(note.tests_ordered)), ("Advice", "; ".join(note.advice)),
        ("Follow-up", f"in {fu.in_days} days: {fu.reason or ''}".strip() if fu.needed and fu.in_days is not None
         else (f"needed, date not set: {fu.reason or ''}".strip() if fu.needed else "")),
        ("Red flags", "; ".join(note.red_flags)),
    ]
    comp = _entry({
        "resourceType": "Composition", "status": "final",
        "type": {"coding": [{"system": "http://loinc.org", "code": "11488-4", "display": "Consult note"}]},
        **common, "date": approved_at, "author": [dr_ref], "title": "Outpatient consultation note",
        "text": _div(sections),
        "section": [
            {"title": "Clinical note", "text": _div(sections)},
            {"title": "Patient instructions (Urdu)", "text": _div([("ہدایات", note.patient_summary_urdu)])},
        ],
    }, sid("note"))
    entries.insert(3, comp)
    return {"resourceType": "Bundle", "type": "transaction", "entry": entries}
