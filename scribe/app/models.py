"""
models.py - the shapes of the data that move through the scribe.

ConsultNote is what the AI drafts from the conversation and what the doctor edits and approves.
Only what was actually said in the consultation goes in. Anything the AI is unsure about goes
in `uncertain`, never into the note as a fact.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------- what the doctor types before recording ----------------
class Patient(BaseModel):
    # str_strip_whitespace: " MR-1 " and "MR-1" must be the same patient everywhere
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1)
    mrn: str = Field(min_length=1, description="Hospital medical record number")
    age: Optional[int] = Field(default=None, ge=0, le=130)
    sex: Optional[Literal["male", "female", "other"]] = None
    phone: Optional[str] = None


class SessionCreate(BaseModel):
    patient: Patient
    doctor: str = Field(min_length=1)
    department: Optional[str] = None
    consent: bool = Field(description="Patient AND doctor agreed to the recording")


# ---------------- the note the AI drafts ----------------
class Turn(BaseModel):
    speaker: Literal["doctor", "patient", "other"]
    text: str


class Symptom(BaseModel):
    name: str
    present: bool = True
    duration: Optional[str] = None      # as said, e.g. "3 weeks"
    detail: Optional[str] = None        # e.g. "worse on standing"


class Diagnosis(BaseModel):
    name: str
    certainty: Literal["confirmed", "suspected", "ruled_out"] = "suspected"
    icd10: Optional[str] = None


class Medication(BaseModel):
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    instructions: Optional[str] = None


class FollowUp(BaseModel):
    needed: bool = False
    in_days: Optional[int] = Field(default=None, ge=0, le=3650)
    reason: Optional[str] = None


class ConsultNote(BaseModel):
    chief_complaint: str = ""
    history: str = ""
    symptoms: list[Symptom] = []
    examination: str = ""
    assessment: list[Diagnosis] = []
    medications: list[Medication] = []
    tests_ordered: list[str] = []
    advice: list[str] = []
    follow_up: FollowUp = FollowUp()
    red_flags: list[str] = []
    patient_summary_urdu: str = ""      # plain Urdu summary the patient can take home
    uncertain: list[str] = []           # things the AI could not make out; the doctor checks these
    dialogue: list[Turn] = []           # transcript split into doctor / patient turns


# ---------------- the shape sent to Claude ----------------
# The API rejects ConsultNote as "Schema is too complex" (nine Optional fields = nine unions).
# So Claude fills this flat version: empty text "" and in_days -1 mean "not said", and
# to_note() turns them back into None. Keep the two in step.
class _SymptomOut(BaseModel):
    name: str
    present: bool
    duration: str
    detail: str


class _DiagnosisOut(BaseModel):
    name: str
    certainty: Literal["confirmed", "suspected", "ruled_out"]
    icd10: str


class _MedicationOut(BaseModel):
    name: str
    dose: str
    frequency: str
    duration: str
    instructions: str


class _FollowUpOut(BaseModel):
    needed: bool
    in_days: int
    reason: str


class ConsultNoteOut(BaseModel):
    chief_complaint: str
    history: str
    symptoms: list[_SymptomOut]
    examination: str
    assessment: list[_DiagnosisOut]
    medications: list[_MedicationOut]
    tests_ordered: list[str]
    advice: list[str]
    follow_up: _FollowUpOut
    red_flags: list[str]
    patient_summary_urdu: str
    uncertain: list[str]
    dialogue: list[Turn]

    def to_note(self) -> ConsultNote:
        n = lambda v: (v or "").strip() or None
        f = self.follow_up
        in_days = f.in_days if 0 <= f.in_days <= 3650 else None
        return ConsultNote(
            chief_complaint=self.chief_complaint, history=self.history,
            symptoms=[Symptom(name=x.name, present=x.present, duration=n(x.duration), detail=n(x.detail))
                      for x in self.symptoms],
            examination=self.examination,
            assessment=[Diagnosis(name=x.name, certainty=x.certainty, icd10=n(x.icd10)) for x in self.assessment],
            medications=[Medication(name=x.name, dose=n(x.dose), frequency=n(x.frequency),
                                    duration=n(x.duration), instructions=n(x.instructions))
                         for x in self.medications],
            tests_ordered=self.tests_ordered, advice=self.advice,
            follow_up=FollowUp(needed=f.needed, in_days=in_days, reason=n(f.reason)),
            red_flags=self.red_flags, patient_summary_urdu=self.patient_summary_urdu,
            uncertain=self.uncertain, dialogue=self.dialogue)


# ---------------- what the API returns ----------------
SessionStatus = Literal[
    "waiting_for_audio", "transcribing", "drafting", "review",
    "sending", "sent", "send_failed", "failed",
]


class SessionView(BaseModel):
    id: str
    status: SessionStatus
    patient: Patient
    doctor: str
    department: Optional[str]
    created_at: str
    transcript: Optional[str]
    transcript_confidence: Optional[float]
    note: Optional[ConsultNote]
    approved_at: Optional[str]
    erp_reference: Optional[str]
    error: Optional[str]
    ready_at: Optional[str] = None       # when the draft was ready for review
    sent_at: Optional[str] = None
    edit_stats: Optional[dict] = None    # what the doctor changed in the AI draft (see edits.py)
