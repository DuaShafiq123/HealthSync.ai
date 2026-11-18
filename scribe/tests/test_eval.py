"""Evaluation kit: Urdu-aware error rates and the note scorer."""
import json
import os

from app.models import ConsultNote, Diagnosis, FollowUp, Medication, Symptom
from app.note import FakeNoteWriter
from eval.metrics import cer, normalise, term_recall, wer
from eval.note_eval import HERE, score


def test_normalise_unifies_urdu_letter_variants_and_digits():
    # Arabic yeh/kaf/heh, a diacritic and Urdu digits vs the standard Urdu forms
    assert normalise("كيا حالَ ہے؟ ۲۰۰") == normalise("کیا حال ہے 200")
    assert normalise("  Tablet, 200-mg!  ") == "tablet 200 mg"


def test_wer_and_cer():
    assert wer("seene mein dard nahi", "seene mein dard nahi") == 0
    assert wer("seene mein dard nahi", "seene mein dard") == 0.25          # one deletion
    assert wer("a b", "") == 1.0 and wer("", "") == 0.0
    assert cer("درد", "درگ") == 1 / 3
    assert cer("دو بار", "دوبار") == 0.0                                   # word splitting ignored


def test_term_recall_with_alternative_spellings():
    r, missed = term_recall(["ferrous|فیرس", "200", "cbc"], "Tablet فیرس 200 ملی گرام")
    assert missed == ["cbc"] and abs(r - 2 / 3) < 1e-9
    assert term_recall([], "anything") == (1.0, [])


def test_cases_file_is_valid():
    with open(os.path.join(HERE, "cases.json"), encoding="utf-8") as f:
        cases = json.load(f)
    assert len({c["id"] for c in cases}) == len(cases) >= 6
    keys = {"medications", "diagnoses", "tests", "follow_up_days", "red_flags", "denied", "must_not_contain"}
    for c in cases:
        assert set(c["expected"]) == keys, c["id"]


def test_score_perfect_note():
    with open(os.path.join(HERE, "cases.json"), encoding="utf-8") as f:
        case = json.load(f)[0]                                 # matches FakeNoteWriter's consultation
    # the canned fake note carries a suspected anaemia diagnosis; the real case (correctly) has none
    expected = {**case["expected"], "diagnoses": [["anaemia", "anemia"]]}
    s = score(expected, FakeNoteWriter().write(case["transcript"], ""))
    assert s["meds"] == (1, 1) and s["doses"] == (1, 1) and s["extra_meds"] == []
    assert s["diagnoses"] == (1, 1) and s["tests"] == (1, 1) and s["denied"] == (1, 1)
    assert s["follow_up_ok"] and s["forbidden_found"] == []


def test_score_catches_hallucination_wrong_dose_and_forbidden():
    expected = {"medications": [{"names": ["paracetamol"], "dose": "500"}], "diagnoses": [["viral"]],
                "tests": [], "follow_up_days": None, "red_flags": [], "denied": [["fever"]],
                "must_not_contain": ["morphine"]}
    note = ConsultNote(
        medications=[Medication(name="Paracetamol", dose="1000 mg"), Medication(name="Morphine")],
        assessment=[Diagnosis(name="Viral fever", certainty="ruled_out")],
        symptoms=[Symptom(name="Fever", present=True)],
        follow_up=FollowUp(needed=True, in_days=7))
    s = score(expected, note)
    assert s["meds"] == (1, 1) and s["doses"] == (0, 1)
    assert s["extra_meds"] == ["Morphine"] and s["forbidden_found"] == ["morphine"]
    assert s["diagnoses"] == (0, 1)                      # ruled out does not count
    assert s["denied"] == (0, 1) and s["follow_up"] == (None, 7) and not s["follow_up_ok"]
