"""ClaudeNoteWriter: request shape and how API outcomes are handled (no real API calls)."""
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.models import ConsultNote, ConsultNoteOut
from app.note import SYSTEM_PROMPT, ClaudeNoteWriter, NoteError


class FakeMessages:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def writer(response):
    msgs = FakeMessages(response)
    client = SimpleNamespace(beta=SimpleNamespace(messages=msgs))
    return ClaudeNoteWriter(Settings(), client=client), msgs


def test_request_shape_and_success():
    out = ConsultNoteOut(
        chief_complaint="Cough", history="", symptoms=[], examination="", assessment=[],
        medications=[{"name": "Syrup", "dose": "5 ml", "frequency": "", "duration": " ", "instructions": ""}],
        tests_ordered=[], advice=[], follow_up={"needed": False, "in_days": -1, "reason": ""},
        red_flags=[], patient_summary_urdu="", uncertain=[], dialogue=[])
    w, msgs = writer(SimpleNamespace(stop_reason="end_turn", parsed_output=out))
    note = w.write("Patient: khansi hai", "department: OPD")
    assert note.chief_complaint == "Cough"
    assert note.medications[0].dose == "5 ml" and note.medications[0].frequency is None
    assert note.medications[0].duration is None and note.follow_up.in_days is None
    k = msgs.kwargs
    assert k["model"] == "claude-opus-5-5"
    assert k["output_format"] is ConsultNoteOut
    assert k["system"] == SYSTEM_PROMPT
    assert k["fallbacks"] == "default" and k["betas"] == ["server-side-fallback-2026-07-01"]
    assert "<transcript>\nPatient: khansi hai\n</transcript>" in k["messages"][0]["content"]


@pytest.mark.parametrize("stop, msg", [("refusal", "declined"), ("max_tokens", "too long")])
def test_stop_reasons_become_clear_errors(stop, msg):
    w, _ = writer(SimpleNamespace(stop_reason=stop, parsed_output=None))
    with pytest.raises(NoteError, match=msg):
        w.write("x", "y")
