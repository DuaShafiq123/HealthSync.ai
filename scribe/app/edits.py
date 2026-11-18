"""
edits.py - how much did the doctor change the AI's draft before approving it?

This is the scribe's main quality measure:
  * an item the doctor ADDED is something the AI missed (an omission)
  * an item the doctor REMOVED is something the AI got wrong or made up (a possible hallucination)
  * an item the doctor MODIFIED was found but recorded wrongly (for example the wrong dose)
`uncertain` and `dialogue` are working material, not part of the note, so they are ignored.
"""
from __future__ import annotations

import difflib
from typing import Callable

from .models import ConsultNote

TEXT_FIELDS = ["chief_complaint", "history", "examination", "patient_summary_urdu"]


def _norm(s: str | None) -> str:
    return " ".join((s or "").lower().split())


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return round(difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio(), 3)


def _compare_list(draft: list, final: list, key: Callable, detail: Callable) -> dict:
    d = {key(x): detail(x) for x in draft}
    f = {key(x): detail(x) for x in final}
    kept = d.keys() & f.keys()
    return {"draft": len(d), "final": len(f),
            "added": sorted(f.keys() - d.keys()), "removed": sorted(d.keys() - f.keys()),
            "modified": sorted(k for k in kept if d[k] != f[k])}


def compare(draft: ConsultNote, final: ConsultNote) -> dict:
    out: dict = {"text": {}, "lists": {}}
    for name in TEXT_FIELDS:
        a, b = getattr(draft, name), getattr(final, name)
        out["text"][name] = {"changed": _norm(a) != _norm(b), "similarity": _similarity(a, b)}

    same = lambda x: ""  # noqa: E731  (presence only)
    out["lists"]["symptoms"] = _compare_list(
        draft.symptoms, final.symptoms, lambda s: _norm(s.name),
        lambda s: (s.present, _norm(s.duration), _norm(s.detail)))
    out["lists"]["assessment"] = _compare_list(
        draft.assessment, final.assessment, lambda d: _norm(d.name), lambda d: (d.certainty, _norm(d.icd10)))
    out["lists"]["medications"] = _compare_list(
        draft.medications, final.medications, lambda m: _norm(m.name),
        lambda m: tuple(_norm(getattr(m, f)) for f in ("dose", "frequency", "duration", "instructions")))
    for name in ("tests_ordered", "advice", "red_flags"):
        out["lists"][name] = _compare_list(getattr(draft, name), getattr(final, name), _norm, same)

    fu_d, fu_f = draft.follow_up, final.follow_up
    out["follow_up_changed"] = (fu_d.needed, fu_d.in_days) != (fu_f.needed, fu_f.in_days)

    lists = out["lists"].values()
    out["items_added"] = sum(len(v["added"]) for v in lists)
    out["items_removed"] = sum(len(v["removed"]) for v in lists)
    out["items_modified"] = sum(len(v["modified"]) for v in lists)
    out["unchanged"] = (not any(t["changed"] for t in out["text"].values())
                        and out["items_added"] == out["items_removed"] == out["items_modified"] == 0
                        and not out["follow_up_changed"])
    return out
