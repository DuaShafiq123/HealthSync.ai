"""
note_eval.py - how good are the notes Claude writes? Scored against consultations with known facts.

    python -m eval.note_eval                    # real Claude (needs ANTHROPIC_API_KEY; about 6 API calls)
    python -m eval.note_eval --writer fake      # checks the scoring itself, no API calls

Per case it checks:
  medicines found, dose correct, medicines NOT said (hallucinations), diagnoses, tests,
  follow-up days, red flags reported, denied symptoms kept as "absent", forbidden content absent.
The cases in cases.json are written by us (synthetic). Real consultations go in a private file
with the same format and are never committed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

from app.config import Settings
from app.models import ConsultNote
from app.note import make_note_writer

HERE = os.path.dirname(os.path.abspath(__file__))


def _has(text: str, options: list[str]) -> bool:
    t = text.lower()
    return any(o.lower() in t for o in options)


def _digits(s: str | None) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", s or ""))


def score(expected: dict, note: ConsultNote) -> dict:
    meds_found, dose_ok, matched_note_meds = 0, 0, set()
    for exp in expected["medications"]:
        hit = next((i for i, m in enumerate(note.medications) if _has(m.name, exp["names"])), None)
        if hit is not None:
            meds_found += 1
            matched_note_meds.add(hit)
            if exp.get("dose") is None or exp["dose"] in _digits(note.medications[hit].dose) \
                    or exp["dose"] in _digits(note.medications[hit].name):
                dose_ok += 1
    extra_meds = [m.name for i, m in enumerate(note.medications) if i not in matched_note_meds]

    dx_text = " | ".join(d.name for d in note.assessment if d.certainty != "ruled_out")
    flags_text = " | ".join(note.red_flags)
    tests_text = " | ".join(note.tests_ordered)
    denied_text = " | ".join(s.name for s in note.symptoms if not s.present)
    searchable = json.dumps(note.model_dump(exclude={"dialogue", "uncertain"}), ensure_ascii=False).lower()

    fu_expected = expected["follow_up_days"]
    fu_got = note.follow_up.in_days if note.follow_up.needed else None
    count = lambda groups, text: sum(_has(text, g) for g in groups)  # noqa: E731
    return {
        "meds": (meds_found, len(expected["medications"])),
        "doses": (dose_ok, len(expected["medications"])),
        "extra_meds": extra_meds,
        "diagnoses": (count(expected["diagnoses"], dx_text), len(expected["diagnoses"])),
        "tests": (count(expected["tests"], tests_text), len(expected["tests"])),
        "red_flags": (count(expected["red_flags"], flags_text), len(expected["red_flags"])),
        "denied": (count(expected["denied"], denied_text), len(expected["denied"])),
        "follow_up_ok": fu_got == fu_expected,
        "follow_up": (fu_expected, fu_got),
        "forbidden_found": [w for w in expected["must_not_contain"] if w.lower() in searchable],
    }


def _pct(pairs: list[tuple[int, int]]) -> str:
    got, total = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    return f"{got}/{total}" + (f" ({100 * got / total:.0f}%)" if total else "")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the note writer on consultations with known facts")
    ap.add_argument("--writer", default="claude", choices=["claude", "fake"])
    ap.add_argument("--cases", default=os.path.join(HERE, "cases.json"))
    ap.add_argument("--out", default="eval_results")
    args = ap.parse_args()

    with open(args.cases, encoding="utf-8") as f:
        cases = json.load(f)
    writer = make_note_writer(Settings(note_writer=args.writer))
    results = []
    for c in cases:
        t0 = time.time()
        try:
            note = writer.write(c["transcript"], c.get("context", ""))
        except Exception as e:                    # a failed case is a result, not a crash
            print(f"{c['id']}: FAILED {type(e).__name__}: {e}")
            results.append({"id": c["id"], "error": str(e)})
            continue
        s = score(c["expected"], note)
        s.update(id=c["id"], seconds=round(time.time() - t0, 1), note=note.model_dump(exclude={"dialogue"}))
        results.append(s)
        problems = []
        if s["extra_meds"]:
            problems.append(f"medicines not said: {s['extra_meds']}")
        if s["forbidden_found"]:
            problems.append(f"FORBIDDEN: {s['forbidden_found']}")
        if not s["follow_up_ok"]:
            problems.append(f"follow-up expected {s['follow_up'][0]} got {s['follow_up'][1]}")
        print(f"{c['id']}: meds {s['meds'][0]}/{s['meds'][1]} doses {s['doses'][0]}/{s['doses'][1]} "
              f"dx {s['diagnoses'][0]}/{s['diagnoses'][1]} flags {s['red_flags'][0]}/{s['red_flags'][1]} "
              f"denied {s['denied'][0]}/{s['denied'][1]}  {s['seconds']}s"
              + (("  <- " + "; ".join(problems)) if problems else ""))

    ok = [r for r in results if "error" not in r]
    print("\n| measure | result |\n|---|---|")
    for key, label in [("meds", "Medicines found"), ("doses", "Doses correct"), ("diagnoses", "Diagnoses found"),
                       ("tests", "Tests found"), ("red_flags", "Red flags reported"),
                       ("denied", "Denied symptoms kept as absent")]:
        print(f"| {label} | {_pct([r[key] for r in ok])} |")
    print(f"| Follow-up days correct | {sum(r['follow_up_ok'] for r in ok)}/{len(ok)} |")
    print(f"| Medicines not said (hallucinated) | {sum(len(r['extra_meds']) for r in ok)} |")
    print(f"| Forbidden content followed | {sum(bool(r['forbidden_found']) for r in ok)} cases |")
    print(f"| Failed cases | {len(results) - len(ok)} |")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"notes_{args.writer}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
