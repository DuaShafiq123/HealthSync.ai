"""
stats.py - numbers that show whether the scribe works, from the consultations themselves.

  * turnaround: recording received -> draft ready (machine time),
                draft ready -> approved (doctor review time)
  * quality:    share of drafts approved without any change, and per section how many items
                doctors added (AI missed them), removed (AI got them wrong) or modified
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import datetime
from typing import Optional

LOW_CONFIDENCE = 0.35


def _seconds(a: Optional[str], b: Optional[str]) -> Optional[float]:
    if not a or not b:
        return None
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()


def _median(xs: list[float]) -> Optional[float]:
    return round(statistics.median(xs), 1) if xs else None


def summarise(rows) -> dict:
    status = Counter(r["status"] for r in rows)
    machine = [s for r in rows if (s := _seconds(r["audio_at"], r["ready_at"])) is not None]
    review = [s for r in rows if (s := _seconds(r["ready_at"], r["approved_at"])) is not None]
    edits = [json.loads(r["edit_stats"]) for r in rows if r["edit_stats"]]
    low = sum(1 for r in rows if r["transcript_confidence"] is not None
              and r["transcript_confidence"] < LOW_CONFIDENCE)

    per_section: dict = {}
    for e in edits:
        for name, v in e["lists"].items():
            d = per_section.setdefault(name, {"draft_items": 0, "added": 0, "removed": 0, "modified": 0})
            d["draft_items"] += v["draft"]
            for k in ("added", "removed", "modified"):
                d[k] += len(v[k])
    text_changed = Counter()
    for e in edits:
        for name, v in e["text"].items():
            text_changed[name] += v["changed"]

    n = len(edits)
    return {
        "consultations": len(rows),
        "by_status": dict(status),
        "low_confidence_transcripts": low,
        "median_seconds_recording_to_draft": _median(machine),
        "median_seconds_doctor_review": _median(review),
        "approved_with_edit_tracking": n,
        "approved_unchanged_pct": round(100 * sum(e["unchanged"] for e in edits) / n, 1) if n else None,
        "mean_items_added_per_note": round(sum(e["items_added"] for e in edits) / n, 2) if n else None,
        "mean_items_removed_per_note": round(sum(e["items_removed"] for e in edits) / n, 2) if n else None,
        "mean_items_modified_per_note": round(sum(e["items_modified"] for e in edits) / n, 2) if n else None,
        "sections": per_section,
        "text_fields_changed_pct": {k: round(100 * v / n, 1) for k, v in text_changed.items()} if n else {},
    }
