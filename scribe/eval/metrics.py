"""
metrics.py - text normalisation and error rates for Urdu / Roman Urdu / English transcripts.

Urdu text from different tools uses different Unicode letters for the same sound (Arabic vs
Urdu yeh, kaf and heh), optional diacritics, and Urdu or Western digits. Without normalising,
an error rate counts those as mistakes. normalise() makes them the same before comparing.
"""
from __future__ import annotations

import re
import unicodedata

# Arabic-script variants -> the Urdu letter
_CHAR_MAP = str.maketrans({
    "ي": "ی", "ى": "ی", "ئ": "ی",     # yeh
    "ك": "ک",                        # kaf
    "ه": "ہ", "ة": "ہ",              # heh
    "ۀ": "ہ", "ۂ": "ہ",
    "أ": "ا", "إ": "ا", "آ": "آ",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4", "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
})
_DIACRITICS = re.compile("[ً-ٰٟۖ-ۭ]")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "").translate(_CHAR_MAP)
    t = _DIACRITICS.sub("", t).replace("‌", " ").replace("‍", "")
    t = _PUNCT.sub(" ", t.lower())
    return " ".join(t.split())


def _edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: (substitutions + deletions + insertions) / words in the reference."""
    ref, hyp = normalise(reference).split(), normalise(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate, spaces ignored. Fairer than WER for Urdu, where word splitting varies."""
    ref = normalise(reference).replace(" ", "")
    hyp = normalise(hypothesis).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(list(ref), list(hyp)) / len(ref)


def term_recall(terms: list[str], hypothesis: str) -> tuple[float, list[str]]:
    """Share of key terms (drug names, doses, numbers) found in the transcript, and the missed ones.
    A term may list spellings separated by '|', e.g. 'ferrous|فیرس'."""
    hyp = " " + normalise(hypothesis) + " "
    hyp_nospace = hyp.replace(" ", "")
    missed = []
    for term in terms:
        options = [normalise(o) for o in term.split("|") if o.strip()]
        if not any(f" {o} " in hyp or o.replace(" ", "") in hyp_nospace for o in options):
            missed.append(term)
    return (1 - len(missed) / len(terms) if terms else 1.0), missed
