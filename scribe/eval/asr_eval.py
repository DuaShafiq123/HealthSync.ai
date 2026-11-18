"""
asr_eval.py - compare speech-to-text models on OUR OWN recorded consultation clips.

Put clips in a folder (never commit them; they are real voices):
    clips/visit01.wav      the recording (wav, webm, m4a, mp3, ogg)
    clips/visit01.txt      what was actually said, typed by a person (the reference)
    clips/visit01.terms    optional: key terms, one per line (drug names, doses, numbers).
                           Alternative spellings on one line with '|', e.g.  ferrous|فیرس

Run (in the Docker image, or anywhere faster-whisper is installed):
    python -m eval.asr_eval clips/ --models large-v3-turbo kingabzpro/whisper-large-v3-urdu-ct2
Writes eval_results/asr_<time>.csv and prints a summary table.
Medical words matter more than filler words, so look at term recall, not only WER.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from dataclasses import replace

from app.config import Settings
from app.transcribe import WhisperTranscriber
from eval.metrics import cer, term_recall, wer

AUDIO = (".wav", ".webm", ".m4a", ".mp3", ".ogg")


def find_clips(folder: str) -> list[tuple[str, str, list[str]]]:
    clips = []
    for name in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in AUDIO:
            continue
        ref_path = os.path.join(folder, stem + ".txt")
        if not os.path.exists(ref_path):
            print(f"skip {name}: no {stem}.txt reference")
            continue
        with open(ref_path, encoding="utf-8") as f:
            ref = f.read()
        terms_path = os.path.join(folder, stem + ".terms")
        terms = []
        if os.path.exists(terms_path):
            with open(terms_path, encoding="utf-8") as f:
                terms = [t.strip() for t in f if t.strip()]
        clips.append((os.path.join(folder, name), ref, terms))
    return clips


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Whisper models on our own clips")
    ap.add_argument("folder")
    ap.add_argument("--models", nargs="+", default=["large-v3-turbo"])
    ap.add_argument("--language", default="ur")
    ap.add_argument("--out", default="eval_results")
    args = ap.parse_args()

    clips = find_clips(args.folder)
    if not clips:
        raise SystemExit("No clips with a matching .txt reference found")
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"asr_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    rows = []
    for model in args.models:
        print(f"\n=== {model} ===")
        tr = WhisperTranscriber(replace(Settings(), whisper_model=model, whisper_language=args.language))
        for path, ref, terms in clips:
            t0 = time.time()
            t = tr.transcribe(path)
            secs = time.time() - t0
            recall, missed = term_recall(terms, t.text)
            row = {"model": model, "clip": os.path.basename(path), "wer": round(wer(ref, t.text), 3),
                   "cer": round(cer(ref, t.text), 3), "term_recall": round(recall, 3),
                   "missed_terms": "; ".join(missed), "confidence": t.confidence,
                   "audio_s": t.duration_s, "seconds": round(secs, 1), "transcript": t.text}
            rows.append(row)
            print(f"{row['clip']}: WER {row['wer']}  CER {row['cer']}  terms {row['term_recall']}  "
                  f"{secs:.1f}s" + (f"  missed: {row['missed_terms']}" if missed else ""))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n| model | clips | mean WER | mean CER | mean term recall | real-time factor |")
    print("|---|---|---|---|---|---|")
    for model in args.models:
        r = [x for x in rows if x["model"] == model]
        rtf = sum(x["seconds"] for x in r) / max(sum(x["audio_s"] for x in r), 0.1)
        print(f"| {model} | {len(r)} | {statistics.mean(x['wer'] for x in r):.3f} | "
              f"{statistics.mean(x['cer'] for x in r):.3f} | "
              f"{statistics.mean(x['term_recall'] for x in r):.3f} | {rtf:.2f} |")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
