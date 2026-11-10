"""
rppg_frozen.py - frozen rPPG heart-rate pipeline (HealthSync, Module A)

Method (chosen on the debug subjects, then frozen):
  1. Find the face in the first 30 frames, take the median box, and keep it fixed.
  2. Take the mean R, G, B of forehead + both cheeks for every frame.
  3. Turn R, G, B into one pulse signal with POS.
  4. Band-pass filter 0.7-4 Hz.
  5. Count the beats (peaks) in 30 s windows that move forward by 1 s.
     The first 3 s of every video are skipped.
  6. Compare with two references: the oximeter wave and the hr numbers in the file.

Run from the project folder (with the .venv active):
  python rppg_frozen.py
  python rppg_frozen.py --subjects subject4 subject5
  python rppg_frozen.py --root DATASET_2 --out outputs/frozen

Do NOT change CONFIG after the final run has started.
"""
import os
import re
import json
import argparse

import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from rppg_step1_3 import (load_ground_truth, detect_face_box,
                          skin_regions_from_box, mean_rgb_rect)

# ---------------- FROZEN SETTINGS ----------------
CONFIG = {
    "skip_sec": 3,                # skip the start of every video (camera adjusts brightness)
    "win_sec": 30,                # window length in seconds
    "step_sec": 1,                # how far the window moves each time
    "band_low_hz": 0.7,           # 42 BPM
    "band_high_hz": 4.0,          # 240 BPM
    "peak_prominence": 0.6,       # how tall a peak must be (times the signal's std) to count as a beat
    "reference_prominence": 0.3,  # same rule for the oximeter wave reference
    "min_beats": 3,               # fewer beats than this = window failed
    "max_hr_bpm": 200,            # beats closer than this rate are not counted twice
    "pos_win_sec": 1.6,           # POS window length
    "n_init_frames": 30,          # frames used to find the fixed face box
}

# Subjects that were used to build the method. They are NOT a fair test.
DEBUG_SUBJECTS = ["subject1", "subject3", "subject8", "subject13", "subject22"]


# ---------------- signal helpers ----------------
def bandpass(x, fs, cfg=CONFIG):
    low, high = cfg["band_low_hz"], cfg["band_high_hz"]
    b, a = butter(3, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def pos_signal(rgb, fs, win_sec=1.6):
    """POS: combine the R, G, B averages into one pulse signal.
    rgb has shape (N, 3): the mean R, G, B of every frame."""
    N = len(rgb)
    L = int(win_sec * fs)
    H = np.zeros(N)
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    for n in range(L - 1, N):
        m = n - L + 1
        C = rgb[m:n + 1].T                       # shape (3, L)
        Cn = C / C.mean(axis=1, keepdims=True)   # divide each color by its own average
        S = P @ Cn
        h = S[0] + (S[0].std() / (S[1].std() + 1e-9)) * S[1]
        H[m:n + 1] += h - h.mean()
    return H


def peak_rate(x_filtered, fs, prominence, cfg=CONFIG):
    """Heart rate in BPM from counting the beats. Returns nan if too few beats are found."""
    x = x_filtered - np.mean(x_filtered)
    min_gap = int(fs * 60 / cfg["max_hr_bpm"])
    peaks, _ = find_peaks(x, distance=min_gap, prominence=prominence * np.std(x))
    if len(peaks) < cfg["min_beats"]:
        return np.nan
    return 60.0 * (len(peaks) - 1) / ((peaks[-1] - peaks[0]) / fs)


# ---------------- video -> mean RGB ----------------
def extract_rgb_fixed(video_path, cfg=CONFIG):
    """Mean R, G, B of forehead + both cheeks for every frame, with one fixed face box."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    dets = []
    for _ in range(cfg["n_init_frames"]):
        ok, f = cap.read()
        if not ok:
            break
        b = detect_face_box(f)
        if b is not None:
            dets.append(b)
    if not dets:
        cap.release()
        raise RuntimeError("No face found in the first frames")
    box = np.median(np.array(dets, dtype=float), axis=0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rows, idx = [], 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        h_img, w_img = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        regions = skin_regions_from_box(box, w_img, h_img)
        means = [mean_rgb_rect(frame_rgb, rect) for rect in regions.values()]
        row = {"frame": idx, "time_s": idx / fps}
        for k, c in enumerate("RGB"):
            row[f"combined_{c}"] = float(np.nanmean([m[k] for m in means]))
        rows.append(row)
        idx += 1
    cap.release()
    print(f"[video] frames: {idx}, fps: {fps:.2f}, face box: {np.round(box, 1)}")
    return pd.DataFrame(rows), fps


# ---------------- one subject ----------------
def run_subject(subject_dir, cfg=CONFIG):
    video_path = os.path.join(subject_dir, "vid.avi")
    gt = load_ground_truth(os.path.join(subject_dir, "ground_truth.txt"))
    if "time" not in gt:
        raise RuntimeError("ground_truth.txt did not have 3 rows (ppg, hr, time)")

    df, fs = extract_rgb_fixed(video_path, cfg)
    t_v = df["time_s"].values
    pos = pos_signal(df[["combined_R", "combined_G", "combined_B"]].values,
                     fs, cfg["pos_win_sec"])
    v = bandpass(pos, fs, cfg)
    w = bandpass(gt["ppg"], gt["fs"], cfg)

    rows = []
    a = cfg["skip_sec"]
    end = min(t_v[-1], gt["time"][-1])
    while a + cfg["win_sec"] <= end:
        mv = (t_v >= a) & (t_v < a + cfg["win_sec"])
        mw = (gt["time"] >= a) & (gt["time"] < a + cfg["win_sec"])
        rows.append({
            "start_s": a,
            "video_hr": peak_rate(v[mv], fs, cfg["peak_prominence"], cfg),
            "wave_hr": peak_rate(w[mw], gt["fs"], cfg["reference_prominence"], cfg),
            "file_hr": gt["hr"][mw].mean(),
        })
        a += cfg["step_sec"]

    out = pd.DataFrame(rows)
    out["err_vs_wave"] = (out["video_hr"] - out["wave_hr"]).abs()
    out["err_vs_file"] = (out["video_hr"] - out["file_hr"]).abs()
    return out


# ---------------- all subjects ----------------
def subject_number(name):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 10 ** 9


def find_subjects(root):
    return sorted(
        [d for d in os.listdir(root)
         if d.startswith("subject")
         and os.path.exists(os.path.join(root, d, "vid.avi"))
         and os.path.exists(os.path.join(root, d, "ground_truth.txt"))],
        key=subject_number)


def main():
    ap = argparse.ArgumentParser(description="Frozen rPPG pipeline")
    ap.add_argument("--root", default="DATASET_2", help="folder with the subject folders")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="subject folder names to run (default: every subject found)")
    ap.add_argument("--out", default="outputs/frozen", help="folder for the result files")
    args = ap.parse_args()

    subjects = args.subjects if args.subjects else find_subjects(args.root)
    subjects = sorted(subjects, key=subject_number)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config_used.json"), "w") as f:
        json.dump({"CONFIG": CONFIG, "DEBUG_SUBJECTS": DEBUG_SUBJECTS}, f, indent=2)
    print("Subjects to run:", subjects)

    frames, status = [], {}
    for s in subjects:
        print(f"\n===== {s} =====")
        try:
            out = run_subject(os.path.join(args.root, s))
            out.insert(0, "subject", s)
            frames.append(out)
            status[s] = "ok"
        except Exception as e:                      # a failed video is a result, not something to hide
            print(f"FAILED: {e}")
            status[s] = f"crashed: {e}"

    rows = []
    if frames:
        all_windows = pd.concat(frames, ignore_index=True)
        all_windows.to_csv(os.path.join(args.out, "windows.csv"), index=False)
        g = all_windows.groupby("subject").agg(
            windows=("start_s", "count"),
            failed_windows=("video_hr", lambda x: int(x.isna().sum())),
            mae_vs_wave=("err_vs_wave", "mean"),
            mae_vs_file=("err_vs_file", "mean"),
            over5_vs_wave=("err_vs_wave", lambda x: int((x > 5).sum())),
            over5_vs_file=("err_vs_file", lambda x: int((x > 5).sum())),
        )
        for s in subjects:
            row = g.loc[s].to_dict() if s in g.index else {}
            row.update({"subject": s, "status": status[s]})
            rows.append(row)
    else:
        rows = [{"subject": s, "status": status[s]} for s in subjects]

    summary = pd.DataFrame(rows).set_index("subject")
    summary["debug_subject"] = summary.index.isin(DEBUG_SUBJECTS)
    summary = summary.round(2)
    summary.to_csv(os.path.join(args.out, "summary.csv"))

    print("\n===== SUMMARY =====")
    print(summary.to_string())

    ok = summary[summary["status"] == "ok"]
    for label, part in [("UNSEEN subjects (fair test)", ok[~ok["debug_subject"]]),
                        ("DEBUG subjects (used to build the method)", ok[ok["debug_subject"]])]:
        print(f"\n{label}: {len(part)} subjects")
        if len(part):
            print("  mean MAE vs oximeter wave:", round(part["mae_vs_wave"].mean(), 2), "BPM")
            print("  mean MAE vs file hr numbers:", round(part["mae_vs_file"].mean(), 2), "BPM")
    n_crash = int((summary["status"] != "ok").sum())
    print(f"\nCrashed subjects: {n_crash} of {len(summary)}")
    print(f"Files saved in: {args.out}")


if __name__ == "__main__":
    main()