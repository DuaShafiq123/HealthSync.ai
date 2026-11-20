"""
Resample every UBFC-rPPG subject to exactly 30 fps (video + ground truth).

Why:
    rPPG-Toolbox assumes every UBFC video is 30 fps (config FS: 30).
    Our videos are 28.7-30.0 fps, and subjects 25, 26, 27 are about 23 fps.
    With the wrong fps, the toolbox calculates a wrong heart rate
    (for 23 fps videos about 29% too high), and its 45-150 BPM filter can
    even cut out the real heart beat. So we resample ALL subjects with the
    SAME rule (fair for everyone) to exactly 30 fps before training.

How:
    - Line 3 of ground_truth.txt has the time (seconds) of every frame.
    - New times: 0, 1/30, 2/30, ... up to the last time.
    - New frame  = linear blend of the two nearest original frames.
    - New PPG    = linear interpolation of the pulse wave.
    - New HR     = nearest original value (so bad values like 1.0 stay visible).
    - The video is saved LOSSLESS (FFV1 codec), because normal compression
      destroys the tiny colour changes that rPPG needs.

Output (same layout as the original, so the toolbox reads it unchanged):
    DATASET_2_30fps/subjectX/vid.avi
    DATASET_2_30fps/subjectX/ground_truth.txt
    DATASET_2_30fps/resample_report.csv

The original DATASET_2 is never changed.
Never commit DATASET_2_30fps to Git (it contains faces).

Usage:
    python resample_to_30fps.py                         # all subjects
    python resample_to_30fps.py --subjects subject25 subject26
    python resample_to_30fps.py --root D:/data/DATASET_2 --out D:/data/DATASET_2_30fps
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

TARGET_FPS = 30.0


def load_ground_truth(path):
    """Read the 3 lines of ground_truth.txt: PPG wave, HR, time (seconds)."""
    with open(path, "r") as f:
        lines = [ln for ln in f.read().split("\n") if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"Expected 3 lines in {path}, found {len(lines)}")
    ppg = np.array([float(x) for x in lines[0].split()])
    hr = np.array([float(x) for x in lines[1].split()])
    t = np.array([float(x) for x in lines[2].split()])
    if not (len(ppg) == len(hr) == len(t)):
        raise ValueError(f"Line lengths differ in {path}: {len(ppg)}, {len(hr)}, {len(t)}")
    if np.any(np.diff(t) < 0):
        raise ValueError(f"Time goes backwards in {path}. Check this subject by hand.")
    return ppg, hr, t


def fix_repeated_times(t):
    """
    The time line is saved in milliseconds, so sometimes two frames get the
    SAME time (found once in subjects 1, 4, 5 and 10). We spread each group of
    repeated times evenly up to the next different time. If the repeat is at
    the very end, we continue with the normal (median) step.
    Times that are already increasing are not changed.
    Returns the fixed times and how many values were changed.
    """
    t = t.astype(float).copy()
    step = np.median(np.diff(t)[np.diff(t) > 0])
    changed = 0
    i = 0
    n = len(t)
    while i < n - 1:
        if t[i + 1] != t[i]:
            i += 1
            continue
        j = i + 1                                   # find the end of the repeated group
        while j < n and t[j] == t[i]:
            j += 1
        k = j - i                                   # number of samples in the group
        if j < n:                                   # next different time exists
            new_vals = t[i] + (t[j] - t[i]) * np.arange(1, k) / k
        else:                                       # repeat at the very end
            new_vals = t[i] + step * np.arange(1, k)
        t[i + 1:j] = new_vals
        changed += k - 1
        i = j
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time line is still not increasing after the fix.")
    return t, changed


def count_frames(video_path):
    """Count frames by reading them (the header count is not always correct)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    header_fps = cap.get(cv2.CAP_PROP_FPS)
    n = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        n += 1
    cap.release()
    return n, header_fps


def resample_subject(subject_dir, out_dir):
    gt_path = os.path.join(subject_dir, "ground_truth.txt")
    vid_path = os.path.join(subject_dir, "vid.avi")
    ppg, hr, t = load_ground_truth(gt_path)
    n_frames, header_fps = count_frames(vid_path)

    # One ground-truth sample per frame is expected in UBFC DATASET_2.
    if n_frames != len(t):
        raise ValueError(
            f"Frames ({n_frames}) and ground-truth samples ({len(t)}) differ. "
            "Check this subject by hand before resampling."
        )

    t, times_fixed = fix_repeated_times(t)
    if times_fixed:
        print(f"[note] {times_fixed} repeated time value(s) fixed")
    t = t - t[0]                                   # start the time at 0
    real_fps = (len(t) - 1) / t[-1]
    new_t = np.arange(0.0, t[-1] + 1e-9, 1.0 / TARGET_FPS)

    # For every new time: index of the original frame just before it, and blend weight.
    left = np.searchsorted(t, new_t, side="right") - 1
    left = np.clip(left, 0, len(t) - 2)
    weight = (new_t - t[left]) / (t[left + 1] - t[left])
    weight = np.clip(weight, 0.0, 1.0)

    # ---- ground truth ----
    new_ppg = np.interp(new_t, t, ppg)
    nearest = np.where(weight < 0.5, left, left + 1)
    new_hr = hr[nearest]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ground_truth.txt"), "w") as f:
        f.write(" ".join(f"{v:.10e}" for v in new_ppg) + "\n")
        f.write(" ".join(f"{v:.10e}" for v in new_hr) + "\n")
        f.write(" ".join(f"{v:.10e}" for v in new_t) + "\n")

    # ---- video (streamed, only 2 original frames kept in memory) ----
    cap = cv2.VideoCapture(vid_path)
    ok, frame_a = cap.read()
    ok_b, frame_b = cap.read()
    if not ok or not ok_b:
        raise RuntimeError(f"Could not read the first frames of {vid_path}")
    h, w = frame_a.shape[:2]

    out_vid = os.path.join(out_dir, "vid.avi")
    writer = cv2.VideoWriter(out_vid, cv2.VideoWriter_fourcc(*"FFV1"), TARGET_FPS, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open the FFV1 (lossless) video writer.")

    loaded = 1          # index of frame_b in the original video
    first_written = None
    for k in range(len(new_t)):
        target_left = left[k]
        while loaded < target_left + 1:             # move forward until frame_b = left + 1
            frame_a = frame_b
            ok_b, frame_b = cap.read()
            if not ok_b:
                raise RuntimeError(f"Video ended early at frame {loaded + 1} in {vid_path}")
            loaded += 1
        wk = weight[k]
        if wk == 0.0:
            out = frame_a
        elif wk == 1.0:
            out = frame_b
        else:
            blend = (1.0 - wk) * frame_a.astype(np.float32) + wk * frame_b.astype(np.float32)
            out = np.clip(np.rint(blend), 0, 255).astype(np.uint8)
        if first_written is None:
            first_written = out.copy()
        writer.write(out)
    writer.release()
    cap.release()

    # ---- checks on the written files ----
    out_frames, out_header_fps = count_frames(out_vid)
    cap = cv2.VideoCapture(out_vid)
    _, first_back = cap.read()
    cap.release()
    lossless = bool(np.array_equal(first_back, first_written))

    _, _, t_check = load_ground_truth(os.path.join(out_dir, "ground_truth.txt"))
    return {
        "frames_before": n_frames,
        "fps_before": round(real_fps, 3),
        "header_fps_before": round(header_fps, 3),
        "frames_after": out_frames,
        "gt_samples_after": len(t_check),
        "fps_after": round((len(t_check) - 1) / t_check[-1], 3),
        "header_fps_after": round(out_header_fps, 3),
        "duration_s": round(t[-1], 3),
        "repeated_times_fixed": times_fixed,
        "lossless_check": lossless,
        "video_size_mb": round(os.path.getsize(out_vid) / 1e6, 1),
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Resample UBFC-rPPG subjects to exactly 30 fps.")
    parser.add_argument("--root", default=os.path.join(here, "DATASET_2"))
    parser.add_argument("--out", default=os.path.join(here, "DATASET_2_30fps"))
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"Dataset folder not found: {args.root}")
    subjects = args.subjects or sorted(
        [d for d in os.listdir(args.root) if d.startswith("subject")],
        key=lambda s: int(s.replace("subject", "")),
    )
    print(f"Subjects to resample: {len(subjects)}")

    os.makedirs(args.out, exist_ok=True)
    rows, failed = [], []
    for s in subjects:
        start = time.time()
        print(f"\n===== {s} =====")
        try:
            info = resample_subject(os.path.join(args.root, s), os.path.join(args.out, s))
        except Exception as e:                       # keep going, report at the end
            print(f"[FAILED] {s}: {e}")
            failed.append(s)
            continue
        info = {"subject": s, **info}
        rows.append(info)
        print(
            f"fps {info['fps_before']} -> {info['fps_after']}, "
            f"frames {info['frames_before']} -> {info['frames_after']}, "
            f"gt samples {info['gt_samples_after']}, lossless: {info['lossless_check']}, "
            f"{info['video_size_mb']} MB, {time.time() - start:.0f} s"
        )

    # Merge with an older report, so re-running a few subjects keeps the rest.
    report = os.path.join(args.out, "resample_report.csv")
    all_rows = {}
    if os.path.isfile(report):
        with open(report, "r", newline="") as f:
            for r in csv.DictReader(f):
                all_rows[r["subject"]] = r
    for r in rows:
        all_rows[r["subject"]] = r
    if all_rows:
        fields = list(rows[0].keys()) if rows else list(next(iter(all_rows.values())).keys())
        ordered = sorted(all_rows.values(), key=lambda r: int(r["subject"].replace("subject", "")))
        with open(report, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
            wr.writeheader()
            wr.writerows(ordered)
        print(f"Subjects in the report: {len(ordered)}")

    # ---- final summary ----
    bad = [r["subject"] for r in rows
           if r["frames_after"] != r["gt_samples_after"]
           or abs(r["fps_after"] - TARGET_FPS) > 0.01
           or not r["lossless_check"]]
    print("\n===== SUMMARY =====")
    print(f"Done: {len(rows)}   Failed: {len(failed)} {failed}")
    print(f"Subjects with a problem after resampling: {bad if bad else 'none'}")
    print(f"Total video size: {sum(r['video_size_mb'] for r in rows) / 1000:.1f} GB")
    print(f"Report saved to: {report}")


if __name__ == "__main__":
    main()