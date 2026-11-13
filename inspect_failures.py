"""
inspect_failures.py - look at WHY some subjects failed (does NOT change the method)

For each subject it saves one figure (outputs/failures/<subject>.png) with:
  Row 1: first, middle and last frame with the FIXED face box and skin regions
         -> did the face stay inside the box?
  Row 2: face movement (how far the detected face is from the fixed box),
         brightness of the skin regions, and heart rate over time
         (video vs oximeter wave vs file hr)
  Row 3: 10 s of the video pulse signal and of the oximeter wave, taken from the
         window with the biggest error, with the counted beats marked
         -> were beats missed or counted twice?

The figures contain faces: do NOT push outputs/failures to GitHub.

Run from the project folder (with the .venv active), after analyze_frozen.py:
  python inspect_failures.py --root "C:\\data\\DATASET_2" --subjects subject25 subject26 subject38 subject45 subject34
"""
import os
import argparse

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from rppg_frozen import CONFIG, pos_signal, bandpass
from rppg_step1_3 import (load_ground_truth, detect_face_box,
                          skin_regions_from_box, mean_rgb_rect)

DETECT_EVERY = 5      # run the face detector on every 5th frame to measure movement
ZOOM_SEC = 10         # length of the zoomed pulse plots


def read_subject(video_path, cfg=CONFIG):
    """One pass over the video: fixed box (same rule as rppg_frozen.py), mean RGB,
    face movement and three sample frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
    box_center = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])

    keep = {0: None, n_total // 2: None, max(n_total - 1, 0): None}
    rgb, move_t, move_px = [], [], []
    idx, last_frame = 0, None
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        h_img, w_img = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        regions = skin_regions_from_box(box, w_img, h_img)
        means = [mean_rgb_rect(frame_rgb, r) for r in regions.values()]
        rgb.append([np.nanmean([m[k] for m in means]) for k in range(3)])

        if idx % DETECT_EVERY == 0:
            b = detect_face_box(frame_bgr)
            move_t.append(idx / fps)
            if b is None:
                move_px.append(np.nan)            # no face found in this frame
            else:
                c = np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])
                move_px.append(float(np.linalg.norm(c - box_center)))
        if idx in keep:
            keep[idx] = frame_rgb.copy()
        last_frame = frame_rgb
        idx += 1
    cap.release()

    frames = [f for f in keep.values() if f is not None]
    if len(frames) < 3 and last_frame is not None:
        frames.append(last_frame)
    t = np.arange(idx) / fps
    return {"fps": fps, "box": box, "rgb": np.array(rgb), "t": t,
            "move_t": np.array(move_t), "move_px": np.array(move_px),
            "frames": frames[:3], "frame_times": [0, (idx // 2) / fps, (idx - 1) / fps]}


def draw_box(frame_rgb, box):
    vis = frame_rgb.copy()
    h_img, w_img = vis.shape[:2]
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 2)
    for (x1, y1, x2, y2) in skin_regions_from_box(box, w_img, h_img).values():
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return vis


def count_peaks(x, fs, prominence, cfg=CONFIG):
    """Same beat rule as peak_rate() in rppg_frozen.py, but returns the peak positions."""
    x = x - np.mean(x)
    min_gap = int(fs * 60 / cfg["max_hr_bpm"])
    peaks, _ = find_peaks(x, distance=min_gap, prominence=prominence * np.std(x))
    return peaks


def inspect(subject, root, win_df, out_dir, cfg=CONFIG):
    d = read_subject(os.path.join(root, subject, "vid.avi"), cfg)
    gt = load_ground_truth(os.path.join(root, subject, "ground_truth.txt"))
    fs = d["fps"]
    v = bandpass(pos_signal(d["rgb"], fs, cfg["pos_win_sec"]), fs, cfg)
    wv = bandpass(gt["ppg"], gt["fs"], cfg)

    w = win_df[win_df["subject"] == subject].copy()
    w["center"] = w["start_s"] + cfg["win_sec"] / 2
    ref = "file_hr_clean" if "file_hr_clean" in w.columns else "file_hr"
    w["err"] = (w["video_hr"] - w["wave_hr"]).abs()

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 6)

    # Row 1: three frames
    for k, (f, ft) in enumerate(zip(d["frames"], d["frame_times"])):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        ax.imshow(draw_box(f, d["box"]))
        ax.set_title(f"{subject}: frame at {ft:.1f} s (fixed box)")
        ax.axis("off")

    # Row 2: movement, brightness, heart rate over time
    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(d["move_t"], d["move_px"])
    ax.axhline(0.1 * d["box"][2], color="gray", linestyle="--", linewidth=1,
               label="10% of box width")
    n_lost = int(np.isnan(d["move_px"]).sum())
    ax.set_title(f"Face movement from fixed box (face not found: {n_lost} checks)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (pixels)")
    ax.legend()

    ax = fig.add_subplot(gs[1, 2:4])
    ax.plot(d["t"], d["rgb"][:, 1])
    ax.axvline(cfg["skip_sec"], color="gray", linestyle="--", linewidth=1, label="skip ends")
    ax.set_title("Brightness of skin regions (mean green)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean G")
    ax.legend()

    ax = fig.add_subplot(gs[1, 4:6])
    ax.plot(w["center"], w["video_hr"], marker=".", label="Video HR")
    ax.plot(w["center"], w["wave_hr"], marker=".", label="Oximeter wave HR")
    ax.plot(w["center"], w[ref], marker=".", label=f"File HR ({ref})")
    ax.set_title("Heart rate per 30 s window")
    ax.set_xlabel("Window center (s)")
    ax.set_ylabel("BPM")
    ax.legend()

    # Row 3: 10 s zoom from the worst window
    if len(w):
        worst = w.loc[w["err"].idxmax()]
        a = float(worst["start_s"])
    else:
        worst, a = None, float(cfg["skip_sec"])
    for col, (sig, t_sig, f_sig, prom, name) in enumerate([
            (v, d["t"], fs, cfg["peak_prominence"], "Video pulse (POS, filtered)"),
            (wv, gt["time"], gt["fs"], cfg["reference_prominence"], "Oximeter wave (filtered)")]):
        m = (t_sig >= a) & (t_sig < a + cfg["win_sec"])
        seg, tseg = sig[m], t_sig[m]
        pk = count_peaks(seg, f_sig, prom, cfg)
        z = tseg < a + ZOOM_SEC
        ax = fig.add_subplot(gs[2, 3 * col:3 * col + 3])
        ax.plot(tseg[z], seg[z])
        pz = pk[tseg[pk] < a + ZOOM_SEC]
        ax.plot(tseg[pz], seg[pz], "x", color="red", label="counted beat")
        ax.set_title(f"{name}: first {ZOOM_SEC} s of worst window "
                     f"({len(pk)} beats in 30 s)")
        ax.set_xlabel("Time (s)")
        ax.legend()

    title = f"{subject} | fps {fs:.2f}"
    if worst is not None:
        title += (f" | worst window starts at {a:.0f} s: video {worst['video_hr']:.1f}, "
                  f"wave {worst['wave_hr']:.1f}, file {worst[ref]:.1f} BPM")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{subject}.png")
    plt.savefig(path, dpi=110)
    plt.close()

    print(f"{subject}: fps {fs:.2f}, face not found in {n_lost} of {len(d['move_px'])} checks, "
          f"max movement {np.nanmax(d['move_px']) if len(d['move_px']) and not np.all(np.isnan(d['move_px'])) else float('nan'):.1f} px "
          f"(box width {d['box'][2]:.0f} px) -> {path}")


def main():
    ap = argparse.ArgumentParser(description="Look at failed subjects")
    ap.add_argument("--root", default="DATASET_2")
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--windows", default="outputs/analysis/windows_analysis.csv",
                    help="windows_analysis.csv from analyze_frozen.py")
    ap.add_argument("--out", default="outputs/failures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    win_df = pd.read_csv(args.windows)

    for s in args.subjects:
        print(f"\n===== {s} =====")
        try:
            inspect(s, args.root, win_df, args.out)
        except Exception as e:
            print(f"FAILED: {e}")


if __name__ == "__main__":
    main()
