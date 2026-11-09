import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
SUBJECT_DIR = "DATASET_2/subject1"
OUT_DIR = "outputs/subject1"
MAX_FRAMES = 300          # use 300 for a quick test, None for the full video

os.makedirs(OUT_DIR, exist_ok=True)
VIDEO_PATH = os.path.join(SUBJECT_DIR, "vid.avi")
GT_PATH = os.path.join(SUBJECT_DIR, "ground_truth.txt")


# ---------------- STEP 1: Inspect the video ----------------
def inspect_video(path):
    """Read FPS, resolution and frame count from the video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count_reported": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_sec_reported"] = (
        info["frame_count_reported"] / info["fps"] if info["fps"] else None)
    cap.release()
    return info


# ---------------- STEP 2: Load and check ground truth ----------------
def load_ground_truth(path):
    """Load ground_truth.txt and print its shape and timing.
    UBFC files usually have 3 rows: PPG signal, heart rate, time."""
    arr = np.loadtxt(path)
    print(f"[GT] raw shape: {arr.shape}")

    if arr.ndim == 2 and arr.shape[0] != 3 and arr.shape[1] == 3:
        arr = arr.T
        print(f"[GT] transposed -> {arr.shape}")

    if arr.ndim != 2 or arr.shape[0] < 3:
        print("[GT] WARNING: did not find 3 rows (ppg, hr, time). Check the format manually.")
        return {"raw": arr}

    gt = {"ppg": arr[0], "hr": arr[1], "time": arr[2]}
    n = arr.shape[1]
    duration = gt["time"][-1] - gt["time"][0]
    gt_fs_avg = (n - 1) / duration          # average rate over the whole recording
    print(f"[GT] samples: {n}")
    print(f"[GT] time range: {gt['time'][0]:.2f}s -> {gt['time'][-1]:.2f}s")
    print(f"[GT] average sampling frequency: {gt_fs_avg:.2f} Hz")
    print(f"[GT] HR min/mean/max: {gt['hr'].min():.1f} / {gt['hr'].mean():.1f} / {gt['hr'].max():.1f} BPM")
    gt["fs"] = gt_fs_avg
    return gt


# ---------------- STEP 3: Face box, skin regions, RGB signal ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    raise RuntimeError("Could not load the Haar cascade file")


def detect_face_box(frame_bgr):
    """Return the biggest face box (x, y, w, h), or None if no face is found."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])


def smooth_box(prev_box, new_box, alpha=0.3):
    """Blend the new box with the previous one so the regions do not jitter."""
    if prev_box is None:
        return np.array(new_box, dtype=np.float64)
    return (1 - alpha) * prev_box + alpha * np.array(new_box, dtype=np.float64)


def skin_regions_from_box(box, w_img, h_img):
    """Cut forehead and cheek rectangles out of the face box using fixed ratios.
    Returns a dict: name -> (x1, y1, x2, y2)."""
    x, y, w, h = box

    def rect(fx1, fy1, fx2, fy2):
        x1 = int(np.clip(x + fx1 * w, 0, w_img - 1))
        x2 = int(np.clip(x + fx2 * w, 0, w_img - 1))
        y1 = int(np.clip(y + fy1 * h, 0, h_img - 1))
        y2 = int(np.clip(y + fy2 * h, 0, h_img - 1))
        return (x1, y1, x2, y2)

    return {
        "forehead":    rect(0.30, 0.05, 0.70, 0.22),
        "left_cheek":  rect(0.15, 0.50, 0.38, 0.72),
        "right_cheek": rect(0.62, 0.50, 0.85, 0.72),
    }


def mean_rgb_rect(frame_rgb, rect):
    """Average R, G, B over a rectangle."""
    x1, y1, x2, y2 = rect
    patch = frame_rgb[y1:y2, x1:x2]
    if patch.size == 0:
        return (np.nan, np.nan, np.nan)
    m = patch.reshape(-1, 3).mean(axis=0)
    return float(m[0]), float(m[1]), float(m[2])


def save_debug_image(frame_bgr, box, regions, path):
    """Draw the face box and skin regions on one frame so we can check them by eye."""
    vis = frame_bgr.copy()
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 1)
    colors = {"forehead": (0, 255, 0), "left_cheek": (255, 0, 0), "right_cheek": (0, 0, 255)}
    for name, (x1, y1, x2, y2) in regions.items():
        cv2.rectangle(vis, (x1, y1), (x2, y2), colors[name], 2)
    cv2.imwrite(path, vis)


def extract_rgb_signals(video_path, fps, max_frames=None):
    """Go through the video frame by frame and save the mean RGB of each region."""
    cap = cv2.VideoCapture(video_path)
    rows = []
    idx, not_detected, debug_saved = 0, 0, False
    box = None

    while True:
        ok, frame_bgr = cap.read()
        if not ok or (max_frames is not None and idx >= max_frames):
            break

        h_img, w_img = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # OpenCV gives BGR

        new_box = detect_face_box(frame_bgr)
        detected = new_box is not None
        if detected:
            box = smooth_box(box, new_box)
        else:
            not_detected += 1      # reuse the last box, but count this frame

        row = {"frame": idx, "time_s": idx / fps, "face_detected": int(detected)}

        if box is not None:
            regions = skin_regions_from_box(box, w_img, h_img)
            for name, rect in regions.items():
                r, g, b = mean_rgb_rect(frame_rgb, rect)
                row[f"{name}_R"], row[f"{name}_G"], row[f"{name}_B"] = r, g, b
            for c in "RGB":
                row[f"combined_{c}"] = float(np.nanmean(
                    [row[f"{n}_{c}"] for n in regions]))
            if not debug_saved:
                save_debug_image(frame_bgr, box, regions,
                                 os.path.join(OUT_DIR, "debug_roi_frame0.png"))
                debug_saved = True
        rows.append(row)
        idx += 1

    cap.release()
    df = pd.DataFrame(rows)
    print(f"[RGB] frames processed: {idx}")
    print(f"[RGB] frames with no fresh face detection: {not_detected} "
          f"(last known box was reused for these)")
    return df


# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("=== STEP 1: Video info ===")
    info = inspect_video(VIDEO_PATH)
    for k, v in info.items():
        print(f"{k}: {v}")

    print("\n=== STEP 2: Ground truth info ===")
    gt = load_ground_truth(GT_PATH)

    print("\n=== STEP 3: Face + skin regions + RGB extraction ===")
    df = extract_rgb_signals(VIDEO_PATH, info["fps"], MAX_FRAMES)
    csv_path = os.path.join(OUT_DIR, "rgb_signals.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SAVE] {csv_path}")

    if "time" in gt:
        print(f"\n[CHECK] frames processed: {len(df)} | GT samples: {len(gt['time'])}")

    if "combined_G" in df.columns:
        plt.figure(figsize=(10, 3))
        plt.plot(df["time_s"], df["combined_G"])
        plt.title("Mean green channel (combined regions) vs time")
        plt.xlabel("Time (s)")
        plt.ylabel("Mean G")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "green_signal.png"), dpi=120)
        plt.show()