import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ===============================================================
# SETTINGS
# ===============================================================
SUBJECTS = ["subject3", "subject5", "subject11", "subject18",
            "subject20", "subject24", "subject37", "subject39"]

WINDOW_S = 30               # window length in seconds
HR_MIN, HR_MAX = 40, 180    # file HR outside this range is impossible
MIN_VALID_FRACTION = 0.8    # file window needs at least 80% good values
MIN_BEAT_GAP_S = 0.33       # at most 180 beats per minute
PROMINENCE_FACTOR = 0.3     # peak must stick out by 30% of signal range
KEEP_LOW, KEEP_HIGH = 0.6, 1.6   # keep intervals between 0.6x and 1.6x the median
FLAG_DIFF_BPM = 1.5         # flag windows where two methods differ by more than this

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "reference_check")
os.makedirs(OUT_DIR, exist_ok=True)

windows = pd.read_csv(os.path.join(SCRIPT_DIR, "outputs", "frozen", "windows.csv"))
ids = windows["subject"].astype(str).str.replace("subject", "", regex=False).str.strip()


# ===============================================================
# HELPER FUNCTIONS
# ===============================================================
def load_ground_truth(subject):
    path = os.path.join(SCRIPT_DIR, "DATASET_2", subject, "ground_truth.txt")
    if not os.path.exists(path):
        return None
    lines = open(path).read().strip().split("\n")
    wave = np.array(lines[0].split(), dtype=float)
    hr = np.array(lines[1].split(), dtype=float)
    t = np.array(lines[2].split(), dtype=float)
    return wave, hr, t - t[0]


def detect_peaks(w, fs):
    signal_range = np.percentile(w, 95) - np.percentile(w, 5)
    peaks, _ = find_peaks(w, distance=max(1, int(fs * MIN_BEAT_GAP_S)),
                          prominence=PROMINENCE_FACTOR * signal_range)
    return peaks


def hr_by_count(tw, peaks):
    """Method 2: (beats - 1) / time from first to last beat."""
    if len(peaks) < 2:
        return np.nan
    return (len(peaks) - 1) / (tw[peaks[-1]] - tw[peaks[0]]) * 60


def hr_by_median(tw, peaks):
    """Method 3: 60 / median interval between beats."""
    if len(peaks) < 3:
        return np.nan
    return 60 / np.median(np.diff(tw[peaks]))


def hr_by_clean_mean(tw, peaks):
    """Method 4: remove strange intervals, then 60 / mean of the rest.
    Returns (hr, number of intervals removed)."""
    if len(peaks) < 3:
        return np.nan, 0
    intervals = np.diff(tw[peaks])
    med = np.median(intervals)
    keep = (intervals >= KEEP_LOW * med) & (intervals <= KEEP_HIGH * med)
    if keep.sum() < 3:
        return np.nan, int((~keep).sum())
    return 60 / np.mean(intervals[keep]), int((~keep).sum())


def mean_abs(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    return round(float(np.mean(np.abs(d))), 2) if len(d) else np.nan


# ===============================================================
# MAIN LOOP
# ===============================================================
all_rows = []
summary = []

for subject in SUBJECTS:
    sub = windows[ids == subject.replace("subject", "")].copy()
    gt = load_ground_truth(subject)

    if sub.empty:
        print(subject, "-> no rows in windows.csv, skipped")
        continue
    if gt is None:
        print(subject, "-> ground_truth.txt not found in DATASET_2, skipped")
        continue

    wave, hr, t = gt
    fs = (len(t) - 1) / t[-1]
    valid = (hr >= HR_MIN) & (hr <= HR_MAX)

    for _, row in sub.iterrows():
        start = row["start_s"]
        m = (t >= start) & (t < start + WINDOW_S)
        w, tw = wave[m], t[m]

        peaks = detect_peaks(w, fs) if len(w) > fs * 10 else np.array([], dtype=int)
        count_hr = hr_by_count(tw, peaks)
        median_hr = hr_by_median(tw, peaks)
        clean_hr, removed = hr_by_clean_mean(tw, peaks)

        frac = valid[m].mean() if m.any() else 0.0
        clean_file = hr[m & valid].mean() if frac >= MIN_VALID_FRACTION else np.nan

        all_rows.append({
            "subject": subject,
            "start_s": start,
            "video_hr": row["video_hr"],
            "old_wave_hr": row["wave_hr"],
            "count_wave_hr": count_hr,
            "median_wave_hr": median_hr,
            "cleanmean_wave_hr": clean_hr,
            "intervals_removed": removed,
            "old_file_hr": row["file_hr"],
            "clean_file_hr": clean_file,
            "good_file_%": round(100 * frac, 1),
        })

    s = pd.DataFrame([r for r in all_rows if r["subject"] == subject])

    summary.append({
        "subject": subject,
        # Table A: wave methods vs clean file (decide with this)
        "old_vs_file": mean_abs(s["old_wave_hr"], s["clean_file_hr"]),
        "count_vs_file": mean_abs(s["count_wave_hr"], s["clean_file_hr"]),
        "median_vs_file": mean_abs(s["median_wave_hr"], s["clean_file_hr"]),
        "cleanmean_vs_file": mean_abs(s["cleanmean_wave_hr"], s["clean_file_hr"]),
        "usable_file_windows": int(s["clean_file_hr"].notna().sum()),
        # Table B: video vs each reference (information only)
        "video_vs_old": mean_abs(s["video_hr"], s["old_wave_hr"]),
        "video_vs_count": mean_abs(s["video_hr"], s["count_wave_hr"]),
        "video_vs_median": mean_abs(s["video_hr"], s["median_wave_hr"]),
        "video_vs_cleanmean": mean_abs(s["video_hr"], s["cleanmean_wave_hr"]),
        "video_vs_file": mean_abs(s["video_hr"], s["clean_file_hr"]),
        # Table C: warning signs
        "windows": len(s),
        "windows_with_removed": int((s["intervals_removed"] > 0).sum()),
        "max_removed_in_window": int(s["intervals_removed"].max()),
        "count_vs_cleanmean_differ": int(np.sum(
            np.abs(s["count_wave_hr"] - s["cleanmean_wave_hr"]) > FLAG_DIFF_BPM)),
    })

# ===============================================================
# PRINT AND SAVE
# ===============================================================
if not summary:
    print("\nNo subject could be checked (no rows in windows.csv or no ground_truth.txt "
          "in DATASET_2). Nothing to save.")
    raise SystemExit(1)

win_df = pd.DataFrame(all_rows).round(2)
sum_df = pd.DataFrame(summary)
pd.set_option("display.width", 250)

print("\n" + "=" * 70)
print("TABLE A: wave methods vs CLEAN FILE  (decide with this)")
print("Trusted judge only for: subject5, subject11, subject18, subject39")
print("=" * 70)
print(sum_df[["subject", "old_vs_file", "count_vs_file", "median_vs_file",
              "cleanmean_vs_file", "usable_file_windows"]].to_string(index=False))

print("\n" + "=" * 70)
print("TABLE B: VIDEO vs each reference  (information only, do NOT decide with this)")
print("=" * 70)
print(sum_df[["subject", "video_vs_old", "video_vs_count", "video_vs_median",
              "video_vs_cleanmean", "video_vs_file"]].to_string(index=False))

print("\n" + "=" * 70)
print("TABLE C: where did the clean-mean method remove intervals?")
print("=" * 70)
print(sum_df[["subject", "windows", "windows_with_removed",
              "max_removed_in_window", "count_vs_cleanmean_differ"]].to_string(index=False))

win_path = os.path.join(OUT_DIR, "wave_methods_windows.csv")
sum_path = os.path.join(OUT_DIR, "wave_methods_summary.csv")
win_df.to_csv(win_path, index=False)
sum_df.to_csv(sum_path, index=False)
print("\nSaved:", win_path)
print("Saved:", sum_path)

# ===============================================================
# PLOT: Table A as a bar chart (4 methods)
# ===============================================================
x = np.arange(len(sum_df))
bw = 0.2
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x - 1.5 * bw, sum_df["old_vs_file"], bw, label="Old counting")
ax.bar(x - 0.5 * bw, sum_df["count_vs_file"], bw, label="Fixed count")
ax.bar(x + 0.5 * bw, sum_df["median_vs_file"], bw, label="Median interval")
ax.bar(x + 1.5 * bw, sum_df["cleanmean_vs_file"], bw, label="Clean mean (new)")
ax.set_xticks(x)
ax.set_xticklabels(sum_df["subject"])
ax.set_ylabel("Mean difference vs clean file (bpm)")
ax.set_title("Which wave method agrees best with the clean file? (lower is better)")
ax.legend()
plt.tight_layout()
png_path = os.path.join(OUT_DIR, "wave_methods_vs_cleanfile.png")
plt.savefig(png_path, dpi=120)
print("Saved:", png_path)
plt.show()