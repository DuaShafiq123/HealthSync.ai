import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ===============================================================
# SETTINGS (final rules, same as compare_wave_methods.py)
# ===============================================================
WINDOW_S = 30                    # window length in seconds
MIN_BEAT_GAP_S = 0.33            # at most 180 beats per minute
PROMINENCE_FACTOR = 0.3          # peak must stick out by 30% of signal range
KEEP_LOW, KEEP_HIGH = 0.6, 1.6   # clean mean: keep intervals between 0.6x and 1.6x the median

HR_MIN, HR_MAX = 40, 180         # file HR outside this range is impossible
MIN_VALID_FRACTION = 0.8         # file window needs at least 80% good values

# Extra file rule - NOT agreed with the team yet, so it is OFF by default
EXTRA_FILE_RULE = False          # True = also treat 127 and the seconds after a dropout as bad
AFTER_DROPOUT_S = 10             # seconds after a dropout to treat as bad (only if rule is on)

PASS_MAE = 5.0                   # reporting threshold for "pass" (agree this with the team)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Look for the dataset in these places (first one that exists is used)
DATA_DIR_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "DATASET_2"),   # Dua's laptop
    r"C:\data\DATASET_2",                    # Abeera's laptop
]
DATA_DIR = next((d for d in DATA_DIR_CANDIDATES if os.path.isdir(d)), None)
if DATA_DIR is None:
    print("Dataset folder not found. Checked:", DATA_DIR_CANDIDATES)
    raise SystemExit
print("Using dataset folder:", DATA_DIR)

# Read the MERGED run (all subjects). Made by merge_frozen_runs.py.
IN_DIR = os.path.join(SCRIPT_DIR, "outputs", "frozen_all")
IN_PATH = os.path.join(IN_DIR, "windows.csv")
if not os.path.exists(IN_PATH):
    print("Not found:", IN_PATH)
    print("Run this first:  python merge_frozen_runs.py")
    raise SystemExit
print("Using results file:", IN_PATH)

# Debug subjects (used to build the method) - read from the run's own config
with open(os.path.join(IN_DIR, "config_used.json")) as f:
    DEBUG_SUBJECTS = set(json.load(f).get("DEBUG_SUBJECTS", []))
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "frozen_cleaned")
os.makedirs(OUT_DIR, exist_ok=True)


# ===============================================================
# HELPER FUNCTIONS
# ===============================================================
def load_ground_truth(subject):
    path = os.path.join(DATA_DIR, subject, "ground_truth.txt")
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
    """Fixed count (used for the sensitivity check)."""
    if len(peaks) < 2:
        return np.nan
    return (len(peaks) - 1) / (tw[peaks[-1]] - tw[peaks[0]]) * 60


def hr_by_clean_mean(tw, peaks):
    """Clean mean (main method): remove strange intervals, then 60 / mean."""
    if len(peaks) < 3:
        return np.nan, 0
    intervals = np.diff(tw[peaks])
    med = np.median(intervals)
    keep = (intervals >= KEEP_LOW * med) & (intervals <= KEEP_HIGH * med)
    if keep.sum() < 3:
        return np.nan, int((~keep).sum())
    return 60 / np.mean(intervals[keep]), int((~keep).sum())


def file_valid_mask(hr, t):
    in_range = (hr >= HR_MIN) & (hr <= HR_MAX)
    valid = in_range.copy()
    if EXTRA_FILE_RULE:
        valid &= (hr != 127)
        bad_times = t[~in_range]
        if len(bad_times):
            idx = np.searchsorted(bad_times, t, side="left")
            prev_bad = np.where(idx > 0, bad_times[np.maximum(idx - 1, 0)], -np.inf)
            valid &= ~((t - prev_bad) <= AFTER_DROPOUT_S)
    return valid


def mae(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    return float(np.mean(np.abs(d))) if len(d) else np.nan


def rmse(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    return float(np.sqrt(np.mean(d ** 2))) if len(d) else np.nan


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


# ===============================================================
# MAIN LOOP: recalculate the references for every window
# ===============================================================
windows = pd.read_csv(IN_PATH)
windows["subject_id"] = windows["subject"].astype(str).str.replace("subject", "", regex=False).str.strip()
subject_ids = sorted(windows["subject_id"].unique(), key=lambda s: int(s) if s.isdigit() else 9999)

rows = []
missing = []

for sid in subject_ids:
    name = "subject" + sid
    sub = windows[windows["subject_id"] == sid]
    gt = load_ground_truth(name)
    if gt is None:
        missing.append(name)
        continue

    wave, hr, t = gt
    fs = (len(t) - 1) / t[-1]
    valid = file_valid_mask(hr, t)

    for _, row in sub.iterrows():
        start = row["start_s"]
        m = (t >= start) & (t < start + WINDOW_S)
        w, tw = wave[m], t[m]

        peaks = detect_peaks(w, fs) if len(w) > fs * 10 else np.array([], dtype=int)
        count_hr = hr_by_count(tw, peaks)
        clean_hr, removed = hr_by_clean_mean(tw, peaks)

        frac = valid[m].mean() if m.any() else 0.0
        clean_file = hr[m & valid].mean() if frac >= MIN_VALID_FRACTION else np.nan

        rows.append({
            "subject": name,
            "start_s": start,
            "video_hr": row["video_hr"],
            "old_wave_hr": row["wave_hr"],
            "old_file_hr": row["file_hr"],
            "wave_hr_cleanmean": clean_hr,
            "wave_hr_count": count_hr,
            "intervals_removed": removed,
            "clean_file_hr": clean_file,
            "good_file_%": round(100 * frac, 1),
        })

df = pd.DataFrame(rows)

if df.empty:
    print("No subjects could be processed. Check the dataset folder.")
    raise SystemExit

# ===============================================================
# PER-SUBJECT RESULTS
# ===============================================================
per_rows = []
for name, s in df.groupby("subject", sort=False):
    mae_main = mae(s["video_hr"], s["wave_hr_cleanmean"])
    mae_count = mae(s["video_hr"], s["wave_hr_count"])
    status_main = "pass" if mae_main <= PASS_MAE else "fail"
    status_count = "pass" if mae_count <= PASS_MAE else "fail"
    per_rows.append({
        "subject": name,
        "group": "DEBUG" if name in DEBUG_SUBJECTS else "UNSEEN",
        "windows": len(s),
        "mae_old_wave": mae(s["video_hr"], s["old_wave_hr"]),
        "mae_old_file": mae(s["video_hr"], s["old_file_hr"]),
        "mae_wave_MAIN": mae_main,
        "mae_wave_count_SENS": mae_count,
        "mae_clean_file": mae(s["video_hr"], s["clean_file_hr"]),
        "usable_file_windows": int(s["clean_file_hr"].notna().sum()),
        "refs_agreement": mae(s["wave_hr_cleanmean"], s["clean_file_hr"]),
        "intervals_removed": int(s["intervals_removed"].sum()),
        "status_MAIN": status_main,
        "status_count": status_count,
        "status_changes": status_main != status_count,
    })
per = pd.DataFrame(per_rows)

# ===============================================================
# OVERALL RESULTS
# ===============================================================
comparisons = [
    ("video vs OLD wave (original)", "old_wave_hr"),
    ("video vs OLD file (original)", "old_file_hr"),
    ("video vs WAVE clean mean (MAIN)", "wave_hr_cleanmean"),
    ("video vs WAVE fixed count (sensitivity)", "wave_hr_count"),
    ("video vs CLEAN file (second check)", "clean_file_hr"),
]
overall_rows = []
df["group"] = np.where(df["subject"].isin(DEBUG_SUBJECTS), "DEBUG", "UNSEEN")
for group in ["UNSEEN", "DEBUG", "ALL"]:
    part = df if group == "ALL" else df[df["group"] == group]
    for label, col in comparisons:
        subj_mae = np.array([mae(s["video_hr"], s[col]) for _, s in part.groupby("subject")])
        overall_rows.append({
            "group": group,
            "subjects": int(part["subject"].nunique()),
            "comparison": label,
            "windows_used": int(part[col].notna().sum()),
            "MAE_all_windows": mae(part["video_hr"], part[col]),
            "RMSE_all_windows": rmse(part["video_hr"], part[col]),
            "Pearson_r": pearson(part["video_hr"], part[col]),
            "mean_MAE_per_subject": float(np.nanmean(subj_mae)),
            "subjects_MAE_le_threshold": int(np.sum(subj_mae <= PASS_MAE)),
            "subjects_with_data": int(np.sum(~np.isnan(subj_mae))),
        })
overall = pd.DataFrame(overall_rows)

# ===============================================================
# PRINT
# ===============================================================
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 50)

print("Subjects processed:", per["subject"].nunique(), "of", len(subject_ids))
if missing:
    print("MISSING ground_truth.txt (skipped):", ", ".join(missing))
print("Extra file rule (127 + after dropout):", "ON" if EXTRA_FILE_RULE else "OFF")

print("\n" + "=" * 90)
print("OVERALL RESULTS  (UNSEEN = fair test; DEBUG = used to build the method; report UNSEEN as the main result)")
print("=" * 90)
print(overall.round(2).to_string(index=False))

print("\n" + "=" * 90)
print("PER SUBJECT (sorted by MAIN error, worst first)")
print("=" * 90)
print(per.sort_values("mae_wave_MAIN", ascending=False).round(2).to_string(index=False))

print("\n" + "=" * 90)
print("SENSITIVITY CHECK: does pass/fail change between clean mean and fixed count?")
print("=" * 90)
changed = per[per["status_changes"]]
if changed.empty:
    print("No subject changes status. The results do not depend on this choice.")
else:
    print(changed[["subject", "mae_wave_MAIN", "mae_wave_count_SENS",
                   "status_MAIN", "status_count"]].round(2).to_string(index=False))

# ===============================================================
# SAVE
# ===============================================================
df.round(3).to_csv(os.path.join(OUT_DIR, "windows_cleaned.csv"), index=False)
per.round(3).to_csv(os.path.join(OUT_DIR, "per_subject_cleaned.csv"), index=False)
overall.round(3).to_csv(os.path.join(OUT_DIR, "overall_cleaned.csv"), index=False)

settings = {
    "window_s": WINDOW_S, "min_beat_gap_s": MIN_BEAT_GAP_S,
    "prominence_factor": PROMINENCE_FACTOR,
    "clean_mean_keep_range": [KEEP_LOW, KEEP_HIGH],
    "file_hr_valid_range": [HR_MIN, HR_MAX],
    "file_min_valid_fraction": MIN_VALID_FRACTION,
    "extra_file_rule": EXTRA_FILE_RULE, "after_dropout_s": AFTER_DROPOUT_S,
    "pass_mae_threshold": PASS_MAE,
    "subjects_processed": per["subject"].tolist(),
    "subjects_missing_ground_truth": missing,
    "input_file": "outputs/frozen_all/windows.csv",
    "debug_subjects": sorted(DEBUG_SUBJECTS),
    "note": "Video HR is unchanged from the frozen run. Only references were recalculated.",
}
with open(os.path.join(OUT_DIR, "settings_used.json"), "w") as f:
    json.dump(settings, f, indent=2)

print("\nSaved in", OUT_DIR, ":")
print("  windows_cleaned.csv, per_subject_cleaned.csv, overall_cleaned.csv, settings_used.json")

# ===============================================================
# PLOT: per-subject error, old reference vs cleaned reference
# ===============================================================
p = per.sort_values("mae_wave_MAIN").reset_index(drop=True)
x = np.arange(len(p))
bw = 0.4
fig, ax = plt.subplots(figsize=(max(12, len(p) * 0.4), 5))
ax.bar(x - bw / 2, p["mae_old_wave"], bw, label="vs OLD wave")
ax.bar(x + bw / 2, p["mae_wave_MAIN"], bw, label="vs CLEANED wave (main)")
ax.axhline(PASS_MAE, color="gray", linestyle="--", label="Threshold (" + str(PASS_MAE) + " bpm)")
ax.set_xticks(x)
ax.set_xticklabels([n.replace("subject", "s") + ("*" if g == "DEBUG" else "")
                    for n, g in zip(p["subject"], p["group"])], rotation=90)
ax.set_ylabel("Mean absolute error (bpm)")
ax.set_title("Video HR error per subject: old vs cleaned reference (* = debug subject)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "mae_old_vs_cleaned.png"), dpi=120)
print("  mae_old_vs_cleaned.png")
plt.show()