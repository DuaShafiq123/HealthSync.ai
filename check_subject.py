import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ===============================================================
# CHANGE ONLY THIS LINE  (e.g. "subject20", "subject11", "subject24", "subject18")
# ===============================================================
SUBJECT = "subject24"
# ===============================================================

# Rules for the file HR
HR_MIN, HR_MAX = 40, 180      # values outside this range are impossible
MIN_VALID_FRACTION = 0.8      # a window needs at least 80% good file values

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------
# PART 1: results from windows.csv for this subject
# ---------------------------------------------------------------
df = pd.read_csv(os.path.join(SCRIPT_DIR, "outputs", "frozen", "windows.csv"))
ids = df["subject"].astype(str).str.replace("subject", "", regex=False).str.strip()
sub = df[ids == SUBJECT.replace("subject", "")].copy()

if sub.empty:
    print("No rows for", SUBJECT, "in windows.csv")
    raise SystemExit

pd.set_option("display.width", 200)
print("=== PART 1:", SUBJECT, "results from windows.csv ===")
print("Mean error vs wave:", round(sub["err_vs_wave"].abs().mean(), 2))
print("Mean error vs file:", round(sub["err_vs_file"].abs().mean(), 2))

# ---------------------------------------------------------------
# PART 2: read the ground truth file
# ---------------------------------------------------------------
GT_PATH = os.path.join(SCRIPT_DIR, "DATASET_2", SUBJECT, "ground_truth.txt")

if not os.path.exists(GT_PATH):
    print("\nFile not found:", GT_PATH)
    folder = os.path.join(SCRIPT_DIR, "DATASET_2", SUBJECT)
    if os.path.exists(folder):
        print("Files inside this folder:", os.listdir(folder))
    else:
        print("The folder DATASET_2\\" + SUBJECT + " does not exist.")
    raise SystemExit

lines = open(GT_PATH).read().strip().split("\n")
wave = np.array(lines[0].split(), dtype=float)
hr   = np.array(lines[1].split(), dtype=float)
t    = np.array(lines[2].split(), dtype=float)
t    = t - t[0]
fs   = (len(t) - 1) / t[-1]

valid = (hr >= HR_MIN) & (hr <= HR_MAX)

print("\n=== PART 2: file check ===")
print("Duration:", round(t[-1], 2), "s    Samples per second:", round(fs, 2))
print("File HR -> min:", round(hr.min(), 1), "  max:", round(hr.max(), 1))
print("Impossible file values (outside", HR_MIN, "-", HR_MAX, "):",
      int((~valid).sum()), "out of", len(hr),
      "(" + str(round(100 * (~valid).mean(), 1)) + "%)")
if (~valid).any():
    print("Impossible values happen between",
          round(t[~valid].min(), 1), "s and", round(t[~valid].max(), 1), "s")

# ---------------------------------------------------------------
# PART 3: fixed wave HR + clean file HR, window by window
# ---------------------------------------------------------------
def window_mask(start):
    return (t >= start) & (t < start + 30)

def fixed_wave_hr(start):
    m = window_mask(start)
    w, tw = wave[m], t[m]
    signal_range = np.percentile(w, 95) - np.percentile(w, 5)
    peaks, _ = find_peaks(w, distance=max(1, int(fs * 0.33)), prominence=0.3 * signal_range)
    if len(peaks) < 2:
        return np.nan, w, tw, peaks
    return (len(peaks) - 1) / (tw[peaks[-1]] - tw[peaks[0]]) * 60, w, tw, peaks

def clean_file_hr(start):
    m = window_mask(start)
    frac = valid[m].mean()
    if frac < MIN_VALID_FRACTION:
        return np.nan, frac
    return hr[m & valid].mean(), frac

sub["fixed_wave_hr"] = [fixed_wave_hr(s)[0] for s in sub["start_s"]]
cf = [clean_file_hr(s) for s in sub["start_s"]]
sub["clean_file_hr"] = [c[0] for c in cf]
sub["good_file_%"]   = [round(100 * c[1], 1) for c in cf]

print("\n=== PART 3: per window ===")
print(sub[["start_s", "video_hr", "wave_hr", "fixed_wave_hr",
           "file_hr", "clean_file_hr", "good_file_%"]].round(2).to_string(index=False))

dropped = sub["clean_file_hr"].isna().sum()
print("\nWindows where file is too broken to use:", dropped, "out of", len(sub))
print("Mean error video vs OLD wave:   ", round((sub["video_hr"] - sub["wave_hr"]).abs().mean(), 2))
print("Mean error video vs FIXED wave: ", round((sub["video_hr"] - sub["fixed_wave_hr"]).abs().mean(), 2))
print("Mean error video vs OLD file:   ", round((sub["video_hr"] - sub["file_hr"]).abs().mean(), 2))
print("Mean error video vs CLEAN file: ", round((sub["video_hr"] - sub["clean_file_hr"]).abs().mean(), 2),
      "(only windows where file is usable)")

# ---------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(14, 11))

# Plot 1: file HR over the whole recording, impossible values in red
ax[0].plot(t, hr, color="purple", label="File HR")
ax[0].plot(t[~valid], hr[~valid], "r.", markersize=4, label="Impossible values")
ax[0].set_title(SUBJECT + ": HR stored in the file (whole recording)")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("HR (beats per minute)")
ax[0].legend()

# Plot 2: wave in the window where the file is most broken
worst = sub.sort_values("good_file_%")["start_s"].iloc[0]
_, w, tw, peaks = fixed_wave_hr(worst)
ax[1].plot(tw, w, label="Wave")
ax[1].plot(tw[peaks], w[peaks], "rx", markersize=8, label="Counted peaks (fixed)")
ax[1].set_title(SUBJECT + ": wave in the window with the most broken file values ("
                + str(worst) + "-" + str(worst + 30) + " s)")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Wave value")
ax[1].legend()

# Plot 3: all HR values per window
ax[2].plot(sub["start_s"], sub["video_hr"], "o-", label="Video HR (our method)")
ax[2].plot(sub["start_s"], sub["fixed_wave_hr"], "^-", label="Wave HR (fixed counting)")
ax[2].plot(sub["start_s"], sub["file_hr"], "x--", label="File HR (old)")
ax[2].plot(sub["start_s"], sub["clean_file_hr"], "s-", label="File HR (clean)")
ax[2].set_title(SUBJECT + ": HR per 30 s window")
ax[2].set_xlabel("Window start (s)")
ax[2].set_ylabel("HR (beats per minute)")
ax[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, SUBJECT + "_check.png"), dpi=120)
print("\nPlot saved as", SUBJECT + "_check.png")
plt.show()