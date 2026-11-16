import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ===============================================================
# CHANGE ONLY THIS LINE  (e.g. "subject3", "subject37", "subject39")
# ===============================================================
SUBJECT = "subject39"
# ===============================================================

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
print(sub[["start_s", "video_hr", "wave_hr", "file_hr"]].round(2).to_string(index=False))
print("\nMean error vs wave:", round(sub["err_vs_wave"].abs().mean(), 2))
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
if len(lines) < 3:
    print("The file does not have 3 lines.")
    raise SystemExit

wave = np.array(lines[0].split(), dtype=float)   # line 1: the wave
hr   = np.array(lines[1].split(), dtype=float)   # line 2: HR from the file
t    = np.array(lines[2].split(), dtype=float)   # line 3: time of each sample
t    = t - t[0]
fs   = (len(t) - 1) / t[-1]

print("\n=== PART 2: file check ===")
print("Duration:", round(t[-1], 2), "s    Samples per second:", round(fs, 2))
print("File HR -> min:", round(hr.min(), 1), "  max:", round(hr.max(), 1), "  mean:", round(hr.mean(), 1))
bad = np.sum((hr < 40) | (hr > 180))
print("File HR values outside 40-180 (impossible):", bad, "out of", len(hr))

# ---------------------------------------------------------------
# PART 3: count wave peaks with the FIXED method, window by window
# ---------------------------------------------------------------
def count_hr(start):
    mask = (t >= start) & (t < start + 30)
    w, tw = wave[mask], t[mask]
    signal_range = np.percentile(w, 95) - np.percentile(w, 5)
    peaks, _ = find_peaks(w, distance=max(1, int(fs * 0.33)), prominence=0.3 * signal_range)
    if len(peaks) < 2:
        return np.nan, w, tw, peaks
    hr_value = (len(peaks) - 1) / (tw[peaks[-1]] - tw[peaks[0]]) * 60
    return hr_value, w, tw, peaks

sub["fixed_wave_hr"] = [round(count_hr(s)[0], 2) for s in sub["start_s"]]

print("\n=== PART 3: with fixed peak counting ===")
print(sub[["start_s", "video_hr", "wave_hr", "fixed_wave_hr", "file_hr"]].round(2).to_string(index=False))
print("\nMean error video vs OLD wave:  ", round((sub["video_hr"] - sub["wave_hr"]).abs().mean(), 2))
print("Mean error video vs FIXED wave:", round((sub["video_hr"] - sub["fixed_wave_hr"]).abs().mean(), 2))
print("Mean error video vs file:      ", round((sub["video_hr"] - sub["file_hr"]).abs().mean(), 2))

# ---------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(14, 11))

# Plot 1: the HR line stored in the file (whole recording)
ax[0].plot(t, hr, color="purple")
ax[0].set_title(SUBJECT + ": HR stored in the file (whole recording)")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("HR (beats per minute)")

# Plot 2: wave with counted peaks in the first window
first_start = sub["start_s"].iloc[0]
_, w, tw, peaks = count_hr(first_start)
ax[1].plot(tw, w, label="Wave")
ax[1].plot(tw[peaks], w[peaks], "rx", markersize=8, label="Counted peaks (fixed)")
ax[1].set_title(SUBJECT + ": wave with counted peaks (" + str(first_start) + "-" + str(first_start + 30) + " s)")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Wave value")
ax[1].legend()

# Plot 3: all HR values per window
ax[2].plot(sub["start_s"], sub["video_hr"], "o-", label="Video HR (our method)")
ax[2].plot(sub["start_s"], sub["wave_hr"], "s--", label="Wave HR (old counting)")
ax[2].plot(sub["start_s"], sub["fixed_wave_hr"], "^-", label="Wave HR (fixed counting)")
ax[2].plot(sub["start_s"], sub["file_hr"], "x-", label="File HR")
ax[2].set_title(SUBJECT + ": HR per 30 s window")
ax[2].set_xlabel("Window start (s)")
ax[2].set_ylabel("HR (beats per minute)")
ax[2].legend()

plt.tight_layout()
out_png = os.path.join(SCRIPT_DIR, SUBJECT + "_check.png")
plt.savefig(out_png, dpi=120)
print("\nPlot saved as", SUBJECT + "_check.png")
plt.show()