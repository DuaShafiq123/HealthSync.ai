# Reference Check Findings – UBFC-rPPG (Frozen Run)

**Date:** 17 Nov 2026
**Pipeline checked:** frozen run (POS + peak counting, 30 s windows, first 3 s skipped)
**Results file used:** `outputs/frozen/windows.csv`
**Check script:** `check_subject.py` (change `SUBJECT = "..."` on line 9 to check another subject)

---

## 1. Short summary

- The **"wave" reference has a bug**. Its peak counter sometimes counts the small bump that comes after each heartbeat (the dicrotic notch) as an extra beat. This makes wave HR too high. It was very large for subjects 5 and 39 (about +20) and small for subjects 3 and 37 (about +2).
- A fix (adding a `prominence` rule to `find_peaks`) removes the extra peaks. It was tested on subjects 5, 3, 37 and 39.
- The **"file" reference (HR line in `ground_truth.txt`) is not always reliable.** It sometimes shows impossible values (group a), reads 7–9 bpm too high, or reacts late (subjects 3, 37).
- After fixing the references, **subjects 5, 3 and 37 are NOT failures of our method** (error below 1 bpm).
- **Subject 39 is only partly a reference problem.** After the fix, our method still has an error of about 4–5 bpm, so it moves to group (c).
- **The method (video side) was NOT changed.** Only the references were checked.

---

## 2. The fix tested for the wave reference

Old rule, as it really is in `rppg_frozen.py` (`peak_rate()` with `reference_prominence = 0.3`). It already has a prominence rule, but it is 0.3 × the **standard deviation** of the **band-passed** (0.7–4 Hz) wave, with a minimum gap of 60/200 = 0.3 s. That is too small to reject the dicrotic notch on subjects 5 and 39:
```python
w = bandpass(gt["ppg"], gt["fs"])            # 0.7-4 Hz
x = w_window - np.mean(w_window)
peaks, _ = find_peaks(x, distance=int(fs * 60 / 200), prominence=0.3 * np.std(x))
```

Fixed (only counts tall peaks):
```python
signal_range = np.percentile(w, 95) - np.percentile(w, 5)
peaks, _ = find_peaks(w, distance=int(fs * 0.33), prominence=0.3 * signal_range)
```

- `distance = 0.33 s` means at most 180 bpm.
- `prominence = 0.3 × signal range` means a peak must stick out clearly above the dips around it. Real beats do; the small bumps do not.
- **Note:** this fix was tested in `check_subject.py`, not yet in `rppg_frozen.py`. The fixed rule also runs on the **raw** (not band-passed) wave and uses the **range** (95th − 5th percentile), not the std. So it differs from the frozen rule in three ways (filtering, prominence scale, gap 0.33 s vs 0.3 s), not only by "adding prominence".

---

## 3. Results per subject

### Subject 5: wave bug (CONFIRMED, method OK)

| Check | Value |
|---|---|
| Video HR (all windows) | about 98–99.7 |
| File HR | about 98–100 |
| Old wave HR | about 118–124 (ratio 1.18–1.24 vs file in all 20 windows) |
| Samples per second | 29.58 (time is correct) |
| Duration | 52.37 s |
| Peaks in 3–33 s, OLD counting | 61 → 122.2 bpm |
| Peaks in 3–33 s, FIXED counting | 50 → **99.8 bpm** |
| File HR in 3–33 s | 99.9 bpm |
| Mean error vs old wave / vs file | 20.48 / 0.38 |

The plot showed about 11 extra red X marks on the small bump after the beats. With the fix, every X is on a real beat.

### Subject 3: file too high (CONFIRMED, method OK)

| Comparison | Mean error |
|---|---|
| Video vs old wave | 2.03 |
| Video vs **fixed wave** | **0.63** |
| Video vs file | 8.15 |

- File HR: min 92, max 108, mean 101.2. No impossible values (0 of 1801 outside 40–180).
- File HR is about 7–8 bpm higher than both video and fixed wave in every window.
- File HR moves in whole-number steps, so it is calculated by the device's own (unknown) method.

### Subject 37: file reacts late (CONFIRMED, method mostly OK)

| Comparison | Mean error |
|---|---|
| Video vs old wave | 1.65 |
| Video vs **fixed wave** | **0.74** |
| Video vs file | 6.54 |

- File HR: min 56, max 67, mean 63. No impossible values (0 of 2032).
- The real heart rate goes down from about 62 to about 55 (video and fixed wave agree). The file stays flat at about 64–65 and only drops near the end, so it reacts late.
- **Small method error:** in windows 32–38, video rises to about 58 while the wave stays at about 55 (error about 2–3 bpm).

### Subject 39: wave bug + real method error (PARTLY a method failure)

| Comparison | Mean error |
|---|---|
| Video vs old wave | 20.15 |
| Video vs **fixed wave** | **5.16** |
| Video vs file | 3.64 |

- Old wave was about 104–116 bpm, which is wrong. The fixed wave is about 83–85 and agrees with the file (about 85).
- Both references now agree, but **video is too high in windows 11–25** (about 91–92 vs a real value of about 84, around 8 bpm off).
- Windows 30–38 are fine (video about 85–86).
- **Conclusion:** subject 39 should move to group (c) as a small failure. Possible cause: movement, a light change or face tracking at about 11–55 s of the video (not checked yet).

---

## 4. Updated groups

### Group (a): file HR has impossible values (finger clip lost signal)
Subjects 11, 18, 20, 24. The file shows HR = 1.0 in some places.

| Subject | Error vs wave | Error vs file |
|---|---|---|
| subject20 | 5.18 | 91.39 |
| subject11 | 1.70 | 40.37 |
| subject24 | 1.34 | 34.15 |

**Planned rule:** file HR below 40 or above 180 counts as invalid. Do not use the file reference for those windows.

### Group (b): the two references disagree
| Subject | Reference problem | After fixing, is our method OK? |
|---|---|---|
| subject5 | Wave counted extra bumps | Yes (about 0.4) |
| subject3 | File about 7–8 too high | Yes (0.63) |
| subject37 | File reacts late | Mostly yes (0.74, small error at the end) |
| subject39 | Wave counted extra bumps | **No, partly** (still about 4–5) → move to group (c) |

### Group (c): real failures of our method (NOT re-checked in this round)
Subjects 25, 26, 38, 45, 18, a little 17 and 41, and now **39** (small).
- Subjects 25, 26 and 27 were recorded at about 23 fps (others about 30 fps). 25 and 26 are the worst, but 27 is not as bad, so fps alone may not explain it.

---

## 5. Rules we follow (fairness)

1. **Do not change the method (video side)** based on these test subjects. The test set is our "exam paper".
2. Fixing **reference mistakes** is allowed only if:
   - the same rule is applied to **all 41 subjects**,
   - results are shown **before and after** the fix, and
   - every change is written down (this file).
3. **Honesty note:** the wave fix rule was chosen after looking at test results. We will say this openly in the report.
4. Face images are **not pushed to Git** (privacy). The dataset stays out of Git.

---

## 6. Next steps

- [x] Check how `rppg_frozen.py` counts wave peaks: NOT the same as the snippet first written here (see section 2, corrected).
- [ ] Use one file-HR rule everywhere: `analyze_frozen.py` uses 40–220 BPM and keeps any window with at least one good sample; `check_subject.py` and `compare_wave_methods.py` use 40–180 BPM and need 80% good samples.
- [ ] Discuss the fix with the team and supervisor before changing anything (the pipeline is frozen).
- [ ] Write `clean_references.py`: apply the wave fix + the file 40–180 rule to all 41 subjects and save to `outputs/frozen_cleaned/` (the frozen results stay unchanged).
- [ ] Needs all 41 `ground_truth.txt` files (text only, no videos).
- [ ] Safety check: good subjects should barely change after the fix.
- [ ] Report a table with old results and cleaned results side by side.
- [ ] Failure analysis for group (c) using `inspect_failures.py` (locally, images not pushed), including subject 39 at about 11–55 s.
- [ ] Future method improvements: test on another dataset (e.g. PURE), then run UBFC only once at the end.

---

## Appendix: full logs from `check_subject.py`

### Subject 5 (first version of the check script, window 3–33 s)
```
Number of lines in file: 3
Lengths -> wave: 1550  hr: 1550  time: 1550
First 5 time values: [0.    0.033 0.067 0.102 0.132]
Duration: 52.37
Samples per second: 29.58

OLD counting:   Peaks found: 61   HR from peaks: 122.2   Average HR from file: 99.9
FIXED counting: Peaks found: 50   HR from peaks: 99.8    Average HR from file: 99.9
```

Subject 5 per-window results (from windows.csv, old counting):
```
start_s  video_hr  wave_hr  file_hr  ratio_wave_vs_file
   3     99.33    121.42    99.91    1.22
   4     99.32    121.14    99.97    1.21
   5     99.33    119.69    99.97    1.20
   6     99.56    119.53    99.96    1.20
   7     99.66    124.25    99.96    1.24
   8     99.66    117.78    99.95    1.18
   9     99.56    117.64    99.91    1.18
  10     99.32    118.73    99.86    1.19
  11     99.10    119.26    99.82    1.19
  12     99.56    118.86    99.70    1.19
  13     99.21    118.45    99.56    1.19
  14     99.22    118.45    99.44    1.19
  15     98.87    120.73    99.30    1.22
  16     98.64    119.58    99.17    1.21
  17     98.64    119.67    99.02    1.21
  18     98.64    120.91    98.85    1.22
  19     98.42    118.73    98.69    1.20
  20     98.19    118.32    98.48    1.20
  21     98.08    118.59    98.25    1.21
  22     97.83    118.04    98.01    1.20
Mean error vs wave: 20.48
Mean error vs file: 0.38
```

### Subject 3
```
Duration: 60.93 s    Samples per second: 29.54
File HR -> min: 92.0   max: 108.0   mean: 101.2
File HR values outside 40-180 (impossible): 0 out of 1801

start_s  video_hr  wave_hr  fixed_wave_hr  file_hr
   3     92.32    94.37     94.51     101.35
   4     92.23    94.37     94.66     101.71
   5     92.23    95.21     94.63     102.04
   6     92.34    96.20     94.63     102.34
   7     93.53    96.09     94.49     102.53
   8     93.85    95.81     94.95     102.62
   9     95.05    96.76     95.13     102.73
  10     95.78    96.91     95.85     102.87
  11     97.00    98.36     96.90     103.09
  12     97.34    99.04     97.51     103.36
  13     97.37    99.40     97.91     103.63
  14     97.23    99.26     97.64     103.89
  15     96.33    98.70     97.23     104.16
  16     95.94    98.02     96.47     104.39
  17     95.67    97.90     96.00     104.57
  18     95.72    97.68     95.81     104.78
  19     95.56    97.67     96.06     104.93
  20     96.40    97.80     96.15     104.98
  21     96.44    98.03     96.86     104.93
  22     96.81    98.82     97.02     104.80
  23     96.89    98.81     97.20     104.61
  24     96.44    98.59     97.00     104.39
  25     96.00    97.80     96.16     104.17
  26     95.67    97.78     95.51     103.95
  27     95.16    97.13     95.20     103.63
  28     94.83    96.87     94.73     103.29
  29     95.16    96.98     94.92     102.87
  30     95.45    97.35     95.28     102.38

Mean error video vs OLD wave:   2.03
Mean error video vs FIXED wave: 0.63
Mean error video vs file:       8.15
```

### Subject 37
```
Duration: 68.15 s    Samples per second: 29.8
File HR -> min: 56.0   max: 67.0   mean: 63.0
File HR values outside 40-180 (impossible): 0 out of 2032

start_s  video_hr  wave_hr  fixed_wave_hr  file_hr
   3     61.92    64.16     62.87     64.22
   4     61.64    63.79     62.39     64.31
   5     61.57    63.35     62.02     64.41
   6     61.00    63.18     61.56     64.50
   7     60.56    62.89     61.05     64.57
   8     59.94    62.09     60.22     64.64
   9     59.51    61.87     60.00     64.70
  10     59.30    61.45     59.84     64.75
  11     59.16    61.22     59.45     64.78
  12     58.74    61.00     59.29     64.85
  13     58.54    60.58     58.69     64.93
  14     58.20    60.23     58.44     64.99
  15     57.79    60.02     58.18     65.02
  16     57.39    59.40     57.79     65.06
  17     56.87    59.18     57.44     65.09
  18     56.64    58.76     56.96     65.18
  19     56.45    58.62     57.04     65.19
  20     56.25    58.49     56.68     65.16
  21     56.31    58.56     56.63     65.08
  22     56.31    58.28     56.47     64.98
  23     56.25    56.20     56.32     64.86
  24     55.99    56.01     55.91     64.73
  25     55.60    55.75     55.79     64.58
  26     54.61    55.11     55.13     64.43
  27     54.93    54.92     54.72     64.26
  28     54.74    54.99     54.69     64.08
  29     54.78    55.02     54.89     63.82
  30     55.09    54.95     54.86     63.55
  31     55.13    55.21     55.03     63.31
  32     56.45    55.21     55.02     63.07
  33     57.79    55.34     55.06     62.84
  34     57.73    55.41     55.31     62.62
  35     57.87    55.41     55.19     62.32
  36     57.80    55.74     55.64     62.05
  37     57.80    55.74     55.65     61.78
  38     58.15    55.75     55.46     61.51

Mean error video vs OLD wave:   1.65
Mean error video vs FIXED wave: 0.74
Mean error video vs file:       6.54
```

### Subject 39
```
Duration: 68.54 s    Samples per second: 29.71
File HR -> min: 82.0   max: 88.0   mean: 85.0
File HR values outside 40-180 (impossible): 0 out of 2037

start_s  video_hr  wave_hr  fixed_wave_hr  file_hr
   3     86.70   113.03     84.63     84.82
   4     88.66   113.59     84.73     84.86
   5     88.66   113.46     84.61     84.92
   6     88.25   111.27     84.41     84.99
   7     88.35   108.45     84.11     85.06
   8     87.95   106.41     84.07     85.13
   9     88.13   106.56     83.95     85.19
  10     87.75   106.16     83.87     85.26
  11     90.93   108.08     83.79     85.34
  12     91.93   106.16     83.84     85.44
  13     91.83   107.34     83.97     85.54
  14     91.93   108.08     83.79     85.64
  15     91.83   111.14     83.78     85.71
  16     91.89   112.83     83.67     85.73
  17     91.51   109.62     83.52     85.70
  18     91.68   111.52     83.47     85.67
  19     91.51   107.59     83.47     85.60
  20     91.68   106.62     83.53     85.56
  21     89.59   108.27     83.56     85.49
  22     91.41   107.83     83.53     85.44
  23     91.36   109.12     83.41     85.39
  24     91.20   111.14     83.50     85.36
  25     91.46   109.37     83.48     85.32
  26     89.07   106.28     83.53     85.30
  27     87.75   106.53     83.65     85.31
  28     87.92   106.28     83.77     85.31
  29     87.65   103.50     83.54     85.31
  30     86.30   103.77     83.30     85.30
  31     85.03   104.72     82.95     85.27
  32     84.93   104.72     82.85     85.21
  33     85.03   107.89     82.85     85.14
  34     85.41   109.08     83.07     85.04
  35     85.71   112.42     83.50     84.98
  36     86.13   115.73     83.96     84.95
  37     86.30   115.25     84.05     84.95
  38     86.40   115.52     84.31     84.95

Mean error video vs OLD wave:   20.15
Mean error video vs FIXED wave: 5.16
Mean error video vs file:       3.64
```

---

## Addendum: reproduction and failure figures (17 Nov 2026, second machine)

**Reproduction.** `rppg_frozen.py` was re-run on a second laptop (macOS, Python 3.15, pandas 3.1) on the 12 subjects whose videos were downloaded (5, 16, 17, 18, 34, 35, 36, 38, 39, 41, 42, 49). All 415 windows match `outputs/frozen/windows.csv` exactly (difference 0.0 for video, wave and file HR). `analyze_frozen.py` on the official ground-truth files reproduces `overall.csv`, `per_subject.csv` and `windows_analysis.csv` byte for byte.

**One mismatch.** `compare_wave_methods.py` reproduces every row except **subject24** (for example, count_vs_file is 4.19 instead of 4.03). The script logic was not changed, so the local `DATASET_2/subject24/ground_truth.txt` may differ from the official file, or the CSV came from an earlier version of the script. Please check.

**Failure figures** (`inspect_failures.py`, kept locally, not committed: they show faces):

| Subject | What the figure shows | Likely cause |
|---|---|---|
| subject38 | Face drifts slowly by about 32 px over the video (fixed box); forehead box sits on the hair fringe; skin brightness falls from 130 to 115. Video misses beats and is about 10 BPM low. | Slow drift + bad forehead region |
| subject41 | Sudden head movements at about 10, 27 and 60 s (up to 100 px) with brightness jumps; cheek boxes sit on the glasses. The wave reference also counts notch peaks. | Movement + glasses; reference also unreliable |
| subject39 | Face steady (under 11 px). Wave counter marks the dicrotic notch as beats (visible in the zoom). | Confirms the wave bug; the video error at 11–25 s is not caused by movement |

**Dataset note** (from the UBFC readme and Agreement.xlsx): the faces of subjects 21 and 27 must not appear in any publication, presentation or report. The dataset authors evaluated against HR from the contact PPG **wave**, not the HR numbers in the file.
