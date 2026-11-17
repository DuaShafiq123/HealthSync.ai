# HealthSync.ai: Camera Heart Rate (rPPG)

This project measures heart rate from an ordinary face video, with no contact. It is Module A of the HealthSync final-year project.

We test the method on the public **UBFC-rPPG (DATASET_2)** set: 41 subjects, each filmed with a webcam while a finger oximeter records the true pulse.

## Method (frozen)

All settings are in `CONFIG` in `rppg_frozen.py`. We chose them on 5 debug subjects and then froze them before testing.

1. **Face box.** Detect the face in the first 30 frames (OpenCV Haar cascade), take the median box, and keep it fixed.
2. **Skin colour.** Average the R, G and B values of the forehead and both cheeks in every frame.
3. **Pulse signal.** Combine R, G and B into one pulse signal with POS (Plane-Orthogonal-to-Skin, 1.6 s window).
4. **Filter.** Keep only heart-rate frequencies with a 0.7–4 Hz band-pass (42–240 BPM).
5. **Heart rate.** Count the beats (peaks) in 30 s windows that move forward 1 s at a time. The first 3 s of each video are skipped.
6. **Check against two references:**
   - **wave:** beats counted in the oximeter's pulse wave
   - **file:** the HR numbers stored in `ground_truth.txt`

Debug subjects (used to build the method, so not a fair test): `subject1, 3, 8, 13, 22`. Every other subject is **unseen**.

## Results

These come from `outputs/frozen_cleaned/overall_cleaned.csv` (made by `clean_references.py`).
The video heart rate is the frozen result and was **not** recalculated. Only the reference (the "true" heart rate) was cleaned: a beat-counting bug in the oximeter wave was fixed and impossible file values were removed. See `decision.md` (D-14 to D-17) and `outputs/analysis/reference_check_findings.md`.

| Subjects | Reference | MAE (BPM) | RMSE | Pearson r | Subjects with MAE ≤ 5 BPM |
|---|---|---|---|---|---|
| Unseen (37) | oximeter wave, original | 3.62 | 6.61 | 0.93 | 28 of 37 |
| Unseen (37) | **oximeter wave, cleaned (main result)** | **2.86** | 4.96 | 0.96 | 29 of 37 |
| Unseen (37) | oximeter wave, cleaned, simple beat count (check) | 2.75 | 4.84 | 0.96 | 29 of 37 |
| Unseen (36) | file HR, cleaned | 3.40 | 5.38 | 0.95 | 28 of 36 |
| Debug (5) | oximeter wave, cleaned | 0.76 | 1.15 | 0.98 | 5 of 5 |

- File HR row has 36 subjects because subject20 has no window with enough good file values.
- For half of the unseen subjects, the average error is 1.25 BPM or less (median of per-subject MAE). Most of the overall error comes from a few subjects (26, 25, 18, 38, 41, 45).
- `outputs/analysis/` is the **older** analysis made by `analyze_frozen.py` (file rule 40–220 BPM). It is kept for history only; quote the numbers above.

## Setup

Tested with Python 3.14.3 (Windows). The results were also reproduced with Python 3.15 on macOS. `requirements.txt` has the exact library versions.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download UBFC-rPPG DATASET_2 and place it in the project folder like this (it is git-ignored and must never be committed):

```
DATASET_2/
  subject1/
    vid.avi
    ground_truth.txt
  subject3/
  ...
```

## How to run

Run these from the project folder, in this order:

```bash
# 1. Frozen pipeline on every subject  ->  outputs/frozen/
python rppg_frozen.py
python rppg_frozen.py --subjects subject4 --out outputs/frozen_subject4   # one subject only

# 2. Metrics, reference cleaning and plots  ->  outputs/analysis/
python analyze_frozen.py --windows outputs/frozen/windows.csv outputs/frozen_subject4/windows.csv   # older analysis, kept for history

# 2b. Join all frozen runs into one file (42 subjects)  ->  outputs/frozen_all/
python merge_frozen_runs.py

# 2c. Clean the references (MAIN results)  ->  outputs/frozen_cleaned/
python clean_references.py

# 3. Look at why a subject failed  ->  outputs/failures/   (images show faces: do NOT commit)
python inspect_failures.py --subjects subject25 subject26 subject38

# 4. Reference checks (need outputs/frozen_all/windows.csv + DATASET_2)
python check_subject.py            # set SUBJECT = "..." at the top first
python compare_wave_methods.py     # compares 4 ways of reading the oximeter wave
```

If DATASET_2 is somewhere else, add `--root <path>` to `rppg_frozen.py`, `analyze_frozen.py` and `inspect_failures.py`. For `clean_references.py`, set the `HEALTHSYNC_DATA` environment variable first (PowerShell: `$env:HEALTHSYNC_DATA = "C:\path\to\DATASET_2"`). `check_subject.py` and `compare_wave_methods.py` always look for `DATASET_2/` inside the project folder.

## Files

| File | What it does |
|---|---|
| `rppg_frozen.py` | The frozen pipeline: video → heart rate per window, plus both references |
| `rppg_step1_3.py` | Shared helpers: reading the video, loading ground truth, face box, skin regions |
| `merge_frozen_runs.py` | Joins all frozen runs into one file (42 subjects): `outputs/frozen_all/` |
| `clean_references.py` | Recalculates the references with the fixed rules and makes the **main results**: `outputs/frozen_cleaned/` |
| `analyze_frozen.py` | Older analysis (MAE / RMSE / r, file rule 40–220 BPM, Bland-Altman plot); kept for history |
| `inspect_failures.py` | One diagnostic figure per subject: face movement, brightness, counted beats |
| `check_subject.py` | Detailed reference check for one subject |
| `compare_wave_methods.py` | Compares 4 ways of counting beats in the oximeter wave |
| `explore.ipynb` | Early exploration notebook (subject1) |
| `outputs/` | Committed result tables, plots and logs (no face images) |
| `outputs/reference_check/` | Plots and tables made by `check_subject.py` and `compare_wave_methods.py` |
| `decision.md` | Every important decision, why we made it, and open questions |

## Ground rules

1. **The method is frozen.** Do not change the video side because of test-subject results. The unseen subjects are the exam paper.
2. **Reference fixes** are allowed only if the same rule is applied to all 41 subjects, results are shown before and after, and the change is written down in `reference_check_findings.md`.
3. **Privacy.** Never commit the dataset, videos or any image that shows a face.

## Open work

Open decisions are listed at the end of `decision.md`. In short:
- get the supervisor's approval for the reference fixes before quoting 2.86 BPM as the main result
- decide the extra file rule (value 127 and 10 s after a dropout) and the 5 BPM pass line
- `analyze_frozen.py` still uses the old file rule (40–220 BPM); the main results use 40–180
- check subject24's `ground_truth.txt` against the official file
- find out why group (c) subjects fail (26, 25, 18, 38, 41, 45, 39)
- test on a second dataset (for example PURE) before any method change
