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

These come from `outputs/analysis/overall.csv`. Unseen subjects = 36 UBFC subjects plus a separate run of subject4.

| Subjects | Reference | MAE (BPM) | RMSE | Pearson r | Windows within 5 BPM |
|---|---|---|---|---|---|
| Unseen (37) | oximeter wave | **3.62** | 6.61 | 0.93 | 78.8% |
| Unseen (37) | file HR, cleaned | 3.47 | 5.43 | 0.95 | 76.9% |
| Unseen (37) | wave, only where the two references agree | 2.83 | 5.13 | 0.96 | 82.4% |
| Debug (5) | oximeter wave | 0.90 | 1.29 | 0.98 | 100% |

For half of the unseen subjects, the average error is about 1.4 BPM or less (median of per-subject MAE). Most of the overall error comes from a few subjects. The references themselves are sometimes wrong, so read `outputs/analysis/reference_check_findings.md` before quoting any number.

## Setup

Requires Python 3.10 or newer.

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
python analyze_frozen.py --windows outputs/frozen/windows.csv outputs/frozen_subject4/windows.csv

# 3. Look at why a subject failed  ->  outputs/failures/   (images show faces: do NOT commit)
python inspect_failures.py --subjects subject25 subject26 subject38

# 4. Reference checks (need outputs/frozen/windows.csv + DATASET_2)
python check_subject.py            # set SUBJECT = "..." at the top first
python compare_wave_methods.py     # compares 4 ways of reading the oximeter wave
```

If DATASET_2 is somewhere else, add `--root <path>` to `rppg_frozen.py`, `analyze_frozen.py` and `inspect_failures.py`. `check_subject.py` and `compare_wave_methods.py` always look for `DATASET_2/` inside the project folder.

## Files

| File | What it does |
|---|---|
| `rppg_frozen.py` | The frozen pipeline: video → heart rate per window, plus both references |
| `rppg_step1_3.py` | Shared helpers: reading the video, loading ground truth, face box, skin regions |
| `analyze_frozen.py` | MAE / RMSE / r, file-HR cleaning, per-subject table, Bland-Altman plot |
| `inspect_failures.py` | One diagnostic figure per subject: face movement, brightness, counted beats |
| `check_subject.py` | Detailed reference check for one subject |
| `compare_wave_methods.py` | Compares 4 ways of counting beats in the oximeter wave |
| `explore.ipynb` | Early exploration notebook (subject1) |
| `test_cv.py` | Quick check that OpenCV and the face detector work |
| `outputs/` | Committed result tables, plots and logs (no face images) |
| `subject*_check.png` | Plots made by `check_subject.py` |

## Ground rules

1. **The method is frozen.** Do not change the video side because of test-subject results. The unseen subjects are the exam paper.
2. **Reference fixes** are allowed only if the same rule is applied to all 41 subjects, results are shown before and after, and the change is written down in `reference_check_findings.md`.
3. **Privacy.** Never commit the dataset, videos or any image that shows a face.

## Open work

See the "Next steps" section of `outputs/analysis/reference_check_findings.md`. In short:
- apply one cleaning rule to both references for all subjects (`clean_references.py`)
- use one file-HR rule everywhere (the scripts currently use 40–220 vs 40–180 BPM)
- find out why group (c) subjects fail
- test on a second dataset (for example PURE) before any method change
