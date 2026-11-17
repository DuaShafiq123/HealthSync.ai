# Decision Log: HealthSync.ai, Module A (Camera Heart Rate / rPPG)

This file lists the **important decisions** we made and **why** we made them.
It does not list every small change. Git history already has that.

**Rules for this file**
- Add a decision only if it changes **what** we measure, **how** we measure it, **which data** we use, or **what we report**.
- Never delete an old decision. If we change our mind, mark the old one `REPLACED by D-xx` and add a new one.
- Always write the **reason** and the **evidence** (a number, a file or a plot).
- Status: `ACTIVE` = we follow it now · `REPLACED` = no longer used · `OPEN` = not decided yet.

**Template for a new decision** (copy this):
```
### D-xx: <short title>
- Date / who:
- Decision:
- Why:
- Evidence:
- Status:
```

---

## Part 1: Project setup

### D-01: Start with the camera heart-rate module, on the UBFC-rPPG dataset
- **Date / who:** early Nov 2026, team
- **Decision:** Build the face-video heart-rate module (Module A) first. Develop and test it on **UBFC-rPPG DATASET_2**. PURE, MMPD and our own phone recordings come later.
- **Why:** UBFC is public and widely used. Each subject has a webcam face video plus a finger pulse sensor (oximeter) recorded at the same time, so we have a "true" heart rate to compare against.
- **Status:** ACTIVE

### D-02: Never put the dataset or any face image on GitHub
- **Date / who:** 9 Nov 2026 (`.gitignore`), confirmed 13 Nov (Abeera, `inspect_failures.py` images not committed)
- **Decision:** `DATASET_2/`, all videos (`*.avi`, `*.mp4`) and every picture that shows a face stay on our laptops only.
- **Why:** These are real people's faces, and the dataset agreement does not let us share them. Extra rule from the UBFC readme and `Agreement.xlsx`: the faces of **subjects 21 and 27 must not appear in any report, presentation or publication**.
- **How:** `.gitignore` ignores `DATASET_2`, videos, and the folders where scripts save face pictures (`outputs/failures/`, `outputs/subject1/`). Notebook outputs are cleared before committing.
- **Status:** ACTIVE. ⚠️ **Partly fixed (17 Nov):** `explore.ipynb` had 5 face images saved inside it (subjects 1, 3, 8, 13, 22). The outputs were cleared on branch `cleanup/repo-tidy`, but the images are **still in old commits** (see O-09).

---

## Part 2: How the video method was built (on debug subjects only)

All tests below were done in `explore.ipynb` on the debug subjects (see D-10).

### D-03: Find the face with the OpenCV Haar detector; keep OpenCV below version 5
- **Date / who:** 9 Nov 2026, Dua
- **Decision:** Use OpenCV's Haar cascade face detector. `requirements.txt` has `opencv-python<5`.
- **Why:** Simple, fast, needs no extra model files. OpenCV 5.0.0 did not include `CascadeClassifier`, so the code broke on version 5.
- **Status:** ACTIVE

### D-04: Take skin colour from the forehead and both cheeks
- **Date / who:** 9 Nov 2026, Dua
- **Decision:** Average the R, G and B values inside three boxes (forehead, left cheek, right cheek) in every frame.
- **Why:** These areas have a lot of visible skin and blood flow, with little hair or eyes.
- **Status:** ACTIVE. (Known weak point: the forehead box can land on hair (subject 38) and cheek boxes on glasses (subject 41).)

### D-05: Keep the face box FIXED instead of following the face
- **Date / who:** 9–10 Nov 2026, Dua
- **Decision:** Detect the face in the first 30 frames, take the median box, and keep that box for the whole video.
- **Why:** When the box followed the face frame by frame, small detector jumps caused big sudden drops in the colour signal (notebook plot "old (smoothed box)": the signal jumps by about 60 units). With a fixed box the signal changed by only about 2 units. The jumps looked like fake beats.
- **Evidence:** `explore.ipynb` cells 2 and 7 (fixed vs gated/smoothed box).
- **Trade-off:** If the person moves a lot, the fixed box drifts off the face (subjects 38 and 41).
- **Status:** ACTIVE (frozen)

### D-06: Skip the first 3 seconds of every video
- **Date / who:** 10 Nov 2026, Dua
- **Decision:** Ignore the first 3 s.
- **Why:** At the start the camera is still adjusting its brightness, which adds a fake trend.
- **Evidence:** Subject 8, green channel: error went from 3.19 to 1.43 BPM when the first 3 s were skipped.
- **Status:** ACTIVE (frozen)

### D-07: Count beats (peaks) instead of using FFT
- **Date / who:** 10 Nov 2026, Dua
- **Decision:** Get heart rate by counting peaks in the signal, not by finding the strongest frequency (FFT).
- **Why:** Counting was much more accurate.
- **Evidence:** 3 subjects, 30 s windows, average error vs oximeter wave: **peaks 0.76 BPM vs FFT 2.95 BPM**.
- **Status:** ACTIVE (frozen)

### D-08: Use 30-second windows that move 1 second at a time
- **Date / who:** 10 Nov 2026, Dua
- **Decision:** Window length 30 s (not 15 s), step 1 s.
- **Why:** Longer windows contain more beats, so one missed or extra beat changes the result less.
- **Evidence:** Peak counting vs file HR: subject 1, 2.24 BPM (15 s) vs 1.34 (30 s); subject 8, 1.51 vs 1.10.
- **Trade-off:** We get one heart-rate value per 30 s, so fast changes are smoothed out.
- **Status:** ACTIVE (frozen)

### D-09: Check against TWO references: the oximeter wave and the file HR numbers
- **Date / who:** 10 Nov 2026, Dua
- **Decision:** Report error against (1) beats counted in the oximeter's pulse **wave** and (2) the **HR numbers** stored in `ground_truth.txt`.
- **Why:** The two did not agree. On subject 3 the wave and the file differed by about 10 BPM. If we used only one and it was wrong, we could not tell.
- **Evidence:** `explore.ipynb` cell 11 ("oximeter wave vs hr numbers": subject 3 = 10.01 BPM).
- **Status:** ACTIVE. (The way each reference is calculated was later changed, see D-14 and D-15.)

### D-10: Build the method on 5 "debug" subjects only, then FREEZE it
- **Date / who:** 10 Nov 2026, Dua
- **Decision:** Subjects **1, 3, 8, 13, 22** are debug subjects. All method choices were made on them. After that, the settings in `CONFIG` (`rppg_frozen.py`) are frozen and must not change because of results on other subjects.
- **Why:** For a fair test. The other subjects are the "exam paper". If we changed the method after seeing their results, our error numbers would look better than they really are.
- **Note:** Debug subjects are always reported separately and never mixed into the main result.
- **Status:** ACTIVE

### D-11: Use POS (not the green channel) with peak prominence 0.6
- **Date / who:** 10 Nov 2026, Dua
- **Decision:** Combine R, G and B with **POS** (1.6 s window), band-pass 0.7–4 Hz, and count a peak only if it is at least **0.6 × std** tall.
- **Why:** Subjects 13 and 22 failed badly with the green channel (the video found about 12–16% too many beats). POS fixed them. A prominence of 0.6 was the best of the values we tried.
- **Evidence (average error vs wave over the 5 debug subjects):**
  - green, peaks 0.3: 6.15 BPM → **POS, peaks 0.6: 0.89 BPM**
  - prominence sweep with POS: 0.3 = 2.51, 0.4 = 2.09, 0.5 = 1.87, **0.6 = 0.89**, 0.8 = 1.80, 1.0 = 2.48
  - band-pass to 3 Hz instead of 4 Hz made it worse (6.63)
- **Note:** CHROM was planned as a second method but has **not been tested** yet.
- **Status:** ACTIVE (frozen)

---

## Part 3: Running the test

### D-12: Run the frozen method once on all subjects; put everything in one results file
- **Date / who:** 13 Nov 2026, Abeera (run); 17 Nov 2026, Abeera (merge)
- **Decision:** Run `rppg_frozen.py` once on 41 subjects (`outputs/frozen/`). Subject 4 was run separately with the **same settings** (`outputs/frozen_subject4/`). Both are joined by `merge_frozen_runs.py` into **`outputs/frozen_all/` (42 subjects)**, and every later script should read this file.
- **Why:** One run with fixed settings is the fair test. One merged file means every script uses the same data. The merge script stops if the two runs used different settings or a subject appears twice.
- **Status:** ACTIVE. The raw frozen folders are never edited.

### D-13: Result tables must always split DEBUG and UNSEEN subjects
- **Date / who:** 13 Nov 2026 (Abeera, `analyze_frozen.py`), kept in `clean_references.py`
- **Decision:** The main result is for the **37 unseen subjects**. The 5 debug subjects are shown separately.
- **Why:** The debug subjects were used to build the method, so including them would make the result look better (debug error 0.90 vs unseen 3.62).
- **Status:** ACTIVE

---

## Part 4: Fixing the reference ("answer key")

### D-14: Fix how beats are counted in the oximeter wave
- **Date / who:** 17 Nov 2026, Dua (found and tested); Abeera applied it to all subjects (`clean_references.py`)
- **Decision:**
  - **Old rule** (in the frozen `CONFIG`, `reference_prominence = 0.3`): band-pass the wave 0.7–4 Hz, peak must be 0.3 × **std** tall, gap at least 0.3 s.
  - **New rule:** use the **raw** wave, peak must be 0.3 × **(95th − 5th percentile range)** tall, gap at least **0.33 s** (max 180 BPM).
- **Why:** After each real heartbeat the pulse wave has a small second bump (the dicrotic notch). The old rule often counted this bump as another beat, so the "true" heart rate came out too high. Our video was marked wrong when it was actually right.
- **Evidence:** Subject 5: old wave 122 BPM, new wave 99.8, file 99.9. Subject 39: video error vs old wave 20.15, vs new wave 5.16. Plots: `subject5_wave_peaks.png`, `subject39_check.png`.
- **Honesty note:** we found this bug by looking at subjects where the video looked bad, and we will say so in the report. To keep it fair: the bug is visible in the sensor wave alone, the new rule is used for **all 42 subjects**, and old and new results are both reported.
- **Status:** ACTIVE. ⚠️ **Supervisor approval still pending** (see O-01).

### D-15: Main wave heart rate = "clean mean"; fixed count only as a check
- **Date / who:** 17 Nov 2026, Dua (compared 4 ways in `compare_wave_methods.py`); used by Abeera in `clean_references.py`
- **Decision:** After finding the beats (D-14), remove beat gaps that are strange (shorter than 0.6× or longer than 1.6× the median gap), then heart rate = 60 / average gap. The simple "count the beats" result is also saved as a sensitivity check.
- **Why:** One missed or extra beat at a sensor glitch can pull a simple count off. Removing strange gaps protects against that. The two ways give almost the same overall result (unseen: 2.86 vs 2.75 BPM), so the choice does not change our conclusion.
- **Evidence:** `outputs/reference_check/wave_methods_summary.csv`, `outputs/frozen_cleaned/overall_cleaned.csv`.
- **Status:** ACTIVE. *(Please add here if there was another reason for choosing clean mean over median.)*

### D-16: Remove impossible values from the file HR
- **Date / who:** 17 Nov 2026, Dua (`check_subject.py`); Abeera (`clean_references.py`)
- **Decision:** File HR below **40** or above **180** BPM is treated as bad. A 30 s window uses the file reference only if **at least 80%** of its values are good. Otherwise that window has no file reference.
- **Why:** When the finger clip loses contact, the file stores impossible numbers (for example HR = 1.0 on subjects 11, 18, 20, 24). These made the error look like 34–91 BPM on those subjects.
- **Replaces:** the older rule in `analyze_frozen.py` (40–220 BPM, keep a window if it has even one good value).
- **Status:** ACTIVE in `clean_references.py`, `check_subject.py`, `compare_wave_methods.py`. ⚠️ `analyze_frozen.py` still uses the old rule (see O-03).

### D-17: Never overwrite frozen results; report old and cleaned results side by side
- **Date / who:** 17 Nov 2026, team (written in `reference_check_findings.md`)
- **Decision:** Cleaned results go to a new folder (`outputs/frozen_cleaned/`). The video heart-rate numbers are never recalculated. The report shows both:
  - unseen, original wave reference: **MAE 3.62 BPM**, r = 0.93
  - unseen, cleaned wave reference: **MAE 2.86 BPM**, r = 0.96 (main result)
  - unseen, cleaned file reference: MAE 3.40 BPM, r = 0.95
- **Why:** The reader can see exactly how much of the improvement comes from fixing the reference, and that the method itself did not change. Video HR in cleaned vs frozen results differs by at most 0.0005 BPM (rounding only).
- **Status:** ACTIVE

### D-18: Fix exact library versions in `requirements.txt`
- **Date / who:** 17 Nov 2026, Dua
- **Decision:** `requirements.txt` lists exact versions (`==`): opencv-python 4.14.0.94, numpy 2.5.4, pandas 3.1.1, matplotlib 3.11.2, scipy 1.18.1, tested with Python 3.14.3 on Windows.
- **Why:** Without versions, each person gets whatever is newest on the day they install, so results could change later without anyone noticing. With exact versions, the supervisor, examiners and teammates get the same libraries.
- **Evidence:** Mehroz reproduced all 415 windows exactly with different versions (Python 3.15, pandas 3.1, macOS), so the current results do not depend on these exact versions. This is a safety measure for the future.
- **Status:** ACTIVE. Abeera and Mehroz should test `pip install -r requirements.txt` once. If a version does not install on their Python, loosen only that line to a range (for example `numpy>=2,<3`).

---

## Open decisions (not decided yet)

| ID | Question | Notes |
|---|---|---|
| O-01 | Does the **supervisor** agree with the reference fixes (D-14, D-16)? | Must be asked **before** we use 2.86 BPM as the main number. |
| O-02 | Turn on the **extra file rule** (also drop value 127 and 10 s after a dropout)? | Exists in `clean_references.py` as `EXTRA_FILE_RULE = False`. Team has not agreed. |
| O-03 | Use **one file-HR rule everywhere**? | `analyze_frozen.py` still uses 40–220. For now the README marks `outputs/analysis/` as history only and quotes `outputs/frozen_cleaned/`. Decide whether to also update `analyze_frozen.py`. |
| O-04 | **"Pass" threshold**: is 5 BPM error the line between pass and fail? | `PASS_MAE = 5.0` in `clean_references.py`, comment says "agree this with the team". |
| O-05 | Why do **group (c)** subjects fail (26, 25, 18, 38, 41, 45, 39)? | 25 and 26 were recorded at about 23 fps (others about 30). 38 = slow drift and hair, 41 = movement and glasses. The method can **not** be changed on these subjects (D-10). Any new method must be built on another dataset first. |
| O-06 | Test **CHROM** and a second dataset (**PURE**) | Planned in D-01 and D-11, not started. |
| O-07 | **Reference sensor** for our own phone recordings | Polar H10 (about PKR 40,000) or finger oximeter (about PKR 4,000). |
| O-08 | **Subject 24** gives 4.19 on Mehroz's laptop vs 4.03 in our CSV | Check whether the local `ground_truth.txt` matches the official file. Subject 24 also changed the most after cleaning (1.34 → 2.58 BPM), together with subject 46 (0.48 → 2.22). |
| O-09 | **Remove the face images from old Git history** | Needs `git filter-repo` and a force push; everyone must clone again afterwards. Team must agree first. |
