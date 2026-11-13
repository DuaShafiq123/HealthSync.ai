"""
analyze_frozen.py - analysis of the frozen rPPG run (does NOT change the method)

It reads the windows.csv file(s) made by rppg_frozen.py and:
  1. Cleans the "file" reference: hr samples outside 40-220 BPM are oximeter
     dropouts (for example the value 1.0), so they are ignored in each window.
  2. Checks how much the two references (oximeter wave vs file hr) agree.
  3. Computes MAE, RMSE, Pearson r and "% windows within 5 BPM"
     for UNSEEN and DEBUG subjects separately.
  4. Saves tables and two plots in outputs/analysis.

The video heart rate (video_hr) is NOT recomputed. Only the references are checked.

Run from the project folder (with the .venv active):
  python analyze_frozen.py --root "C:\\data\\DATASET_2"
  python analyze_frozen.py --root "C:\\data\\DATASET_2" --windows outputs/frozen/windows.csv outputs/frozen_subject4/windows.csv
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rppg_frozen import CONFIG, DEBUG_SUBJECTS, subject_number
from rppg_step1_3 import load_ground_truth

VALID_HR = (40, 220)      # file hr samples outside this range are treated as dropouts
AGREE_BPM = 5             # the two references "agree" if they differ by 5 BPM or less


def clean_file_hr(windows, root):
    """Recompute the file reference per window, ignoring dropout samples."""
    out = []
    for s, part in windows.groupby("subject", sort=False):
        gt = load_ground_truth(os.path.join(root, s, "ground_truth.txt"))
        hr, t = gt["hr"], gt["time"]
        valid = (hr >= VALID_HR[0]) & (hr <= VALID_HR[1])
        part = part.copy()
        clean, bad_frac = [], []
        for a in part["start_s"]:
            m = (t >= a) & (t < a + CONFIG["win_sec"])     # same window rule as rppg_frozen.py
            mv = m & valid
            clean.append(hr[mv].mean() if mv.any() else np.nan)
            bad_frac.append(1 - mv.sum() / max(m.sum(), 1))
        part["file_hr_clean"] = clean
        part["file_dropout_frac"] = bad_frac
        out.append(part)
    return pd.concat(out, ignore_index=True)


def metrics(est, ref):
    """MAE, RMSE, Pearson r and % within 5 BPM, using only windows where both values exist."""
    m = ~(np.isnan(est) | np.isnan(ref))
    e, r = est[m], ref[m]
    if len(e) == 0:
        return {"n_windows": 0}
    err = e - r
    return {
        "n_windows": int(len(e)),
        "MAE": np.abs(err).mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "pearson_r": (np.corrcoef(e, r)[0, 1]
                      if len(e) > 2 and e.std() > 0 and r.std() > 0 else np.nan),
        "pct_within_5": 100 * (np.abs(err) <= 5).mean(),
    }


def main():
    ap = argparse.ArgumentParser(description="Analysis of the frozen rPPG run")
    ap.add_argument("--root", default="DATASET_2", help="folder with the subject folders")
    ap.add_argument("--windows", nargs="+", default=["outputs/frozen/windows.csv"],
                    help="one or more windows.csv files from rppg_frozen.py")
    ap.add_argument("--out", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    w = pd.concat([pd.read_csv(p) for p in args.windows], ignore_index=True)
    w = w.drop_duplicates(subset=["subject", "start_s"])
    w = clean_file_hr(w, args.root)
    w["ref_gap"] = (w["wave_hr"] - w["file_hr_clean"]).abs()
    w["refs_agree"] = w["ref_gap"] <= AGREE_BPM
    w["group"] = np.where(w["subject"].isin(DEBUG_SUBJECTS), "DEBUG", "UNSEEN")
    w.to_csv(os.path.join(args.out, "windows_analysis.csv"), index=False)

    # ---------- per subject ----------
    rows = []
    for s, p in w.groupby("subject"):
        rows.append({
            "subject": s,
            "group": p["group"].iloc[0],
            "windows": len(p),
            "file_dropout_windows": int((p["file_dropout_frac"] > 0).sum()),
            "ref_gap_mean": p["ref_gap"].mean(),
            "refs_agree_pct": 100 * p["refs_agree"].mean(),
            "mae_vs_wave": (p["video_hr"] - p["wave_hr"]).abs().mean(),
            "mae_vs_file_raw": (p["video_hr"] - p["file_hr"]).abs().mean(),
            "mae_vs_file_clean": (p["video_hr"] - p["file_hr_clean"]).abs().mean(),
        })
    per_subj = pd.DataFrame(rows)
    per_subj = per_subj.iloc[per_subj["subject"].map(subject_number).argsort()].round(2)
    per_subj.to_csv(os.path.join(args.out, "per_subject.csv"), index=False)
    print("\n===== PER SUBJECT =====")
    print(per_subj.to_string(index=False))

    # ---------- overall, UNSEEN and DEBUG kept separate ----------
    overall = []
    for g in ["UNSEEN", "DEBUG"]:
        p = w[w["group"] == g]
        n_subj = p["subject"].nunique()
        for ref_name, ref_col, mask in [
            ("vs wave", "wave_hr", np.ones(len(p), bool)),
            ("vs file (raw)", "file_hr", np.ones(len(p), bool)),
            ("vs file (clean)", "file_hr_clean", np.ones(len(p), bool)),
            ("vs wave, refs agree", "wave_hr", p["refs_agree"].values),
            ("vs file clean, refs agree", "file_hr_clean", p["refs_agree"].values),
        ]:
            q = p[mask]
            row = {"group": g, "subjects": n_subj, "reference": ref_name}
            row.update(metrics(q["video_hr"].values.astype(float), q[ref_col].values.astype(float)))
            # subject-level MAE (each subject counts once)
            subj_mae = q.groupby("subject").apply(
                lambda x: (x["video_hr"] - x[ref_col]).abs().mean())
            row["mean_of_subject_MAE"] = subj_mae.mean()
            row["median_of_subject_MAE"] = subj_mae.median()
            overall.append(row)
    overall = pd.DataFrame(overall).round(2)
    overall.to_csv(os.path.join(args.out, "overall.csv"), index=False)
    print("\n===== OVERALL =====")
    print(overall.to_string(index=False))

    # ---------- plot 1: MAE per subject ----------
    ps = per_subj.reset_index(drop=True)
    x = np.arange(len(ps))
    fig, ax = plt.subplots(figsize=(16, 4.5))
    ax.bar(x - 0.2, ps["mae_vs_wave"], width=0.4, label="MAE vs oximeter wave")
    ax.bar(x + 0.2, ps["mae_vs_file_clean"], width=0.4, label="MAE vs file hr (clean)")
    ax.axhline(5, color="gray", linestyle="--", linewidth=1, label="5 BPM")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s.replace('subject', 'S')}{'*' if g == 'DEBUG' else ''}"
                        for s, g in zip(ps["subject"], ps["group"])], rotation=90)
    ax.set_ylabel("MAE (BPM)")
    ax.set_title("MAE per subject (* = debug subject, used to build the method)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "mae_per_subject.png"), dpi=150)
    plt.close()

    # ---------- plot 2: Bland-Altman, one point per subject (UNSEEN only) ----------
    u = w[w["group"] == "UNSEEN"].groupby("subject")[["video_hr", "wave_hr"]].mean().dropna()
    mean_pair = (u["video_hr"] + u["wave_hr"]) / 2
    diff = u["video_hr"] - u["wave_hr"]
    bias, sd = diff.mean(), diff.std()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean_pair, diff)
    for lab, val, ls in [("bias", bias, "-"), ("+1.96 SD", bias + 1.96 * sd, "--"),
                         ("-1.96 SD", bias - 1.96 * sd, "--")]:
        ax.axhline(val, color="gray", linestyle=ls, linewidth=1)
        ax.text(ax.get_xlim()[1], val, f" {lab}: {val:.1f}", va="center")
    ax.set_xlabel("Mean of video and wave HR (BPM)")
    ax.set_ylabel("Video HR - wave HR (BPM)")
    ax.set_title("Bland-Altman, unseen subjects (one point per subject)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "bland_altman_unseen.png"), dpi=150)
    plt.close()

    print(f"\nBland-Altman (unseen, per subject): bias {bias:.2f} BPM, "
          f"limits {bias - 1.96 * sd:.2f} to {bias + 1.96 * sd:.2f} BPM")
    print(f"Files saved in: {args.out}")


if __name__ == "__main__":
    main()
