"""
merge_frozen_runs.py - put all frozen-run results (all subjects) into ONE place.

Why: the main frozen run (outputs/frozen) has 41 subjects. subject4 was not in it
and was run later on its own (outputs/frozen_subject4) with the SAME frozen settings.
This script joins them into outputs/frozen_all, so every later script reads one file.

Safety checks:
  - stops if the runs used different settings (config_used.json must be identical)
  - stops if the same subject appears in two runs

The original folders are NOT changed. The video heart rates are NOT recomputed.

Run from the project folder (with the .venv active):
  python merge_frozen_runs.py
"""
import os
import re
import json

import pandas as pd

# Runs to merge (add more folders here if more subjects are run separately later)
RUN_FOLDERS = [
    os.path.join("outputs", "frozen"),
    os.path.join("outputs", "frozen_subject4"),
]
OUT_FOLDER = os.path.join("outputs", "frozen_all")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def subject_number(name):
    m = re.search(r"(\d+)$", str(name))
    return int(m.group(1)) if m else 10 ** 9


def main():
    configs, frames = [], []
    for folder in RUN_FOLDERS:
        path = os.path.join(SCRIPT_DIR, folder)
        cfg_path = os.path.join(path, "config_used.json")
        win_path = os.path.join(path, "windows.csv")
        if not (os.path.exists(cfg_path) and os.path.exists(win_path)):
            print("Missing config_used.json or windows.csv in:", folder)
            raise SystemExit
        with open(cfg_path) as f:
            configs.append(json.load(f))
        w = pd.read_csv(win_path)
        w["source_run"] = folder.replace("\\", "/")
        frames.append(w)
        print(f"{folder}: {w['subject'].nunique()} subjects, {len(w)} windows")

    # Check 1: every run used exactly the same frozen settings
    for folder, cfg in zip(RUN_FOLDERS[1:], configs[1:]):
        if cfg != configs[0]:
            print("STOP: settings are different in", folder, "- these runs cannot be merged.")
            raise SystemExit
    print("Check 1 passed: all runs used the same frozen settings.")

    # Check 2: no subject appears in two runs
    seen = {}
    for folder, w in zip(RUN_FOLDERS, frames):
        for s in w["subject"].unique():
            if s in seen:
                print(f"STOP: {s} is in both {seen[s]} and {folder}.")
                raise SystemExit
            seen[s] = folder
    print("Check 2 passed: no subject is duplicated.")

    allw = pd.concat(frames, ignore_index=True)
    allw["_n"] = allw["subject"].map(subject_number)
    allw = allw.sort_values(["_n", "start_s"]).drop(columns="_n").reset_index(drop=True)

    # Summary per subject (same columns as rppg_frozen.py makes)
    debug = set(configs[0].get("DEBUG_SUBJECTS", []))
    g = allw.groupby("subject").agg(
        windows=("start_s", "count"),
        failed_windows=("video_hr", lambda x: int(x.isna().sum())),
        mae_vs_wave=("err_vs_wave", "mean"),
        mae_vs_file=("err_vs_file", "mean"),
        over5_vs_wave=("err_vs_wave", lambda x: int((x > 5).sum())),
        over5_vs_file=("err_vs_file", lambda x: int((x > 5).sum())),
        source_run=("source_run", "first"),
    )
    g["status"] = "ok"
    g["debug_subject"] = g.index.isin(debug)
    g = g.iloc[g.index.map(subject_number).argsort()].round(2)

    out = os.path.join(SCRIPT_DIR, OUT_FOLDER)
    os.makedirs(out, exist_ok=True)
    allw.to_csv(os.path.join(out, "windows.csv"), index=False)
    g.to_csv(os.path.join(out, "summary.csv"))
    merged_cfg = dict(configs[0])
    merged_cfg["MERGED_FROM"] = [f.replace("\\", "/") for f in RUN_FOLDERS]
    with open(os.path.join(out, "config_used.json"), "w") as f:
        json.dump(merged_cfg, f, indent=2)

    n_unseen = int((~g["debug_subject"]).sum())
    n_debug = int(g["debug_subject"].sum())
    print(f"\nMerged: {len(g)} subjects ({n_unseen} unseen + {n_debug} debug), {len(allw)} windows")
    print("Saved in", OUT_FOLDER, ": windows.csv, summary.csv, config_used.json")


if __name__ == "__main__":
    main()
