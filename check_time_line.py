"""
Check the time line (line 3) of ground_truth.txt for problems.
Shows where the time stays the same or goes backwards.

Usage:
    python check_time_line.py subject1 subject4 subject5
"""

import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(here, "DATASET_2")
subjects = sys.argv[1:] or ["subject1", "subject4", "subject5"]

for s in subjects:
    path = os.path.join(root, s, "ground_truth.txt")
    with open(path, "r") as f:
        lines = [ln for ln in f.read().split("\n") if ln.strip()]
    t = np.array([float(x) for x in lines[2].split()])
    raw_text = lines[2].split()
    dt = np.diff(t)
    same = np.where(dt == 0)[0]
    back = np.where(dt < 0)[0]
    normal = dt[dt > 0]

    print(f"\n===== {s} =====")
    print(f"samples: {len(t)}, first time: {t[0]}, last time: {t[-1]}")
    print(f"normal step: median {np.median(normal):.5f} s, min {normal.min():.5f} s, max {normal.max():.5f} s")
    print(f"time stays the same: {len(same)} times")
    print(f"time goes backwards: {len(back)} times")
    print(f"example raw values (first 5): {raw_text[:5]}")
    for name, idx in (("SAME", same), ("BACKWARDS", back)):
        for i in idx[:5]:
            lo, hi = max(0, i - 2), min(len(t), i + 4)
            print(f"  {name} at sample {i}: " + ", ".join(raw_text[lo:hi]))
