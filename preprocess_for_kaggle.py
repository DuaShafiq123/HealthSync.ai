"""
Prepare the 30 fps UBFC-rPPG copy for PhysNet training with rPPG-Toolbox.

Why our own script:
    The toolbox's main.py needs Python 3.8 and the mamba packages, which do not
    install on Windows. But the preparation (preprocessing) step only needs
    OpenCV and NumPy. So this script contains an EXACT copy of the toolbox's
    preprocessing code (rPPG-Toolbox commit b7500b8, files
    dataset/data_loader/BaseLoader.py and UBFCrPPGLoader.py).
    The output files are the same files the toolbox would make, so on Kaggle we
    set DO_PREPROCESS: False and the toolbox uses them directly.

Settings (same as the toolbox config UBFC-rPPG_UBFC-rPPG_PURE_PHYSNET_BASIC.yaml):
    DATA_TYPE DiffNormalized, LABEL_TYPE DiffNormalized, CHUNK_LENGTH 128,
    box made 1.5x larger, resized to 72 x 72.

ONE change from the toolbox: how the face is found.
    The toolbox runs a loose Haar detector on the FIRST frame only and keeps
    the biggest box. On subject1 that box was the chin, neck and chest, not
    the face. So we use the face rule of our own POS pipeline instead
    (D-03, D-05, function detect_face_box in rppg_step1_3.py): grey image,
    minNeighbors 5, faces at least 80 x 80, biggest face in each of the first
    30 frames, then the median box. After that, everything is the toolbox code.

Box check: outputs/frozen_run_log.txt has the face box our POS run used for
each subject (those boxes gave good POS results). The report compares our new
box with it. If a side moved more than 20 px, or there is no POS box
(subject4 is not in that log), the subject is listed under
"Subjects to check by picture": look at it with check_face_box.py.

Input:
    DATASET_2_30fps/subjectX/vid.avi and ground_truth.txt (from resample_to_30fps.py)

Output:
    PreprocessedData/<EXP_DATA_NAME>/subjectX_input0.npy, subjectX_label0.npy, ...
    PreprocessedData/preprocess_report.csv
    PreprocessedData_PhysNet.zip   (only with --zip; this is the file for Kaggle)

All 42 subjects are prepared. The train / validation / test split is chosen
later on Kaggle, so it does not change anything here.

Usage:
    python preprocess_for_kaggle.py                      # all subjects
    python preprocess_for_kaggle.py --subjects subject1  # one subject (quick test)
    python preprocess_for_kaggle.py --zip                # all subjects, then make the zip
"""

import argparse
import csv
import glob
import os
import re
import sys
import time
import zipfile
from math import ceil

import cv2
import numpy as np

# ---------------- Settings (do not change: they must match the Kaggle config) ----------------
DATASET_NAME = "UBFC-rPPG"
DATA_TYPE = ["DiffNormalized"]
DATA_AUG = ["None"]
LABEL_TYPE = "DiffNormalized"
CHUNK_LENGTH = 128
DO_CROP_FACE = True
BACKEND = "HC"
USE_LARGE_FACE_BOX = True
LARGE_BOX_COEF = 1.5
DO_DYNAMIC_DETECTION = False
DYNAMIC_DETECTION_FREQUENCY = 32
USE_MEDIAN_FACE_BOX = False
RESIZE_W = 72
RESIZE_H = 72

# Same folder name the toolbox builds by itself (config.py), so nothing has to be renamed.
EXP_DATA_NAME = "_".join([
    DATASET_NAME,
    "SizeW{0}".format(RESIZE_W),
    "SizeH{0}".format(RESIZE_W),  # the toolbox also uses W here
    "ClipLength{0}".format(CHUNK_LENGTH),
    "DataType{0}".format("_".join(DATA_TYPE)),
    "DataAug{0}".format("_".join(DATA_AUG)),
    "LabelType{0}".format(LABEL_TYPE),
    "Crop_face{0}".format(DO_CROP_FACE),
    "Backend{0}".format(BACKEND),
    "Large_box{0}".format(USE_LARGE_FACE_BOX),
    "Large_size{0}".format(LARGE_BOX_COEF),
    "Dyamic_Det{0}".format(DO_DYNAMIC_DETECTION),  # spelling copied from the toolbox
    "det_len{0}".format(DYNAMIC_DETECTION_FREQUENCY),
    "Median_face_box{0}".format(USE_MEDIAN_FACE_BOX),
])

# Our POS face rule (rppg_step1_3.detect_face_box and rppg_frozen.py, D-03 and D-05).
N_INIT_FRAMES = 30
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (80, 80)

# A box is "the same" as the POS box if no side moved more than this (pixels).
# Small moves are normal: the 30 fps frames are blends of the original frames.
POS_BOX_MAX_DIFF_PX = 20

# The toolbox uses its own copy of this file; it is byte-for-byte the same as OpenCV's.
HAAR_FILE = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")


# ---------------- Exact copies of the toolbox code ----------------
def read_video(video_file):
    """UBFCrPPGLoader.read_video: reads a video, returns frames (T, H, W, 3) in RGB."""
    VidObj = cv2.VideoCapture(video_file)
    VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
    success, frame = VidObj.read()
    frames = list()
    while success:
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame)
        frames.append(frame)
        success, frame = VidObj.read()
    return np.asarray(frames)


def read_wave(bvp_file):
    """UBFCrPPGLoader.read_wave: reads line 1 (the PPG wave) of ground_truth.txt."""
    with open(bvp_file, "r") as f:
        str1 = f.read()
        str1 = str1.split("\n")
        bvp = [float(x) for x in str1[0].split()]
    return np.asarray(bvp)


def find_faces(frame_rgb, detector):
    """All face boxes in one frame, with the same settings as our POS pipeline."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)  # same grey image as BGR2GRAY on the BGR frame
    return detector.detectMultiScale(gray, scaleFactor=HAAR_SCALE_FACTOR,
                                     minNeighbors=HAAR_MIN_NEIGHBORS, minSize=HAAR_MIN_SIZE)


def find_face_box(frames, detector):
    """OUR face rule (not the toolbox's): biggest face in each of the first 30 frames,
    then the median box. Returns the box (x, y, w, h) before enlarging, plus counts."""
    dets = []
    no_face = 0
    many_faces = 0
    n_checked = min(N_INIT_FRAMES, len(frames))
    for i in range(n_checked):
        faces = find_faces(frames[i], detector)
        if len(faces) == 0:
            no_face += 1
            continue
        if len(faces) > 1:
            many_faces += 1
        dets.append(max(faces, key=lambda b: b[2] * b[3]))
    if not dets:
        raise ValueError(f"no face found in the first {n_checked} frames")
    box = np.median(np.array(dets, dtype=float), axis=0)
    return box, n_checked, no_face, many_faces


def enlarge_box(box, larger_box_coef):
    """Toolbox code (BaseLoader.face_detection, use_larger_box part).
    The box is kept as whole numbers, like the toolbox's face box."""
    face_box_coor = np.array(np.round(box), dtype=np.int32)
    face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
    face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
    face_box_coor[2] = larger_box_coef * face_box_coor[2]
    face_box_coor[3] = larger_box_coef * face_box_coor[3]
    return face_box_coor


def crop_resize(frames, face_region, width, height):
    """Toolbox code (BaseLoader.crop_face_resize) for one fixed face box."""
    total_frames, _, _, channels = frames.shape
    resized_frames = np.zeros((total_frames, height, width, channels))
    for i in range(0, total_frames):
        frame = frames[i]
        frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
        resized_frames[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return resized_frames


def diff_normalize_data(data):
    """BaseLoader.diff_normalize_data."""
    n, h, w, c = data.shape
    diffnormalized_len = n - 1
    diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
    diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    for j in range(diffnormalized_len):
        diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
    diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
    diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
    diffnormalized_data[np.isnan(diffnormalized_data)] = 0
    return diffnormalized_data


def diff_normalize_label(label):
    """BaseLoader.diff_normalize_label."""
    diff_label = np.diff(label, axis=0)
    diffnormalized_label = diff_label / np.std(diff_label)
    diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
    diffnormalized_label[np.isnan(diffnormalized_label)] = 0
    return diffnormalized_label


def chunk(frames, bvps, chunk_length):
    """BaseLoader.chunk."""
    clip_num = frames.shape[0] // chunk_length
    frames_clips = [frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
    bvps_clips = [bvps[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
    return np.array(frames_clips), np.array(bvps_clips)


def preprocess(frames, bvps, detector):
    """BaseLoader.preprocess with the PhysNet settings above (face box: our rule)."""
    box, n_checked, no_face, many_faces = find_face_box(frames, detector)
    face_box = enlarge_box(box, LARGE_BOX_COEF)
    frames = crop_resize(frames, face_box, RESIZE_W, RESIZE_H)
    data = list()
    for data_type in DATA_TYPE:
        f_c = frames.copy()
        if data_type == "DiffNormalized":
            data.append(diff_normalize_data(f_c))
        else:
            raise ValueError("Only DiffNormalized is used for PhysNet.")
    data = np.concatenate(data, axis=-1)
    if LABEL_TYPE == "DiffNormalized":
        bvps = diff_normalize_label(bvps)
    else:
        raise ValueError("Only DiffNormalized is used for PhysNet.")
    frames_clips, bvps_clips = chunk(data, bvps, CHUNK_LENGTH)
    face_info = {"frames_checked": n_checked, "frames_no_face": no_face, "frames_many_faces": many_faces,
                 "median_box": box}
    return frames_clips, bvps_clips, face_box, face_info


def save(frames_clips, bvps_clips, cached_path, filename):
    """BaseLoader.save_multi_process: same file names as the toolbox."""
    os.makedirs(cached_path, exist_ok=True)
    for i in range(len(bvps_clips)):
        np.save(cached_path + os.sep + "{0}_input{1}.npy".format(filename, str(i)), frames_clips[i])
        np.save(cached_path + os.sep + "{0}_label{1}.npy".format(filename, str(i)), bvps_clips[i])
    return len(bvps_clips)


# ---------------- Our own part: loop, checks, report, zip ----------------
def subject_number(path):
    return int(re.search(r"subject(\d+)", path).group(1))


def load_pos_boxes(log_path):
    """Face boxes our frozen POS run used, read from outputs/frozen_run_log.txt.
    Returns {subject: [x, y, w, h]}. These boxes gave good POS results, so our new
    box (before the 1.5x enlargement) should be almost the same."""
    boxes = {}
    if not os.path.exists(log_path):
        return boxes
    subject = None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"===== (subject\d+) =====", line.strip())
            if m:
                subject = m.group(1)
                continue
            m = re.search(r"face box: \[([^\]]+)\]", line)
            if m and subject:
                boxes[subject] = [float(v) for v in m.group(1).split()]
    return boxes


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Prepare UBFC 30 fps data for PhysNet (rPPG-Toolbox).")
    parser.add_argument("--data", default=os.path.join(here, "DATASET_2_30fps"))
    parser.add_argument("--out", default=os.path.join(here, "PreprocessedData"))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--zip", action="store_true", help="make PreprocessedData_PhysNet.zip at the end")
    parser.add_argument("--pos-log", default=os.path.join(here, "outputs", "frozen_run_log.txt"),
                        help="POS run log with the face box of each subject (for the box check)")
    args = parser.parse_args()

    detector = cv2.CascadeClassifier(HAAR_FILE)
    if detector.empty():
        sys.exit(f"Could not load the face detector file: {HAAR_FILE}")

    dirs = sorted(glob.glob(os.path.join(args.data, "subject*")), key=subject_number)
    if args.subjects:
        wanted = set(args.subjects)
        dirs = [d for d in dirs if os.path.basename(d) in wanted]
        missing = wanted - {os.path.basename(d) for d in dirs}
        if missing:
            sys.exit(f"Not found in {args.data}: {sorted(missing)}")
    if not dirs:
        sys.exit(f"No subject folders found in {args.data}")

    cached_path = os.path.join(args.out, EXP_DATA_NAME)
    pos_boxes = load_pos_boxes(args.pos_log)
    print(f"POS face boxes found for the box check: {len(pos_boxes)} subjects")
    print(f"Subjects to prepare: {len(dirs)}")
    print(f"Output folder: {cached_path}\n")

    report_path = os.path.join(args.out, "preprocess_report.csv")
    old_rows = {}
    if os.path.exists(report_path):  # keep rows of subjects done in an earlier run (same columns only)
        with open(report_path, newline="") as f:
            reader = csv.DictReader(f)
            if "diff_vs_pos_px" in (reader.fieldnames or []):
                old_rows = {r["subject"]: r for r in reader}
            else:
                print("Old report has the old face columns: starting a new report.\n")

    rows, failed, problems = [], [], []
    for d in dirs:
        name = os.path.basename(d)
        print(f"===== {name} =====")
        start = time.time()
        try:
            frames = read_video(os.path.join(d, "vid.avi"))
            bvps = read_wave(os.path.join(d, "ground_truth.txt"))
            if frames.ndim != 4 or len(frames) == 0:
                raise ValueError("video could not be read")
            if len(frames) != len(bvps):
                raise ValueError(f"frames ({len(frames)}) and PPG samples ({len(bvps)}) differ")

            # Remove old files of this subject first, so no extra old chunk stays behind.
            for old in glob.glob(os.path.join(cached_path, f"{name}_input*.npy")) + \
                    glob.glob(os.path.join(cached_path, f"{name}_label*.npy")):
                os.remove(old)

            frames_clips, bvps_clips, face_box, face_info = preprocess(frames, bvps, detector)
            n_chunks = save(frames_clips, bvps_clips, cached_path, name)
            has_nan = bool(np.isnan(frames_clips).any() or np.isnan(bvps_clips).any())
            # Box check against the POS box (the box before the 1.5x enlargement).
            pos_box = pos_boxes.get(name)
            if pos_box is not None:
                diff = int(round(max(abs(a - b) for a, b in zip(face_info["median_box"], pos_box))))
            else:
                diff = None
            notes, is_problem = [], False
            if diff is None:
                notes.append("no POS box to compare: check picture")
                is_problem = True
            elif diff > POS_BOX_MAX_DIFF_PX:
                notes.append(f"box is {diff} px away from the POS box: check picture")
                is_problem = True
            if face_info["frames_many_faces"] > 0:
                notes.append(f"2+ faces in {face_info['frames_many_faces']} of {face_info['frames_checked']} frames")
            if face_info["frames_no_face"] > 0:
                notes.append(f"no face in {face_info['frames_no_face']} of {face_info['frames_checked']} frames")
            if has_nan:
                notes.append("NaN in output")
                is_problem = True
            note = "; ".join(notes)
            if is_problem:
                problems.append(f"{name}: {note}")
            row = {
                "subject": name,
                "frames": len(frames),
                "chunks": n_chunks,
                "frames_used": n_chunks * CHUNK_LENGTH,
                "frame_size": f"{frames.shape[2]}x{frames.shape[1]}",
                "frames_checked": face_info["frames_checked"],
                "frames_no_face": face_info["frames_no_face"],
                "frames_many_faces": face_info["frames_many_faces"],
                "face_box_x_y_w_h": " ".join(str(int(v)) for v in face_box),
                "box_before_1.5x": " ".join(str(int(round(v))) for v in face_info["median_box"]),
                "pos_box": " ".join(str(int(round(v))) for v in pos_box) if pos_box is not None else "",
                "diff_vs_pos_px": "" if diff is None else diff,
                "note": note,
                "seconds": round(time.time() - start),
            }
            rows.append(row)
            print(f"frames {len(frames)}, chunks {n_chunks}, "
                  f"face: 2+ faces in {face_info['frames_many_faces']}, none in {face_info['frames_no_face']} "
                  f"of {face_info['frames_checked']} frames, "
                  f"box {row['box_before_1.5x']} (POS box {row['pos_box'] or 'none'}, diff {row['diff_vs_pos_px'] if row['diff_vs_pos_px'] != '' else '-'} px), "
                  f"{row['seconds']} s" + (f"  <-- {note}" if note else ""))
        except Exception as e:
            failed.append(name)
            print(f"[FAILED] {name}: {e}")
        print()

    # Merge with older rows and save the report (sorted by subject number).
    for r in rows:
        old_rows[r["subject"]] = r
    all_rows = sorted(old_rows.values(), key=lambda r: subject_number(r["subject"]))
    os.makedirs(args.out, exist_ok=True)
    fields = ["subject", "frames", "chunks", "frames_used", "frame_size", "frames_checked",
              "frames_no_face", "frames_many_faces", "face_box_x_y_w_h", "box_before_1.5x", "pos_box",
              "diff_vs_pos_px", "note", "seconds"]
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    n_files = len(glob.glob(os.path.join(cached_path, "*.npy")))
    size_gb = sum(os.path.getsize(p) for p in glob.glob(os.path.join(cached_path, "*.npy"))) / 1e9
    print("===== SUMMARY =====")
    print(f"Done: {len(rows)}   Failed: {len(failed)} {failed}")
    print(f"Subjects to check by picture: {problems if problems else 'none'}")
    print(f"Subjects in report: {len(all_rows)}   Total chunks: {sum(int(r['chunks']) for r in all_rows)}")
    print(f"Files in output folder: {n_files}   Size: {size_gb:.1f} GB")
    print(f"Report saved to: {report_path}")

    if args.zip:
        if failed:
            sys.exit("Zip NOT made, because some subjects failed. Fix them first.")
        if problems:
            print("\nNOTE: zip is made, but check the pictures of the subjects above before uploading it.")
        zip_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "PreprocessedData_PhysNet.zip")
        print(f"\nMaking zip (no compression, the numbers do not shrink much): {zip_path}")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            z.write(report_path, arcname="preprocess_report.csv")
            for p in sorted(glob.glob(os.path.join(cached_path, "*.npy"))):
                z.write(p, arcname=EXP_DATA_NAME + "/" + os.path.basename(p))
        print(f"Zip done: {os.path.getsize(zip_path) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
