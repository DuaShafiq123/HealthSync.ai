"""
Check which face box preprocess_for_kaggle.py uses, by saving a picture.

It uses EXACTLY the same face rule as preprocess_for_kaggle.py (our POS rule,
D-05 / D-21): biggest face in each of the first 30 frames, median box,
then made 1.5x larger. On the first frame it draws:
    RED boxes   = every face the detector found in the first frame
    GREEN box   = the box that is really used for the whole video
It also saves the green area resized to 72 x 72 (made bigger so you can see
it), which is what the model sees.

Pictures are saved in PreprocessedData/face_check/ (this folder is in .gitignore,
because the pictures show faces: never commit or share them).

Usage:
    python check_face_box.py subject1
    python check_face_box.py subject1 subject4 subject5
"""

import os
import sys

import cv2
import numpy as np

from preprocess_for_kaggle import (HAAR_FILE, LARGE_BOX_COEF, N_INIT_FRAMES, RESIZE_H, RESIZE_W,
                                   enlarge_box, find_face_box, find_faces)


def read_first_frames(video, n):
    """First n frames in RGB (same colour order as the preparation script)."""
    cap = cv2.VideoCapture(video)
    frames = []
    while len(frames) < n:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(frames)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(here, "DATASET_2_30fps")
    out = os.path.join(here, "PreprocessedData", "face_check")
    os.makedirs(out, exist_ok=True)
    subjects = sys.argv[1:]
    if not subjects:
        sys.exit("Give at least one subject, for example: python check_face_box.py subject1")

    detector = cv2.CascadeClassifier(HAAR_FILE)
    for name in subjects:
        video = os.path.join(data, name, "vid.avi")
        frames = read_first_frames(video, N_INIT_FRAMES)
        if len(frames) == 0:
            print(f"{name}: could not read {video}\n")
            continue
        try:
            box, n_checked, no_face, many_faces = find_face_box(frames, detector)
        except ValueError as e:
            print(f"{name}: {e}\n")
            continue
        x, y, w, h = [int(v) for v in enlarge_box(box, LARGE_BOX_COEF)]

        first_rgb = frames[0]
        first_faces = find_faces(first_rgb, detector)
        picture = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR)
        for (fx, fy, fw, fh) in first_faces:
            cv2.rectangle(picture, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
        cv2.rectangle(picture, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(picture, f"{name}: red = found in frame 1 ({len(first_faces)}), green = used",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(out, f"{name}_boxes.png"), picture)

        crop = first_rgb[max(y, 0):min(y + h, first_rgb.shape[0]), max(x, 0):min(x + w, first_rgb.shape[1])]
        crop = cv2.resize(crop, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_AREA)
        crop = cv2.resize(crop, (RESIZE_W * 4, RESIZE_H * 4), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out, f"{name}_model_view.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        print(f"{name}: of the first {n_checked} frames, 2+ faces in {many_faces}, no face in {no_face}")
        for i, (fx, fy, fw, fh) in enumerate(first_faces):
            print(f"    red box {i + 1} (frame 1): x {fx}, y {fy}, width {fw}, height {fh}")
        print(f"    green box (used): x {x}, y {y}, width {w}, height {h}\n")
    print(f"Pictures saved in: {out}")


if __name__ == "__main__":
    main()
