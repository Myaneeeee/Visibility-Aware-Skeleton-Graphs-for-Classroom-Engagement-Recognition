import os
import glob
import csv

import torch

import transforms

from paths import EXCLUDED_WINDOWS_FILE

CLASS_NAMES = ["Focused", "Distracted"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# The time dimension the NTU-60 backbone was pretrained on. Windows are stored at
# their clip's native frame count so the annotation UI stays frame-aligned with the
# video; the model-facing loader is what normalizes them to this length.
TARGET_T = 90


def make_window_id(w):
    return f"{w['video_name']}_id_{w['track_id']}_f{w['start_frame']}_f{w['end_frame']}"


def prenormalize_2d(keypoints, threshold=0.01):
    """pyskl PreNormalize2D(mode='auto'): person-centric, confidence-masked
    center+scale of a (3, T, V) window's x,y to ~[-1, 1]. conf channel untouched.
    """
    t = keypoints.clone().float()          # (3, T, V) = x, y, conf
    x, y, score = t[0], t[1], t[2]
    mask = score > threshold               # (T, V) confident joints
    if bool(mask.any()):
        xm, ym = x[mask], y[mask]
        x_max, x_min = float(xm.max()), float(xm.min())
        y_max, y_min = float(ym.max()), float(ym.min())
    else:
        x_max = x_min = y_max = y_min = 0.0

    if (x_max - x_min) > 10 and (y_max - y_min) > 10:
        t[0] = (x - (x_max + x_min) / 2.0) / (x_max - x_min) * 2.0
        t[1] = (y - (y_max + y_min) / 2.0) / (y_max - y_min) * 2.0

    maskout = score <= threshold           # zero x,y of low-confidence joints
    t[0][maskout] = 0.0
    t[1][maskout] = 0.0
    return t


def load_excluded_ids(excluded_csv=EXCLUDED_WINDOWS_FILE):
    """Window ids listed in the exclusion CSV, or an empty set if it does not exist."""
    if not os.path.exists(excluded_csv):
        return set()
    with open(excluded_csv, newline="") as f:
        return {row["window_id"] for row in csv.DictReader(f)}


def load_labeled_data(pt_dir, annotations_csv, excluded_csv=EXCLUDED_WINDOWS_FILE):
    """Load labeled windows, prenormalized and stretched to TARGET_T."""
    excluded = load_excluded_ids(excluded_csv)
    labels_dict = {}
    with open(annotations_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) == 2 and row[1] in CLASS_TO_IDX and row[0] not in excluded:
                labels_dict[row[0]] = CLASS_TO_IDX[row[1]]

    tensors, labels, video_names, window_ids = [], [], [], []
    for pt_path in sorted(glob.glob(os.path.join(pt_dir, "*_windows.pt"))):
        for w in torch.load(pt_path, weights_only=False):
            wid = make_window_id(w)
            if wid not in labels_dict:
                continue
            t = prenormalize_2d(w["keypoints"].float())
            if t.shape[1] != TARGET_T:
                t = transforms.resample_length(t, out_len=TARGET_T)
            tensors.append(t)
            labels.append(labels_dict[wid])
            video_names.append(w["video_name"])
            window_ids.append(wid)

    if not tensors:
        raise ValueError("No labeled windows found.")
    return {"tensors": tensors, "labels": labels,
            "video_names": video_names, "window_ids": window_ids}
