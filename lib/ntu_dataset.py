import re
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

import transforms
from common import prenormalize_2d

_SUBJECT_RE = re.compile(r"P(\d+)")


def parse_subject(frame_dir):
    m = _SUBJECT_RE.search(frame_dir)
    if not m:
        raise ValueError(f"no subject id in {frame_dir!r}")
    return int(m.group(1))


def select_primary_person(keypoint, keypoint_score):
    """keypoint: (M,T,V,2), keypoint_score: (M,T,V). Pick person with max mean
    score (mean over T,V folds in confidence AND completeness; padded persons
    score ~0)."""
    kp = np.asarray(keypoint, dtype=np.float32)
    sc = np.asarray(keypoint_score, dtype=np.float32)
    if kp.ndim == 3:                      # already single-person (T,V,2)
        return kp, sc
    quality = sc.reshape(sc.shape[0], -1).mean(axis=1)   # (M,)
    m = int(np.argmax(quality))
    return kp[m], sc[m]


def resample_frames(kp, target_T=90):
    """kp: (3, T, V) -> (3, target_T, V) by linear interpolation over time."""
    return transforms.resample_length(kp, out_len=target_T)


def anno_to_window(anno, target_T=90):
    kp, sc = select_primary_person(anno["keypoint"], anno["keypoint_score"])
    # kp: (T,V,2) -> stack x,y,conf into (3,T,V)
    x = torch.from_numpy(kp[..., 0]).float()
    y = torch.from_numpy(kp[..., 1]).float()
    conf = torch.from_numpy(sc).float()
    win = torch.stack([x, y, conf], dim=0)            # (3, T, V)
    win = resample_frames(win, target_T=target_T)     # (3, 90, V)
    return prenormalize_2d(win)


def load_split(pkl_path, split="xsub_train", limit=None):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    wanted = set(data["split"][split])
    annos = [a for a in data["annotations"] if a["frame_dir"] in wanted]
    annos.sort(key=lambda a: a["frame_dir"])          # deterministic order
    if limit is not None:
        annos = annos[:limit]
    return annos


def subject_holdout(annos, val_frac=0.1, seed=42):
    subjects = sorted({parse_subject(a["frame_dir"]) for a in annos})
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_frac))) if len(subjects) > 1 else 0
    val_subj = set(subjects[:n_val])
    tr = [a for a in annos if parse_subject(a["frame_dir"]) not in val_subj]
    va = [a for a in annos if parse_subject(a["frame_dir"]) in val_subj]
    return tr, va


class NTUDataset(Dataset):
    def __init__(self, annos, target_T=90):
        self.annos = annos
        self.target_T = target_T

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        a = self.annos[idx]
        return anno_to_window(a, target_T=self.target_T), int(a["label"])
