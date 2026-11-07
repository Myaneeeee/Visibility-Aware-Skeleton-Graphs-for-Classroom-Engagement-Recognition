"""Hand-built skeleton feature families for the representation probes.

Each family maps one (3, T, 17) window to a flat float32 vector. The definitions
match analysis/representation_sweep.py, so a trained MLP on a family is comparable to
the linear probe on the same family.
"""
import numpy as np

import baselines as bl
from model import OBSERVED_JOINTS_13

BONES_17 = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8),
            (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
            (14, 16), (5, 6)]

FEATURE_DIMS = {"marg85": 85, "marg65": 65, "dist272": 272, "dist156": 156,
                "bone144": 144}


def _marginals(xy, joints):
    sub = xy[:, :, joints]
    speed = np.sqrt((np.diff(sub, axis=1) ** 2).sum(0))
    return np.concatenate([sub.mean(1).ravel(), sub.std(1).ravel(), speed.mean(0)])


def _distances(xy, joints):
    sub = xy[:, :, joints]
    dm = np.sqrt(((sub[:, :, :, None] - sub[:, :, None, :]) ** 2).sum(0))
    iu = np.triu_indices(len(joints), k=1)
    return np.concatenate([dm.mean(0)[iu], dm.std(0)[iu]])


def _bones(xy):
    bv = np.stack([xy[:, :, b] - xy[:, :, a] for a, b in BONES_17], -1)
    length = np.sqrt((bv ** 2).sum(0))
    angle = np.arctan2(bv[1], bv[0])
    return np.concatenate([bv.mean(1).ravel(), bv.std(1).ravel(),
                           length.mean(0), length.std(0),
                           np.cos(angle).mean(0), np.sin(angle).mean(0)])


def features(name, window):
    """Feature vector of family `name` for one (3, T, 17) window."""
    if name == "marg85":
        return bl.pose_stat_features(window)
    xy = window[:2].numpy().astype(np.float64)
    if name == "marg65":
        vec = _marginals(xy, OBSERVED_JOINTS_13)
    elif name == "dist272":
        vec = _distances(xy, list(range(17)))
    elif name == "dist156":
        vec = _distances(xy, OBSERVED_JOINTS_13)
    elif name == "bone144":
        vec = _bones(xy)
    else:
        raise ValueError(f"unknown feature family {name!r}; "
                         f"expected one of {sorted(FEATURE_DIMS)}")
    return np.nan_to_num(vec.astype(np.float32))
