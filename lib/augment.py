"""On-the-fly skeleton augmentation for config C.

Every transform here runs on a window that common.prenormalize_2d has ALREADY
normalized (it is applied at load time, lib/common.py:73). Two consequences
shape this module:

* Masked joints sit at exactly (0, 0). Rotation and flip both map (0, 0) to
  (0, 0), so the confidence mask survives them unchanged. Any transform added
  later must preserve that property -- reviving a masked joint invents a
  coordinate where the pipeline recorded "unknown".
* Translation and isotropic scale are deliberately absent. Normalization removes
  both deterministically, so they cannot vary at test time and augmenting them
  buys nothing. See the design doc, Sec. 2.2.

Magnitudes live in DEFAULT_AUGMENT and are pre-registered (design doc Sec. 2.3).
They are derived from measured between-video geometry, NOT tuned -- tuning them
on the reported folds would be tuning on the test set.
"""
import math
import random

import torch
import torch.utils.data

import transforms

# COCO-17 left/right mirror. Index i holds the joint that lands at position i
# after a horizontal flip: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows,
# 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles.
COCO_FLIP_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def rotate(window, degrees):
    """Rotate x,y about the origin by `degrees`. window: (3, T, V) -> (3, T, V).

    Windows are already centered by prenormalize_2d, so the origin is the body
    center and no translation is needed before rotating.
    """
    theta = math.radians(degrees)
    cos, sin = math.cos(theta), math.sin(theta)
    out = window.clone()
    x, y = window[0], window[1]
    out[0] = x * cos - y * sin
    out[1] = x * sin + y * cos
    return out


def flip_perm_for(joints):
    """COCO_FLIP_PERM reindexed onto a subset of COCO-17 joints.

    Raises if a joint's mirror was dropped from the subset: with no partner to
    swap onto, a flip would leave a left joint sitting on the body's right.
    """
    pos = {j: i for i, j in enumerate(joints)}
    orphans = [j for j in joints if COCO_FLIP_PERM[j] not in pos]
    if orphans:
        raise ValueError(
            f"joints {orphans} have no mirror in the subset {list(joints)}, "
            "so a horizontal flip is not defined on it")
    return [pos[COCO_FLIP_PERM[j]] for j in joints]


def hflip(window, flip_perm=None):
    """Mirror horizontally: negate x AND swap left/right joint indices.

    The index swap is not optional. Negating x alone puts the left shoulder on
    the body's right side, which is anatomically impossible and fails silently.
    `flip_perm` defaults to COCO-17; a subset graph needs flip_perm_for().
    """
    out = window.clone()
    out[0] = -out[0]
    return out[:, :, COCO_FLIP_PERM if flip_perm is None else flip_perm]


def temporal_crop(window, start, length):
    """Keep frames [start, start+length) and stretch them back over T frames.

    Simulates the clip's own boundary uncertainty. cut_clips.py established the
    clip-index -> timestamp mapping empirically and records a possible residual
    sub-second phase offset, so the 3s span's alignment to the labeled behavior
    is only approximately known. This trains the model not to depend on it.

    Randomness is the caller's (SkeletonAugment), so this stays pure and testable.
    """
    T = window.shape[1]
    cropped = window[:, start:start + length, :]
    return transforms.resample_length(cropped, out_len=T)


# Pre-registered magnitudes (design doc Sec. 2.3). Rotation matches the measured
# between-video spread of torso angle (videos span -11.8 to +8.3 degrees, so ~+/-10)
# with ~1.5x margin. The crop floor of 0.8 clears the 2/3 needed to retain SAV's
# keyframe at frame 30 of 90 under every draw. Fixed BEFORE any C result exists.
DEFAULT_AUGMENT = {"max_rotation_deg": 15.0, "flip_prob": 0.5, "min_crop_ratio": 0.8}


class SkeletonAugment:
    """Draws random parameters and applies rotate -> hflip -> temporal_crop.

    Owns its own RNG so a run is reproducible independently of torch's global
    state (the DataLoader's shuffling consumes that one).
    """

    def __init__(self, max_rotation_deg=15.0, flip_prob=0.5, min_crop_ratio=0.8,
                 seed=0, flip_perm=None):
        self.max_rotation_deg = max_rotation_deg
        self.flip_prob = flip_prob
        self.min_crop_ratio = min_crop_ratio
        self.flip_perm = flip_perm
        self.rng = random.Random(seed)

    def _draw_crop_ratio(self):
        return self.rng.uniform(self.min_crop_ratio, 1.0)

    def __call__(self, window):
        # One angle for the whole window: a per-frame angle would simulate a
        # shaking camera rather than a differently-mounted one.
        out = rotate(window, self.rng.uniform(-self.max_rotation_deg,
                                              self.max_rotation_deg))
        if self.rng.random() < self.flip_prob:
            out = hflip(out, flip_perm=self.flip_perm)
        T = window.shape[1]
        length = max(1, int(round(T * self._draw_crop_ratio())))
        start = self.rng.randint(0, T - length)
        return temporal_crop(out, start, length)


class AugmentedTensorDataset(torch.utils.data.Dataset):
    """TensorDataset that applies `augment`, then `transform`, per __getitem__.

    Per-item rather than precomputed, so every epoch sees fresh randomness. A
    precomputed enlarged dataset is a materially weaker augmentation.

    `transform` runs last and is not random: it is the input representation
    (rung E3's bone vectors), which must be derived from the augmented
    coordinates rather than augmented itself.
    """

    def __init__(self, X, y, augment=None, transform=None):
        self.X = X
        self.y = y
        self.augment = augment
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment is not None:
            x = self.augment(x)
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[i]
