import re

import numpy as np
from sklearn.model_selection import GroupKFold

# Captures the source video id from either a clip id ('20230118_34_136') or a
# source id ('20230118_34'). The optional trailing group is the clip index.
_SAV_NAME_RE = re.compile(r"^(\d{8}_[^_]+)(?:_\d+)?$")

# Below this many source videos, k=5 folds are too small for the fold spread to mean
# anything. Fix by downloading more of the shortlist, not by loosening the grouping.
MIN_GROUPS_FOR_STABLE_CV = 15


def cv_group_key(video_name):
    """Reduce a clip id to its source video id, the CV grouping unit.

    Raises ValueError on an unrecognized name: a silent collapse to one group would
    degrade GroupKFold to a single fold with nothing in the output to reveal it.
    """
    match = _SAV_NAME_RE.match(str(video_name))
    if not match:
        raise ValueError(
            f"cannot derive a CV video group from video_name {video_name!r}: expected "
            "a SAV clip id like '20230118_34_136' or a source id like '20230118_34'.")
    return match.group(1)


def grouping_report(groups, k=5):
    """Summarize the CV grouping. Returns (text, ok_to_report)."""
    n_groups = len(set(groups))
    folds = min(k, n_groups)
    text = (f"CV grouping: {len(groups)} windows across {n_groups} source videos "
            f"-> {folds} folds (~{n_groups / folds:.1f} videos per fold)")
    ok = n_groups >= MIN_GROUPS_FOR_STABLE_CV
    if not ok:
        text += (f"\n  WARNING: too few videos ({n_groups} < {MIN_GROUPS_FOR_STABLE_CV})"
                 " for a stable cross-video CV.")
    return text, ok


def source_video_folds(video_names, k=5):
    """GroupKFold over source videos: clips of one recording never straddle a fold."""
    return group_kfold_indices([cv_group_key(v) for v in video_names], k=k)


def group_kfold_indices(video_names, k=5):
    groups = np.asarray(video_names)
    n_groups = len(set(groups.tolist()))
    gkf = GroupKFold(n_splits=min(k, n_groups))
    return [(tr, te) for tr, te in gkf.split(np.zeros((len(groups), 1)), groups=groups)]
