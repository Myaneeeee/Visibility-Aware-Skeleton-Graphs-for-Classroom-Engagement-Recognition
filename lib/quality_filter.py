"""Skeleton-window quality metrics + usability decision for pre-annotation filtering."""
import numpy as np

CONF_JOINT_THRESHOLD = 0.5


def _to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def pose_quality_metrics(keypoints, bboxes):
    """keypoints: (3,T,17) = x,y,conf ; bboxes: (T,4) = cx,cy,w,h.
    T is the clip's native frame count (54 for an 18fps 3s clip), not a fixed 90.
    Returns mean_conf, frac_conf_joints, and bbox-normalized temporal jitter.

    Jitter is per frame-to-frame step, so it is comparable across clip lengths --
    but only at equal fps. An 18fps clip steps 1/18s per frame where a 30fps clip
    steps 1/30s, so the same motion reads as more jitter at the lower fps."""
    kp = _to_np(keypoints).astype(np.float32)
    bb = _to_np(bboxes).astype(np.float32)
    x, y, c = kp[0], kp[1], kp[2]                      # each (T,17)

    mean_conf = float(c.mean())
    frac_conf_joints = float((c > CONF_JOINT_THRESHOLD).mean())

    # per-frame scale = max(bbox w, h), guards against tiny/degenerate boxes
    scale = np.maximum(bb[:, 2], bb[:, 3])
    scale = np.clip(scale, 1.0, None)                  # (T,)

    # frame-to-frame displacement of confident joints, normalized by scale
    dx = np.diff(x, axis=0)                            # (T-1,17)
    dy = np.diff(y, axis=0)
    dist = np.sqrt(dx * dx + dy * dy)                  # (T-1,17)
    both_conf = (c[:-1] > CONF_JOINT_THRESHOLD) & (c[1:] > CONF_JOINT_THRESHOLD)
    norm_dist = dist / scale[1:, None]
    jitter = float(norm_dist[both_conf].mean()) if both_conf.any() else 1.0

    return {"mean_conf": mean_conf, "frac_conf_joints": frac_conf_joints, "jitter": jitter}


def is_usable(metrics, min_mean_conf=0.35, min_frac_conf=0.40, max_jitter=0.25):
    return (metrics["mean_conf"] >= min_mean_conf
            and metrics["frac_conf_joints"] >= min_frac_conf
            and metrics["jitter"] <= max_jitter)
