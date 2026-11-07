import numpy as np
import quality_filter as qf

def _stable_window(conf=0.9):
    # a still, high-confidence skeleton at a fixed pose
    kp = np.zeros((3, 90, 17), dtype=np.float32)
    base = np.random.default_rng(0).uniform(100, 300, size=(2, 17)).astype(np.float32)
    kp[0] = base[0][None, :]      # x constant over time
    kp[1] = base[1][None, :]      # y constant over time
    kp[2] = conf
    bboxes = np.tile([200, 200, 120, 200], (90, 1)).astype(np.float32)
    return kp, bboxes

def test_stable_high_conf_window_is_usable():
    kp, bb = _stable_window(conf=0.9)
    m = qf.pose_quality_metrics(kp, bb)
    assert m["mean_conf"] > 0.8
    assert m["jitter"] < 0.05
    assert qf.is_usable(m) is True

def test_low_confidence_window_is_dropped():
    kp, bb = _stable_window(conf=0.1)
    m = qf.pose_quality_metrics(kp, bb)
    assert qf.is_usable(m) is False

def test_jittery_window_is_dropped():
    rng = np.random.default_rng(1)
    kp = np.zeros((3, 90, 17), dtype=np.float32)
    kp[0] = rng.uniform(100, 300, size=(90, 17))   # x jumps every frame
    kp[1] = rng.uniform(100, 300, size=(90, 17))   # y jumps every frame
    kp[2] = 0.9
    bb = np.tile([200, 200, 120, 200], (90, 1)).astype(np.float32)
    m = qf.pose_quality_metrics(kp, bb)
    assert m["jitter"] > 0.25
    assert qf.is_usable(m) is False
