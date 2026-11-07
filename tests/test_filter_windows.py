import numpy as np
import torch
import filter_windows as fsw

def _win(conf):
    kp = np.zeros((3, 90, 17), dtype=np.float32)
    base = np.random.default_rng(0).uniform(100, 300, size=(2, 17)).astype(np.float32)
    kp[0] = base[0][None, :]; kp[1] = base[1][None, :]; kp[2] = conf
    return {"keypoints": kp, "bboxes": np.tile([200, 200, 120, 200], (90, 1)).astype(np.float32)}

def test_filter_keeps_good_drops_bad():
    kept, stats = fsw.filter_windows([_win(0.9), _win(0.05), _win(0.9)])
    assert stats == {"total": 3, "kept": 2, "dropped": 1}
    assert len(kept) == 2

def test_filter_keeps_good_window_with_torch_tensor_keypoints_and_list_bboxes():
    # Real p2 pipeline output: keypoints is a torch.Tensor, bboxes is a Python list.
    base = np.random.default_rng(0).uniform(100, 300, size=(2, 17)).astype(np.float32)
    kp = np.zeros((3, 90, 17), dtype=np.float32)
    kp[0] = base[0][None, :]; kp[1] = base[1][None, :]; kp[2] = 0.9
    keypoints = torch.tensor(kp)
    bboxes = [[200.0, 200.0, 120.0, 200.0] for _ in range(90)]

    win = {"keypoints": keypoints, "bboxes": bboxes}
    kept, stats = fsw.filter_windows([win])

    assert stats["kept"] == 1
    assert len(kept) == 1
