import pickle
import numpy as np
import torch
import pytest
import ntu_dataset as nd


def _anno(frame_dir, label, T, kp=None, score=None):
    if kp is None:
        kp = (np.random.rand(1, T, 17, 2) * 1000).astype(np.float32)
    if score is None:
        score = (np.random.rand(kp.shape[0], T, 17) * 0.4 + 0.5).astype(np.float32)
    return {"frame_dir": frame_dir, "label": label, "img_shape": (1080, 1920),
            "total_frames": T, "keypoint": kp, "keypoint_score": score}


@pytest.fixture
def fake_pkl(tmp_path):
    data = {
        "split": {
            "xsub_train": ["S001C001P001R001A001", "S001C001P002R001A005"],
            "xsub_val": ["S001C001P003R001A010"],
        },
        "annotations": [
            _anno("S001C001P001R001A001", 0, 50),
            _anno("S001C001P002R001A005", 4, 120),
            _anno("S001C001P003R001A010", 9, 90),
        ],
    }
    p = tmp_path / "ntu60_hrnet_fake.pkl"
    pickle.dump(data, open(p, "wb"))
    return str(p)


def test_parse_subject():
    assert nd.parse_subject("S001C001P017R002A033") == 17


def test_select_primary_person_prefers_confident():
    T = 8
    kp = np.zeros((2, T, 17, 2), dtype=np.float32)
    kp[1] = np.random.rand(T, 17, 2) * 500          # person 1 real
    score = np.zeros((2, T, 17), dtype=np.float32)
    score[1] = 0.9                                   # person 0 padded (score 0)
    p_kp, p_score = nd.select_primary_person(kp, score)
    assert p_kp.shape == (T, 17, 2)
    assert float(p_score.mean()) > 0.5               # picked the real person


def test_resample_frames_changes_length_only():
    kp = torch.rand(3, 50, 17)
    out = nd.resample_frames(kp, target_T=90)
    assert out.shape == (3, 90, 17)
    down = nd.resample_frames(kp, target_T=25)
    assert down.shape == (3, 25, 17)


def test_anno_to_window_shape_and_norm():
    a = _anno("S001C001P001R001A001", 0, 37)
    w = nd.anno_to_window(a)
    assert w.shape == (3, 90, 17)
    conf = w[2] > 0.01
    assert float(w[0][conf].abs().max()) <= 1.0 + 1e-4


def test_load_split_filters_and_limits(fake_pkl):
    annos = nd.load_split(fake_pkl, split="xsub_train")
    assert len(annos) == 2
    assert {a["frame_dir"] for a in annos} == {
        "S001C001P001R001A001", "S001C001P002R001A005"}
    assert len(nd.load_split(fake_pkl, split="xsub_train", limit=1)) == 1


def test_subject_holdout_disjoint(fake_pkl):
    annos = nd.load_split(fake_pkl, split="xsub_train")
    tr, va = nd.subject_holdout(annos, val_frac=0.5, seed=0)
    tr_subj = {nd.parse_subject(a["frame_dir"]) for a in tr}
    va_subj = {nd.parse_subject(a["frame_dir"]) for a in va}
    assert tr_subj.isdisjoint(va_subj)
    assert len(tr) > 0 and len(va) > 0            # neither split is empty
    assert len(tr) + len(va) == len(annos)        # no annotations dropped or duplicated


def test_dataset_getitem(fake_pkl):
    annos = nd.load_split(fake_pkl, split="xsub_train")
    ds = nd.NTUDataset(annos)
    x, y = ds[0]
    assert x.shape == (3, 90, 17) and isinstance(int(y), int)
    assert len(ds) == 2
